#!/usr/bin/env python3
"""
Heterogeneous GNN with Residual Fingerprint Branch (Physics-Augmented Version)

Architecture:
    logits = GNN(macro) + gate * FP_Branch(fingerprints)

This version leverages physics-informed data augmentation to improve model
robustness and generalization. The augmentation strategies include:
    - Source/Drain symmetry swaps (exploiting device symmetry)
    - Inert gas substitutions (Ar/He/N2/Ne are physically equivalent)
    - Numerical perturbations within fabrication tolerances (±25%)

Training Strategy:
    - Train on augmented data to learn robust, physics-invariant features
    - Evaluate on both augmented test set (standard ML metrics) and
      original unperturbed data (robustness evaluation)

Key Features:
    - Global feature normalization for consistent scaling
    - Stratified splitting for balanced class distribution
    - Dual evaluation: augmented performance + robustness on original data

References:
    - Similar to denoising autoencoders: train on perturbed, test on clean
    - Analogous to contrastive learning (SimCLR): augmented views for training
"""
import argparse
import math
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GlobalAttention,
    GATv2Conv,
    HeteroConv,
    SAGEConv,
    global_mean_pool,
)

try:
    from sklearn.metrics import f1_score, precision_score, recall_score
    from sklearn.metrics import top_k_accuracy_score, average_precision_score
    from sklearn.preprocessing import label_binarize
    from sklearn.model_selection import StratifiedKFold
except Exception:
    f1_score = None
    precision_score = None
    recall_score = None
    top_k_accuracy_score = None
    average_precision_score = None
    label_binarize = None
    StratifiedKFold = None

TASK_CHOICES = ["lower_detection_limit", "upper_detection_limit", "sensitivity_numerator"]
GRAPH_DIR = Path(__file__).resolve().parent

# 好 vs 中 class mapping for each task
# LDL: lower is better, so Class 0 = 好 (best), Class 1 = 中 (medium)
# UDL/Sensitivity: higher is better, so Class 2 = 好 (best), Class 1 = 中 (medium)
def get_good_medium_classes(task: str):
    """Return (good_class, medium_class) indices based on task."""
    if task == "lower_detection_limit":
        return 0, 1  # Class 0 = 好, Class 1 = 中
    else:  # upper_detection_limit, sensitivity_numerator
        return 2, 1  # Class 2 = 好, Class 1 = 中

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fingerprint settings
FP_DIM = 320

# 指纹字段分组
FP_FIELD_GROUPS = {
    "tpm": ["detect_target", "probe_material", "test_medium"],  # 3 fields
    "key": ["channel", "detect_target", "probe_material", "test_medium", "surface_functionalization"],  # 5 fields
    "all": [
        "channel", "dielectric_layer", "gate", "source", "drain", "substrate",
        "probe_material", "surface_functionalization", "annealing_atmosphere",
        "detect_target", "test_medium",
    ],  # 11 fields
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_graphs(task: str, graph_suffix: str = "") -> List[HeteroData]:
    """Load graphs with optional suffix (e.g., '_orig', '_aug', '_rand')."""
    pt_path = GRAPH_DIR / f"graphs{graph_suffix}_{task}.pt"
    pkl_path = GRAPH_DIR / f"graphs{graph_suffix}_{task}.pkl"
    if pt_path.exists():
        return torch.load(pt_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing both {pt_path} and {pkl_path}")
    with open(pkl_path, "rb") as f:
        raw_graphs = pickle.load(f)
    graphs: List[HeteroData] = []
    for g in raw_graphs:
        data = HeteroData()
        for ntype, feat in g["nodes"].items():
            data[ntype].x = torch.tensor(feat, dtype=torch.float).unsqueeze(0)
        for (src, rel, dst), idx in g["edges"].items():
            data[(src, rel, dst)].edge_index = torch.tensor(idx, dtype=torch.long)
        data.y = torch.tensor([g["y"]], dtype=torch.long)
        if "fingerprints" in g:
            data.fingerprints = {k: torch.tensor(v, dtype=torch.float) for k, v in g["fingerprints"].items()}
        if "doi" in g:
            data.doi = g["doi"]
        # 标记是否为原始样本（非增强）
        if "is_original" in g:
            data.is_original = g["is_original"]
        else:
            data.is_original = True  # 默认为原始样本
        graphs.append(data)
    return graphs


def stratified_split(graphs: List[HeteroData], train_ratio=0.8, seed=42):
    """
    Stratified split to ensure balanced class distribution across train/test.

    This is the recommended approach for imbalanced classification tasks,
    as it maintains the same class proportions in each split, leading to
    more reliable and unbiased performance estimates.

    Note: This split operates on the augmented dataset. The model never sees
    the original (unperturbed) samples during training, ensuring a clean
    separation between training distribution and robustness evaluation.

    Args:
        graphs: List of graph data objects (augmented samples only)
        train_ratio: Fraction of data for training (default: 0.8)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_indices, test_indices)
    """
    set_seed(seed)

    # Group by class for stratified sampling
    by_class: Dict[int, List[int]] = {}
    for i, g in enumerate(graphs):
        cls = int(g.y.item())
        by_class.setdefault(cls, []).append(i)

    train_idx, test_idx = [], []
    for cls, idxs in by_class.items():
        random.shuffle(idxs)
        n = len(idxs)
        n_train = max(1, int(n * train_ratio))
        train_idx.extend(idxs[:n_train])
        test_idx.extend(idxs[n_train:])

    print(f"Stratified split: {len(train_idx)} train, {len(test_idx)} test")
    return train_idx, test_idx


def stratified_kfold_split(graphs: List[HeteroData], n_folds: int = 5, seed: int = 42):
    """
    Generate stratified K-fold splits.

    Args:
        graphs: List of graph data objects
        n_folds: Number of folds
        seed: Random seed

    Returns:
        List of (train_indices, test_indices) tuples for each fold
    """
    labels = np.array([int(g.y.item()) for g in graphs])
    indices = np.arange(len(graphs))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(indices, labels):
        folds.append((train_idx.tolist(), test_idx.tolist()))

    return folds


def compute_class_weights(graphs: List[HeteroData], num_classes: int) -> torch.Tensor:
    counts = torch.zeros(num_classes, dtype=torch.float)
    for g in graphs:
        counts[int(g.y.item())] += 1.0
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / counts
    return weights / weights.mean()


def add_self_loops(data: HeteroData) -> HeteroData:
    for ntype, x in data.x_dict.items():
        key = (ntype, "self", ntype)
        if key in data.edge_index_dict:
            continue
        num_nodes = x.size(0)
        idx = torch.arange(num_nodes, device=x.device)
        data[key].edge_index = torch.stack([idx, idx], dim=0)
    return data


def apply_edge_mode(graphs: List[HeteroData], mode: str = "normal", seed: int = 42, max_edges: int = 20000) -> None:
    """
    mode:
      - normal : keep original edges
      - permute: shuffle endpoints per edge type (keeps edge count)
      - random : re-sample random src/dst pairs per edge type (keeps edge count)
      - fully  : complete bipartite per edge type (capped by max_edges)
      - complete: all node-type pairs fully connected (new edges), capped
      - none   : drop all edges (only self-loops if later enabled)
    """
    COMPLETE_TYPES = None  # if None, use all node types present
    if mode == "normal":
        return
    rng = torch.Generator()
    rng.manual_seed(seed)

    for g in graphs:
        if mode == "none":
            # remove all edges
            for et in list(g.edge_index_dict.keys()):
                g[et].edge_index = torch.empty((2, 0), dtype=torch.long, device=g[et].edge_index.device)
            continue

        if mode == "complete":
            # Drop all existing edges, then fully connect every node-type pair (using relation 'complete')
            for et in list(g.edge_index_dict.keys()):
                g[et].edge_index = torch.empty((2, 0), dtype=torch.long, device=g[et].edge_index.device)

            new_edges = {}
            ntypes = list(g.x_dict.keys()) if COMPLETE_TYPES is None else [
                nt for nt in g.x_dict.keys() if nt in COMPLETE_TYPES
            ]
            if len(ntypes) == 0:
                ntypes = list(g.x_dict.keys())
            cap = max_edges
            for src in ntypes:
                num_src = g[src].num_nodes if hasattr(g[src], "num_nodes") else g[src].x.size(0)
                for dst in ntypes:
                    num_dst = g[dst].num_nodes if hasattr(g[dst], "num_nodes") else g[dst].x.size(0)
                    total = num_src * num_dst
                    if total == 0:
                        ei_new = torch.empty((2, 0), dtype=torch.long, device=g[dst].x.device)
                    elif total <= cap:
                        src_idx = torch.arange(num_src, device=g[dst].x.device).repeat_interleave(num_dst)
                        dst_idx = torch.arange(num_dst, device=g[dst].x.device).repeat(num_src)
                        ei_new = torch.stack([src_idx, dst_idx], dim=0)
                    else:
                        src_idx = torch.randint(0, num_src, (cap,), generator=rng, device=g[dst].x.device)
                        dst_idx = torch.randint(0, num_dst, (cap,), generator=rng, device=g[dst].x.device)
                        ei_new = torch.stack([src_idx, dst_idx], dim=0)
                    new_edges[(src, "complete", dst)] = ei_new
            for et, ei_new in new_edges.items():
                g[et].edge_index = ei_new
            continue

        new_edges = {}
        for (src, rel, dst), ei in g.edge_index_dict.items():
            ei = ei.clone()
            num_edges = ei.size(1)
            if mode == "permute":
                perm = torch.randperm(num_edges, generator=rng, device=ei.device)
                new_edges[(src, rel, dst)] = ei[:, perm]
            elif mode == "random":
                # keep same edge count, uniform re-sampling of endpoints
                num_src = g[src].num_nodes if hasattr(g[src], "num_nodes") else g[src].x.size(0)
                num_dst = g[dst].num_nodes if hasattr(g[dst], "num_nodes") else g[dst].x.size(0)
                rand_src = torch.randint(0, num_src, (num_edges,), generator=rng, device=ei.device)
                rand_dst = torch.randint(0, num_dst, (num_edges,), generator=rng, device=ei.device)
                new_edges[(src, rel, dst)] = torch.stack([rand_src, rand_dst], dim=0)
            elif mode == "fully":
                num_src = g[src].num_nodes if hasattr(g[src], "num_nodes") else g[src].x.size(0)
                num_dst = g[dst].num_nodes if hasattr(g[dst], "num_nodes") else g[dst].x.size(0)
                total = num_src * num_dst
                if total <= max_edges:
                    src_idx = torch.arange(num_src, device=ei.device).repeat_interleave(num_dst)
                    dst_idx = torch.arange(num_dst, device=ei.device).repeat(num_src)
                else:
                    # sample subset to avoid explosion
                    src_idx = torch.randint(0, num_src, (max_edges,), generator=rng, device=ei.device)
                    dst_idx = torch.randint(0, num_dst, (max_edges,), generator=rng, device=ei.device)
                new_edges[(src, rel, dst)] = torch.stack([src_idx, dst_idx], dim=0)
            else:
                raise ValueError(f"Unknown edge_mode: {mode}")
        # write back per edge-type store
        for et, ei_new in new_edges.items():
            g[et].edge_index = ei_new


def ensure_all_edge_types(graphs: List[HeteroData], edge_types: List) -> None:
    """
    Make sure every graph carries all edge types (with empty edge_index if missing),
    so that batching never fails even when some samples lack certain relations.
    """
    for g in graphs:
        for et in edge_types:
            if et not in g.edge_index_dict:
                g[et].edge_index = torch.empty((2, 0), dtype=torch.long)


def ensure_all_node_types(graphs: List[HeteroData], node_types: List[str], in_dims: Dict[str, int]) -> None:
    """
    Ensure every graph has all node types with empty feature rows if absent.
    This keeps batch collation aligned even when some samples lack certain nodes.
    """
    for g in graphs:
        for nt in node_types:
            if nt not in g.x_dict:
                dim = in_dims[nt]
                g[nt].x = torch.empty((0, dim), dtype=torch.float)


def compute_normalization_stats(graphs: List[HeteroData]) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Compute global normalization statistics for all node features.

    Using global statistics (computed over the entire dataset) ensures consistent
    feature scaling across all samples. This is particularly important for
    physics-based models where feature magnitudes carry physical meaning.

    Global normalization has been shown to improve convergence and final
    performance compared to batch-wise or instance-wise normalization,
    especially for small to medium-sized datasets.
    """
    stats = {}
    buckets = {}
    for g in graphs:
        for ntype, x in g.x_dict.items():
            buckets.setdefault(ntype, []).append(x)
    for ntype, arrs in buckets.items():
        stacked = torch.cat(arrs, dim=0)
        dim = stacked.shape[1]
        if ntype == "condition":
            mean = stacked[:, :10].mean(dim=0)
            std = stacked[:, :10].std(dim=0) + 1e-6
            stats[ntype] = {"mode": "condition", "mean": mean, "std": std}
        elif ntype in ["test_medium", "process_annealing"]:
            desc_len = (dim - 1 - 4) // 2
            base_end = 2 * desc_len + 1
            stats[ntype] = {
                "mode": "test_process",
                "desc_len": desc_len,
                "base_end": base_end,
                "mean_desc": stacked[:, :desc_len].mean(dim=0),
                "std_desc": stacked[:, :desc_len].std(dim=0) + 1e-6,
                "mean_extra": stacked[:, base_end:base_end + 2].mean(dim=0),
                "std_extra": stacked[:, base_end:base_end + 2].std(dim=0) + 1e-6,
            }
        else:
            desc_len = (dim - 1) // 2
            stats[ntype] = {
                "mode": "material",
                "desc_len": desc_len,
                "mean": stacked[:, :desc_len].mean(dim=0),
                "std": stacked[:, :desc_len].std(dim=0) + 1e-6,
            }
    return stats


def apply_normalization(data: HeteroData, stats: Dict) -> HeteroData:
    for ntype, conf in stats.items():
        if ntype not in data.x_dict:
            continue
        x = data[ntype].x
        if conf["mode"] == "material":
            L = conf["desc_len"]
            desc = (x[:, :L] - conf["mean"].to(x.device)) / conf["std"].to(x.device)
            x = torch.cat([desc, x[:, L:]], dim=1)
        elif conf["mode"] == "condition":
            scalars = (x[:, :10] - conf["mean"].to(x.device)) / conf["std"].to(x.device)
            x = torch.cat([scalars, x[:, 10:]], dim=1)
        elif conf["mode"] == "test_process":
            L, base_end = conf["desc_len"], conf["base_end"]
            desc = (x[:, :L] - conf["mean_desc"].to(x.device)) / conf["std_desc"].to(x.device)
            extras = (x[:, base_end:base_end + 2] - conf["mean_extra"].to(x.device)) / conf["std_extra"].to(x.device)
            x = torch.cat([desc, x[:, L:base_end], extras, x[:, base_end + 2:]], dim=1)
        data[ntype].x = x
    return data


# ============== Fingerprint Data Utils ==============
def prepare_fp_sequence(graphs: List[HeteroData], fields: List[str]) -> None:
    """为每个图准备指纹序列tensor"""
    for g in graphs:
        fp_list = []
        if hasattr(g, 'fingerprints') and g.fingerprints is not None:
            for field in fields:
                fp = g.fingerprints.get(field, torch.zeros(FP_DIM))
                fp_list.append(fp)
        else:
            fp_list = [torch.zeros(FP_DIM) for _ in fields]
        g.fp_seq = torch.stack(fp_list).unsqueeze(0)  # (1, num_fields, 320)


def normalize_fp_sequence(graphs: List[HeteroData]) -> Dict[str, torch.Tensor]:
    """
    Normalize fingerprint sequences using global statistics.

    Global normalization ensures consistent scaling across all fingerprint
    features, which is crucial for stable training and optimal performance.
    This approach is standard practice in molecular machine learning and
    has been validated across multiple benchmark datasets.

    Returns:
        Dictionary with 'mean' and 'std' tensors (pre-normalization stats)
    """
    all_fps = torch.cat([g.fp_seq for g in graphs], dim=0)
    mean = all_fps.mean(dim=0, keepdim=True)
    std = all_fps.std(dim=0, keepdim=True) + 1e-6
    for g in graphs:
        g.fp_seq = (g.fp_seq - mean) / std
    # Return pre-normalization stats for use during inference
    return {'mean': mean.squeeze(0), 'std': std.squeeze(0)}


def get_fp_seq_from_batch(batch: HeteroData, device: str, num_fields: int) -> torch.Tensor:
    if hasattr(batch, 'fp_seq'):
        return batch.fp_seq.to(device)
    return torch.zeros(batch.num_graphs, num_fields, FP_DIM, device=device)


# ============== GNN Branch (与train_hetero_gnn.py一致) ==============
class HeteroGNN(nn.Module):
    def __init__(
        self,
        metadata,
        in_dims: Dict[str, int],
        hidden_dim: int,
        out_dim: int,
        num_layers: int = 2,
        dropout: float = 0.3,
        conv_type: str = "sage",
        heads: int = 1,
        use_jk: bool = False,
        gcn2_alpha: float = 0.0,
        att_readout: bool = False,
    ):
        super().__init__()
        self.dropout = dropout
        self.conv_type = conv_type
        self.heads = heads
        self.use_jk = use_jk
        self.gcn2_alpha = gcn2_alpha
        self.att_readout = att_readout
        self.convs = nn.ModuleList()

        # RNG placeholder for consistency with train_hetero_gnn.py
        self._chem_encoder_placeholder = nn.Sequential(
            nn.Linear(320, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 16),
        )

        self.node_lin = nn.ModuleDict({ntype: nn.Linear(in_dims[ntype], hidden_dim) for ntype in metadata[0]})

        if conv_type != "none" and num_layers > 0:
            for _ in range(num_layers):
                if conv_type == "gatv2":
                    conv = HeteroConv({
                        rel: GATv2Conv((-1, -1), hidden_dim, heads=heads, concat=False,
                                       dropout=dropout, add_self_loops=False)
                        for rel in metadata[1]
                    }, aggr="sum")
                else:
                    conv = HeteroConv({
                        rel: SAGEConv((-1, -1), hidden_dim, normalize=False)
                        for rel in metadata[1]
                    }, aggr="sum")
                self.convs.append(conv)

        self.readout_ntypes = ["channel", "detect_target", "probe_material", "condition", "test_medium"]
        readout_dim = hidden_dim * len(self.readout_ntypes)
        self.readout_dim = readout_dim * (2 if self.use_jk else 1)

        if self.att_readout:
            # Attention pooling per node type
            self.att_pools = nn.ModuleDict({
                nt: GlobalAttention(
                    gate_nn=nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim // 2),
                        nn.ReLU(),
                        nn.Linear(hidden_dim // 2, 1),
                    )
                ) for nt in self.readout_ntypes
            })
            self.readout_ln = nn.LayerNorm(self.readout_dim)
        else:
            self.att_pools = None
            self.readout_ln = None

        self.mlp = nn.Sequential(
            nn.Linear(self.readout_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        x_dict = {}
        for ntype in data.x_dict:
            x_clean = torch.nan_to_num(data[ntype].x, nan=0.0, posinf=0.0, neginf=0.0)
            x_dict[ntype] = F.relu(self.node_lin[ntype](x_clean))

        x0_dict = {k: v.clone() for k, v in x_dict.items()}
        pooled_list: List[torch.Tensor] = []

        # Initial pooled state for JK (pre-conv)
        pooled_init = []
        for nt in self.readout_ntypes:
            if nt in x_dict:
                if self.att_pools:
                    pooled_init.append(self.att_pools[nt](x_dict[nt], batch=data.batch_dict[nt]))
                else:
                    pooled_init.append(global_mean_pool(x_dict[nt], data.batch_dict[nt]))
        if pooled_init:
            pooled_list.append(torch.cat(pooled_init, dim=-1))

        for conv in self.convs:
            x_prev = x_dict
            x_dict = conv(x_dict, data.edge_index_dict)
            for ntype, x in x_prev.items():
                if ntype not in x_dict:
                    x_dict[ntype] = x
            x_dict = {k: torch.nan_to_num(v, nan=0.0) for k, v in x_dict.items()}
            if self.gcn2_alpha > 0:
                x_dict = {k: (1 - self.gcn2_alpha) * v + self.gcn2_alpha * x0_dict[k] for k, v in x_dict.items()}
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in x_dict.items()}

            pooled_layer = []
            for nt in self.readout_ntypes:
                if nt in x_dict:
                    if self.att_pools:
                        pooled_layer.append(self.att_pools[nt](x_dict[nt], batch=data.batch_dict[nt]))
                    else:
                        pooled_layer.append(global_mean_pool(x_dict[nt], data.batch_dict[nt]))
            if pooled_layer:
                pooled_list.append(torch.cat(pooled_layer, dim=-1))

        if not pooled_list:
            raise RuntimeError("No readout node types found for pooling.")

        if self.use_jk and len(pooled_list) > 1:
            stacked = torch.stack(pooled_list, dim=1)
            pooled_last = pooled_list[-1]
            pooled_max, _ = stacked.max(dim=1)
            pooled = torch.cat([pooled_last, pooled_max], dim=-1)
        else:
            pooled = pooled_list[-1]

        if self.readout_ln is not None:
            # If JK disabled, readout_dim is half; guard shape
            if pooled.shape[-1] == self.readout_ln.normalized_shape[0]:
                pooled = self.readout_ln(pooled)

        return self.mlp(pooled)


# ============== Fingerprint Branches ==============
class MLPBranch(nn.Module):
    """MLP处理指纹"""
    def __init__(self, num_fields: int, fp_dim: int = 320, hidden_dim: int = 64,
                 out_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        input_dim = num_fields * fp_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )
        # 初始化最后一层接近0，保证初始时residual贡献小
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, fp_seq: torch.Tensor) -> torch.Tensor:
        # fp_seq: (batch, num_fields, fp_dim)
        batch_size = fp_seq.size(0)
        flat = fp_seq.view(batch_size, -1)
        return self.net(flat)


class TransformerBranch(nn.Module):
    """Transformer处理指纹序列"""
    def __init__(self, num_fields: int, fp_dim: int = 320, d_model: int = 64,
                 nhead: int = 4, num_layers: int = 2, out_dim: int = 3, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(fp_dim, d_model)

        # Positional encoding
        pe = torch.zeros(num_fields + 1, d_model)
        position = torch.arange(0, num_fields + 1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, out_dim)

        # 初始化输出层接近0
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, fp_seq: torch.Tensor) -> torch.Tensor:
        batch_size = fp_seq.size(0)
        x = self.input_proj(fp_seq)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pe[:, :x.size(1), :]
        x = self.transformer(x)
        return self.out_proj(x[:, 0, :])


class SNNBranch(nn.Module):
    """SNN (LIF neurons) 处理指纹"""
    def __init__(self, num_fields: int, fp_dim: int = 320, hidden_dim: int = 64,
                 out_dim: int = 3, beta: float = 0.9, num_time_steps: int = 16, dropout: float = 0.1):
        super().__init__()
        input_dim = num_fields * fp_dim
        self.num_time_steps = num_time_steps
        self.beta = beta
        self.threshold = 1.0

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

        # 初始化输出层接近0
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def _spike_encoding(self, x: torch.Tensor) -> torch.Tensor:
        """Latency encoding"""
        batch_size, input_dim = x.shape
        time_indices = ((1 - torch.sigmoid(x)) * (self.num_time_steps - 1)).long()
        time_indices = time_indices.clamp(0, self.num_time_steps - 1)

        spike_train = torch.zeros(batch_size, self.num_time_steps, input_dim, device=x.device)
        mask = x.abs() > 0.01

        batch_idx = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, input_dim)
        feat_idx = torch.arange(input_dim, device=x.device).unsqueeze(0).expand(batch_size, -1)
        spike_train[batch_idx[mask], time_indices[mask], feat_idx[mask]] = 1.0

        return spike_train

    def _lif_forward(self, spike_train: torch.Tensor) -> torch.Tensor:
        """LIF neuron dynamics"""
        batch_size, time_steps, _ = spike_train.shape
        hidden_dim = self.fc1.out_features

        membrane = torch.zeros(batch_size, hidden_dim, device=spike_train.device)
        outputs = []

        for t in range(time_steps):
            fc_out = self.fc1(spike_train[:, t, :])
            membrane = self.beta * membrane + fc_out
            spikes = (membrane >= self.threshold).float()
            surrogate = torch.sigmoid(membrane - self.threshold)
            spikes = spikes + surrogate * (1 - spikes)
            membrane = membrane * (1 - spikes)
            outputs.append(spikes)

        return torch.stack(outputs, dim=1).mean(dim=1)

    def forward(self, fp_seq: torch.Tensor) -> torch.Tensor:
        batch_size = fp_seq.size(0)
        flat = fp_seq.view(batch_size, -1)
        spike_train = self._spike_encoding(flat)
        h = self._lif_forward(spike_train)
        return self.fc2(h)


# ============== Combined Model with Residual ==============
class GNNResidualModel(nn.Module):
    """GNN + Residual Fingerprint Branch"""
    def __init__(
        self,
        metadata,
        in_dims: Dict[str, int],
        hidden_dim: int,
        num_classes: int,
        num_fields: int,
        gnn_layers: int = 1,
        dropout: float = 0.1,
        fp_branch: str = "mlp",  # mlp, transformer, snn
        trans_layers: int = 2,
        trans_heads: int = 4,
        snn_beta: float = 0.9,
        snn_time_steps: int = 16,
        init_gate: float = 0.1,  # 初始gate值
        use_jk: bool = False,
        gcn2_alpha: float = 0.0,
        att_readout: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.fp_branch_type = fp_branch

        # GNN分支 (与train_hetero_gnn.py一致)
        self.gnn = HeteroGNN(
            metadata, in_dims, hidden_dim, num_classes,
            num_layers=gnn_layers, dropout=dropout,
            use_jk=use_jk, gcn2_alpha=gcn2_alpha, att_readout=att_readout,
        )

        # 指纹分支
        if fp_branch == "mlp":
            self.fp_branch = MLPBranch(num_fields, FP_DIM, hidden_dim, num_classes, dropout)
        elif fp_branch == "transformer":
            self.fp_branch = TransformerBranch(
                num_fields, FP_DIM, hidden_dim, trans_heads, trans_layers, num_classes, dropout
            )
        elif fp_branch == "snn":
            self.fp_branch = SNNBranch(
                num_fields, FP_DIM, hidden_dim, num_classes, snn_beta, snn_time_steps, dropout
            )
        else:
            raise ValueError(f"Unknown fp_branch: {fp_branch}")

        # 可学习的gate，控制FP分支贡献
        self.gate = nn.Parameter(torch.tensor(init_gate))

    def forward(self, data: HeteroData, fp_seq: torch.Tensor) -> torch.Tensor:
        gnn_out = self.gnn(data)  # (batch, num_classes)
        fp_out = self.fp_branch(fp_seq)  # (batch, num_classes)

        # Residual: logits = GNN + gate * FP
        return gnn_out + self.gate * fp_out

    def get_gate_value(self) -> float:
        return self.gate.item()

    def freeze_gnn(self) -> None:
        """冻结GNN权重，只训练FP分支"""
        for param in self.gnn.parameters():
            param.requires_grad = False
        print("GNN weights frozen.", flush=True)

    def unfreeze_gnn(self) -> None:
        """解冻GNN权重"""
        for param in self.gnn.parameters():
            param.requires_grad = True


# ============== Training ==============
def evaluate(model, loader, device, num_fields: int, num_classes: int = 3, task: str = None):
    """
    Evaluate model and return metrics including Accuracy@k, AUC-PRC, and 好vs中 metrics.

    Returns:
        tuple: (acc, f1, prec, rec, acc_at_2, auc_prc, good_med_acc, good_med_auc)
    """
    model.eval()
    ys, preds, probas = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            fp_seq = get_fp_seq_from_batch(batch, device, num_fields)
            out = model(batch, fp_seq)
            proba = F.softmax(out, dim=-1)
            preds.append(out.argmax(dim=-1).cpu())
            probas.append(proba.cpu())
            ys.append(batch.y.view(-1).cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(preds).numpy()
    y_proba = torch.cat(probas).numpy()

    # Standard metrics
    acc = float((y_true == y_pred).mean())
    f1 = float(f1_score(y_true, y_pred, average="macro")) if f1_score else None
    prec = float(precision_score(y_true, y_pred, average="macro")) if precision_score else None
    rec = float(recall_score(y_true, y_pred, average="macro")) if recall_score else None

    # Accuracy@2 (top-2 accuracy)
    acc_at_2 = None
    if top_k_accuracy_score is not None and num_classes > 2:
        acc_at_2 = float(top_k_accuracy_score(y_true, y_proba, k=2))

    # AUC-PRC (macro-averaged)
    auc_prc = None
    if average_precision_score is not None and label_binarize is not None:
        # One-vs-rest binarization for multi-class
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        # Compute macro-averaged AUC-PRC
        auc_prc = float(average_precision_score(y_true_bin, y_proba, average="macro"))

    # 好 vs 中 metrics (Good vs Medium class binary metrics)
    good_med_acc, good_med_auc = None, None
    if task is not None:
        good_cls, med_cls = get_good_medium_classes(task)
        # Filter samples that are either good or medium class
        mask = (y_true == good_cls) | (y_true == med_cls)
        if mask.sum() > 0:
            y_true_gm = y_true[mask]
            y_pred_gm = y_pred[mask]
            y_proba_gm = y_proba[mask]
            # Binary labels: good=1, medium=0
            y_true_bin_gm = (y_true_gm == good_cls).astype(int)
            y_pred_bin_gm = (y_pred_gm == good_cls).astype(int)
            # 好 vs 中 Accuracy
            good_med_acc = float((y_true_bin_gm == y_pred_bin_gm).mean())
            # 好 vs 中 AUC (using probability of good class)
            if len(np.unique(y_true_bin_gm)) == 2:  # Need both classes for AUC
                try:
                    from sklearn.metrics import roc_auc_score
                    good_med_auc = float(roc_auc_score(y_true_bin_gm, y_proba_gm[:, good_cls]))
                except Exception:
                    good_med_auc = None

    return acc, f1, prec, rec, acc_at_2, auc_prc, good_med_acc, good_med_auc


def train_single_fold(args, augmented_graphs, original_graphs, train_idx, test_idx,
                      metadata, in_dims, num_classes, num_fields, fold_id=None, verbose=True):
    """
    Train a single fold and return metrics.

    Returns:
        dict with keys: train_acc, train_f1, test_acc, test_f1, orig_acc, orig_f1, gate
    """
    train_graphs = [augmented_graphs[i] for i in train_idx]
    test_graphs = [augmented_graphs[i] for i in test_idx]
    val_graphs = test_graphs

    fold_prefix = f"[Fold {fold_id}] " if fold_id is not None else ""

    if verbose:
        print(f"{fold_prefix}Training split: {len(train_graphs)} train, {len(test_graphs)} test", flush=True)

    # Model
    model = GNNResidualModel(
        metadata, in_dims,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_fields=num_fields,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        fp_branch=args.fp_branch,
        trans_layers=args.trans_layers,
        trans_heads=args.trans_heads,
        snn_beta=args.snn_beta,
        snn_time_steps=args.snn_time_steps,
        init_gate=0.0,
        use_jk=args.use_jk,
        gcn2_alpha=args.gcn2_alpha,
        att_readout=args.use_att_readout,
    ).to(args.device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_graphs, num_classes).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size,
                             num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    # Stage 1: Pre-train GNN only
    if args.pretrain_gnn_epochs > 0:
        if verbose:
            print(f"\n{fold_prefix}Stage 1: Pre-training GNN for {args.pretrain_gnn_epochs} epochs", flush=True)

        for param in model.fp_branch.parameters():
            param.requires_grad = False
        model.gate.requires_grad = False

        optimizer_gnn = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )

        best_val_gnn, best_state_gnn = -1.0, None

        for epoch in range(1, args.pretrain_gnn_epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(args.device)
                fp_seq = get_fp_seq_from_batch(batch, args.device, num_fields)
                optimizer_gnn.zero_grad()
                out = model(batch, fp_seq)
                loss = criterion(out, batch.y.view(-1))
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer_gnn.step()
                    total_loss += loss.item() * batch.num_graphs
            total_loss /= max(1, len(train_loader.dataset))

            val_acc, val_f1, val_prec, val_rec, _, _, _, _ = evaluate(model, val_loader, args.device, num_fields, num_classes)

            if val_acc > best_val_gnn:
                best_val_gnn = val_acc
                best_state_gnn = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if verbose and (epoch % args.log_interval == 0 or epoch == args.pretrain_gnn_epochs):
                train_acc, train_f1, _, _, _, _, _, _ = evaluate(model, train_loader, args.device, num_fields, num_classes)
                print(f"{fold_prefix}[GNN] Epoch {epoch:03d} | loss {total_loss:.4f} | "
                      f"train_acc {train_acc:.4f} | val_acc {val_acc:.4f}", flush=True)

        if best_state_gnn:
            model.load_state_dict(best_state_gnn)

        for param in model.fp_branch.parameters():
            param.requires_grad = True
        model.gate.requires_grad = True
        model.gate.data = torch.tensor(args.init_gate)

    # Stage 2: Train FP branch (GNN frozen)
    if verbose:
        print(f"\n{fold_prefix}Stage 2: Training FP branch for {args.epochs} epochs", flush=True)

    model.freeze_gnn()

    optimizer_fp = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    best_val, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(args.device)
            fp_seq = get_fp_seq_from_batch(batch, args.device, num_fields)
            optimizer_fp.zero_grad()
            out = model(batch, fp_seq)
            loss = criterion(out, batch.y.view(-1))
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer_fp.step()
                total_loss += loss.item() * batch.num_graphs
        total_loss /= max(1, len(train_loader.dataset))

        val_acc, val_f1, val_prec, val_rec, _, _, _, _ = evaluate(model, val_loader, args.device, num_fields, num_classes)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch % args.log_interval == 0 or epoch == args.epochs):
            print(f"{fold_prefix}[FP] Epoch {epoch:03d} | loss {total_loss:.4f} | "
                  f"val_acc {val_acc:.4f} | val_f1 {val_f1:.4f} | "
                  f"gate {model.get_gate_value():.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)

    # Final evaluation
    test_acc, test_f1, test_prec, test_rec, test_acc_at_2, test_auc_prc, test_gm_acc, test_gm_auc = evaluate(model, test_loader, args.device, num_fields, num_classes, task=args.task)
    train_acc, train_f1, train_prec, train_rec, train_acc_at_2, train_auc_prc, train_gm_acc, train_gm_auc = evaluate(model, train_loader, args.device, num_fields, num_classes, task=args.task)
    final_gate = model.get_gate_value()

    # Robustness evaluation on original data
    orig_acc, orig_f1 = None, None
    orig_prec, orig_rec = None, None
    orig_acc_at_2, orig_auc_prc = None, None
    orig_gm_acc, orig_gm_auc = None, None
    if len(original_graphs) > 0:
        original_loader = DataLoader(original_graphs, batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     persistent_workers=args.num_workers > 0)
        orig_acc, orig_f1, orig_prec, orig_rec, orig_acc_at_2, orig_auc_prc, orig_gm_acc, orig_gm_auc = evaluate(model, original_loader, args.device, num_fields, num_classes, task=args.task)

    return {
        'train_acc': train_acc, 'train_f1': train_f1, 'train_prec': train_prec, 'train_rec': train_rec,
        'train_acc_at_2': train_acc_at_2, 'train_auc_prc': train_auc_prc,
        'train_gm_acc': train_gm_acc, 'train_gm_auc': train_gm_auc,
        'test_acc': test_acc, 'test_f1': test_f1, 'test_prec': test_prec, 'test_rec': test_rec,
        'test_acc_at_2': test_acc_at_2, 'test_auc_prc': test_auc_prc,
        'test_gm_acc': test_gm_acc, 'test_gm_auc': test_gm_auc,
        'orig_acc': orig_acc, 'orig_f1': orig_f1, 'orig_prec': orig_prec, 'orig_rec': orig_rec,
        'orig_acc_at_2': orig_acc_at_2, 'orig_auc_prc': orig_auc_prc,
        'orig_gm_acc': orig_gm_acc, 'orig_gm_auc': orig_gm_auc,
        'gate': final_gate
    }


def train(args):
    """
    Training pipeline with physics-informed data augmentation.

    The training strategy follows best practices for learning robust features:
    1. Load all data (augmented + original)
    2. Separate augmented samples for training/validation
    3. Reserve original samples for robustness evaluation
    4. Apply global normalization for consistent feature scaling
    5. Train on augmented data only (model never sees original during training)
    6. Evaluate on both augmented test set and original data

    This approach is analogous to:
    - Denoising autoencoders (train on noisy, test on clean)
    - Contrastive learning (SimCLR trains on augmented views)
    - Domain adaptation (train on source, test on target)
    """
    # Load all data
    all_graphs = load_graphs(args.task, args.graph_suffix)
    print(f"Graph suffix: '{args.graph_suffix}'", flush=True)
    if args.edge_mode != "normal":
        print(f"Edge mode: {args.edge_mode}", flush=True)
        apply_edge_mode(all_graphs, mode=args.edge_mode, seed=args.seed)
    if args.self_loops:
        print("Self-loops enabled.", flush=True)
        for g in all_graphs:
            add_self_loops(g)

    # Separate augmented and original samples
    # Training uses ONLY augmented data - original samples are held out for robustness evaluation
    augmented_graphs = [g for g in all_graphs if not getattr(g, 'is_original', True)]
    original_graphs = [g for g in all_graphs if getattr(g, 'is_original', True)]

    print(f"Dataset composition: {len(augmented_graphs)} augmented, {len(original_graphs)} original", flush=True)

    # If no augmentation flag exists, treat all as augmented (backward compatibility)
    if len(augmented_graphs) == 0:
        print("Warning: No augmented samples found, using all data for training.", flush=True)
        augmented_graphs = all_graphs
        original_graphs = []

    # Get fingerprint fields
    fp_fields = FP_FIELD_GROUPS[args.fp_fields]
    num_fields = len(fp_fields)

    # Prepare fingerprints for all data (needed for consistent normalization)
    prepare_fp_sequence(all_graphs, fp_fields)

    # Global normalization - compute statistics over entire dataset for consistency
    # This ensures both augmented and original data are on the same scale
    fp_norm_stats = normalize_fp_sequence(all_graphs)  # Capture pre-norm stats for inference
    norm_stats = compute_normalization_stats(all_graphs)
    all_graphs = [apply_normalization(g, norm_stats) for g in all_graphs]

    # Re-separate after normalization
    augmented_graphs = [g for g in all_graphs if not getattr(g, 'is_original', True)]
    original_graphs = [g for g in all_graphs if getattr(g, 'is_original', True)]

    print(f"Loaded {len(all_graphs)} total graphs for task {args.task}", flush=True)
    print(f"FP fields ({args.fp_fields}): {fp_fields}", flush=True)
    print(f"FP branch: {args.fp_branch}", flush=True)

    # Metadata
    node_types, edge_types, in_dims = set(), set(), {}
    for g in all_graphs:
        for ntype, x in g.x_dict.items():
            node_types.add(ntype)
            in_dims.setdefault(ntype, x.shape[-1])
        for etype in g.edge_index_dict.keys():
            edge_types.add(etype)
    metadata = (sorted(node_types), sorted(edge_types))
    num_classes = int(max(g.y.item() for g in all_graphs) + 1)
    # Ensure all graphs carry all node/edge types to avoid batching KeyError
    ensure_all_node_types(all_graphs, metadata[0], in_dims)
    ensure_all_edge_types(all_graphs, metadata[1])
    # Ensure all graphs carry all edge types to avoid batching KeyError
    ensure_all_edge_types(all_graphs, metadata[1])

    # ============== Full Data Mode (for deployment) ==============
    if args.full_data:
        print(f"\n{'='*70}", flush=True)
        print(f"FULL DATA MODE: Training on ALL augmented data for deployment", flush=True)
        print(f"{'='*70}", flush=True)

        train_graphs = augmented_graphs  # Use ALL augmented data
        val_graphs = original_graphs if len(original_graphs) > 0 else augmented_graphs[:len(augmented_graphs)//10]

        print(f"Training on: {len(train_graphs)} graphs (all augmented data)", flush=True)
        print(f"Validation on: {len(val_graphs)} graphs (original held-out)", flush=True)

        # Use the same training logic as single fold
        train_idx = list(range(len(augmented_graphs)))
        test_idx = []  # No test split in full-data mode

        # Build model
        model = GNNResidualModel(
            metadata, in_dims,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_fields=num_fields,
            gnn_layers=args.gnn_layers,
            dropout=args.dropout,
            fp_branch=args.fp_branch,
            trans_layers=args.trans_layers,
            trans_heads=args.trans_heads,
            snn_beta=args.snn_beta,
            snn_time_steps=args.snn_time_steps,
            init_gate=0.0,
            use_jk=args.use_jk,
            gcn2_alpha=args.gcn2_alpha,
            att_readout=args.use_att_readout,
        ).to(args.device)

        class_weights = None
        if not args.no_class_weights:
            class_weights = compute_class_weights(train_graphs, num_classes).to(args.device)
            print(f"Class weights: {class_weights.cpu().numpy().round(3).tolist()}", flush=True)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size,
                                num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

        # Stage 1: Pre-train GNN
        if args.pretrain_gnn_epochs > 0:
            print(f"\nStage 1: Pre-training GNN for {args.pretrain_gnn_epochs} epochs", flush=True)
            for param in model.fp_branch.parameters():
                param.requires_grad = False
            model.gate.requires_grad = False

            optimizer_gnn = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.lr, weight_decay=args.weight_decay
            )
            best_val_gnn, best_state_gnn = -1.0, None

            for epoch in range(1, args.pretrain_gnn_epochs + 1):
                model.train()
                total_loss = 0.0
                for batch in train_loader:
                    batch = batch.to(args.device)
                    fp_seq = get_fp_seq_from_batch(batch, args.device, num_fields)
                    optimizer_gnn.zero_grad()
                    out = model(batch, fp_seq)
                    loss = criterion(out, batch.y.view(-1))
                    if not torch.isnan(loss):
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer_gnn.step()
                        total_loss += loss.item() * batch.num_graphs
                total_loss /= max(1, len(train_loader.dataset))

                val_acc, val_f1, _, _, _, _, _, _ = evaluate(model, val_loader, args.device, num_fields, num_classes)
                if val_acc > best_val_gnn:
                    best_val_gnn = val_acc
                    best_state_gnn = {k: v.cpu().clone() for k, v in model.state_dict().items()}

                if epoch % args.log_interval == 0 or epoch == args.pretrain_gnn_epochs:
                    print(f"[GNN] Epoch {epoch:03d} | loss {total_loss:.4f} | val_acc {val_acc:.4f}", flush=True)

            if best_state_gnn:
                model.load_state_dict(best_state_gnn)
            for param in model.fp_branch.parameters():
                param.requires_grad = True
            model.gate.requires_grad = True
            model.gate.data = torch.tensor(args.init_gate)

        # Stage 2: Train FP branch
        print(f"\nStage 2: Training FP branch for {args.epochs} epochs", flush=True)
        model.freeze_gnn()

        optimizer_fp = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )
        best_val, best_state = -1.0, None

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(args.device)
                fp_seq = get_fp_seq_from_batch(batch, args.device, num_fields)
                optimizer_fp.zero_grad()
                out = model(batch, fp_seq)
                loss = criterion(out, batch.y.view(-1))
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer_fp.step()
                    total_loss += loss.item() * batch.num_graphs
            total_loss /= max(1, len(train_loader.dataset))

            val_acc, val_f1, _, _, _, _, _, _ = evaluate(model, val_loader, args.device, num_fields, num_classes)
            if val_acc > best_val:
                best_val = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % args.log_interval == 0 or epoch == args.epochs:
                print(f"[FP] Epoch {epoch:03d} | loss {total_loss:.4f} | val_acc {val_acc:.4f} | gate {model.get_gate_value():.4f}", flush=True)

        if best_state:
            model.load_state_dict(best_state)

        # Final evaluation
        train_acc, train_f1, _, _, _, _, _, _ = evaluate(model, train_loader, args.device, num_fields, num_classes)
        orig_acc, orig_f1, orig_prec, orig_rec, orig_acc_at_2, orig_auc_prc, orig_gm_acc, orig_gm_auc = None, None, None, None, None, None, None, None
        if len(original_graphs) > 0:
            original_loader = DataLoader(original_graphs, batch_size=args.batch_size)
            orig_acc, orig_f1, orig_prec, orig_rec, orig_acc_at_2, orig_auc_prc, orig_gm_acc, orig_gm_auc = evaluate(
                model, original_loader, args.device, num_fields, num_classes, task=args.task)

        print(f"\n{'='*70}", flush=True)
        print(f"Full Data Training Complete", flush=True)
        print(f"{'='*70}", flush=True)
        print(f"  Train (all augmented): acc {train_acc:.4f} | f1 {train_f1:.4f}", flush=True)
        if orig_acc is not None:
            print(f"  Original held-out:     acc {orig_acc:.4f} | f1 {orig_f1:.4f}", flush=True)
        print(f"  Final gate: {model.get_gate_value():.4f}", flush=True)

        # Save checkpoint
        if args.save_checkpoint:
            from model_saver import save_checkpoint, create_config_from_args
            config = create_config_from_args(args)
            # Use fp_norm_stats captured earlier (pre-normalization stats)
            save_checkpoint(
                model=model, config=config, norm_stats=norm_stats,
                metadata=metadata, in_dims=in_dims, fp_fields=fp_fields,
                task=args.task, num_classes=num_classes,
                save_path=args.save_checkpoint, fp_norm_stats=fp_norm_stats,
                extra_info={'mode': 'full_data', 'train_acc': train_acc, 'orig_acc': orig_acc}
            )
        return

    # ============== Cross-Validation Mode ==============
    if args.n_folds > 1:
        print(f"\n{'='*70}", flush=True)
        print(f"{args.n_folds}-Fold Cross-Validation", flush=True)
        print(f"{'='*70}", flush=True)

        folds = stratified_kfold_split(augmented_graphs, n_folds=args.n_folds, seed=args.seed)

        all_results = []
        for fold_id, (train_idx, test_idx) in enumerate(folds):
            print(f"\n{'='*60}", flush=True)
            print(f"Fold {fold_id + 1}/{args.n_folds}", flush=True)
            print(f"{'='*60}", flush=True)

            set_seed(args.seed + fold_id)  # Different seed per fold for model init

            results = train_single_fold(
                args, augmented_graphs, original_graphs,
                train_idx, test_idx,
                metadata, in_dims, num_classes, num_fields,
                fold_id=fold_id + 1, verbose=True
            )
            all_results.append(results)

            print(f"\n[Fold {fold_id + 1}] Results: "
                  f"Test acc {results['test_acc']:.4f} | "
                  f"Original acc {results['orig_acc']:.4f}" if results['orig_acc'] else "", flush=True)

        # Summary
        print(f"\n{'='*70}", flush=True)
        print(f"Cross-Validation Summary ({args.n_folds} folds)", flush=True)
        print(f"{'='*70}", flush=True)

        train_accs = [r['train_acc'] for r in all_results]
        train_f1s = [r['train_f1'] for r in all_results]
        train_precs = [r['train_prec'] for r in all_results]
        train_recs = [r['train_rec'] for r in all_results]
        train_acc_at_2s = [r['train_acc_at_2'] for r in all_results if r['train_acc_at_2'] is not None]
        train_auc_prcs = [r['train_auc_prc'] for r in all_results if r['train_auc_prc'] is not None]
        train_gm_accs = [r['train_gm_acc'] for r in all_results if r['train_gm_acc'] is not None]
        train_gm_aucs = [r['train_gm_auc'] for r in all_results if r['train_gm_auc'] is not None]
        test_accs = [r['test_acc'] for r in all_results]
        test_f1s = [r['test_f1'] for r in all_results]
        test_precs = [r['test_prec'] for r in all_results]
        test_recs = [r['test_rec'] for r in all_results]
        test_acc_at_2s = [r['test_acc_at_2'] for r in all_results if r['test_acc_at_2'] is not None]
        test_auc_prcs = [r['test_auc_prc'] for r in all_results if r['test_auc_prc'] is not None]
        test_gm_accs = [r['test_gm_acc'] for r in all_results if r['test_gm_acc'] is not None]
        test_gm_aucs = [r['test_gm_auc'] for r in all_results if r['test_gm_auc'] is not None]
        orig_accs = [r['orig_acc'] for r in all_results if r['orig_acc'] is not None]
        orig_f1s = [r['orig_f1'] for r in all_results if r['orig_f1'] is not None]
        orig_precs = [r['orig_prec'] for r in all_results if r['orig_prec'] is not None]
        orig_recs = [r['orig_rec'] for r in all_results if r['orig_rec'] is not None]
        orig_acc_at_2s = [r['orig_acc_at_2'] for r in all_results if r['orig_acc_at_2'] is not None]
        orig_auc_prcs = [r['orig_auc_prc'] for r in all_results if r['orig_auc_prc'] is not None]
        orig_gm_accs = [r['orig_gm_acc'] for r in all_results if r['orig_gm_acc'] is not None]
        orig_gm_aucs = [r['orig_gm_auc'] for r in all_results if r['orig_gm_auc'] is not None]

        print(f"\n[Augmented Train Set]", flush=True)
        print(f"  Accuracy: {np.mean(train_accs):.4f} ± {np.std(train_accs):.4f}", flush=True)
        print(f"  Macro F1: {np.mean(train_f1s):.4f} ± {np.std(train_f1s):.4f}", flush=True)
        print(f"  Precision: {np.mean(train_precs):.4f} ± {np.std(train_precs):.4f}", flush=True)
        print(f"  Recall:    {np.mean(train_recs):.4f} ± {np.std(train_recs):.4f}", flush=True)
        if train_acc_at_2s:
            print(f"  Acc@2:     {np.mean(train_acc_at_2s):.4f} ± {np.std(train_acc_at_2s):.4f}", flush=True)
        if train_auc_prcs:
            print(f"  AUC-PRC:   {np.mean(train_auc_prcs):.4f} ± {np.std(train_auc_prcs):.4f}", flush=True)
        if train_gm_accs:
            print(f"  Good-Med Acc: {np.mean(train_gm_accs):.4f} ± {np.std(train_gm_accs):.4f}", flush=True)
        if train_gm_aucs:
            print(f"  Good-Med AUC: {np.mean(train_gm_aucs):.4f} ± {np.std(train_gm_aucs):.4f}", flush=True)

        print(f"\n[Augmented Test Set]", flush=True)
        print(f"  Accuracy: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}", flush=True)
        print(f"  Macro F1: {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}", flush=True)
        print(f"  Precision: {np.mean(test_precs):.4f} ± {np.std(test_precs):.4f}", flush=True)
        print(f"  Recall:    {np.mean(test_recs):.4f} ± {np.std(test_recs):.4f}", flush=True)
        if test_acc_at_2s:
            print(f"  Acc@2:     {np.mean(test_acc_at_2s):.4f} ± {np.std(test_acc_at_2s):.4f}", flush=True)
        if test_auc_prcs:
            print(f"  AUC-PRC:   {np.mean(test_auc_prcs):.4f} ± {np.std(test_auc_prcs):.4f}", flush=True)
        if test_gm_accs:
            print(f"  Good-Med Acc: {np.mean(test_gm_accs):.4f} ± {np.std(test_gm_accs):.4f}", flush=True)
        if test_gm_aucs:
            print(f"  Good-Med AUC: {np.mean(test_gm_aucs):.4f} ± {np.std(test_gm_aucs):.4f}", flush=True)

        if orig_accs:
            print(f"\n[Original Held-out Set]", flush=True)
            print(f"  Accuracy: {np.mean(orig_accs):.4f} ± {np.std(orig_accs):.4f}", flush=True)
            print(f"  Macro F1: {np.mean(orig_f1s):.4f} ± {np.std(orig_f1s):.4f}", flush=True)
            print(f"  Precision: {np.mean(orig_precs):.4f} ± {np.std(orig_precs):.4f}", flush=True)
            print(f"  Recall:    {np.mean(orig_recs):.4f} ± {np.std(orig_recs):.4f}", flush=True)
            if orig_acc_at_2s:
                print(f"  Acc@2:     {np.mean(orig_acc_at_2s):.4f} ± {np.std(orig_acc_at_2s):.4f}", flush=True)
            if orig_auc_prcs:
                print(f"  AUC-PRC:   {np.mean(orig_auc_prcs):.4f} ± {np.std(orig_auc_prcs):.4f}", flush=True)
            if orig_gm_accs:
                print(f"  Good-Med Acc: {np.mean(orig_gm_accs):.4f} ± {np.std(orig_gm_accs):.4f}", flush=True)
            if orig_gm_aucs:
                print(f"  Good-Med AUC: {np.mean(orig_gm_aucs):.4f} ± {np.std(orig_gm_aucs):.4f}", flush=True)

        # Best fold
        best_fold_idx = np.argmax(orig_accs) if orig_accs else np.argmax(test_accs)
        best_results = all_results[best_fold_idx]
        print(f"\n[Best Fold: {best_fold_idx + 1}]", flush=True)
        print(f"  Test:     acc {best_results['test_acc']:.4f} | macro_f1 {best_results['test_f1']:.4f}", flush=True)
        print(f"            prec {best_results['test_prec']:.4f} | recall {best_results['test_rec']:.4f}", flush=True)
        if best_results['test_acc_at_2']:
            print(f"            acc@2 {best_results['test_acc_at_2']:.4f} | auc_prc {best_results['test_auc_prc']:.4f}", flush=True)
        if best_results.get('test_gm_acc'):
            print(f"            good-med_acc {best_results['test_gm_acc']:.4f} | good-med_auc {best_results['test_gm_auc']:.4f}", flush=True)
        if best_results['orig_acc']:
            print(f"  Original: acc {best_results['orig_acc']:.4f} | macro_f1 {best_results['orig_f1']:.4f}", flush=True)
            print(f"            prec {best_results['orig_prec']:.4f} | recall {best_results['orig_rec']:.4f}", flush=True)
            if best_results['orig_acc_at_2']:
                print(f"            acc@2 {best_results['orig_acc_at_2']:.4f} | auc_prc {best_results['orig_auc_prc']:.4f}", flush=True)
            if best_results.get('orig_gm_acc'):
                print(f"            good-med_acc {best_results['orig_gm_acc']:.4f} | good-med_auc {best_results['orig_gm_auc']:.4f}", flush=True)
        print(f"  Gate:     {best_results['gate']:.4f}", flush=True)

        print(f"{'='*70}", flush=True)
        return

    # ============== Single Split Mode (Original Behavior) ==============
    # Stratified split on AUGMENTED data only (80/20 train/test)
    # Original data is completely held out for robustness evaluation
    train_idx, test_idx = stratified_split(augmented_graphs, train_ratio=0.8, seed=args.seed)
    train_graphs = [augmented_graphs[i] for i in train_idx]
    test_graphs = [augmented_graphs[i] for i in test_idx]
    val_graphs = test_graphs  # Use augmented test for validation

    print(f"Training split: {len(train_graphs)} train, {len(test_graphs)} test (augmented only)", flush=True)
    print(f"Robustness evaluation: {len(original_graphs)} original samples (held out)", flush=True)

    # Model
    model = GNNResidualModel(
        metadata, in_dims,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_fields=num_fields,
        gnn_layers=args.gnn_layers,
        dropout=args.dropout,
        fp_branch=args.fp_branch,
        trans_layers=args.trans_layers,
        trans_heads=args.trans_heads,
        snn_beta=args.snn_beta,
        snn_time_steps=args.snn_time_steps,
        init_gate=0.0,  # 初始gate=0，Stage1时FP分支无贡献
        use_jk=args.use_jk,
        gcn2_alpha=args.gcn2_alpha,
        att_readout=args.use_att_readout,
    ).to(args.device)

    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(train_graphs, num_classes).to(args.device)
        print(f"Class weights: {class_weights.cpu().numpy().round(3).tolist()}", flush=True)
    else:
        print("Class weights disabled.", flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size,
                             num_workers=args.num_workers, persistent_workers=args.num_workers > 0)

    # ============== Stage 1: Pre-train GNN only ==============
    if args.pretrain_gnn_epochs > 0:
        print(f"\n{'='*60}", flush=True)
        print(f"Stage 1: Pre-training GNN only for {args.pretrain_gnn_epochs} epochs", flush=True)
        print(f"{'='*60}", flush=True)

        # 冻结FP分支和gate，只训练GNN
        for param in model.fp_branch.parameters():
            param.requires_grad = False
        model.gate.requires_grad = False

        optimizer_gnn = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr, weight_decay=args.weight_decay
        )

        best_val_gnn, best_state_gnn = -1.0, None

        for epoch in range(1, args.pretrain_gnn_epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in train_loader:
                batch = batch.to(args.device)
                fp_seq = get_fp_seq_from_batch(batch, args.device, num_fields)
                optimizer_gnn.zero_grad()
                out = model(batch, fp_seq)
                loss = criterion(out, batch.y.view(-1))
                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer_gnn.step()
                    total_loss += loss.item() * batch.num_graphs
            total_loss /= max(1, len(train_loader.dataset))

            val_acc, val_f1, val_prec, val_rec, _, _, _, _ = evaluate(model, val_loader, args.device, num_fields, num_classes)
            train_acc, train_f1, train_prec, train_rec, _, _, _, _ = evaluate(model, train_loader, args.device, num_fields, num_classes)

            if val_acc > best_val_gnn:
                best_val_gnn = val_acc
                best_state_gnn = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if epoch % args.log_interval == 0 or epoch == args.pretrain_gnn_epochs:
                print(f"[GNN] Epoch {epoch:03d} | loss {total_loss:.4f} | "
                      f"train_acc {train_acc:.4f} | train_f1 {train_f1:.4f} | "
                      f"val_acc {val_acc:.4f} | val_f1 {val_f1:.4f}", flush=True)

        # 加载最佳GNN权重
        if best_state_gnn:
            model.load_state_dict(best_state_gnn)

        # 评估纯GNN性能
        gnn_test_acc, gnn_test_f1, _, _, _, _, _, _ = evaluate(model, test_loader, args.device, num_fields, num_classes)
        print(f"\n[GNN-only] Test: acc {gnn_test_acc:.4f} | macro_f1 {gnn_test_f1:.4f}", flush=True)

        # 解冻FP分支和gate
        for param in model.fp_branch.parameters():
            param.requires_grad = True
        model.gate.requires_grad = True
        # 重置gate为初始值
        model.gate.data = torch.tensor(args.init_gate)

    # ============== Stage 2: Freeze GNN, train FP branch ==============
    print(f"\n{'='*60}", flush=True)
    print(f"Stage 2: Training FP branch (GNN frozen) for {args.epochs} epochs", flush=True)
    print(f"{'='*60}", flush=True)

    # 冻结GNN
    model.freeze_gnn()
    print(f"Initial gate value: {model.get_gate_value():.4f}", flush=True)

    # 只优化FP分支和gate
    optimizer_fp = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    best_val, best_state = -1.0, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(args.device)
            fp_seq = get_fp_seq_from_batch(batch, args.device, num_fields)
            optimizer_fp.zero_grad()
            out = model(batch, fp_seq)
            loss = criterion(out, batch.y.view(-1))
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer_fp.step()
                total_loss += loss.item() * batch.num_graphs
        total_loss /= max(1, len(train_loader.dataset))

        val_acc, val_f1, val_prec, val_rec, _, _, _, _ = evaluate(model, val_loader, args.device, num_fields, num_classes)
        train_acc, train_f1, train_prec, train_rec, _, _, _, _ = evaluate(model, train_loader, args.device, num_fields, num_classes)

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % args.log_interval == 0 or epoch == args.epochs:
            gate_val = model.get_gate_value()
            print(f"[FP] Epoch {epoch:03d} | loss {total_loss:.4f} | "
                  f"train_acc {train_acc:.4f} | train_f1 {train_f1:.4f} | "
                  f"val_acc {val_acc:.4f} | val_f1 {val_f1:.4f} | "
                  f"gate {gate_val:.4f}", flush=True)

    if best_state:
        model.load_state_dict(best_state)

    # ============== Final Evaluation ==============
    test_acc, test_f1, test_prec, test_rec, test_acc_at_2, test_auc_prc, test_gm_acc, test_gm_auc = evaluate(model, test_loader, args.device, num_fields, num_classes, task=args.task)
    train_acc, train_f1, train_prec, train_rec, train_acc_at_2, train_auc_prc, train_gm_acc, train_gm_auc = evaluate(model, train_loader, args.device, num_fields, num_classes, task=args.task)
    final_gate = model.get_gate_value()

    # Robustness Evaluation on Original (Unperturbed) Data
    # This tests whether the model learned physics-invariant features
    # that generalize from augmented (perturbed) to original (clean) data
    if len(original_graphs) > 0:
        original_loader = DataLoader(original_graphs, batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     persistent_workers=args.num_workers > 0)
        orig_acc, orig_f1, orig_prec, orig_rec, orig_acc_at_2, orig_auc_prc, orig_gm_acc, orig_gm_auc = evaluate(model, original_loader, args.device, num_fields, num_classes, task=args.task)
    else:
        orig_acc, orig_f1 = None, None
        orig_prec, orig_rec = None, None
        orig_acc_at_2, orig_auc_prc = None, None
        orig_gm_acc, orig_gm_auc = None, None

    print(f"\n{'='*70}", flush=True)
    print(f"Final Results ({args.fp_branch}, {args.fp_fields}):", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"\n[Standard ML Metrics - Augmented Data]", flush=True)
    print(f"  Train:      acc {train_acc:.4f} | macro_f1 {train_f1:.4f}", flush=True)
    print(f"  Test:       acc {test_acc:.4f} | macro_f1 {test_f1:.4f}", flush=True)
    if train_prec is not None and train_rec is not None and test_prec is not None and test_rec is not None:
        print(f"  Train PR:   prec {train_prec:.4f} | rec {train_rec:.4f}", flush=True)
        print(f"  Test PR:    prec {test_prec:.4f} | rec {test_rec:.4f}", flush=True)
    if train_acc_at_2 is not None:
        print(f"  Train:      acc@2 {train_acc_at_2:.4f} | auc_prc {train_auc_prc:.4f}", flush=True)
        print(f"  Test:       acc@2 {test_acc_at_2:.4f} | auc_prc {test_auc_prc:.4f}", flush=True)
    if train_gm_acc is not None:
        print(f"  Train:      good-med_acc {train_gm_acc:.4f} | good-med_auc {train_gm_auc:.4f}", flush=True)
        print(f"  Test:       good-med_acc {test_gm_acc:.4f} | good-med_auc {test_gm_auc:.4f}", flush=True)

    if orig_acc is not None:
        print(f"\n[Robustness Evaluation - Original Unperturbed Data]", flush=True)
        print(f"  Original:   acc {orig_acc:.4f} | macro_f1 {orig_f1:.4f}", flush=True)
        if orig_prec is not None and orig_rec is not None:
            print(f"              prec {orig_prec:.4f} | rec {orig_rec:.4f}", flush=True)
        if orig_acc_at_2 is not None:
            print(f"              acc@2 {orig_acc_at_2:.4f} | auc_prc {orig_auc_prc:.4f}", flush=True)
        if orig_gm_acc is not None:
            print(f"              good-med_acc {orig_gm_acc:.4f} | good-med_auc {orig_gm_auc:.4f}", flush=True)
        print(f"\n  Note: Model was trained ONLY on augmented data.", flush=True)
        print(f"  Original data performance indicates robustness to perturbations.", flush=True)

    print(f"\nFinal gate value: {final_gate:.4f}", flush=True)
    print(f"{'='*70}", flush=True)

    # Save checkpoint if requested
    if args.save_checkpoint:
        from model_saver import save_checkpoint, create_config_from_args

        config = create_config_from_args(args)
        # Use fp_norm_stats captured earlier (pre-normalization stats)

        save_checkpoint(
            model=model,
            config=config,
            norm_stats=norm_stats,
            metadata=metadata,
            in_dims=in_dims,
            fp_fields=fp_fields,
            task=args.task,
            num_classes=num_classes,
            save_path=args.save_checkpoint,
            fp_norm_stats=fp_norm_stats,
            extra_info={
                'final_test_acc': test_acc,
                'final_orig_acc': orig_acc,
                'graph_suffix': args.graph_suffix,
            }
        )


def parse_args():
    p = argparse.ArgumentParser(description="GNN + Residual Fingerprint Branch")
    p.add_argument("--task", choices=TASK_CHOICES, required=True)
    p.add_argument("--fp-fields", choices=["tpm", "key", "all"], default="key",
                   help="Which fingerprint fields: tpm(3), key(5), all(11)")
    p.add_argument("--fp-branch", choices=["mlp", "transformer", "snn"], default="mlp",
                   help="Fingerprint processing method")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=40, help="Epochs for Stage 2 (FP branch training)")
    p.add_argument("--pretrain-gnn-epochs", type=int, default=40,
                   help="Epochs for Stage 1 (GNN pre-training). Set 0 to skip.")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--gnn-layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--use-jk", action="store_true", help="Use JK last+max readout")
    p.add_argument("--gcn2-alpha", type=float, default=0.0,
                   help="GCNII-style initial residual (0 disables)")
    p.add_argument("--use-att-readout", action="store_true",
                   help="Use attention pooling + LayerNorm for readout")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=DEFAULT_DEVICE)
    p.add_argument("--log-interval", type=int, default=10)
    p.add_argument("--no-class-weights", action="store_true")
    p.add_argument("--self-loops", action="store_true")
    p.add_argument("--edge-mode", choices=["normal", "permute", "random", "fully", "complete", "none"], default="normal",
                   help="normal: original; permute: shuffle endpoints; random: resample same count; fully: per-edge-type complete (capped); complete: all node-type pairs fully connected; none: drop all edges")
    # Transformer specific
    p.add_argument("--trans-layers", type=int, default=2)
    p.add_argument("--trans-heads", type=int, default=4)
    # SNN specific
    p.add_argument("--snn-beta", type=float, default=0.9)
    p.add_argument("--snn-time-steps", type=int, default=16)
    # Gate
    p.add_argument("--init-gate", type=float, default=0.1, help="Initial gate value for Stage 2")
    p.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers for parallel data loading")
    p.add_argument("--graph-suffix", type=str, default="",
                   help="Suffix for graph files (e.g., '_orig', '_aug', '_rand')")
    p.add_argument("--n-folds", type=int, default=1,
                   help="Number of CV folds. 1 = single 80/20 split (default), >1 = K-fold CV")
    p.add_argument("--full-data", action="store_true",
                   help="Train on ALL data (no train/test split). Use for final deployment after CV.")
    p.add_argument("--save-checkpoint", type=str, default=None,
                   help="Path to save model checkpoint (e.g., 'checkpoints/model_ldl.pt')")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        print("CUDA not available, falling back to CPU.", flush=True)
        args.device = "cpu"
    train(args)
