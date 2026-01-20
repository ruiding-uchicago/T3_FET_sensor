#!/usr/bin/env python3
"""
Inference Utilities for DTE-GNN

Provides:
- load_checkpoint: Load saved model checkpoint
- build_model_from_checkpoint: Rebuild model from checkpoint
- InferenceEngine: Class for running batch inference
"""
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np

# Import model classes from training script
from train_hetero_gnn_residual_sgnn import (
    GNNResidualModel,
    FP_DIM,
    FP_FIELD_GROUPS,
    add_self_loops,
)


def load_checkpoint(checkpoint_path: str, device: str = 'cpu') -> Dict[str, Any]:
    """
    Load checkpoint from file.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load tensors to

    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Checkpoint loaded from: {checkpoint_path}")
    print(f"  Task: {checkpoint['task']}")
    print(f"  Num classes: {checkpoint['num_classes']}")
    print(f"  FP fields: {checkpoint['fp_fields']}")
    print(f"  Gate value: {checkpoint.get('gate_value', 'N/A')}")
    return checkpoint


def build_model_from_checkpoint(
    checkpoint: Dict[str, Any],
    device: str = 'cpu'
) -> GNNResidualModel:
    """
    Rebuild model from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dictionary
        device: Device to place model on

    Returns:
        GNNResidualModel with loaded weights
    """
    config = checkpoint['config']
    metadata = checkpoint['metadata']
    in_dims = checkpoint['in_dims']
    num_classes = checkpoint['num_classes']
    num_fields = len(checkpoint['fp_fields'])

    model = GNNResidualModel(
        metadata=metadata,
        in_dims=in_dims,
        hidden_dim=config['hidden_dim'],
        num_classes=num_classes,
        num_fields=num_fields,
        gnn_layers=config['gnn_layers'],
        dropout=config['dropout'],
        fp_branch=config['fp_branch'],
        trans_layers=config.get('trans_layers', 2),
        trans_heads=config.get('trans_heads', 4),
        snn_beta=config.get('snn_beta', 0.9),
        snn_time_steps=config.get('snn_time_steps', 16),
        init_gate=0.0,
        use_jk=config['use_jk'],
        gcn2_alpha=config['gcn2_alpha'],
        att_readout=config['att_readout'],
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Model rebuilt and loaded on {device}")
    return model


class InferenceEngine:
    """
    Engine for running DTE-GNN inference.

    Usage:
        engine = InferenceEngine.from_checkpoint('model_ldl.pt', device='cuda')
        probs = engine.predict(graph)  # Single graph
        probs_batch = engine.predict_batch(graphs)  # Multiple graphs
    """

    def __init__(
        self,
        model: GNNResidualModel,
        norm_stats: Dict[str, Dict[str, torch.Tensor]],
        fp_norm_stats: Optional[Dict[str, torch.Tensor]],
        fp_fields: List[str],
        config: Dict[str, Any],
        device: str = 'cpu',
    ):
        self.model = model
        self.norm_stats = norm_stats
        self.fp_norm_stats = fp_norm_stats
        self.fp_fields = fp_fields
        self.config = config
        self.device = device
        self.num_fields = len(fp_fields)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str = 'cpu') -> 'InferenceEngine':
        """
        Create InferenceEngine from checkpoint file.

        Args:
            checkpoint_path: Path to checkpoint
            device: Device to run inference on

        Returns:
            InferenceEngine instance
        """
        checkpoint = load_checkpoint(checkpoint_path, device)
        model = build_model_from_checkpoint(checkpoint, device)

        return cls(
            model=model,
            norm_stats=checkpoint['norm_stats'],
            fp_norm_stats=checkpoint.get('fp_norm_stats'),
            fp_fields=checkpoint['fp_fields'],
            config=checkpoint['config'],
            device=device,
        )

    def _apply_normalization(self, data) -> None:
        """Apply node feature normalization in-place."""
        for ntype, conf in self.norm_stats.items():
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

    def _prepare_fp_sequence(self, data) -> torch.Tensor:
        """Prepare fingerprint sequence tensor."""
        fp_list = []
        if hasattr(data, 'fingerprints') and data.fingerprints is not None:
            for field in self.fp_fields:
                fp = data.fingerprints.get(field, torch.zeros(FP_DIM))
                if isinstance(fp, np.ndarray):
                    fp = torch.tensor(fp, dtype=torch.float)
                fp_list.append(fp)
        else:
            fp_list = [torch.zeros(FP_DIM) for _ in self.fp_fields]

        fp_seq = torch.stack(fp_list).unsqueeze(0)  # (1, num_fields, 320)

        # Apply FP normalization if available
        if self.fp_norm_stats is not None:
            mean = self.fp_norm_stats['mean'].to(fp_seq.device)
            std = self.fp_norm_stats['std'].to(fp_seq.device)
            fp_seq = (fp_seq - mean) / std

        return fp_seq

    def predict(self, data, return_logits: bool = False) -> np.ndarray:
        """
        Run inference on a single graph.

        Args:
            data: HeteroData graph
            return_logits: If True, return raw logits instead of probabilities

        Returns:
            Probability array of shape (num_classes,) or logits if return_logits=True
        """
        from torch_geometric.loader import DataLoader

        # Add self-loops if configured
        if self.config.get('self_loops', False):
            add_self_loops(data)

        # Apply normalization
        self._apply_normalization(data)

        # Prepare fingerprints
        data.fp_seq = self._prepare_fp_sequence(data)

        # Use DataLoader to add batch information
        loader = DataLoader([data], batch_size=1, shuffle=False)
        batch = next(iter(loader))

        # Move to device
        batch = batch.to(self.device)
        fp_seq = batch.fp_seq.to(self.device)

        # Inference
        self.model.eval()
        with torch.no_grad():
            logits = self.model(batch, fp_seq)
            if return_logits:
                return logits.cpu().numpy().squeeze()
            probs = F.softmax(logits, dim=-1)
            return probs.cpu().numpy().squeeze()

    def predict_batch(
        self,
        graphs: List,
        batch_size: int = 32,
        return_logits: bool = False,
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Run inference on multiple graphs.

        Args:
            graphs: List of HeteroData graphs
            batch_size: Batch size for inference
            return_logits: If True, return raw logits
            show_progress: If True, print progress

        Returns:
            Array of shape (num_graphs, num_classes)
        """
        from torch_geometric.loader import DataLoader

        # Preprocess all graphs
        for g in graphs:
            if self.config.get('self_loops', False):
                add_self_loops(g)
            self._apply_normalization(g)
            g.fp_seq = self._prepare_fp_sequence(g)

        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)

        all_outputs = []
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(loader):
                if show_progress and i % 10 == 0:
                    print(f"  Batch {i+1}/{len(loader)}", end='\r')

                batch = batch.to(self.device)
                fp_seq = batch.fp_seq.to(self.device)

                logits = self.model(batch, fp_seq)

                if return_logits:
                    all_outputs.append(logits.cpu().numpy())
                else:
                    probs = F.softmax(logits, dim=-1)
                    all_outputs.append(probs.cpu().numpy())

        if show_progress:
            print()

        return np.vstack(all_outputs)

    def get_class_names(self, task: str) -> List[str]:
        """
        Get human-readable class names for a task.

        Args:
            task: Task name

        Returns:
            List of class names
        """
        if task == 'lower_detection_limit':
            return ['Good (Low LDL)', 'Medium', 'Poor (High LDL)']
        elif task == 'upper_detection_limit':
            return ['Poor (Low UDL)', 'Medium', 'Good (High UDL)']
        elif task == 'sensitivity_numerator':
            return ['Poor (Low Sens)', 'Medium', 'Good (High Sens)']
        else:
            return [f'Class {i}' for i in range(self.model.num_classes)]
