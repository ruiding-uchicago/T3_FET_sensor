#!/usr/bin/env python3
"""
Model Checkpoint Saver for DTE-GNN

Saves complete checkpoint containing:
- model_state_dict: trained weights
- config: hyperparameters (hidden_dim, gnn_layers, etc.)
- norm_stats: normalization statistics for node features
- fp_norm_stats: normalization statistics for fingerprints
- metadata: (node_types, edge_types)
- in_dims: input dimensions per node type
- fp_fields: fingerprint field names used
- task: classification task name
- num_classes: number of output classes
"""
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional


def save_checkpoint(
    model: torch.nn.Module,
    config: Dict[str, Any],
    norm_stats: Dict[str, Dict[str, torch.Tensor]],
    metadata: tuple,
    in_dims: Dict[str, int],
    fp_fields: List[str],
    task: str,
    num_classes: int,
    save_path: str,
    fp_norm_stats: Optional[Dict[str, torch.Tensor]] = None,
    extra_info: Optional[Dict] = None,
) -> None:
    """
    Save complete model checkpoint for later inference.

    Args:
        model: Trained GNNResidualModel
        config: Dictionary of hyperparameters
        norm_stats: Node feature normalization statistics
        metadata: (node_types, edge_types) tuple
        in_dims: Input dimensions per node type
        fp_fields: List of fingerprint field names
        task: Task name (e.g., 'lower_detection_limit')
        num_classes: Number of output classes
        save_path: Path to save checkpoint
        fp_norm_stats: Fingerprint normalization stats (mean, std)
        extra_info: Optional extra metadata
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'norm_stats': norm_stats,
        'metadata': metadata,
        'in_dims': in_dims,
        'fp_fields': fp_fields,
        'task': task,
        'num_classes': num_classes,
        'gate_value': model.get_gate_value() if hasattr(model, 'get_gate_value') else None,
    }

    if fp_norm_stats is not None:
        checkpoint['fp_norm_stats'] = fp_norm_stats

    if extra_info is not None:
        checkpoint['extra_info'] = extra_info

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to: {save_path}")


def create_config_from_args(args) -> Dict[str, Any]:
    """
    Extract config dictionary from argparse namespace.

    Args:
        args: argparse.Namespace from training script

    Returns:
        Config dictionary with all hyperparameters
    """
    return {
        'hidden_dim': args.hidden_dim,
        'gnn_layers': args.gnn_layers,
        'dropout': args.dropout,
        'fp_branch': args.fp_branch,
        'trans_layers': getattr(args, 'trans_layers', 2),
        'trans_heads': getattr(args, 'trans_heads', 4),
        'snn_beta': getattr(args, 'snn_beta', 0.9),
        'snn_time_steps': getattr(args, 'snn_time_steps', 16),
        'use_jk': args.use_jk,
        'gcn2_alpha': args.gcn2_alpha,
        'att_readout': args.use_att_readout,
        'self_loops': args.self_loops,
        'fp_fields_group': args.fp_fields,  # 'key', 'tpm', or 'all'
    }


def compute_fp_norm_stats(graphs: List) -> Dict[str, torch.Tensor]:
    """
    Compute fingerprint normalization statistics.

    Args:
        graphs: List of HeteroData with fp_seq attribute

    Returns:
        Dictionary with 'mean' and 'std' tensors
    """
    all_fps = torch.cat([g.fp_seq for g in graphs if hasattr(g, 'fp_seq')], dim=0)
    return {
        'mean': all_fps.mean(dim=0),
        'std': all_fps.std(dim=0) + 1e-6,
    }
