#!/usr/bin/env python3
"""
Build heterogeneous graph datasets from extracted FET records (V2 Graph + Augmentation Support).

This script builds graph datasets with:
- V2 graph topology (expert-recommended edges for probe-target-channel interactions)
- Support for multiple data sources (original, augmented_v2, augmented_v2_random)
- is_original flag to distinguish augmented vs original samples

Usage:
    # Build from original (non-augmented) data
    python build_graph_dataset_augmented.py --data-source original

    # Build from AUGMENTED_V2 data (systematic augmentation)
    python build_graph_dataset_augmented.py --data-source augmented_v2

    # Build from AUGMENTED_V2_RANDOM data (random augmentation)
    python build_graph_dataset_augmented.py --data-source augmented_v2_random

Outputs: graphs_{task}.pkl or graphs_{task}.pt
"""
import argparse
import json
import os
import pickle
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

USE_TORCH = os.environ.get("USE_TORCH", "0") == "1"
if USE_TORCH:
    import torch
    from torch_geometric.data import HeteroData

# Paths (resolve relative to this script so it works from repo root or experiment-aug/)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR
OUTPUT_DIR = SCRIPT_DIR

# Data source configurations
DATA_SOURCES = {
    "original": {
        "file": "extracted_records_FINAL_pH_FIXED.jsonl",
        "is_augmented": False,
        "description": "Original non-augmented data",
    },
    "augmented_v2": {
        "file": "extracted_records_AUGMENTED_V2.jsonl",
        "is_augmented": True,
        "description": "Systematic physics-informed augmentation",
    },
    "augmented_v2_random": {
        "file": "extracted_records_AUGMENTED_V2_RANDOM.jsonl",
        "is_augmented": True,
        "description": "Random physics-informed augmentation (V2)",
    },
    "augmented_v3": {
        "file": "extracted_records_AUGMENTED_V3.jsonl",
        "is_augmented": True,
        "description": "Systematic physics-informed augmentation (V3 - reduced magnitude)",
    },
    "augmented_v3_random_same_mag": {
        "file": "extracted_records_AUGMENTED_V3_RANDOM_SAME_MAG.jsonl",
        "is_augmented": True,
        "description": "Random augmentation with SAME magnitude as Physics but NO constraints",
    },
    "augmented_v3_random": {
        "file": "extracted_records_AUGMENTED_V3_RANDOM.jsonl",
        "is_augmented": True,
        "description": "Destructive random augmentation (V3 - high-level perturbation)",
    },
}

# Task configs
TASKS = {
    "lower_detection_limit": {
        "field": "lower_detection_limit_logppm",
        # 3-class merge: Cat1 | (Cat2+Cat3) | (Cat4+Cat5)
        "bins": [-np.inf, -6, 0, np.inf],
        "use_abs": False,
    },
    "upper_detection_limit": {
        "field": "upper_detection_limit_logppm",
        # 3-class merge: Cat1 | (Cat2+Cat3) | (Cat4+Cat5)
        "bins": [-np.inf, 0, 2, np.inf],
        "use_abs": False,
    },
    "sensitivity_numerator": {
        "field": "sensitivity_numerator_value",
        # 3-class merge: Cat1 | (Cat2+Cat3) | (Cat4+Cat5)
        "bins": [-np.inf, 0, 2, np.inf],
        "use_abs": True,  # log10(|value|)
    },
}

# Category order to infer design from one-hot when raw string is absent
DESIGN_TYPES = ["standard", "remote", "floating_gate", "dual_gate"]

# Fixed sizes (drop fingerprint; keep 25 macro + 5 type one-hot = 30)
DESC_DIM = 30
MAT_FEATURE_DIM = DESC_DIM * 2 + 1  # descriptor + mask + num_components
TEST_MEDIUM_EXTRA = 2 * 2  # pH values + pH masks
PROCESS_EXTRA = 2 * 2  # anneal temp/time + masks

# Fingerprint dimension for SNN branch
FP_DIM = 320

# All material fields with fingerprints (11 fields)
MATERIAL_FIELDS = [
    "channel",
    "dielectric_layer",
    "gate",
    "source",
    "drain",
    "substrate",
    "probe_material",
    "surface_functionalization",
    "annealing_atmosphere",
    "detect_target",
    "test_medium",
]


def safe_float(x: Optional[float]) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def aggregate_fingerprints(materials: Optional[List[Dict]]) -> np.ndarray:
    """
    Aggregate fingerprints across a list of materials (mean).
    Returns 320-dim vector (zeros if no valid fingerprint).
    """
    if not materials:
        return np.zeros(FP_DIM, dtype=float)

    fp_list = []
    for mat in materials:
        fp = mat.get("fingerprint") or []
        if len(fp) == FP_DIM:
            # Convert None to 0, keep valid values
            fp_arr = np.array([safe_float(v) if v is not None else 0.0 for v in fp], dtype=float)
            # Check if it's not all zeros (has valid data)
            if np.any(fp_arr != 0):
                fp_list.append(fp_arr)

    if not fp_list:
        return np.zeros(FP_DIM, dtype=float)

    return np.mean(fp_list, axis=0)


def aggregate_materials(materials: Optional[List[Dict]]) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Aggregate material descriptors and masks across a list (mean), with per-dim mask.
    Returns (desc_mean, mask_any, num_components).
    """
    if not materials:
        return np.zeros(DESC_DIM, dtype=float), np.zeros(DESC_DIM, dtype=float), 0

    desc_list = []
    mask_list = []

    for mat in materials:
        mv = mat.get("macro_vec") or []
        oh = mat.get("substance_type_onehot") or []

        mv_arr = np.zeros(25, dtype=float)
        mv_mask = np.zeros(25, dtype=float)
        if len(mv) == 25:
            mv_arr = np.array([safe_float(v) if v is not None else 0.0 for v in mv], dtype=float)
            mv_mask = np.array([1.0 if v is not None else 0.0 for v in mv], dtype=float)

        oh_arr = np.array(oh, dtype=float) if len(oh) == 5 else np.zeros(5, dtype=float)
        oh_mask = np.ones(5, dtype=float) if len(oh) == 5 else np.zeros(5, dtype=float)

        desc = np.concatenate([mv_arr, oh_arr])
        mask = np.concatenate([mv_mask, oh_mask])

        desc_list.append(desc)
        mask_list.append(mask)

    desc_mean = np.mean(desc_list, axis=0)
    mask_any = (np.sum(mask_list, axis=0) > 0).astype(float)
    num_components = len(materials)

    return desc_mean, mask_any, num_components


def build_material_node(materials: Optional[List[Dict]]) -> np.ndarray:
    desc, mask, num_components = aggregate_materials(materials)
    return np.concatenate([desc, mask, np.array([float(num_components)], dtype=float)])


def build_test_medium_node(materials: Optional[List[Dict]], ph_min: Optional[float], ph_max: Optional[float]) -> np.ndarray:
    base = build_material_node(materials)
    extras = []
    masks = []
    for val in (ph_min, ph_max):
        v = safe_float(val)
        extras.append(0.0 if v is None else v)
        masks.append(0.0 if v is None else 1.0)
    return np.concatenate([base, np.array(extras, dtype=float), np.array(masks, dtype=float)])


def build_process_node(rec: Dict) -> np.ndarray:
    base = build_material_node(rec.get("annealing_atmosphere"))
    extras = []
    masks = []
    for field in ("annealing_temperature", "annealing_time"):
        v = safe_float(rec.get(field))
        extras.append(0.0 if v is None else v)
        masks.append(0.0 if v is None else 1.0)
    return np.concatenate([base, np.array(extras, dtype=float), np.array(masks, dtype=float)])


def build_condition_node(rec: Dict, task: str = "sensitivity_numerator") -> np.ndarray:
    """
    Build condition node features.
    - For LDL/UDL tasks: only device info (no sensitivity-related features)
    - For sensitivity task: include denominator value and units
    """
    if task in ("lower_detection_limit", "upper_detection_limit"):
        # Detection limit tasks: device-only features
        scalar_fields = [
            "dielectric_layer_thickness",
            "substrate_thickness",
            "test_operating_temperature_celcius",
            "annealing_temperature",
            "annealing_time",
            "hydrothermal_temperature",
            "hydrothermal_time",
            "pH_min",
            "pH_max",
        ]
        onehots = [
            rec.get("sensor_type_onehot", [0, 0, 0]),
            rec.get("structure_dimensionality_onehot", [0, 0, 0]),
            rec.get("structure_design_type_onehot", [0, 0, 0, 0]),
        ]
    else:
        # Sensitivity task: include denominator value and units
        scalar_fields = [
            "dielectric_layer_thickness",
            "substrate_thickness",
            "test_operating_temperature_celcius",
            "annealing_temperature",
            "annealing_time",
            "hydrothermal_temperature",
            "hydrothermal_time",
            "pH_min",
            "pH_max",
            "sensitivity_denominator_value",
        ]
        onehots = [
            rec.get("sensor_type_onehot", [0, 0, 0]),
            rec.get("structure_dimensionality_onehot", [0, 0, 0]),
            rec.get("structure_design_type_onehot", [0, 0, 0, 0]),
            rec.get("sensitivity_numerator_unit_onehot", [0, 0, 0]),
            rec.get("sensitivity_denominator_unit_onehot", [0, 0, 0]),
        ]

    values = []
    masks = []
    for f in scalar_fields:
        v = safe_float(rec.get(f))
        values.append(0.0 if v is None else v)
        masks.append(0.0 if v is None else 1.0)

    flat_onehot = [float(x) for arr in onehots for x in arr]
    return np.concatenate([np.array(values, dtype=float), np.array(flat_onehot, dtype=float), np.array(masks, dtype=float)])


def infer_design(rec: Dict) -> str:
    val = rec.get("structure_design_type")
    if isinstance(val, str) and val in DESIGN_TYPES:
        return val
    oh = rec.get("structure_design_type_onehot")
    if oh and len(oh) == len(DESIGN_TYPES):
        idx = int(np.argmax(oh))
        return DESIGN_TYPES[idx]
    return "standard"


def add_edge(data: Union["HeteroData", Dict], src: str, rel: str, dst: str, bidir: bool = False) -> None:
    if USE_TORCH:
        if src not in data.node_types or dst not in data.node_types:
            return
        idx = torch.tensor([[0], [0]], dtype=torch.long)
        key = (src, rel, dst)
        if "edge_index" in data[key]:
            data[key].edge_index = torch.cat([data[key].edge_index, idx], dim=1)
        else:
            data[key].edge_index = idx
        if bidir:
            add_edge(data, dst, rel, src, bidir=False)
    else:
        if src not in data["nodes"] or dst not in data["nodes"]:
            return
        idx = np.array([[0], [0]], dtype=np.int64)
        key = (src, rel, dst)
        if key in data["edges"]:
            data["edges"][key] = np.concatenate([data["edges"][key], idx], axis=1)
        else:
            data["edges"][key] = idx
        if bidir:
            add_edge(data, dst, rel, src, bidir=False)


def build_edges(
    data: Union["HeteroData", Dict],
    design: str,
    has_bottom: bool = False,
    has_floating: bool = False,
    is_remote: bool = False,
) -> None:
    """
    Build graph edges with V2 topology (expert-recommended edges).

    V2 adds 4 new edges based on expert consensus:
    1. probe_material ↔ test_medium (probe stability/solubility)
    2. detect_target ↔ channel (direct sensing for small molecules)
    3. probe_material ↔ channel (probe grafted onto channel)
    4. channel ↔ dielectric_top (interface trap states)
    """
    # Electrical path
    add_edge(data, "source", "electrical", "channel", bidir=True)
    add_edge(data, "drain", "electrical", "channel", bidir=True)
    add_edge(data, "substrate", "electrical", "channel", bidir=True)

    # Capacitive/gating
    if design == "remote":
        add_edge(data, "gate_top", "capacitive", "electrolyte", bidir=True)
        add_edge(data, "electrolyte", "capacitive", "channel", bidir=True)
    elif design == "floating_gate":
        add_edge(data, "gate_top", "capacitive", "dielectric_top", bidir=True)
        add_edge(data, "dielectric_top", "capacitive", "floating_gate", bidir=True)
        add_edge(data, "floating_gate", "capacitive", "dielectric_bottom", bidir=True)
        add_edge(data, "dielectric_bottom", "capacitive", "channel", bidir=True)
    elif design == "dual_gate":
        add_edge(data, "gate_top", "capacitive", "dielectric_top", bidir=True)
        add_edge(data, "dielectric_top", "capacitive", "channel", bidir=True)
        add_edge(data, "gate_bottom", "capacitive", "dielectric_bottom", bidir=True)
        add_edge(data, "dielectric_bottom", "capacitive", "channel", bidir=True)
    else:  # standard/default
        add_edge(data, "gate_top", "capacitive", "dielectric_top", bidir=True)
        add_edge(data, "dielectric_top", "capacitive", "channel", bidir=True)

    # Chemical / interface (sensing chain)
    add_edge(data, "channel", "chemical", "surface_functionalization", bidir=True)
    add_edge(data, "surface_functionalization", "chemical", "probe_material", bidir=True)
    add_edge(data, "probe_material", "chemical", "detect_target", bidir=True)
    add_edge(data, "test_medium", "chemical", "detect_target", bidir=True)
    add_edge(data, "test_medium", "chemical", "channel", bidir=True)

    # V2: Expert-recommended edges (reusing existing edge types)
    # 1. Probe-medium interaction (stability, solubility, conformation)
    add_edge(data, "probe_material", "chemical", "test_medium", bidir=True)
    # 2. Direct target-channel sensing (small molecules, Debye screening bypass)
    add_edge(data, "detect_target", "chemical", "channel", bidir=True)
    # 3. Direct probe-channel interaction (probe grafted onto channel, charge transfer)
    add_edge(data, "probe_material", "chemical", "channel", bidir=True)

    # V2: Channel-dielectric interface (trap states, mobility)
    add_edge(data, "channel", "capacitive", "dielectric_top", bidir=True)

    # Environment/process/condition
    add_edge(data, "process_annealing", "process", "channel", bidir=False)
    add_edge(data, "condition", "condition", "channel", bidir=False)
    add_edge(data, "condition", "condition", "gate_top", bidir=False)
    add_edge(data, "condition", "condition", "test_medium", bidir=False)
    if design == "dual_gate":
        add_edge(data, "condition", "condition", "gate_bottom", bidir=False)
    if is_remote:
        add_edge(data, "electrolyte", "environment", "test_medium", bidir=True)


def record_to_graph(rec: Dict, task: str = "sensitivity_numerator") -> Union["HeteroData", Dict]:
    # Nodes (for GNN branch - macro properties)
    node_features: Dict[str, np.ndarray] = {
        "channel": build_material_node(rec.get("channel")),
        "gate_top": build_material_node(rec.get("gate")),
        "gate_bottom": build_material_node(rec.get("gate")),
        "dielectric_top": build_material_node(rec.get("dielectric_layer")),
        "dielectric_bottom": build_material_node(rec.get("dielectric_layer")),
        "floating_gate": build_material_node(rec.get("gate")),
        "source": build_material_node(rec.get("source")),
        "drain": build_material_node(rec.get("drain")),
        "substrate": build_material_node(rec.get("substrate")),
        "surface_functionalization": build_material_node(rec.get("surface_functionalization")),
        "probe_material": build_material_node(rec.get("probe_material")),
        "detect_target": build_material_node(rec.get("detect_target")),
        "test_medium": build_test_medium_node(rec.get("test_medium"), rec.get("pH_min"), rec.get("pH_max")),
        "electrolyte": build_test_medium_node(rec.get("test_medium"), rec.get("pH_min"), rec.get("pH_max")),
        "process_annealing": build_process_node(rec),
        "condition": build_condition_node(rec, task=task),
    }

    # Fingerprints (for SNN branch - all 11 material fields)
    fingerprints: Dict[str, np.ndarray] = {}
    for field in MATERIAL_FIELDS:
        fingerprints[field] = aggregate_fingerprints(rec.get(field))

    if USE_TORCH:
        data: Union["HeteroData", Dict] = HeteroData()
        for ntype, feat in node_features.items():
            data[ntype].x = torch.tensor(feat, dtype=torch.float).unsqueeze(0)
        # Store fingerprints as a dict of tensors
        data.fingerprints = {k: torch.tensor(v, dtype=torch.float) for k, v in fingerprints.items()}
    else:
        data = {"nodes": node_features, "edges": {}, "fingerprints": fingerprints}

    design = infer_design(rec)
    is_remote = design == "remote"
    has_floating = design == "floating_gate"
    has_bottom = design == "dual_gate"

    build_edges(data, design, has_bottom=has_bottom, has_floating=has_floating, is_remote=is_remote)

    doi = rec.get("doi")
    if doi:
        if USE_TORCH:
            data.doi = doi
        else:
            data["doi"] = doi

    # 标记是否为原始样本（非增强）
    is_original = rec.get("is_original", True)
    if USE_TORCH:
        data.is_original = is_original
    else:
        data["is_original"] = is_original
    return data


def bin_target(value: Optional[float], bins: List[float], use_abs: bool) -> Optional[int]:
    if value is None:
        return None
    val = safe_float(value)
    if val is None:
        return None
    if use_abs:
        if val == 0:
            return None
        val = np.log10(abs(val))
    return int(np.clip(np.digitize(val, bins) - 1, 0, len(bins) - 2))


def load_records(path: Path) -> List[Dict]:
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
    return records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build graph datasets with V2 topology from FET records",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Data sources:
  original           Original non-augmented data (~2000 samples)
  augmented_v2       Systematic physics-informed augmentation (~100k samples)
  augmented_v2_random Random physics-informed augmentation (~100k samples)

Examples:
  python build_graph_dataset_augmented.py --data-source original
  python build_graph_dataset_augmented.py --data-source augmented_v2
  USE_TORCH=1 python build_graph_dataset_augmented.py --data-source augmented_v2_random
        """,
    )
    parser.add_argument(
        "--data-source",
        choices=list(DATA_SOURCES.keys()),
        default="augmented_v2",
        help="Which data source to use (default: augmented_v2)",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Optional suffix for output files (e.g., '_v2' -> graphs_v2_{task}.pkl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Get data source configuration
    source_cfg = DATA_SOURCES[args.data_source]
    input_file = REPO_ROOT / source_cfg["file"]
    is_augmented_source = source_cfg["is_augmented"]

    print(f"Data source: {args.data_source}")
    print(f"  Description: {source_cfg['description']}")
    print(f"  Input file: {input_file}")
    print(f"  Is augmented: {is_augmented_source}")

    if not input_file.exists():
        print(f"ERROR: Input file not found: {input_file}")
        print("Available data sources:")
        for name, cfg in DATA_SOURCES.items():
            path = REPO_ROOT / cfg["file"]
            exists = "✓" if path.exists() else "✗"
            print(f"  [{exists}] {name}: {path}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records = load_records(input_file)
    print(f"Loaded {len(records)} records")

    per_task_graphs: Dict[str, List[Union["HeteroData", Dict]]] = {t: [] for t in TASKS}
    counts = {t: 0 for t in TASKS}
    label_counts = {t: {} for t in TASKS}
    orig_counts = {t: 0 for t in TASKS}

    for rec in records:
        for task_name, cfg in TASKS.items():
            raw = rec.get(cfg["field"])
            label = bin_target(raw, cfg["bins"], cfg["use_abs"])
            if label is None:
                continue

            # Build graph with task-specific condition node
            g = record_to_graph(rec, task=task_name)

            if USE_TORCH:
                g.y = torch.tensor([label], dtype=torch.long)
            else:
                g["y"] = label

            # Track original vs augmented
            is_orig = g.is_original if USE_TORCH else g.get("is_original", True)
            if is_orig:
                orig_counts[task_name] += 1

            per_task_graphs[task_name].append(g)
            counts[task_name] += 1
            label_counts[task_name][label] = label_counts[task_name].get(label, 0) + 1

    # Save graphs
    suffix = args.output_suffix
    for task_name, graphs in per_task_graphs.items():
        if USE_TORCH:
            out_path = OUTPUT_DIR / f"graphs{suffix}_{task_name}.pt"
            torch.save(graphs, out_path)
        else:
            out_path = OUTPUT_DIR / f"graphs{suffix}_{task_name}.pkl"
            with open(out_path, "wb") as f:
                pickle.dump(graphs, f)

        n_aug = counts[task_name] - orig_counts[task_name]
        print(f"\n{task_name}:")
        print(f"  Total: {len(graphs)} graphs")
        print(f"  Original: {orig_counts[task_name]}, Augmented: {n_aug}")
        print(f"  Label distribution: {sorted(label_counts[task_name].items())}")
        print(f"  Saved to: {out_path}")

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"  Data source: {args.data_source}")
    print(f"  Graph topology: V2 (with expert-recommended edges)")
    print(f"  Output format: {'PyTorch (.pt)' if USE_TORCH else 'Pickle (.pkl)'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
