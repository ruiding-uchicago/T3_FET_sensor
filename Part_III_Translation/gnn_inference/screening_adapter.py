#!/usr/bin/env python3
"""
Adapter to convert template sensor + candidate molecule to training-compatible record format.

This bridges the gap between:
- Template JSON format (from 10.1038_s44221-025-00505-9.json)
- Molecule database format (from query_molecules_million)
- Training record format (expected by original_graph_builder.py)
"""
import json
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path

# Import the original graph builder
import os
os.environ["USE_TORCH"] = "1"
import torch
from original_graph_builder import record_to_graph, FP_DIM


def convert_material_to_record_format(material_data: dict) -> List[Dict]:
    """
    Convert a material dict from template JSON to record format (list of dicts).

    Template format:
        {"name": "...", "macroproperties": {...}, "fingerprint": [...]}

    Record format (expected by original_graph_builder):
        [{"macro_vec": [...], "substance_type_onehot": [...], "fingerprint": [...]}]
    """
    if not material_data or not isinstance(material_data, dict):
        return []

    result = {
        "macro_vec": None,
        "substance_type_onehot": [0, 0, 0, 0, 0],  # Default
        "fingerprint": None,
    }

    # Extract macro_vec
    if "macro_vec" in material_data:
        result["macro_vec"] = material_data["macro_vec"]
    elif "macroproperties" in material_data:
        # Build macro_vec from macroproperties (25D)
        props = material_data["macroproperties"]
        # Molecule properties order (verified from training data)
        mol_keys = [
            'Charge', 'aromatic_rings', 'FeatureRingCount3D', 'FeatureCationCount3D',
            'FeatureAnionCount3D', 'TPSA', 'XLogP', 'HBondDonorCount', 'HBondAcceptorCount',
            'FeatureDonorCount3D', 'FeatureAcceptorCount3D', 'MolecularWeight', 'HeavyAtomCount',
            'Volume3D', 'XStericQuadrupole3D', 'YStericQuadrupole3D', 'ZStericQuadrupole3D',
            'RotatableBondCount', 'EffectiveRotorCount3D', 'ConformerModelRMSD3D',
            'Complexity', 'FeatureHydrophobeCount3D', 'FeatureCount3D', 'qed_weighted',
            'np_likeness_score'
        ]
        # Material properties order
        mat_keys = [
            'formation_energy_per_atom', 'energy_above_hull', 'band_gap', 'density',
            'epsilon_x', 'epsilon_y', 'epsilon_z', 'dielectric_total', 'n_x', 'n_y',
            'n_z', 'refractive_index_avg', 'is_stable', 'is_metal', 'is_semiconductor',
            'is_insulator', 'total_magnetization', 'total_magnetization_normalized_vol',
            'num_magnetic_sites', 'theoretical', 'volume', 'nsites',
            'energy_per_atom', 'ordering', 'elements_count'
        ]

        macro_vec = []
        # Try molecule properties first
        for key in mol_keys[:25]:
            val = props.get(key)
            macro_vec.append(val if val is not None else None)

        # If all None, try material properties
        if all(v is None for v in macro_vec):
            macro_vec = []
            for key in mat_keys[:25]:
                val = props.get(key)
                macro_vec.append(val if val is not None else None)

        result["macro_vec"] = macro_vec

    # Extract fingerprint
    if "fingerprint" in material_data:
        result["fingerprint"] = material_data["fingerprint"]

    # Set substance_type_onehot based on substance_type
    # Order verified from training data: [inorganic, molecule, polymer, bio, other]
    stype = material_data.get("substance_type", "").lower()
    if "inorganic" in stype:
        result["substance_type_onehot"] = [1, 0, 0, 0, 0]
    elif "molecule" in stype:
        result["substance_type_onehot"] = [0, 1, 0, 0, 0]
    elif "polymer" in stype:
        result["substance_type_onehot"] = [0, 0, 1, 0, 0]
    elif "bio" in stype:
        result["substance_type_onehot"] = [0, 0, 0, 1, 0]
    else:
        result["substance_type_onehot"] = [0, 0, 0, 0, 1]  # other

    return [result]


def convert_molecule_to_material(molecule: dict) -> List[Dict]:
    """
    Convert a molecule from the database to material format.

    Molecule format:
        {"cid": ..., "name": ..., "macro_vec": [...], "fingerprint": [...], "macroproperties": {...}}
    """
    result = {
        "macro_vec": molecule.get("macro_vec"),
        "substance_type_onehot": [0, 1, 0, 0, 0],  # molecules (idx 1)
        "fingerprint": molecule.get("fingerprint"),
    }

    # If no macro_vec, try to build from macroproperties
    if result["macro_vec"] is None and "macroproperties" in molecule:
        props = molecule["macroproperties"]
        mol_keys = [
            'Charge', 'aromatic_rings', 'FeatureRingCount3D', 'FeatureCationCount3D',
            'FeatureAnionCount3D', 'TPSA', 'XLogP', 'HBondDonorCount', 'HBondAcceptorCount',
            'FeatureDonorCount3D', 'FeatureAcceptorCount3D', 'MolecularWeight', 'HeavyAtomCount',
            'Volume3D', 'XStericQuadrupole3D', 'YStericQuadrupole3D', 'ZStericQuadrupole3D',
            'RotatableBondCount', 'EffectiveRotorCount3D', 'ConformerModelRMSD3D',
            'Complexity', 'FeatureHydrophobeCount3D', 'FeatureCount3D', 'qed_weighted',
            'np_likeness_score'
        ]
        macro_vec = []
        for key in mol_keys[:25]:
            val = props.get(key)
            macro_vec.append(val if val is not None else None)
        result["macro_vec"] = macro_vec

    return [result]


def safe_float(val, default=0.0):
    """Safely convert to float, handling ranges like '8-8' and units like '300 nm'."""
    if val is None:
        return default
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if not val:
            return default

        # Handle range format like "8-8" or "7-8" (but not negative numbers)
        if '-' in val and not val.startswith('-'):
            parts = val.split('-')
            try:
                nums = [float(p.split()[0]) for p in parts if p.strip()]
                if nums:
                    return sum(nums) / len(nums)
            except ValueError:
                pass

        # Extract numeric part from strings with units like "300 nm", "25 °C", "1 h"
        import re
        match = re.match(r'^([-+]?\d*\.?\d+)', val)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return default

        try:
            return float(val)
        except ValueError:
            return default
    return default


def template_to_record(template: dict, candidate_molecule: dict, task: str = "lower_detection_limit") -> dict:
    """
    Convert template + candidate molecule to training record format.

    Args:
        template: Template sensor record (from JSON)
        candidate_molecule: Molecule to use as probe_material
        task: Task name for condition node building

    Returns:
        Record dict compatible with original_graph_builder.record_to_graph()
    """
    record = {}

    # Material fields - convert from template format to record format
    material_fields = [
        "channel", "gate", "dielectric_layer", "source", "drain", "substrate",
        "surface_functionalization", "detect_target", "test_medium"
    ]

    for field in material_fields:
        if field in template and template[field]:
            if isinstance(template[field], list):
                # Already a list of materials
                record[field] = [convert_material_to_record_format(m)[0] for m in template[field] if m]
            else:
                # Single material dict
                record[field] = convert_material_to_record_format(template[field])
        else:
            record[field] = []

    # Replace probe_material with candidate molecule
    record["probe_material"] = convert_molecule_to_material(candidate_molecule)

    # Handle annealing_atmosphere
    if "annealing_atmosphere" in template and template["annealing_atmosphere"]:
        if isinstance(template["annealing_atmosphere"], list):
            record["annealing_atmosphere"] = [convert_material_to_record_format(m)[0] for m in template["annealing_atmosphere"] if m]
        else:
            record["annealing_atmosphere"] = convert_material_to_record_format(template["annealing_atmosphere"])
    else:
        record["annealing_atmosphere"] = []

    # Scalar fields for condition node
    record["dielectric_layer_thickness"] = safe_float(template.get("dielectric_layer_thickness"))
    record["substrate_thickness"] = safe_float(template.get("substrate_thickness"))
    record["test_operating_temperature_celcius"] = safe_float(template.get("test_operating_temperature_celcius", 25))
    record["annealing_temperature"] = safe_float(template.get("annealing_temperature"))
    record["annealing_time"] = safe_float(template.get("annealing_time"))
    record["hydrothermal_temperature"] = safe_float(template.get("hydrothermal_temperature"))
    record["hydrothermal_time"] = safe_float(template.get("hydrothermal_time"))

    # pH values
    ph_val = template.get("pH_value", {})
    if isinstance(ph_val, dict):
        ph_str = ph_val.get("value", "7")
    else:
        ph_str = ph_val if ph_val else "7"

    # Parse pH range (e.g., "7-8" or "8-8")
    if isinstance(ph_str, str) and "-" in ph_str and not ph_str.startswith("-"):
        parts = ph_str.split("-")
        try:
            record["pH_min"] = float(parts[0])
            record["pH_max"] = float(parts[-1])
        except ValueError:
            record["pH_min"] = 7.0
            record["pH_max"] = 7.0
    else:
        ph_float = safe_float(ph_str, 7.0)
        record["pH_min"] = ph_float
        record["pH_max"] = ph_float

    # One-hot encodings (use defaults if not available)
    record["sensor_type_onehot"] = template.get("sensor_type_onehot", [1, 0, 0])  # [solid, liquid, gas]
    record["structure_dimensionality_onehot"] = template.get("structure_dimensionality_onehot", [1, 0, 0])
    record["structure_design_type_onehot"] = template.get("structure_design_type_onehot", [1, 0, 0, 0])  # standard

    # For sensitivity task
    if task == "sensitivity_numerator":
        record["sensitivity_denominator_value"] = safe_float(template.get("sensitivity_denominator_value"))
        record["sensitivity_numerator_unit_onehot"] = template.get("sensitivity_numerator_unit_onehot", [0, 0, 0])
        record["sensitivity_denominator_unit_onehot"] = template.get("sensitivity_denominator_unit_onehot", [0, 0, 0])

    return record


def build_graph_for_screening(template: dict, candidate_molecule: dict, task: str = "lower_detection_limit"):
    """
    Build a HeteroData graph for screening.

    Args:
        template: Template sensor (from JSON)
        candidate_molecule: Candidate molecule for probe_material
        task: Task name

    Returns:
        HeteroData graph compatible with trained model
    """
    record = template_to_record(template, candidate_molecule, task=task)
    graph = record_to_graph(record, task=task)
    return graph


# Cache for template graphs from training data
_template_graph_cache = {}


def load_template_graph_from_training(task: str, doi: str = "10.1038/s44221-025-00505-9"):
    """Load template graph from training data pickle file."""
    import pickle
    cache_key = (task, doi)
    if cache_key in _template_graph_cache:
        return _template_graph_cache[cache_key]

    pkl_path = Path(__file__).parent / f"graphs_aug_{task}.pkl"
    with open(pkl_path, 'rb') as f:
        graphs = pickle.load(f)

    for g in graphs:
        if doi in g.get('doi', '') and g.get('is_original', False):
            _template_graph_cache[cache_key] = g
            return g

    raise ValueError(f"Template graph not found for doi={doi}")


def build_graph_from_training_template(candidate_molecule: dict, task: str = "lower_detection_limit",
                                        doi: str = "10.1038/s44221-025-00505-9"):
    """
    Build screening graph by loading template from training data and replacing probe_material.

    This ensures all material nodes match training data exactly, only probe_material is swapped.
    """
    from torch_geometric.data import HeteroData
    from original_graph_builder import build_material_node, aggregate_fingerprints

    # Load template graph from training data
    template_graph = load_template_graph_from_training(task, doi)

    # Build new probe_material node from candidate molecule
    mol_record = [{
        'macro_vec': candidate_molecule.get('macro_vec'),
        'substance_type_onehot': [0, 1, 0, 0, 0],  # molecule
        'fingerprint': candidate_molecule.get('fingerprint'),
    }]
    new_pm_node = build_material_node(mol_record)
    new_pm_fp = aggregate_fingerprints(mol_record)

    # Create new graph (copy template, replace probe_material)
    data = HeteroData()
    for ntype, feat in template_graph['nodes'].items():
        if ntype == 'probe_material':
            data[ntype].x = torch.tensor(new_pm_node, dtype=torch.float).unsqueeze(0)
        else:
            data[ntype].x = torch.tensor(feat, dtype=torch.float).unsqueeze(0)

    for (src, rel, dst), idx in template_graph['edges'].items():
        data[(src, rel, dst)].edge_index = torch.tensor(idx, dtype=torch.long)

    # Copy fingerprints, replace probe_material
    data.fingerprints = {}
    for k, v in template_graph['fingerprints'].items():
        if k == 'probe_material':
            data.fingerprints[k] = torch.tensor(new_pm_fp, dtype=torch.float)
        else:
            data.fingerprints[k] = torch.tensor(v, dtype=torch.float)

    return data


def load_template_from_json(json_path: str) -> dict:
    """Load template from JSON file."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    if 'records' in data and isinstance(data['records'], list):
        return data['records'][0]
    return data


if __name__ == "__main__":
    # Test the adapter
    # Usage: python screening_adapter.py --molecules /path/to/molecules.jsonl
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecules", type=str, required=True, help="Path to molecules JSONL file")
    args = parser.parse_args()

    template_path = Path(__file__).parent / "10.1038_s44221-025-00505-9.json"

    print("Testing adapter...")

    # Load template
    template = load_template_from_json(template_path)
    print(f"Template loaded: detect_target = {template.get('detect_target', {}).get('name', 'N/A')}")

    # Load one molecule
    with open(args.molecules) as f:
        mol = json.loads(f.readline())
    print(f"Molecule loaded: {mol.get('name', 'N/A')}")

    # Build graph
    graph = build_graph_for_screening(template, mol, task="lower_detection_limit")
    print(f"\nGraph built successfully!")
    print(f"Node types: {list(graph.x_dict.keys())}")
    print(f"Edge types: {len(graph.edge_index_dict)} types")
    print(f"Fingerprint fields: {list(graph.fingerprints.keys())}")

    # Check dimensions
    for ntype, x in graph.x_dict.items():
        print(f"  {ntype}: {x.shape}")
