#!/usr/bin/env python3
"""
Demo script for DTE-GNN inference.

This script demonstrates:
1. How to load a trained checkpoint
2. How to run inference on a single sensor JSON
3. How to interpret the results

Usage:
    python demo_inference.py --checkpoint checkpoints/model_ldl.pt --sensor 10.1038_s44221-025-00505-9.json
"""
import argparse
import json
from pathlib import Path

import torch
import numpy as np


def load_sensor_json(json_path: str) -> dict:
    """Load sensor data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def print_sensor_summary(sensor_data: dict) -> None:
    """Print key information about the sensor."""
    print("\n" + "="*60)
    print("Sensor Summary")
    print("="*60)

    # Handle records wrapper
    if 'records' in sensor_data and isinstance(sensor_data['records'], list):
        record = sensor_data['records'][0]
        print(f"  DOI: {sensor_data.get('DOI', 'N/A')}")
    else:
        record = sensor_data

    # Key fields to display
    key_fields = [
        'detect_target', 'probe_material', 'channel', 'gate',
        'test_medium', 'surface_functionalization',
        'lower_detection_limit', 'upper_detection_limit',
        'sensitivity_numerator', 'response_time'
    ]

    for field in key_fields:
        if field in record:
            value = record[field]
            if isinstance(value, dict):
                # Extract the 'name' from nested dicts
                name = value.get('name', value.get('value', str(value)[:50]))
                print(f"  {field}: {name}")
            else:
                print(f"  {field}: {value}")


def demo_inference():
    """
    Demo: Load checkpoint and run inference on sample sensor.

    Note: This is a demonstration. For actual inference, you need:
    1. A trained checkpoint file
    2. The graph building pipeline to convert JSON to HeteroData
    """
    print("\n" + "="*70)
    print("DTE-GNN Inference Demo")
    print("="*70)

    print("""
This demo shows the inference workflow:

Step 1: Train and Save Model
-----------------------------
    python train_hetero_gnn_residual_sgnn.py \\
        --task lower_detection_limit \\
        --graph-suffix _aug \\
        --use-jk --gcn2-alpha 0.15 --use-att-readout \\
        --hidden-dim 128 --self-loops \\
        --pretrain-gnn-epochs 40 --epochs 20 \\
        --save-checkpoint checkpoints/model_ldl.pt

Step 2: Load and Run Inference
------------------------------
    from inference_utils import InferenceEngine

    # Load the engine
    engine = InferenceEngine.from_checkpoint('checkpoints/model_ldl.pt', device='cuda')

    # Build graph from sensor JSON (requires graph_builder module)
    graph = build_graph_from_json(sensor_json)

    # Run inference
    probs = engine.predict(graph)
    print(f"Class probabilities: {probs}")
    print(f"Predicted class: {probs.argmax()}")

Step 3: Interpret Results
-------------------------
For LDL task:
    - Class 0: Good (Low detection limit = better sensitivity)
    - Class 1: Medium
    - Class 2: Poor (High detection limit)

For UDL/Sensitivity tasks:
    - Class 0: Poor
    - Class 1: Medium
    - Class 2: Good (Higher is better)
""")

    # Check if sample sensor exists
    sensor_path = Path(__file__).parent / "10.1038_s44221-025-00505-9.json"
    if sensor_path.exists():
        sensor_data = load_sensor_json(sensor_path)
        print_sensor_summary(sensor_data)

        print("\n" + "="*60)
        print("Available Fingerprints in this sensor:")
        print("="*60)

        # Handle records wrapper
        if 'records' in sensor_data and isinstance(sensor_data['records'], list):
            record = sensor_data['records'][0]
        else:
            record = sensor_data

        # Check which fields have fingerprints
        fp_fields = ['channel', 'detect_target', 'probe_material', 'test_medium', 'surface_functionalization']
        for field in fp_fields:
            if field in record and isinstance(record[field], dict):
                has_fp = 'fingerprint' in record[field]
                has_macro = 'macroproperties' in record[field]
                status = []
                if has_fp:
                    fp_len = len(record[field]['fingerprint'])
                    status.append(f"fingerprint ({fp_len}D) ✓")
                if has_macro:
                    status.append("macroproperties ✓")
                print(f"  {field}: {', '.join(status) if status else 'no descriptors'}")
            else:
                print(f"  {field}: not found or not a dict")
    else:
        print(f"\nSample sensor file not found: {sensor_path}")

    print("\n" + "="*70)
    print("End of Demo")
    print("="*70)


if __name__ == "__main__":
    demo_inference()
