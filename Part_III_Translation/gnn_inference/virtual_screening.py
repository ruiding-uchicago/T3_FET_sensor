#!/usr/bin/env python3
"""
Virtual Screening Pipeline for DTE-GNN

Screen molecules as potential probe_material candidates.
Keep sensor template fixed, swap probe_material, predict performance.

Usage:
    python virtual_screening.py \
        --checkpoint checkpoints/model_ldl.pt \
        --template 10.1038_s44221-025-00505-9.json \
        --molecules /path/to/molecules_jsonl_directory \
        --output screening_results_ldl.csv \
        --top-k 1000
"""
import argparse
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional, Iterator
import numpy as np
import torch

from inference_utils import InferenceEngine
from screening_adapter import build_graph_from_training_template, load_template_from_json


def load_template(json_path: str) -> dict:
    """Load template sensor JSON."""
    return load_template_from_json(json_path)


def stream_molecules(molecules_dir: str, limit: Optional[int] = None) -> Iterator[dict]:
    """
    Stream molecules from JSONL files.

    Args:
        molecules_dir: Directory containing molecule JSONL files
        limit: Optional limit on total molecules to process

    Yields:
        Molecule dictionaries with cid, name, smiles, macro_vec, fingerprint
    """
    molecules_path = Path(molecules_dir)
    jsonl_files = sorted(molecules_path.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {molecules_dir}")

    print(f"Found {len(jsonl_files)} JSONL files")

    count = 0
    for jsonl_file in jsonl_files:
        with open(jsonl_file, 'r') as f:
            for line in f:
                if limit and count >= limit:
                    return
                try:
                    mol = json.loads(line.strip())
                    yield mol
                    count += 1
                except json.JSONDecodeError:
                    continue

        if limit and count >= limit:
            return


def run_screening(
    engine: InferenceEngine,
    template: dict,
    molecules_dir: str,
    output_path: str,
    top_k: int = 1000,
    batch_size: int = 64,
    limit: Optional[int] = None,
    task: str = 'lower_detection_limit',
):
    """
    Run virtual screening.

    Args:
        engine: InferenceEngine with loaded model
        template: Sensor template
        molecules_dir: Directory with molecule JSONL files
        output_path: Path to save results CSV
        top_k: Number of top candidates to save
        batch_size: Batch size for inference
        limit: Optional limit on molecules to screen
        task: Task name for interpreting results
    """
    print(f"\n{'='*60}")
    print("Virtual Screening")
    print(f"{'='*60}")
    print(f"Task: {task}")
    print(f"Top-K: {top_k}")
    print(f"Batch size: {batch_size}")
    if limit:
        print(f"Limit: {limit} molecules")

    # Determine which class is "good"
    if task == 'lower_detection_limit':
        good_class = 0  # Lower LDL = better
        print("Target: Class 0 (Low detection limit = Good)")
    else:
        good_class = 2  # Higher UDL/Sens = better
        print("Target: Class 2 (High value = Good)")

    # Results storage: (score, cid, name, smiles, probs)
    results = []

    start_time = time.time()
    processed = 0
    batch_molecules = []
    batch_graphs = []

    print("\nProcessing molecules...")

    for mol in stream_molecules(molecules_dir, limit=limit):
        # Build graph using training template (only replace probe_material)
        try:
            graph = build_graph_from_training_template(mol, task=task)
            batch_molecules.append(mol)
            batch_graphs.append(graph)
        except Exception as e:
            continue

        # Process batch
        if len(batch_graphs) >= batch_size:
            probs_batch = engine.predict_batch(batch_graphs, batch_size=batch_size, show_progress=False)

            for mol_data, probs in zip(batch_molecules, probs_batch):
                score = probs[good_class]
                results.append((
                    score,
                    mol_data.get('cid', ''),
                    mol_data.get('name', ''),
                    mol_data.get('smiles', ''),
                    probs.tolist(),
                ))

            processed += len(batch_graphs)
            batch_molecules = []
            batch_graphs = []

            # Progress update
            if processed % 10000 == 0:
                elapsed = time.time() - start_time
                rate = processed / elapsed
                print(f"  Processed: {processed:,} | Rate: {rate:.1f} mol/s | "
                      f"Best score so far: {max(r[0] for r in results):.4f}")

    # Process remaining batch
    if batch_graphs:
        probs_batch = engine.predict_batch(batch_graphs, batch_size=batch_size, show_progress=False)
        for mol_data, probs in zip(batch_molecules, probs_batch):
            score = probs[good_class]
            results.append((
                score,
                mol_data.get('cid', ''),
                mol_data.get('name', ''),
                mol_data.get('smiles', ''),
                probs.tolist(),
            ))
        processed += len(batch_graphs)

    elapsed = time.time() - start_time
    print(f"\nTotal processed: {processed:,} molecules in {elapsed:.1f}s")
    print(f"Average rate: {processed/elapsed:.1f} mol/s")

    # Sort by score (descending) and take top-k
    results.sort(key=lambda x: x[0], reverse=True)
    top_results = results[:top_k]

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['rank', 'cid', 'name', 'smiles', 'score', 'prob_class0', 'prob_class1', 'prob_class2'])
        for rank, (score, cid, name, smiles, probs) in enumerate(top_results, 1):
            writer.writerow([rank, cid, name, smiles, f"{score:.6f}",
                           f"{probs[0]:.6f}", f"{probs[1]:.6f}", f"{probs[2]:.6f}"])

    print(f"\nResults saved to: {output_path}")
    print(f"\nTop 10 candidates:")
    print("-" * 80)
    for rank, (score, cid, name, smiles, probs) in enumerate(top_results[:10], 1):
        print(f"{rank:3d}. {name[:40]:40s} | score={score:.4f} | cid={cid}")

    return top_results


def main():
    parser = argparse.ArgumentParser(description="Virtual Screening with DTE-GNN")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--template", type=str, required=True,
                       help="Path to template sensor JSON")
    parser.add_argument("--molecules", type=str, required=True,
                       help="Directory containing molecule JSONL files")
    parser.add_argument("--output", type=str, default="screening_results.csv",
                       help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=1000,
                       help="Number of top candidates to save")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for inference")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of molecules to screen (for testing)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device for inference")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    engine = InferenceEngine.from_checkpoint(args.checkpoint, device=args.device)

    # Determine task from checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    task = checkpoint.get('task', 'lower_detection_limit')

    # Load template
    print(f"Loading template from: {args.template}")
    template = load_template(args.template)
    print(f"  Template detect_target: {template.get('detect_target', {}).get('name', 'N/A')}")
    print(f"  Template probe_material: {template.get('probe_material', {}).get('name', 'N/A')} (will be replaced)")

    # Run screening
    run_screening(
        engine=engine,
        template=template,
        molecules_dir=args.molecules,
        output_path=args.output,
        top_k=args.top_k,
        batch_size=args.batch_size,
        limit=args.limit,
        task=task,
    )


if __name__ == "__main__":
    main()
