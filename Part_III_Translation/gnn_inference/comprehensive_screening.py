#!/usr/bin/env python3
"""
Comprehensive Virtual Screening Pipeline

Screen molecules for optimal probe_material candidates.
Evaluates LDL, UDL, and Sensitivity (with %/ppt and mV/ppt units).

Output: CSV with all predictions and combined score.
"""
import argparse
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch_geometric.data import HeteroData

from inference_utils import InferenceEngine
from screening_adapter import load_template_graph_from_training, build_graph_from_training_template
from original_graph_builder import build_material_node, aggregate_fingerprints


class ComprehensiveScreener:
    """Screen molecules across LDL, UDL, and Sensitivity tasks."""

    def __init__(self, checkpoint_dir: str = "checkpoints", device: str = "cpu"):
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)

        print("Loading models...")
        self.engine_ldl = InferenceEngine.from_checkpoint(
            self.checkpoint_dir / "model_ldl.pt", device=device
        )
        self.engine_udl = InferenceEngine.from_checkpoint(
            self.checkpoint_dir / "model_udl.pt", device=device
        )
        self.engine_sens = InferenceEngine.from_checkpoint(
            self.checkpoint_dir / "model_sens.pt", device=device
        )

        # Load template for sensitivity (need to extend condition node)
        self.template = load_template_graph_from_training(
            'lower_detection_limit',
            doi='10.1038/s44221-025-00505-9'
        )
        print("Models loaded.\n")

    def predict_sensitivity(self, mol: dict, num_onehot: List[int]) -> np.ndarray:
        """Predict sensitivity with specified numerator unit."""
        mol_record = [{
            'macro_vec': mol.get('macro_vec'),
            'substance_type_onehot': [0, 1, 0, 0, 0],
            'fingerprint': mol.get('fingerprint'),
        }]
        new_pm_node = build_material_node(mol_record)
        new_pm_fp = aggregate_fingerprints(mol_record)

        data = HeteroData()
        for ntype, feat in self.template['nodes'].items():
            if ntype == 'condition':
                # Extend 28D -> 36D for sensitivity
                ldl_scalars = feat[:9]
                ldl_onehots = feat[9:19]
                ldl_masks = feat[19:28]
                sens_scalars = np.concatenate([ldl_scalars, [1e-6]])  # ppt level
                sens_onehots = np.concatenate([ldl_onehots, num_onehot, [1, 0, 0]])  # ppm
                sens_masks = np.concatenate([ldl_masks, [1.0]])
                feat_new = np.concatenate([sens_scalars, sens_onehots, sens_masks])
                data[ntype].x = torch.tensor(feat_new, dtype=torch.float).unsqueeze(0)
            elif ntype == 'probe_material':
                data[ntype].x = torch.tensor(new_pm_node, dtype=torch.float).unsqueeze(0)
            else:
                data[ntype].x = torch.tensor(feat, dtype=torch.float).unsqueeze(0)

        for (src, rel, dst), idx in self.template['edges'].items():
            data[(src, rel, dst)].edge_index = torch.tensor(idx, dtype=torch.long)

        data.fingerprints = {}
        for k, v in self.template['fingerprints'].items():
            if k == 'probe_material':
                data.fingerprints[k] = torch.tensor(new_pm_fp, dtype=torch.float)
            else:
                data.fingerprints[k] = torch.tensor(v, dtype=torch.float)

        return self.engine_sens.predict(data)

    def predict_molecule(self, mol: dict) -> Optional[Dict]:
        """Predict all tasks for a single molecule."""
        try:
            # LDL
            graph_ldl = build_graph_from_training_template(mol, task='lower_detection_limit')
            probs_ldl = self.engine_ldl.predict(graph_ldl)

            # UDL
            graph_udl = build_graph_from_training_template(mol, task='upper_detection_limit')
            probs_udl = self.engine_udl.predict(graph_udl)

            # Sensitivity %/ppt
            probs_sens_pct = self.predict_sensitivity(mol, [1, 0, 0])

            # Sensitivity mV/ppt
            probs_sens_mv = self.predict_sensitivity(mol, [0, 1, 0])

            # Combined score: LDL want C0, UDL want C2, Sens want C2
            ldl_score = float(probs_ldl[0])
            udl_score = float(probs_udl[2])
            sens_score = max(float(probs_sens_pct[2]), float(probs_sens_mv[2]))

            # Avoid zero in product
            combined_score = ldl_score * udl_score * max(sens_score, 1e-6)

            return {
                'cid': mol.get('cid', ''),
                'name': mol.get('name', 'N/A'),
                'smiles': mol.get('smiles', ''),
                # LDL
                'ldl_pred': int(probs_ldl.argmax()),
                'ldl_p0': float(probs_ldl[0]),
                'ldl_p1': float(probs_ldl[1]),
                'ldl_p2': float(probs_ldl[2]),
                # UDL
                'udl_pred': int(probs_udl.argmax()),
                'udl_p0': float(probs_udl[0]),
                'udl_p1': float(probs_udl[1]),
                'udl_p2': float(probs_udl[2]),
                # Sensitivity %/ppt
                'sens_pct_pred': int(probs_sens_pct.argmax()),
                'sens_pct_p0': float(probs_sens_pct[0]),
                'sens_pct_p1': float(probs_sens_pct[1]),
                'sens_pct_p2': float(probs_sens_pct[2]),
                # Sensitivity mV/ppt
                'sens_mv_pred': int(probs_sens_mv.argmax()),
                'sens_mv_p0': float(probs_sens_mv[0]),
                'sens_mv_p1': float(probs_sens_mv[1]),
                'sens_mv_p2': float(probs_sens_mv[2]),
                # Combined score
                'score': combined_score,
            }
        except Exception as e:
            return None

    def stream_molecules(self, molecules_dir: str, limit: Optional[int] = None):
        """Stream molecules from JSONL files."""
        molecules_path = Path(molecules_dir)
        jsonl_files = sorted(molecules_path.glob("*.jsonl"))

        if not jsonl_files:
            raise FileNotFoundError(f"No JSONL files found in {molecules_dir}")

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
        self,
        molecules_dir: str,
        output_path: str,
        limit: Optional[int] = None,
        top_k: Optional[int] = None,
    ):
        """Run comprehensive screening."""
        print("=" * 70)
        print("Comprehensive Virtual Screening")
        print("=" * 70)
        print(f"Molecules dir: {molecules_dir}")
        print(f"Output: {output_path}")
        if limit:
            print(f"Limit: {limit} molecules")
        if top_k:
            print(f"Top-K: {top_k}")
        print()

        results = []
        start_time = time.time()
        processed = 0
        errors = 0

        for mol in self.stream_molecules(molecules_dir, limit=limit):
            result = self.predict_molecule(mol)
            if result:
                results.append(result)
                processed += 1
            else:
                errors += 1

            if (processed + errors) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (processed + errors) / elapsed
                print(f"  Processed: {processed:,} | Errors: {errors} | "
                      f"Rate: {rate:.1f} mol/s | "
                      f"Best UDL P(C2): {max(r['udl_p2'] for r in results):.4f}")

        elapsed = time.time() - start_time
        print()
        print(f"Completed: {processed:,} molecules in {elapsed:.1f}s")
        print(f"Rate: {processed/elapsed:.1f} mol/s")
        print(f"Errors: {errors}")

        # Sort by score
        results.sort(key=lambda x: -x['score'])

        # Keep top-k if specified
        if top_k:
            results = results[:top_k]

        # Add rank
        for i, r in enumerate(results, 1):
            r['rank'] = i

        # Save to CSV
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            'rank', 'cid', 'name', 'smiles', 'score',
            'ldl_pred', 'ldl_p0', 'ldl_p1', 'ldl_p2',
            'udl_pred', 'udl_p0', 'udl_p1', 'udl_p2',
            'sens_pct_pred', 'sens_pct_p0', 'sens_pct_p1', 'sens_pct_p2',
            'sens_mv_pred', 'sens_mv_p0', 'sens_mv_p1', 'sens_mv_p2',
        ]

        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to: {output_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("Summary Statistics")
        print("=" * 70)

        ldl_c0 = sum(1 for r in results if r['ldl_pred'] == 0)
        udl_c2 = sum(1 for r in results if r['udl_pred'] == 2)
        sens_pct_c2 = sum(1 for r in results if r['sens_pct_pred'] == 2)
        sens_mv_c2 = sum(1 for r in results if r['sens_mv_pred'] == 2)

        print(f"LDL Class 0 (Good): {ldl_c0} ({ldl_c0/len(results)*100:.1f}%)")
        print(f"UDL Class 2 (Good): {udl_c2} ({udl_c2/len(results)*100:.1f}%)")
        print(f"Sens %/ppt Class 2: {sens_pct_c2} ({sens_pct_c2/len(results)*100:.1f}%)")
        print(f"Sens mV/ppt Class 2: {sens_mv_c2} ({sens_mv_c2/len(results)*100:.1f}%)")

        print("\nTop 10 Candidates:")
        print("-" * 70)
        for r in results[:10]:
            print(f"{r['rank']:3d}. {r['name'][:40]:40s}")
            print(f"     LDL=C{r['ldl_pred']}({r['ldl_p0']:.3f}) "
                  f"UDL=C{r['udl_pred']}({r['udl_p2']:.3f}) "
                  f"Sens%=C{r['sens_pct_pred']}({r['sens_pct_p2']:.4f}) "
                  f"Score={r['score']:.6f}")

        # Time estimate for full dataset
        print("\n" + "=" * 70)
        print("Time Estimate for Full Dataset")
        print("=" * 70)
        rate = processed / elapsed
        total_mols = 120_000_000
        est_seconds = total_mols / rate
        est_hours = est_seconds / 3600
        est_days = est_hours / 24
        print(f"Current rate: {rate:.1f} mol/s")
        print(f"For {total_mols:,} molecules:")
        print(f"  Estimated time: {est_hours:.1f} hours = {est_days:.1f} days")
        print(f"  With 8x parallel: ~{est_days/8:.1f} days")
        print(f"  With GPU + batch: ~{est_days/50:.1f} days (estimated)")

        return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Virtual Screening")
    parser.add_argument("--molecules", type=str, required=True,
                       help="Directory containing molecule JSONL files")
    parser.add_argument("--output", type=str, default="results/comprehensive_screening.csv",
                       help="Output CSV path")
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit number of molecules (for testing)")
    parser.add_argument("--top-k", type=int, default=None,
                       help="Only save top-k results")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device for inference")
    args = parser.parse_args()

    screener = ComprehensiveScreener(device=args.device)
    screener.run_screening(
        molecules_dir=args.molecules,
        output_path=args.output,
        limit=args.limit,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
