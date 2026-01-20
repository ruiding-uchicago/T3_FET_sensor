#!/usr/bin/env python3
"""
Bayesian optimization runner using FMRG-Ballet (ol_filter_dkbo_one_shot)
for the physics-augmented SGNN baseline.

Features:
    - Builds the discrete grid (1728 configs) and keeps a JSON/CSV record.
    - Picks new candidates with Ballet; falls back to random when cold start.
    - Runs train_hetero_gnn_residual_sgnn.py, parses Original metrics, logs to CSV.
    - Optional fast-smoke mode (n-folds=1, pretrain/epochs=1) to validate the pipeline.

Usage examples:
    # Smoke test (1 eval, n-folds=1, very short epochs)
    PYTORCH_ENABLE_MPS_FALLBACK=1 python bo_ballet.py --max-evals 1 --bo-batch 1 --fast-smoke

    # Full BO (100 evals, n-folds=5)
    PYTORCH_ENABLE_MPS_FALLBACK=1 python bo_ballet.py --max-evals 100 --bo-batch 3
"""
import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

EXPERIMENT_DIR = Path(__file__).resolve().parent
BALLET_DIR = EXPERIMENT_DIR / "FMRG-Ballet"
GNN_TRAINING_DIR = EXPERIMENT_DIR.parent  # parent contains train_hetero_gnn_residual_sgnn.py

sys.path.append(str(BALLET_DIR))
from src.opt.optimization import ol_filter_dkbo_one_shot  # noqa: E402

# Parameter order used for tensors/CSV
PARAM_KEYS = [
    "hidden_dim",
    "dropout",
    "lr",
    "weight_decay",
    "pretrain",
    "epochs",
    "init_gate",
]


def build_grid(grid_json: Path) -> List[Dict[str, float]]:
    """
    Build the full hyperparameter grid and persist to JSON.
    """
    grid = {
        "hidden_dim": [64, 96, 128],
        "dropout": [0.1, 0.15, 0.2, 0.25, 0.3, 0.35],
        "lr": [3e-5, 5e-5, 1e-4, 2e-4, 3e-4, 5e-4],
        "weight_decay": [5e-6, 1e-5, 5e-5, 1e-4],
        "pretrain": [10, 20, 40],
        "epochs": [10, 20, 30],
        "init_gate": [0.1, 0.15, 0.2, 0.25, 0.3],
        "batch_size": [32],  # keep fixed to reduce variance
    }
    keys = list(grid.keys())
    rows = []
    for values in product_lists([grid[k] for k in keys]):
        row = dict(zip(keys, values))
        rows.append(row)
    grid_json.parent.mkdir(parents=True, exist_ok=True)
    with grid_json.open("w") as f:
        json.dump(rows, f, indent=2)
    return rows


def product_lists(lists: Sequence[Sequence[float]]) -> List[List[float]]:
    if not lists:
        return []
    out = [[]]
    for seq in lists:
        out = [row + [x] for row in out for x in seq]
    return out


def grid_to_tensor(grid: List[Dict[str, float]]) -> torch.Tensor:
    data = []
    for row in grid:
        data.append([row[k] for k in PARAM_KEYS])
    return torch.tensor(data, dtype=torch.float)


def load_results(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame(
            columns=PARAM_KEYS
            + [
                "batch_size",
                "macro_f1_orig",
                "acc_orig",
                "macro_f1_aug",
                "acc_aug",
                "status",
                "log_path",
                "runtime_min",
                "grid_index",
                "n_folds",
            ]
        )
    return pd.read_csv(csv_path)


def save_row(csv_path: Path, row: Dict) -> None:
    exists = csv_path.exists()
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def pick_candidates(
    grid: List[Dict[str, float]],
    results: pd.DataFrame,
    batch: int,
    train_times: int = 40,
    beta: float = 5.0,
    acq: str = "ucb",
) -> List[int]:
    grid_tensor = grid_to_tensor(grid)
    done = set(results.get("grid_index", []))
    if len(results) == 0 or results["macro_f1_orig"].notna().sum() == 0:
        pool = list(set(range(len(grid))) - done)
        return random.sample(pool, k=min(batch, len(pool)))

    observed = results.dropna(subset=["macro_f1_orig"])
    # Ensure numeric tensor; coerce objects/strings to float and drop NaNs
    obs_x_df = observed[PARAM_KEYS].apply(pd.to_numeric, errors="coerce")
    observed = observed[obs_x_df.notnull().all(axis=1)]
    obs_x = torch.tensor(obs_x_df.loc[observed.index].values, dtype=torch.float)

    obs_y_np = pd.to_numeric(observed["macro_f1_orig"], errors="coerce").to_numpy(dtype=np.float32)
    # Avoid single-sample issues inside Ballet: duplicate when only one point
    if obs_y_np.size == 1:
        obs_y_np = np.concatenate([obs_y_np, obs_y_np])
        obs_x = torch.cat([obs_x, obs_x], dim=0)
    obs_y = torch.tensor(obs_y_np, dtype=torch.float)

    try:
        cand_tensors = ol_filter_dkbo_one_shot(
            x_tensor=grid_tensor,
            init_x=obs_x,
            init_y=obs_y,
            n_iter=batch,
            train_times=train_times,
            beta=beta,
            acq=acq,
            lr=1e-2,
            fix_seed=False,
            pretrained=False,
        )
    except Exception as e:
        print(f"[WARN] Ballet selector failed ({e}), falling back to random.", flush=True)
        pool = list(set(range(len(grid))) - done)
        return random.sample(pool, k=min(batch, len(pool)))
    idxs: List[int] = []
    for c in cand_tensors:
        dists = torch.sum((grid_tensor - c) ** 2, dim=1)
        idxs.append(int(torch.argmin(dists)))
    # Deduplicate and drop already-finished configs
    done = set(results["grid_index"]) if "grid_index" in results else set()
    dedup = []
    for i in idxs:
        if i in done or i in dedup:
            continue
        dedup.append(i)
    if not dedup:
        # Ballet returned only seen points; fallback to random unseen
        pool = list(set(range(len(grid))) - done)
        return random.sample(pool, k=min(batch, len(pool)))
    return dedup


def parse_metrics(stdout: str) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Robust parser that works for both CV summary blocks and single-run outputs.
    Returns: (aug_acc, aug_f1, orig_acc, orig_f1)
    """
    aug_acc = aug_f1 = orig_acc = orig_f1 = None

    # 1) CV summary blocks
    aug_block = re.search(r"\[Augmented Test Set\](.*?)(?:\n\[|$)", stdout, re.S)
    if aug_block:
        acc_m = re.search(r"Accuracy:\s*([0-9.]+)", aug_block.group(1))
        f1_m = re.search(r"Macro F1:\s*([0-9.]+)", aug_block.group(1))
        if acc_m:
            aug_acc = float(acc_m.group(1))
        if f1_m:
            aug_f1 = float(f1_m.group(1))

    orig_block = re.search(r"\[Original[^\]]*\](.*?)(?:\n\[|$)", stdout, re.S)
    if orig_block:
        acc_m = re.search(r"Accuracy:\s*([0-9.]+)", orig_block.group(1))
        f1_m = re.search(r"Macro F1:\s*([0-9.]+)", orig_block.group(1))
        if acc_m:
            orig_acc = float(acc_m.group(1))
        if f1_m:
            orig_f1 = float(f1_m.group(1))

    # 2) Fallback to single-run final lines, take the last occurrence
    if aug_acc is None or aug_f1 is None:
        tests = re.findall(r"Test:\s*acc\s*([0-9.]+)\s*\|\s*macro_f1\s*([0-9.]+)", stdout)
        if tests:
            aug_acc, aug_f1 = map(float, tests[-1])

    if orig_acc is None or orig_f1 is None:
        origs = re.findall(r"Original:\s*acc\s*([0-9.]+)\s*\|\s*macro_f1\s*([0-9.]+)", stdout)
        if origs:
            orig_acc, orig_f1 = map(float, origs[-1])

    return aug_acc, aug_f1, orig_acc, orig_f1


def run_config(
    cfg: Dict[str, float],
    args: argparse.Namespace,
    grid_index: int,
    log_dir: Path,
    train_script: Path,
) -> Dict:
    run_cfg = cfg.copy()
    n_folds = args.n_folds
    if args.fast_smoke:
        run_cfg["pretrain"] = 1
        run_cfg["epochs"] = 1
        n_folds = 1

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"bo_{grid_index}_{int(time.time())}.log"

    cmd = [
        "python",
        str(train_script),
        "--task",
        args.task,
        "--device",
        args.device,
        "--gnn-layers",
        str(args.gnn_layers),
        "--hidden-dim",
        str(run_cfg["hidden_dim"]),
        "--dropout",
        str(run_cfg["dropout"]),
        "--pretrain-gnn-epochs",
        str(int(run_cfg["pretrain"])),
        "--epochs",
        str(int(run_cfg["epochs"])),
        "--batch-size",
        str(int(cfg["batch_size"])),
        "--lr",
        str(run_cfg["lr"]),
        "--weight-decay",
        str(run_cfg["weight_decay"]),
        "--init-gate",
        str(run_cfg["init_gate"]),
        "--graph-suffix",
        args.graph_suffix,
        "--fp-fields",
        args.fp_fields,
        "--fp-branch",
        args.fp_branch,
        "--n-folds",
        str(n_folds),
        "--seed",
        str(args.seed),
    ]
    if args.self_loops:
        cmd.append("--self-loops")
    if args.no_class_weights:
        cmd.append("--no-class-weights")
    if args.use_jk:
        cmd.append("--use-jk")
    # If grid has gcn2_alpha, override per-config; else use CLI default
    gcn2_val = cfg.get("gcn2_alpha", args.gcn2_alpha)
    if gcn2_val is not None and gcn2_val > 0:
        cmd.extend(["--gcn2-alpha", str(gcn2_val)])
    if args.use_att_readout:
        cmd.append("--use-att-readout")

    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    start = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(EXPERIMENT_DIR), env=env)
    runtime_min = (time.time() - start) / 60.0

    with log_path.open("w") as f:
        f.write(proc.stdout)
        if proc.stderr:
            f.write("\n[stderr]\n")
            f.write(proc.stderr)

    status = "ok" if proc.returncode == 0 else f"fail_{proc.returncode}"
    aug_acc, aug_f1, orig_acc, orig_f1 = parse_metrics(proc.stdout)
    row = {
        **{k: cfg[k] for k in PARAM_KEYS},
        "batch_size": cfg["batch_size"],
        "macro_f1_orig": orig_f1,
        "acc_orig": orig_acc,
        "macro_f1_aug": aug_f1,
        "acc_aug": aug_acc,
        "status": status,
        "log_path": str(log_path),
        "runtime_min": runtime_min,
        "grid_index": grid_index,
        "n_folds": n_folds,
    }
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="BO runner with FMRG-Ballet for SGNN physics baseline")
    p.add_argument("--grid-json", type=Path, default=EXPERIMENT_DIR / "bo_grid.json")
    p.add_argument("--results-csv", type=Path, default=EXPERIMENT_DIR / "bo_results.csv")
    p.add_argument("--log-dir", type=Path, default=EXPERIMENT_DIR / "bo_logs")
    p.add_argument("--max-evals", type=int, default=5)
    p.add_argument("--bo-batch", type=int, default=3)
    p.add_argument("--expected-grid-size", type=int, default=9720,
                   help="Set to 0 to skip size check")
    p.add_argument("--task", default="lower_detection_limit")
    p.add_argument("--graph-suffix", default="_aug")
    p.add_argument("--device", default="cpu")
    p.add_argument("--gnn-layers", type=int, default=1)
    p.add_argument("--fp-fields", choices=["tpm", "key", "all"], default="key")
    p.add_argument("--fp-branch", choices=["mlp", "transformer", "snn"], default="mlp")
    p.add_argument("--train-script", type=Path, default=GNN_TRAINING_DIR / "train_hetero_gnn_residual_sgnn.py")
    p.add_argument("--self-loops", action="store_true", default=True)
    p.add_argument("--no-class-weights", action="store_true", default=True)
    p.add_argument("--use-jk", action="store_true", help="Pass --use-jk to train script")
    p.add_argument("--gcn2-alpha", type=float, default=0.0, help="Pass --gcn2-alpha to train script (overridden by grid if present)")
    p.add_argument("--use-att-readout", action="store_true", help="Pass --use-att-readout to train script")
    p.add_argument("--n-folds", type=int, default=1)
    p.add_argument("--fast-smoke", action="store_true", help="Use n-folds=1 and epochs=1 for a quick pipeline test")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rebuild-grid", action="store_true", help="Force regenerate grid JSON")
    p.add_argument("--include-default-seed", dest="include_default_seed", action="store_true", default=True,
                   help="Run the known good baseline config as the first eval if not already done")
    p.add_argument("--no-include-default-seed", dest="include_default_seed", action="store_false",
                   help="Disable baseline warm start")
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.rebuild_grid or not args.grid_json.exists():
        grid = build_grid(args.grid_json)
    else:
        grid = json.load(args.grid_json.open())
        if args.expected_grid_size and len(grid) != args.expected_grid_size:
            print(f"Warning: grid size is {len(grid)}, expected {args.expected_grid_size}.", flush=True)
            if args.rebuild_grid:
                grid = build_grid(args.grid_json)

    results = load_results(args.results_csv)
    already_done = set(results.get("grid_index", []))
    eval_count = len(already_done)

    print(f"Existing results: {eval_count} rows. Target evals: {args.max_evals}.", flush=True)

    # Optional warm-start with known good baseline (physics sweet spot)
    seed_cfg = {
        "hidden_dim": 64,
        "dropout": 0.25,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "pretrain": 40,
        "epochs": 20,
        "init_gate": 0.3,
        "batch_size": 32,
    }
    if args.include_default_seed:
        try:
            seed_idx = next(
                i for i, row in enumerate(grid)
                if all(abs(row[k] - seed_cfg[k]) < 1e-12 for k in seed_cfg)
            )
        except StopIteration:
            seed_idx = None
        if seed_idx is not None and seed_idx not in already_done and eval_count < args.max_evals:
            print(f"\n==> Warm-start with baseline config at grid idx {seed_idx}: {seed_cfg}", flush=True)
            row = run_config(seed_cfg, args, seed_idx, args.log_dir, args.train_script)
            save_row(args.results_csv, row)
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            eval_count += 1
            already_done.add(seed_idx)
            print(
                f"Baseline done: orig_f1={row['macro_f1_orig']} orig_acc={row['acc_orig']} status={row['status']} runtime_min={row['runtime_min']:.1f}",
                flush=True,
            )

    # Additional random warm-up (ensure at least 5 total initial points)
    target_warm = 5 if args.include_default_seed else 5
    if eval_count < target_warm and eval_count < args.max_evals:
        remaining = target_warm - eval_count
        pool = list(set(range(len(grid))) - set(results.get("grid_index", [])))
        random.shuffle(pool)
        for idx in pool[:remaining]:
            if eval_count >= args.max_evals:
                break
            cfg = grid[idx]
            print(f"\n==> Warm-up random config at grid idx {idx}: {cfg}", flush=True)
            row = run_config(cfg, args, idx, args.log_dir, args.train_script)
            save_row(args.results_csv, row)
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            eval_count += 1
            already_done.add(idx)
            print(
                f"Warm-up done: orig_f1={row['macro_f1_orig']} orig_acc={row['acc_orig']} status={row['status']} runtime_min={row['runtime_min']:.1f}",
                flush=True,
            )

    while eval_count < args.max_evals:
        needed = min(args.bo_batch, args.max_evals - eval_count)
        cand_idxs = pick_candidates(grid, results, batch=needed)
        if not cand_idxs:
            print("No new candidates available.", flush=True)
            break

        for idx in cand_idxs:
            if eval_count >= args.max_evals:
                break
            cfg = grid[idx]
            print(f"\n==> Eval {eval_count+1}/{args.max_evals} | grid idx {idx} | cfg {cfg}", flush=True)
            row = run_config(cfg, args, idx, args.log_dir, args.train_script)
            save_row(args.results_csv, row)
            results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
            eval_count += 1
            print(
                f"Done idx {idx}: orig_f1={row['macro_f1_orig']} orig_acc={row['acc_orig']} status={row['status']} runtime_min={row['runtime_min']:.1f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
