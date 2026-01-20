#!/usr/bin/env python3
"""
Bar plots for ablation results across three tasks (LDL/UDL/Sensitivity).

Inputs: CSVs in ablation_logs_residual[_udl|_sensitivity]/ablation_metrics.csv
Each CSV is expected to have columns:
  config, orig_acc, orig_f1, orig_prec, orig_rec, train_*, test_* ...

Outputs: saves PNGs under plots/ with bars for orig_acc / orig_f1 / orig_prec / orig_rec,
and a "avg_orig_acc" chart averaging three tasks.
"""
import csv
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
TASK_DIRS = {
    "LDL": BASE / "ablation_logs_residual",
    "UDL": BASE / "ablation_logs_residual_udl",
    "Sensitivity": BASE / "ablation_logs_residual_sensitivity",
}
PLOT_DIR = BASE / "plots"
PLOT_DIR.mkdir(exist_ok=True)
# Optional: compare_results.csv for additional baseline comparison
COMPARE_CSV = BASE / "compare_results.csv"  # Set to None or provide path via CLI if needed


def parse_tabular_file(task_key: str):
    """Parse baseline_tabular_<task>.txt lines with '[Orig holdout]' to rows similar to ablation."""
    fname_map = {
        "LDL": "baseline_tabular_lower_detection_limit.txt",
        "UDL": "baseline_tabular_upper_detection_limit.txt",
        "Sensitivity": "baseline_tabular_sensitivity_numerator.txt",
    }
    path = BASE / fname_map[task_key]
    rows = []
    if not path.exists():
        return rows
    # accept both '±' and '+-' by normalizing
    pat = re.compile(
        r"^(.*?) \[Orig holdout\] \| acc ([0-9.]+) \+- ([0-9.]+) \| "
        r"f1 ([0-9.]+) \+- ([0-9.]+) \| prec ([0-9.]+) \+- ([0-9.]+) \| rec ([0-9.]+) \+- ([0-9.]+)"
    )
    for raw in path.read_text().splitlines():
        line = raw.replace("±", "+-").strip()
        m = pat.match(line)
        if not m:
            continue
        name = m.group(1).strip()
        rows.append({
            "config": f"tab_{name}",
            "orig_acc": float(m.group(2)),
            "orig_acc_std": float(m.group(3)),
            "orig_f1": float(m.group(4)),
            "orig_f1_std": float(m.group(5)),
            "orig_prec": float(m.group(6)),
            "orig_prec_std": float(m.group(7)),
            "orig_rec": float(m.group(8)),
            "orig_rec_std": float(m.group(9)),
        })
    return rows


def parse_compare(task_key: str):
    """Load compare_results.csv entries for given task and split=orig."""
    rows = []
    if not COMPARE_CSV.exists():
        return rows
    norm = {
        "LDL": "lower_detection_limit",
        "UDL": "upper_detection_limit",
        "Sensitivity": "sensitivity_numerator",
    }[task_key]
    with COMPARE_CSV.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r["task"] != norm:
                continue
            if r["split"] != "orig":
                continue
            rows.append({
                "config": f"cmp_{r['model']}",
                "orig_acc": float(r["acc_mean"]),
                "orig_acc_std": float(r["acc_std"]),
                "orig_f1": float(r["f1_mean"]),
                "orig_f1_std": float(r["f1_std"]),
                "orig_prec": float(r["prec_mean"]),
                "orig_prec_std": float(r["prec_std"]),
                "orig_rec": float(r["rec_mean"]),
                "orig_rec_std": float(r["rec_std"]),
            })
    return rows


def load_csv(path: Path):
    rows = []
    with path.open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def bar_plot(task_name: str, rows, metric: str):
    """metric keys: orig_acc / orig_f1 / orig_prec / orig_rec"""
    configs = [r["config"] for r in rows]
    vals = np.array([float(r.get(metric, "") or 0.0) for r in rows])
    stds = np.array([float(r.get(f"{metric}_std", "") or 0.0) for r in rows])
    min_val = float(vals.min()) if len(vals) else 0.0
    ylim_low = max(0.4, min_val - 0.05)
    x = np.arange(len(configs))
    fig, ax = plt.subplots(figsize=(16, 8))
    def color_for(cfg: str):
        if cfg.startswith("cmp_"):
            return "#f59e0b"  # amber for compare models
        if cfg.startswith("tab_"):
            return "#9ca3af"  # gray for tabular baselines
        return "#3b82f6"      # blue for ablation configs
    colors = [color_for(c) for c in configs]
    ax.bar(x, vals, yerr=stds, capsize=3, color=colors, alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(configs, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel(metric)
    ax.set_ylim(ylim_low, 1.0)
    # annotate all bars
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    # simple legend handles
    handles = [
        plt.Rectangle((0, 0), 1, 1, color="#3b82f6", alpha=0.9, label="ablation"),
        plt.Rectangle((0, 0), 1, 1, color="#f59e0b", alpha=0.9, label="compare"),
        plt.Rectangle((0, 0), 1, 1, color="#9ca3af", alpha=0.9, label="tabular"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left")
    ax.set_title(f"{task_name} - {metric} (error bars = std)")
    fig.tight_layout()
    out = PLOT_DIR / f"{task_name.lower()}_{metric}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def main():
    # load ablation
    ldl_rows = load_csv(TASK_DIRS["LDL"] / "ablation_metrics.csv")
    udl_rows = load_csv(TASK_DIRS["UDL"] / "ablation_metrics.csv")
    sen_rows = load_csv(TASK_DIRS["Sensitivity"] / "ablation_metrics.csv")
    # append compare + tabular baselines
    ldl_rows += parse_compare("LDL") + parse_tabular_file("LDL")
    udl_rows += parse_compare("UDL") + parse_tabular_file("UDL")
    sen_rows += parse_compare("Sensitivity") + parse_tabular_file("Sensitivity")

    ldl = {r["config"]: float(r["orig_acc"] or 0.0) for r in ldl_rows}
    udl = {r["config"]: float(r["orig_acc"] or 0.0) for r in udl_rows}
    sen = {r["config"]: float(r["orig_acc"] or 0.0) for r in sen_rows}
    all_configs = set(ldl) | set(udl) | set(sen)
    all_configs.discard("gcn2_alpha_015")

    avg_scores = []
    for cfg in all_configs:
        if cfg in ldl and cfg in udl and cfg in sen:
            vals = np.array([ldl[cfg], udl[cfg], sen[cfg]])
            avg = vals.mean()
        else:
            avg = -1  # push missing to end
        avg_scores.append((cfg, avg))

    # anchor first, then by avg desc
    avg_scores.sort(key=lambda x: x[1], reverse=True)
    ordered = [cfg for cfg, _ in avg_scores]
    if "anchor_jk_att" in ordered:
        ordered.remove("anchor_jk_att")
        ordered.insert(0, "anchor_jk_att")
    # save order for debugging/inspection
    order_path = PLOT_DIR / "ordered_configs.txt"
    order_path.write_text("\n".join(ordered))

    # per-task plots with the same order
    for task, rows in [("LDL", ldl_rows), ("UDL", udl_rows), ("Sensitivity", sen_rows)]:
        rows_map = {r["config"]: r for r in rows}
        ordered_rows = [rows_map[cfg] for cfg in ordered if cfg in rows_map]
        for metric in ["orig_acc", "orig_f1", "orig_prec", "orig_rec"]:
            bar_plot(task, ordered_rows, metric)

    # average across tasks plot (using ordered list)
    fig, ax = plt.subplots(figsize=(12, 6))
    xs = np.arange(len(ordered))
    avg_vals = []
    avg_errs = []
    colors = []
    for cfg in ordered:
        vals = np.array([ldl[cfg], udl[cfg], sen[cfg]])
        avg_vals.append(vals.mean())
        avg_errs.append(vals.std())
        if cfg.startswith("cmp_"):
            colors.append("#f59e0b")
        elif cfg.startswith("tab_"):
            colors.append("#9ca3af")
        else:
            colors.append("#10b981")
    min_val = min(avg_vals) if avg_vals else 0.0
    ylim_low = max(0.4, min_val - 0.05)
    ax.bar(xs, avg_vals, yerr=avg_errs, capsize=3, color=colors, alpha=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(ordered, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("avg_orig_acc")
    ax.set_ylim(ylim_low, 1.0)
    for xi, v in zip(xs, avg_vals):
        ax.text(xi, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7, rotation=90)
    ax.set_title("Average Original Accuracy (LDL/UDL/Sensitivity)")
    fig.tight_layout()
    out = PLOT_DIR / "avg_orig_acc.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
