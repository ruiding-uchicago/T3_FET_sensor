#!/usr/bin/env python
"""
Quick-and-dirty tabular baselines (RF/ExtraTrees/LogReg/SVM/kNN/NB/MLP/XGB if available).
- Builds a tabular feature by mean-pooling all node-type features per graph (and optional fingerprints).
- Runs StratifiedKFold CV and reports acc / macro-F1 / precision / recall (mean ± std).
Usage:
  python baseline_tabular.py --task lower_detection_limit --graph-suffix _aug --folds 5 --output-csv tabular_ldl.csv
"""
import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import top_k_accuracy_score, average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold

# 好 vs 中 class mapping for each task
# LDL: lower is better, so Class 0 = 好 (best), Class 1 = 中 (medium)
# UDL/Sensitivity: higher is better, so Class 2 = 好 (best), Class 1 = 中 (medium)
def get_good_medium_classes(task: str):
    """Return (good_class, medium_class) indices based on task."""
    if task == "lower_detection_limit":
        return 0, 1  # Class 0 = 好, Class 1 = 中
    else:  # upper_detection_limit, sensitivity_numerator
        return 2, 1  # Class 2 = 好, Class 1 = 中

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC

GRAPH_DIR = Path(__file__).resolve().parent


def load_graphs(task: str, graph_suffix: str) -> List[torch.Tensor]:
    pt_path = GRAPH_DIR / f"graphs{graph_suffix}_{task}.pt"
    pkl_path = GRAPH_DIR / f"graphs{graph_suffix}_{task}.pkl"
    if pt_path.exists():
        return torch.load(pt_path)
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    raise FileNotFoundError(f"Missing graphs file: {pt_path} / {pkl_path}")


def build_tabular(graphs, use_fp: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = [], []
    is_orig = []
    for g in graphs:
        feats = []
        if hasattr(g, "x_dict"):  # torch geometric HeteroData
            for x in g.x_dict.values():
                feats.append(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).mean(dim=0).cpu().numpy())
        else:  # fallback for dict-like raw graphs
            for x in g["nodes"].values():
                feats.append(np.nan_to_num(np.array(x), nan=0.0).mean(axis=0))

        if use_fp:
            fp_parts = []
            fps = getattr(g, "fingerprints", None)
            if fps is None and isinstance(g, dict):
                fps = g.get("fingerprints")
            if fps:
                fp_parts.extend([np.array(v) for v in fps.values()])
            if fp_parts:
                feats.append(np.concatenate(fp_parts))
        # Clean and guard against empty/0-d
        feats = [np.atleast_1d(np.nan_to_num(f, nan=0.0)) for f in feats if f is not None and np.size(f) > 0]
        if not feats:
            continue  # skip malformed sample
        X.append(np.concatenate(feats))
        y.append(int(g.y.item()) if hasattr(g, "y") else int(g["y"]))
        is_orig.append(bool(getattr(g, "is_original", True) if hasattr(g, "is_original") else g.get("is_original", True)))
    if len(X) == 0:
        raise ValueError("No valid samples built for tabular baseline.")
    is_orig = np.array(is_orig, dtype=bool)
    X = np.vstack(X)
    y = np.array(y)
    # Split into augmented vs original
    X_aug, y_aug = X[~is_orig], y[~is_orig]
    X_orig, y_orig = X[is_orig], y[is_orig]
    return X_aug, y_aug, X_orig, y_orig


def summarize(name: str, scores: List[Dict[str, float]]) -> str:
    def mean_std(key):
        vals = [s.get(key) for s in scores if s.get(key) is not None]
        if not vals:
            return None, None
        arr = np.array(vals, dtype=float)
        return arr.mean(), arr.std()

    if len(scores) == 0:
        return f"{name:15s} | acc N/A | f1 N/A | prec N/A | rec N/A"
    acc_m, acc_s = mean_std("acc")
    f1_m, f1_s = mean_std("f1")
    prec_m, prec_s = mean_std("prec")
    rec_m, rec_s = mean_std("rec")
    acc_at_2_m, acc_at_2_s = mean_std("acc_at_2")
    auc_prc_m, auc_prc_s = mean_std("auc_prc")
    gm_acc_m, gm_acc_s = mean_std("gm_acc")
    gm_auc_m, gm_auc_s = mean_std("gm_auc")

    result = (
        f"{name:15s} | acc {acc_m:.4f} ± {acc_s:.4f} | "
        f"f1 {f1_m:.4f} ± {f1_s:.4f} | "
        f"prec {prec_m:.4f} ± {prec_s:.4f} | "
        f"rec {rec_m:.4f} ± {rec_s:.4f}"
    )
    if acc_at_2_m is not None:
        result += f" | acc@2 {acc_at_2_m:.4f} ± {acc_at_2_s:.4f}"
    if auc_prc_m is not None:
        result += f" | auc_prc {auc_prc_m:.4f} ± {auc_prc_s:.4f}"
    if gm_acc_m is not None:
        result += f" | gm_acc {gm_acc_m:.4f} ± {gm_acc_s:.4f}"
    if gm_auc_m is not None:
        result += f" | gm_auc {gm_auc_m:.4f} ± {gm_auc_s:.4f}"
    return result


def run_cv(model, X, y, X_orig, y_orig, folds: int, seed: int, num_classes: int = 3, task: str = None):
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores_test, scores_orig = [], []

    # Get good and medium class indices if task is specified
    good_cls, med_cls = None, None
    if task is not None:
        good_cls, med_cls = get_good_medium_classes(task)

    for tr, te in skf.split(X, y):
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])

        # Try to get probabilities for Accuracy@2 and AUC-PRC
        proba = None
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X[te])
            elif hasattr(model, "named_steps"):  # Pipeline
                last_step = list(model.named_steps.values())[-1]
                if hasattr(last_step, "predict_proba"):
                    proba = model.predict_proba(X[te])
        except Exception:
            proba = None

        score_dict = {
            "acc": accuracy_score(y[te], pred),
            "f1": f1_score(y[te], pred, average="macro"),
            "prec": precision_score(y[te], pred, average="macro", zero_division=0),
            "rec": recall_score(y[te], pred, average="macro", zero_division=0),
            "acc_at_2": None,
            "auc_prc": None,
            "gm_acc": None,
            "gm_auc": None,
        }

        if proba is not None and num_classes > 2:
            score_dict["acc_at_2"] = top_k_accuracy_score(y[te], proba, k=2)
            y_true_bin = label_binarize(y[te], classes=list(range(num_classes)))
            score_dict["auc_prc"] = average_precision_score(y_true_bin, proba, average="macro")

            # 好 vs 中 metrics
            if good_cls is not None:
                mask = (y[te] == good_cls) | (y[te] == med_cls)
                if mask.sum() > 0:
                    y_gm = y[te][mask]
                    pred_gm = pred[mask]
                    proba_gm = proba[mask]
                    y_bin_gm = (y_gm == good_cls).astype(int)
                    pred_bin_gm = (pred_gm == good_cls).astype(int)
                    score_dict["gm_acc"] = float((y_bin_gm == pred_bin_gm).mean())
                    if len(np.unique(y_bin_gm)) == 2:
                        try:
                            score_dict["gm_auc"] = roc_auc_score(y_bin_gm, proba_gm[:, good_cls])
                        except Exception:
                            pass

        scores_test.append(score_dict)

        if len(X_orig) > 0:
            o_pred = model.predict(X_orig)

            # Try to get probabilities for original data
            o_proba = None
            try:
                if hasattr(model, "predict_proba"):
                    o_proba = model.predict_proba(X_orig)
                elif hasattr(model, "named_steps"):
                    last_step = list(model.named_steps.values())[-1]
                    if hasattr(last_step, "predict_proba"):
                        o_proba = model.predict_proba(X_orig)
            except Exception:
                o_proba = None

            orig_dict = {
                "acc": accuracy_score(y_orig, o_pred),
                "f1": f1_score(y_orig, o_pred, average="macro"),
                "prec": precision_score(y_orig, o_pred, average="macro", zero_division=0),
                "rec": recall_score(y_orig, o_pred, average="macro", zero_division=0),
                "acc_at_2": None,
                "auc_prc": None,
                "gm_acc": None,
                "gm_auc": None,
            }

            if o_proba is not None and num_classes > 2:
                orig_dict["acc_at_2"] = top_k_accuracy_score(y_orig, o_proba, k=2)
                y_orig_bin = label_binarize(y_orig, classes=list(range(num_classes)))
                orig_dict["auc_prc"] = average_precision_score(y_orig_bin, o_proba, average="macro")

                # 好 vs 中 metrics for original data
                if good_cls is not None:
                    mask_orig = (y_orig == good_cls) | (y_orig == med_cls)
                    if mask_orig.sum() > 0:
                        y_gm_orig = y_orig[mask_orig]
                        pred_gm_orig = o_pred[mask_orig]
                        proba_gm_orig = o_proba[mask_orig]
                        y_bin_gm_orig = (y_gm_orig == good_cls).astype(int)
                        pred_bin_gm_orig = (pred_gm_orig == good_cls).astype(int)
                        orig_dict["gm_acc"] = float((y_bin_gm_orig == pred_bin_gm_orig).mean())
                        if len(np.unique(y_bin_gm_orig)) == 2:
                            try:
                                orig_dict["gm_auc"] = roc_auc_score(y_bin_gm_orig, proba_gm_orig[:, good_cls])
                            except Exception:
                                pass

            scores_orig.append(orig_dict)

    return scores_test, scores_orig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--graph-suffix", default="_aug")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-fp", action="store_true", help="Include fingerprints if present")
    ap.add_argument("--output-csv", default=None, help="Optional CSV to save results")
    args = ap.parse_args()

    graphs = load_graphs(args.task, args.graph_suffix)
    X, y, X_orig, y_orig = build_tabular(graphs, use_fp=args.use_fp)
    cls = np.unique(np.concatenate([y, y_orig])) if len(y_orig) > 0 else np.unique(y)
    num_classes = len(cls)
    print(f"Loaded {len(X)+len(X_orig)} samples "
          f"(aug {len(X)}, orig {len(X_orig)}), dim={X.shape[1]}, classes={cls.tolist()}")

    models = {
        # 故意偏保守的弱配置
        "rf": RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1),
        "extra": ExtraTreesClassifier(n_estimators=200, max_depth=12, class_weight="balanced", n_jobs=-1),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced"),
        ),
        "lin_svm": make_pipeline(
            StandardScaler(),
            LinearSVC(C=1.0, class_weight="balanced"),
        ),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=3, gamma="scale", class_weight="balanced", probability=True),
        ),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=10, weights="distance")),
        "gnb": GaussianNB(),
    }
    from sklearn.neural_network import MLPClassifier

    models["mlp"] = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(96,),
            max_iter=150,
            batch_size=64,
            alpha=5e-4,
            early_stopping=True,
            learning_rate_init=1e-3,
        ),
    )

    # Optional XGBoost
    try:
        from xgboost import XGBClassifier

        models["xgb"] = XGBClassifier(
            max_depth=6,
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=4,
        )
    except Exception:
        pass

    lines = []
    for name, model in models.items():
        scores_test, scores_orig = run_cv(model, X, y, X_orig, y_orig, folds=args.folds, seed=args.seed, num_classes=num_classes, task=args.task)
        line_test = summarize(name + " [Aug test]", scores_test)
        print(line_test)
        lines.append(line_test)
        if len(scores_orig) > 0:
            line_orig = summarize(name + " [Orig holdout]", scores_orig)
            print(line_orig)
            lines.append(line_orig)

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.write_text("\n".join(lines))
        print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
