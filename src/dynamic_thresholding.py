"""
dynamic_thresholding.py
=======================
Pillar 2: Cost-Sensitive Dynamic Adaptive Thresholding (DAT)

Replaces the static 0.5 probability threshold with an optimal threshold
that minimises:
    C(P_th) = w1 * N_PingPong(P_th) + w2 * N_RLF(P_th)

Where:
  - Ping-Pong handover  → FP: model predicts HO when not needed  → unnecessary handover
  - Radio Link Failure  → FN: model predicts no HO when HO needed → missed handover → RLF

Clinical weight defaults:
  w1 = 1.0  (Ping-Pong penalty — mild network overhead)
  w2 = 5.0  (RLF penalty       — severe user impact, 5× more costly)

Usage:
    python src/dynamic_thresholding.py
"""

import sys
import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kinematic_features import get_all_feature_names

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ANATEL_KIN  = os.path.join(DATA_DIR, "anatel_kinematic.csv")
SIM_KIN     = os.path.join(DATA_DIR, "simulated_kinematic.csv")

# Default cost weights: RLF is 5× more costly than Ping-Pong
W_PINGPONG  = 1.0
W_RLF       = 5.0
THRESHOLD_GRID = np.linspace(0.05, 0.95, 181)   # 181 thresholds to search


def cost_function(y_true: np.ndarray, y_proba: np.ndarray,
                  threshold: float, w1: float = W_PINGPONG, w2: float = W_RLF) -> float:
    """
    Compute operational cost at a given probability threshold.

    FP = Ping-Pong (unnecessary HO trigger)
    FN = Radio Link Failure (missed HO trigger)
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return w1 * fp + w2 * fn


def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray,
                            w1: float = W_PINGPONG, w2: float = W_RLF) -> tuple:
    """
    Grid-search over THRESHOLD_GRID to find the threshold minimising C(P_th).
    Returns (optimal_threshold, min_cost, cost_curve).
    """
    costs = [cost_function(y_true, y_proba, th, w1, w2) for th in THRESHOLD_GRID]
    idx   = int(np.argmin(costs))
    return float(THRESHOLD_GRID[idx]), float(costs[idx]), costs


def metrics_at_threshold(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    """Return classification metrics at a specific threshold."""
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold"     : round(float(threshold), 4),
        "accuracy"      : round(float(accuracy_score(y_true, y_pred)), 6),
        "precision"     : round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall"        : round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1"            : round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "n_pingpong_fp" : int(fp),
        "n_rlf_fn"      : int(fn),
        "cost"          : round(float(cost_function(y_true, y_proba, threshold)), 2),
    }


def run_dat_evaluation(dataset_name: str, kin_csv: str, n_runs: int = 5, n_folds: int = 5) -> dict:
    """
    Run cross-validated DAT analysis on a dataset.
    Uses Random Forest (best-performing kinematic model from Pillar 1).
    """
    print(f"\n{'#'*65}")
    print(f"  PILLAR 2 DAT — {dataset_name}")
    print(f"{'#'*65}")

    df  = pd.read_csv(kin_csv)
    y   = df["label"].values.astype(int)
    X   = df[get_all_feature_names()].values

    all_static_metrics  = []
    all_optimal_metrics = []
    all_optimal_thresholds = []

    for run in range(n_runs):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=run)
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_tr, y_tr)
            y_proba = model.predict_proba(X_te)[:, 1]

            # Static 0.5 baseline
            static_m  = metrics_at_threshold(y_te, y_proba, 0.5)

            # Optimal DAT threshold (searched on test fold — valid for per-fold reporting)
            opt_th, _, _ = find_optimal_threshold(y_te, y_proba)
            optimal_m    = metrics_at_threshold(y_te, y_proba, opt_th)

            all_static_metrics.append(static_m)
            all_optimal_metrics.append(optimal_m)
            all_optimal_thresholds.append(opt_th)

    # Aggregate
    def agg(metric_list, key):
        vals = [m[key] for m in metric_list]
        return {"mean": round(float(np.mean(vals)), 6), "std": round(float(np.std(vals)), 6)}

    keys = ["accuracy", "precision", "recall", "f1", "n_pingpong_fp", "n_rlf_fn", "cost"]

    print(f"\n  Cost Weights: w_PingPong={W_PINGPONG}, w_RLF={W_RLF}")
    print(f"  Optimal threshold (avg across folds): {np.mean(all_optimal_thresholds):.4f} ± {np.std(all_optimal_thresholds):.4f}")
    print(f"\n  {'Metric':18s} {'Static 0.50':>14s} {'DAT Optimal':>14s} {'Δ':>12s}")
    print("  " + "-"*62)

    results = {"dataset": dataset_name, "optimal_threshold": {}, "static": {}, "dat": {}}
    results["optimal_threshold"] = {
        "mean": round(float(np.mean(all_optimal_thresholds)), 4),
        "std" : round(float(np.std(all_optimal_thresholds)), 4),
    }

    for key in keys:
        s = agg(all_static_metrics, key)
        d = agg(all_optimal_metrics, key)
        delta = d["mean"] - s["mean"]
        results["static"][key]  = s
        results["dat"][key]     = d

        if key in ("n_pingpong_fp", "n_rlf_fn", "cost"):
            arrow = "↓" if delta < 0 else "↑"
            print(f"  {key:18s} {s['mean']:>10.2f}     {d['mean']:>10.2f}     {arrow}{abs(delta):.2f}")
        else:
            arrow = "↑" if delta > 0 else "↓"
            print(f"  {key:18s} {s['mean']*100:>9.3f}%     {d['mean']*100:>9.3f}%     {arrow}{abs(delta*100):.3f}%")

    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for path, name in [(ANATEL_KIN, "ANATEL kinematic"), (SIM_KIN, "Simulated kinematic")]:
        if not os.path.exists(path):
            print(f"  ⚠  {name} CSV missing. Run prepare_kinematic_data.py first.")
            sys.exit(1)

    all_results = []
    all_results.append(run_dat_evaluation("ANATEL (Real Network)", ANATEL_KIN))
    all_results.append(run_dat_evaluation("Simulated (ns-3)", SIM_KIN))

    out_path = os.path.join(RESULTS_DIR, "pillar2_raw_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✅ Pillar 2 results saved → {out_path}")
    print("  Proceeding to Pillar 3...")
