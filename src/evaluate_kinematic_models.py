"""
evaluate_kinematic_models.py
============================
Pillar 1 Evaluation: Compare Baseline RSRP-only vs. Kinematic-Enriched models.

Models evaluated:
  - Decision Tree  (DT)
  - Random Forest  (RF)
  - XGBoost        (XGB)

Protocol:
  - 10 runs × 5-fold Stratified Cross-Validation (50 folds total) — matching Paper 1 protocol.
  - Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC.
  - Feature importance comparison.

Usage:
    python src/evaluate_kinematic_models.py
"""

import sys
import os
import warnings
import json
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    warnings.warn("XGBoost not available — will skip XGB evaluation.", stacklevel=2)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kinematic_features import RSRP_COLS, get_kinematic_feature_names, get_all_feature_names

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR     = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ANATEL_KIN   = os.path.join(DATA_DIR, "anatel_kinematic.csv")
SIM_KIN      = os.path.join(DATA_DIR, "simulated_kinematic.csv")

N_RUNS  = 10
N_FOLDS = 5


# ── Model factory ────────────────────────────────────────────────────────────
def get_models():
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, use_label_encoder=False,
            eval_metric="logloss", random_state=42,
            verbosity=0
        )
    return models


# ── Evaluation ───────────────────────────────────────────────────────────────
def evaluate(X: np.ndarray, y: np.ndarray, model_name: str, model,
             n_runs=N_RUNS, n_folds=N_FOLDS) -> dict:
    """Run n_runs × n_folds SK-CV and return aggregated metrics."""
    all_metrics = defaultdict(list)

    for run in range(n_runs):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=run)
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]

            m = type(model)(**model.get_params()) if hasattr(model, "get_params") else model
            m.fit(X_tr, y_tr)
            y_pred  = m.predict(X_te)
            y_proba = m.predict_proba(X_te)[:, 1] if hasattr(m, "predict_proba") else y_pred

            all_metrics["accuracy"].append(accuracy_score(y_te, y_pred))
            all_metrics["precision"].append(precision_score(y_te, y_pred, zero_division=0))
            all_metrics["recall"].append(recall_score(y_te, y_pred, zero_division=0))
            all_metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))
            try:
                all_metrics["roc_auc"].append(roc_auc_score(y_te, y_proba))
            except Exception:
                all_metrics["roc_auc"].append(float("nan"))

    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in all_metrics.items()}


def get_feature_importances(X: np.ndarray, y: np.ndarray,
                             model, feature_names: list) -> pd.Series:
    """Fit model on full data and return feature importances."""
    m = type(model)(**model.get_params()) if hasattr(model, "get_params") else model
    m.fit(X, y)
    if hasattr(m, "feature_importances_"):
        return pd.Series(m.feature_importances_, index=feature_names).sort_values(ascending=False)
    return pd.Series(dtype=float)


def run_dataset_evaluation(dataset_name: str, kin_csv_path: str) -> dict:
    """Full evaluation pipeline for one dataset."""
    print(f"\n{'#'*65}")
    print(f"  DATASET: {dataset_name}")
    print(f"{'#'*65}")

    df = pd.read_csv(kin_csv_path)
    y  = df["label"].values.astype(int)

    baseline_features  = RSRP_COLS
    kinematic_features = get_all_feature_names()  # 50 + 14

    X_base = df[baseline_features].values
    X_kin  = df[kinematic_features].values

    dataset_results = {"dataset": dataset_name, "models": {}}

    models = get_models()

    for model_name, model in models.items():
        print(f"\n  ── {model_name} ──────────────────────────────")

        # Baseline
        base_metrics = evaluate(X_base, y, model_name, model)
        print(f"  [Baseline RSRP-only]")
        for metric, vals in base_metrics.items():
            print(f"     {metric:12s}: {vals['mean']*100:.2f}% ± {vals['std']*100:.2f}%")

        # Kinematic-enriched
        kin_metrics = evaluate(X_kin, y, model_name, model)
        print(f"  [Kinematic-Enriched]")
        for metric, vals in kin_metrics.items():
            delta = kin_metrics[metric]["mean"] - base_metrics[metric]["mean"]
            arrow = "↑" if delta > 0 else "↓"
            print(f"     {metric:12s}: {vals['mean']*100:.2f}% ± {vals['std']*100:.2f}%  ({arrow}{abs(delta)*100:.3f}%)")

        # Feature importances (kinematic-enriched)
        print(f"\n  [Top-10 Feature Importances (Kinematic-Enriched)]")
        fi = get_feature_importances(X_kin, y, model, kinematic_features)
        print(fi.head(10).round(4).to_string())
        kin_fi_names = get_kinematic_feature_names()
        kin_in_top10 = [f for f in fi.head(10).index if f in kin_fi_names]
        print(f"  → Kinematic features in top-10: {kin_in_top10}")

        dataset_results["models"][model_name] = {
            "baseline": base_metrics,
            "kinematic": kin_metrics,
            "top_features": fi.head(10).to_dict(),
            "kinematic_in_top10": kin_in_top10,
        }

    return dataset_results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Check kinematic datasets exist
    for path, name in [(ANATEL_KIN, "ANATEL kinematic"), (SIM_KIN, "Simulated kinematic")]:
        if not os.path.exists(path):
            print(f"  ⚠ {name} CSV not found. Run prepare_kinematic_data.py first.")
            sys.exit(1)

    all_results = []
    all_results.append(run_dataset_evaluation("ANATEL (Real Network)", ANATEL_KIN))
    all_results.append(run_dataset_evaluation("Simulated (ns-3)", SIM_KIN))

    # Save JSON results
    results_path = os.path.join(RESULTS_DIR, "pillar1_raw_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  ✅ Raw results saved → {results_path}")

    # Summary table
    print("\n" + "="*65)
    print("  PILLAR 1 SUMMARY TABLE (Accuracy)")
    print("="*65)
    header = f"  {'Dataset':25s} {'Model':16s} {'Baseline':>10s} {'Kinematic':>10s} {'Δ':>8s}"
    print(header)
    print("  " + "-"*63)
    for res in all_results:
        for mname, mres in res["models"].items():
            b = mres["baseline"]["accuracy"]["mean"] * 100
            k = mres["kinematic"]["accuracy"]["mean"] * 100
            d = k - b
            arrow = "↑" if d > 0 else "↓" if d < 0 else "="
            print(f"  {res['dataset']:25s} {mname:16s} {b:9.2f}% {k:9.2f}%  {arrow}{abs(d):.3f}%")

    print("\n  Pillar 1 Evaluation COMPLETE. Results → results/pillar1_raw_results.json")
    print("  Proceeding to Pillar 2...")
