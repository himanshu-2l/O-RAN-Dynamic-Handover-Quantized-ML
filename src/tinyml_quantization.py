"""
tinyml_quantization.py  (Pillar 3 - v2, self-corrected)
=========================================================
Edge TinyML Quantization & Latency Profiling

Key corrections vs v1:
  - Added StandardScaler (RSRP dBm values in [-70,-115] range require normalisation for MLP)
  - Removed BatchNorm1d (incompatible with PyTorch dynamic INT8 quantization)
  - Increased epochs to 80 with early stopping based on validation loss
  - INT8 size benefit is the primary O-RAN deployment story; per-sample latency overhead
    on tiny CPU models is a documented limitation of dynamic quantization at small batch.

Usage:
    python src/tinyml_quantization.py
"""

import sys
import os
import json
import time
import warnings
import pickle
import io
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kinematic_features import get_all_feature_names

warnings.filterwarnings("ignore")

DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
ANATEL_KIN  = os.path.join(DATA_DIR, "anatel_kinematic.csv")
SIM_KIN     = os.path.join(DATA_DIR, "simulated_kinematic.csv")

N_FEATURES        = len(get_all_feature_names())   # 64
BATCH_SIZE        = 64
N_LATENCY_REPEATS = 500


# -- INT8-compatible MLP (no BatchNorm) ---------------------------------------
class HandoverMLP(nn.Module):
    """
    Tiny MLP for Near-RT RIC edge deployment.
    Architecture: 64 -> 64 -> 32 -> 16 -> 1  (sigmoid output)
    No BatchNorm -- required for PyTorch dynamic INT8 quantization.
    Inputs MUST be StandardScaler-normalised before training.
    """
    def __init__(self, n_features: int = N_FEATURES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray,
              n_epochs: int = 80, lr: float = 1e-3,
              patience: int = 10) -> "HandoverMLP":
    """
    Train HandoverMLP with early stopping on a 10% validation split.
    X_train must already be StandardScaler-normalised.
    """
    model   = HandoverMLP()
    opt     = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32)

    # 90% train / 10% validation split
    n_val  = max(1, int(0.1 * len(X_t)))
    n_tr   = len(X_t) - n_val
    tr_ds, val_ds = random_split(TensorDataset(X_t, y_t), [n_tr, n_val],
                                  generator=torch.Generator().manual_seed(42))
    tr_dl  = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in tr_dl:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                val_losses.append(loss_fn(model(xb), yb).item())
        val_loss = float(np.mean(val_losses))

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    return model


def get_model_size_kb(model) -> float:
    """Estimate serialised model size in KB."""
    buf = io.BytesIO()
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), buf)
    else:
        pickle.dump(model, buf)
    return len(buf.getvalue()) / 1024.0


def measure_torch_latency_ms(model: nn.Module, X_test: np.ndarray,
                               repeats: int = N_LATENCY_REPEATS) -> dict:
    """Measure per-sample and per-batch inference latency (ms)."""
    model.eval()
    X_t = torch.tensor(X_test, dtype=torch.float32)

    # Warm-up
    with torch.no_grad():
        for _ in range(20):
            _ = model(X_t[:1])

    # Per-sample
    sample = X_t[:1]
    times  = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(sample)
            times.append((time.perf_counter() - t0) * 1000)
    per_sample_ms = float(np.mean(times))

    # Per-batch
    batch = X_t[:min(BATCH_SIZE, len(X_t))]
    times = []
    with torch.no_grad():
        for _ in range(repeats):
            t0 = time.perf_counter()
            _ = model(batch)
            times.append((time.perf_counter() - t0) * 1000)
    per_batch_ms = float(np.mean(times))

    return {
        "per_sample_ms": round(per_sample_ms, 4),
        "per_batch_ms":  round(per_batch_ms, 4),
    }


def measure_sklearn_latency_ms(model, X_test: np.ndarray,
                                 repeats: int = N_LATENCY_REPEATS) -> dict:
    sample = X_test[:1]
    batch  = X_test[:min(BATCH_SIZE, len(X_test))]

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - t0) * 1000)
    per_sample_ms = float(np.mean(times))

    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model.predict(batch)
        times.append((time.perf_counter() - t0) * 1000)
    per_batch_ms = float(np.mean(times))

    return {
        "per_sample_ms": round(per_sample_ms, 4),
        "per_batch_ms":  round(per_batch_ms, 4),
    }


def apply_int8_quantization(model: HandoverMLP) -> nn.Module:
    """Apply PyTorch dynamic INT8 quantization to Linear layers."""
    model.eval()
    quantized = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},
        dtype=torch.qint8
    )
    return quantized


def run_pillar3(dataset_name: str, kin_csv: str) -> dict:
    """Full Pillar 3 benchmark for one dataset."""
    print(f"\n{'#'*65}")
    print(f"  PILLAR 3 TinyML -- {dataset_name}")
    print(f"{'#'*65}")

    df = pd.read_csv(kin_csv)
    y  = df["label"].values.astype(int)
    X  = df[get_all_feature_names()].values.astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # StandardScaler: essential for MLP on raw dBm values
    scaler         = StandardScaler()
    X_train_sc     = scaler.fit_transform(X_train).astype(np.float32)
    X_test_sc      = scaler.transform(X_test).astype(np.float32)

    results = {"dataset": dataset_name, "models": {}}

    # -- 1. Full-Precision MLP --------------------------------------------------
    print(f"\n  [1/4] Training Full-Precision MLP (StandardScaler + early stopping)...")
    mlp_fp = train_mlp(X_train_sc, y_train, n_epochs=80, patience=10)
    with torch.no_grad():
        preds_fp = (mlp_fp(torch.tensor(X_test_sc)).numpy() >= 0.5).astype(int)
    acc_fp  = accuracy_score(y_test, preds_fp)
    size_fp = get_model_size_kb(mlp_fp)
    lat_fp  = measure_torch_latency_ms(mlp_fp, X_test_sc)
    print(f"     Accuracy       : {acc_fp*100:.2f}%")
    print(f"     Size           : {size_fp:.2f} KB")
    print(f"     Latency/sample : {lat_fp['per_sample_ms']:.4f} ms")
    print(f"     Latency/batch  : {lat_fp['per_batch_ms']:.4f} ms")

    results["models"]["MLP_FP32"] = {
        "accuracy": round(acc_fp, 6), "size_kb": round(size_fp, 2), **lat_fp
    }

    # -- 2. INT8 Quantized MLP --------------------------------------------------
    print(f"\n  [2/4] Applying INT8 Dynamic Quantization...")
    mlp_int8 = apply_int8_quantization(mlp_fp)
    with torch.no_grad():
        preds_q = (mlp_int8(torch.tensor(X_test_sc)).numpy() >= 0.5).astype(int)
    acc_q    = accuracy_score(y_test, preds_q)
    size_q   = get_model_size_kb(mlp_int8)
    lat_q    = measure_torch_latency_ms(mlp_int8, X_test_sc)
    acc_drop = (acc_fp - acc_q) * 100
    size_red = (1 - size_q / size_fp) * 100
    speedup  = lat_fp["per_sample_ms"] / lat_q["per_sample_ms"] if lat_q["per_sample_ms"] > 0 else 1.0
    print(f"     Accuracy       : {acc_q*100:.2f}% (drop: {acc_drop:.3f}%)")
    print(f"     Size           : {size_q:.2f} KB  (reduction: {size_red:.1f}%)")
    print(f"     Latency/sample : {lat_q['per_sample_ms']:.4f} ms  (vs FP32: {speedup:.2f}x)")
    print(f"     Latency/batch  : {lat_q['per_batch_ms']:.4f} ms")
    print(f"     [Note] INT8 per-sample latency overhead is expected on tiny CPU models;")
    print(f"            primary benefit is the {size_red:.1f}% model size reduction for edge RAM.")

    results["models"]["MLP_INT8"] = {
        "accuracy": round(acc_q, 6), "size_kb": round(size_q, 2),
        "accuracy_drop_pct": round(acc_drop, 4),
        "size_reduction_pct": round(size_red, 2),
        "speedup_vs_fp32": round(speedup, 3),
        **lat_q
    }

    # -- 3. Random Forest -------------------------------------------------------
    print(f"\n  [3/4] Benchmarking Random Forest (100 trees)...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    acc_rf  = accuracy_score(y_test, rf.predict(X_test))
    size_rf = get_model_size_kb(rf)
    lat_rf  = measure_sklearn_latency_ms(rf, X_test)
    print(f"     Accuracy       : {acc_rf*100:.2f}%")
    print(f"     Size           : {size_rf:.2f} KB")
    print(f"     Latency/sample : {lat_rf['per_sample_ms']:.4f} ms")
    results["models"]["RandomForest"] = {
        "accuracy": round(acc_rf, 6), "size_kb": round(size_rf, 2), **lat_rf
    }

    # -- 4. Pruned Decision Tree ------------------------------------------------
    print(f"\n  [4/4] Benchmarking Pruned Decision Tree (max_depth=8, edge-optimised)...")
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt.fit(X_train, y_train)
    acc_dt  = accuracy_score(y_test, dt.predict(X_test))
    size_dt = get_model_size_kb(dt)
    lat_dt  = measure_sklearn_latency_ms(dt, X_test)
    print(f"     Accuracy       : {acc_dt*100:.2f}%")
    print(f"     Size           : {size_dt:.2f} KB")
    print(f"     Latency/sample : {lat_dt['per_sample_ms']:.4f} ms")
    results["models"]["DecisionTree_Pruned"] = {
        "accuracy": round(acc_dt, 6), "size_kb": round(size_dt, 2), **lat_dt
    }

    # -- Comparison table -------------------------------------------------------
    print(f"\n  ---- Comparison Table: {dataset_name} ----")
    print(f"  {'Model':28s} {'Acc':>8s} {'Size(KB)':>10s} {'Lat/sample(ms)':>16s} {'Lat/batch(ms)':>14s}")
    print("  " + "-"*80)
    for mname, mres in results["models"].items():
        print(f"  {mname:28s} {mres['accuracy']*100:>7.2f}%  {mres['size_kb']:>9.2f}   "
              f"{mres['per_sample_ms']:>13.4f}   {mres['per_batch_ms']:>12.4f}")

    return results


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    for path, name in [(ANATEL_KIN, "ANATEL kinematic"), (SIM_KIN, "Simulated kinematic")]:
        if not os.path.exists(path):
            print(f"  [!] {name} CSV missing. Run prepare_kinematic_data.py first.")
            sys.exit(1)

    all_results = []
    all_results.append(run_pillar3("ANATEL (Real Network)", ANATEL_KIN))
    all_results.append(run_pillar3("Simulated (ns-3)", SIM_KIN))

    out_path = os.path.join(RESULTS_DIR, "pillar3_raw_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  [OK] Pillar 3 results saved -> {out_path}")
