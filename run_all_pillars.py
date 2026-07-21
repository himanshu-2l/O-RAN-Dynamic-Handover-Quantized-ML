"""
run_all_pillars.py
==================
Master orchestration script — executes all 3 Pillars sequentially,
generates results JSON files, and prints a consolidated summary.

Usage:
    python run_all_pillars.py
"""

import sys
import os
import json
import time

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

print("\n" + "="*70)
print("  PAPER 2 -- 3-PILLAR AUTONOMOUS EXECUTION")
print("  Physics-Informed Dynamic Thresholding and TinyML Quantization")
print("  for Low-Latency 5G O-RAN Handover Optimization")
print("="*70)

# -- PILLAR 1 -----------------------------------------------------------------
print("\n\n" + "*"*70)
print("  PILLAR 1: Kinematic Physics-Informed Feature Engineering")
print("*"*70)

t0 = time.time()
from src.prepare_kinematic_data import prepare_dataset, verify_integrity
import pandas as pd
from src.kinematic_features import get_kinematic_feature_names

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
ANATEL_RAW = os.path.join(DATA_DIR, "anatel_classifybase.csv")
SIM_RAW    = os.path.join(DATA_DIR, "classifybase.csv")
ANATEL_KIN = os.path.join(DATA_DIR, "anatel_kinematic.csv")
SIM_KIN    = os.path.join(DATA_DIR, "simulated_kinematic.csv")

df_anatel_raw = pd.read_csv(ANATEL_RAW)
df_anatel_kin = prepare_dataset(ANATEL_RAW, ANATEL_KIN, "ANATEL (Real Network)")
verify_integrity(df_anatel_raw, df_anatel_kin, "ANATEL")

df_sim_raw = pd.read_csv(SIM_RAW)
df_sim_kin = prepare_dataset(SIM_RAW, SIM_KIN, "Simulated (ns-3)")
verify_integrity(df_sim_raw, df_sim_kin, "Simulated")
print(f"\n  [OK] Pillar 1a: Data preparation complete ({time.time()-t0:.1f}s)")

t1 = time.time()
from src.evaluate_kinematic_models import run_dataset_evaluation
p1_anatel = run_dataset_evaluation("ANATEL (Real Network)", ANATEL_KIN)
p1_sim    = run_dataset_evaluation("Simulated (ns-3)", SIM_KIN)
p1_results = [p1_anatel, p1_sim]
with open(os.path.join(RESULTS_DIR, "pillar1_raw_results.json"), "w") as f:
    json.dump(p1_results, f, indent=2)
print(f"\n  [OK] Pillar 1b: Model evaluation complete ({time.time()-t1:.1f}s)")

# -- PILLAR 2 -----------------------------------------------------------------
print("\n\n" + "*"*70)
print("  PILLAR 2: Cost-Sensitive Dynamic Adaptive Thresholding (DAT)")
print("*"*70)

t2 = time.time()
from src.dynamic_thresholding import run_dat_evaluation
p2_anatel = run_dat_evaluation("ANATEL (Real Network)", ANATEL_KIN)
p2_sim    = run_dat_evaluation("Simulated (ns-3)", SIM_KIN)
p2_results = [p2_anatel, p2_sim]
with open(os.path.join(RESULTS_DIR, "pillar2_raw_results.json"), "w") as f:
    json.dump(p2_results, f, indent=2)
print(f"\n  [OK] Pillar 2: DAT evaluation complete ({time.time()-t2:.1f}s)")

# -- PILLAR 3 -----------------------------------------------------------------
print("\n\n" + "*"*70)
print("  PILLAR 3: Edge TinyML Quantization & Latency Profiling")
print("*"*70)

t3 = time.time()
from src.tinyml_quantization import run_pillar3
p3_anatel = run_pillar3("ANATEL (Real Network)", ANATEL_KIN)
p3_sim    = run_pillar3("Simulated (ns-3)", SIM_KIN)
p3_results = [p3_anatel, p3_sim]
with open(os.path.join(RESULTS_DIR, "pillar3_raw_results.json"), "w") as f:
    json.dump(p3_results, f, indent=2)
print(f"\n  [OK] Pillar 3: TinyML evaluation complete ({time.time()-t3:.1f}s)")

# -- CONSOLIDATED SUMMARY -----------------------------------------------------
total_time = time.time() - t0
print("\n\n" + "="*70)
print("  CONSOLIDATED RESULTS SUMMARY")
print("="*70)

print("\n[PILLAR 1] Kinematic Feature Impact (Accuracy, Decision Tree)")
for res in p1_results:
    if "Decision Tree" in res["models"]:
        m = res["models"]["Decision Tree"]
        b = m["baseline"]["accuracy"]["mean"] * 100
        k = m["kinematic"]["accuracy"]["mean"] * 100
        print(f"  {res['dataset']:28s}: Baseline={b:.2f}%  Kinematic={k:.2f}%  Δ={k-b:+.3f}%")

print("\n[PILLAR 2] DAT Threshold Optimisation (Accuracy & Cost Reduction)")
for res in p2_results:
    s_acc  = res["static"]["accuracy"]["mean"] * 100
    d_acc  = res["dat"]["accuracy"]["mean"] * 100
    s_cost = res["static"]["cost"]["mean"]
    d_cost = res["dat"]["cost"]["mean"]
    opt_th = res["optimal_threshold"]["mean"]
    print(f"  {res['dataset']:28s}: Static(0.50)={s_acc:.2f}%, DAT({opt_th:.2f})={d_acc:.2f}%  Cost: {s_cost:.1f}→{d_cost:.1f} ({(s_cost-d_cost)/s_cost*100:.1f}% ↓)")

print("\n[PILLAR 3] TinyML Quantization Speedup (ANATEL)")
for res in p3_results:
    if "MLP_INT8" in res["models"]:
        q = res["models"]["MLP_INT8"]
        f = res["models"]["MLP_FP32"]
        print(f"  {res['dataset']:28s}: FP32={f['per_sample_ms']:.4f}ms → INT8={q['per_sample_ms']:.4f}ms  ({q['speedup_vs_fp32']:.2f}×)  Size: {f['size_kb']:.1f}→{q['size_kb']:.1f}KB ({q['size_reduction_pct']:.1f}% ↓)  AccDrop={q['accuracy_drop_pct']:.3f}%")

print(f"\n  Total execution time: {total_time:.1f}s")
print("\n  All results saved to: results/")
print("="*70)
