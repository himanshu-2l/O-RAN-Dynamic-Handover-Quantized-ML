"""
prepare_kinematic_data.py
=========================
Pillar 1 Pipeline: Load raw datasets → Extract kinematic features → Save enriched CSVs.

Usage:
    python src/prepare_kinematic_data.py
"""

import sys
import os
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.kinematic_features import extract_kinematic_features, get_kinematic_feature_names, RSRP_COLS

# ── Dataset paths ────────────────────────────────────────────────────────────
DATA_DIR      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
ANATEL_RAW    = os.path.join(DATA_DIR, "anatel_classifybase.csv")
SIM_RAW       = os.path.join(DATA_DIR, "classifybase.csv")
ANATEL_OUT    = os.path.join(DATA_DIR, "anatel_kinematic.csv")
SIM_OUT       = os.path.join(DATA_DIR, "simulated_kinematic.csv")


def prepare_dataset(raw_path: str, out_path: str, name: str) -> pd.DataFrame:
    """Load raw CSV, apply kinematic extraction, save, and print summary."""
    print(f"\n{'='*60}")
    print(f"  Processing: {name}")
    print(f"{'='*60}")

    df = pd.read_csv(raw_path)
    print(f"  Raw shape         : {df.shape}")
    print(f"  Label distribution: {df['label'].value_counts().to_dict()}")

    # Extract kinematic features
    df_kin = extract_kinematic_features(df)
    print(f"  Enriched shape    : {df_kin.shape}")
    print(f"  New features added: {get_kinematic_feature_names()}")

    # Print descriptive stats for kinematic features
    kin_names = get_kinematic_feature_names()
    print(f"\n  Kinematic Feature Statistics:")
    stats = df_kin[kin_names].describe().round(4)
    print(stats.to_string())

    # Correlation with label
    print(f"\n  Pearson Correlation with Handover Label:")
    correlations = df_kin[kin_names + ["label"]].corr()["label"].drop("label").sort_values(key=abs, ascending=False)
    print(correlations.round(4).to_string())

    # Save
    df_kin.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    return df_kin


def verify_integrity(df_orig: pd.DataFrame, df_kin: pd.DataFrame, name: str):
    """Ensure no data loss or NaN corruption during processing."""
    errors = []
    if len(df_orig) != len(df_kin):
        errors.append(f"Row count mismatch: {len(df_orig)} → {len(df_kin)}")
    if df_kin[RSRP_COLS].isnull().any().any():
        errors.append("NaN detected in original RSRP columns after processing")
    kin_names = get_kinematic_feature_names()
    if df_kin[kin_names].isnull().any().any():
        errors.append("NaN detected in kinematic feature columns")
    if errors:
        print(f"\n  ❌ INTEGRITY ERRORS for {name}:")
        for e in errors:
            print(f"     - {e}")
        return False
    else:
        print(f"\n  ✅ Integrity check PASSED for {name} — No NaNs, row count preserved.")
        return True


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    # Process both datasets
    df_anatel_raw = pd.read_csv(ANATEL_RAW)
    df_anatel_kin = prepare_dataset(ANATEL_RAW, ANATEL_OUT, "ANATEL (Real Network)")
    verify_integrity(df_anatel_raw, df_anatel_kin, "ANATEL")

    df_sim_raw = pd.read_csv(SIM_RAW)
    df_sim_kin = prepare_dataset(SIM_RAW, SIM_OUT, "Simulated (ns-3)")
    verify_integrity(df_sim_raw, df_sim_kin, "Simulated")

    print("\n" + "="*60)
    print("  Pillar 1 Data Preparation COMPLETE")
    print(f"  ANATEL kinematic  → {ANATEL_OUT}")
    print(f"  Simulated kinematic → {SIM_OUT}")
    print("="*60)
