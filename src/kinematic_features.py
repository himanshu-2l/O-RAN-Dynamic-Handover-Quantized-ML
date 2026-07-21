"""
kinematic_features.py
=====================
Pillar 1: Physics-Informed Kinematic Feature Engineering

Extracts RSRP Velocity (dRSRP/dt) and RSRP Acceleration (d²RSRP/dt²)
from 50-sample RSRP sliding window representations.

Features Added per Sample:
  - v_mean          : mean of velocity sequence (signal drift rate)
  - v_std           : std of velocity sequence (fading volatility)
  - v_min, v_max    : extremes of velocity
  - v_terminal      : terminal velocity (last 5-step average slope)
  - a_mean          : mean of acceleration sequence
  - a_std           : std of acceleration sequence
  - a_min, a_max    : extremes of acceleration
  - a_terminal      : last acceleration step (v[49] - v[48])
  - v_pos_ratio     : fraction of positive-velocity steps (approach)
  - v_neg_ratio     : fraction of negative-velocity steps (departure)
  - energy_velocity : RMS of velocity (kinematic energy proxy)
  - energy_accel    : RMS of acceleration (kinematic jerk energy proxy)
"""

import numpy as np
import pandas as pd


# RSRP column names (string-indexed 0..49)
RSRP_COLS = [str(i) for i in range(50)]
LABEL_COL = "label"


def _compute_velocity(rsrp_array: np.ndarray) -> np.ndarray:
    """
    Compute 1st-order finite difference (velocity) along time axis.
    Input : shape (N, 50)
    Output: shape (N, 49)  — v_t = r_t - r_{t-1}
    """
    return np.diff(rsrp_array, n=1, axis=1)


def _compute_acceleration(rsrp_array: np.ndarray) -> np.ndarray:
    """
    Compute 2nd-order finite difference (acceleration) along time axis.
    Input : shape (N, 50)
    Output: shape (N, 48)  — a_t = r_t - 2*r_{t-1} + r_{t-2}
    """
    return np.diff(rsrp_array, n=2, axis=1)


def extract_kinematic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame with columns '0'..'49' (RSRP sliding window) and 'label',
    returns a new DataFrame that contains the original columns PLUS kinematic
    summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset. Columns '0'...'49' are RSRP dBm values; 'label' is the
        handover decision (0/1).

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame with kinematic summary features appended.
    """
    df = df.copy()

    # --- Extract raw RSRP matrix ---
    rsrp = df[RSRP_COLS].values.astype(np.float32)  # (N, 50)

    # --- Velocity & Acceleration sequences ---
    vel = _compute_velocity(rsrp)      # (N, 49)
    acc = _compute_acceleration(rsrp)  # (N, 48)

    # --- Summary statistics ---
    # Velocity statistics
    df["v_mean"]       = vel.mean(axis=1)
    df["v_std"]        = vel.std(axis=1)
    df["v_min"]        = vel.min(axis=1)
    df["v_max"]        = vel.max(axis=1)

    # Terminal velocity: slope over last 5 samples (indices 44-49 → 4 diff steps)
    df["v_terminal"]   = vel[:, -5:].mean(axis=1)

    # Fraction of time signal is rising vs. falling
    df["v_pos_ratio"]  = (vel > 0).mean(axis=1)
    df["v_neg_ratio"]  = (vel < 0).mean(axis=1)

    # RMS (kinematic energy proxy)
    df["energy_velocity"] = np.sqrt((vel ** 2).mean(axis=1))

    # Acceleration statistics
    df["a_mean"]       = acc.mean(axis=1)
    df["a_std"]        = acc.std(axis=1)
    df["a_min"]        = acc.min(axis=1)
    df["a_max"]        = acc.max(axis=1)

    # Terminal acceleration: last diff of velocity
    df["a_terminal"]   = vel[:, -1] - vel[:, -2]

    # RMS (jerk-energy proxy)
    df["energy_accel"] = np.sqrt((acc ** 2).mean(axis=1))

    return df


def get_kinematic_feature_names() -> list:
    """Return the list of kinematic summary feature column names."""
    return [
        "v_mean", "v_std", "v_min", "v_max",
        "v_terminal", "v_pos_ratio", "v_neg_ratio", "energy_velocity",
        "a_mean", "a_std", "a_min", "a_max",
        "a_terminal", "energy_accel",
    ]


def get_all_feature_names() -> list:
    """Return full feature set: 50 RSRP + 14 kinematic summary."""
    return RSRP_COLS + get_kinematic_feature_names()


if __name__ == "__main__":
    # Quick sanity check: constant slope → constant velocity, zero acceleration
    n = 5
    slope = -2.0
    synthetic = pd.DataFrame(
        {str(i): [slope * i] * n for i in range(50)},
        dtype=float
    )
    synthetic["label"] = 0
    result = extract_kinematic_features(synthetic)

    print("=== Kinematic Sanity Check ===")
    print(f"v_mean  expected={slope:.2f}, got={result['v_mean'].iloc[0]:.4f}")
    print(f"v_std   expected=0.00, got={result['v_std'].iloc[0]:.6f}")
    print(f"a_mean  expected=0.00, got={result['a_mean'].iloc[0]:.6f}")
    print(f"a_std   expected=0.00, got={result['a_std'].iloc[0]:.6f}")
    print("Sanity check PASSED" if abs(result["a_mean"].iloc[0]) < 1e-5 else "FAILED")
