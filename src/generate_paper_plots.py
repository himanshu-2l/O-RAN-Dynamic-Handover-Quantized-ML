"""
generate_paper_plots.py
========================
Generates publication-quality figures for Paper 2:
1. Figure 1: Feature Importance Rankings (Kinematic vs Raw RSRP)
2. Figure 2: Dynamic Adaptive Thresholding (DAT) Cost Curves C(P_th)
3. Figure 3: RLF (FN) vs Ping-Pong (FP) Tradeoff Comparison
4. Figure 4: TinyML Edge Pareto Frontier (Accuracy vs Model Size vs Latency)

Output Directory: paper/figures/
Resolution: 300 DPI (PNG + PDF)
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set publication style using standard matplotlib
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.titlesize': 12,
    'figure.dpi': 300,
})

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PAPER_FIG_DIR = os.path.join(BASE_DIR, "paper", "figures")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)


# ── 1. FIGURE 1: KINEMATIC FEATURE IMPORTANCE ───────────────────────────────
def plot_figure_1():
    print("Generating Figure 1: Kinematic Feature Importance...")
    p1_path = os.path.join(RESULTS_DIR, "pillar1_raw_results.json")
    with open(p1_path, "r") as f:
        p1_data = json.load(f)

    anatel_xgb_fi = p1_data[0]["models"]["XGBoost"]["top_features"]
    sim_xgb_fi = p1_data[1]["models"]["XGBoost"]["top_features"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)

    def get_colors(features):
        kin_names = ["v_mean", "v_std", "v_min", "v_max", "v_terminal", "v_pos_ratio",
                     "v_neg_ratio", "energy_velocity", "a_mean", "a_std", "a_min",
                     "a_max", "a_terminal", "energy_accel"]
        return ['#1f77b4' if f in kin_names else '#aec7e8' for f in features]

    # Subplot 1: ANATEL
    feats_a = list(anatel_xgb_fi.keys())[:10]
    vals_a = list(anatel_xgb_fi.values())[:10]
    colors_a = get_colors(feats_a)
    y_pos_a = np.arange(len(feats_a))
    ax1.barh(y_pos_a, vals_a, align='center', color=colors_a, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos_a)
    ax1.set_yticklabels(feats_a)
    ax1.invert_yaxis()
    ax1.set_xlabel('Feature Importance Score')
    ax1.set_title('(a) ANATEL Real Network (XGBoost)', fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Simulated
    feats_s = list(sim_xgb_fi.keys())[:10]
    vals_s = list(sim_xgb_fi.values())[:10]
    colors_s = get_colors(feats_s)
    y_pos_s = np.arange(len(feats_s))
    ax2.barh(y_pos_s, vals_s, align='center', color=colors_s, edgecolor='black', linewidth=0.5)
    ax2.set_yticks(y_pos_s)
    ax2.set_yticklabels(feats_s)
    ax2.invert_yaxis()
    ax2.set_xlabel('Feature Importance Score')
    ax2.set_title('(b) ns-3 Simulated Network (XGBoost)', fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#1f77b4', edgecolor='black', label='Kinematic Feature (v, a)'),
        Patch(facecolor='#aec7e8', edgecolor='black', label='Raw RSRP Time Step')
    ]
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, frameon=True)

    plt.tight_layout()
    png_path = os.path.join(PAPER_FIG_DIR, "fig1_kinematic_importance.png")
    pdf_path = os.path.join(PAPER_FIG_DIR, "fig1_kinematic_importance.pdf")
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f" Saved -> {png_path}")


# ── 2. FIGURE 2: DAT COST CURVES ─────────────────────────────────────────────
def plot_figure_2():
    print("Generating Figure 2: DAT Operational Cost Curves C(P_th)...")
    sys.path.insert(0, os.path.join(BASE_DIR, "src"))
    from dynamic_thresholding import cost_function, THRESHOLD_GRID, get_all_feature_names
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    data_dir = os.path.join(BASE_DIR, "data")
    df_a = pd.read_csv(os.path.join(data_dir, "anatel_kinematic.csv"))
    df_s = pd.read_csv(os.path.join(data_dir, "simulated_kinematic.csv"))

    feature_cols = get_all_feature_names()

    # ANATEL
    X_a, y_a = df_a[feature_cols].values, df_a["label"].values.astype(int)
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(X_a, y_a, test_size=0.3, random_state=42, stratify=y_a)
    m_a = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xa_tr, ya_tr)
    pa_te = m_a.predict_proba(Xa_te)[:, 1]
    costs_a = [cost_function(ya_te, pa_te, th, 1.0, 5.0) for th in THRESHOLD_GRID]
    opt_th_a = THRESHOLD_GRID[np.argmin(costs_a)]

    # Simulated
    X_s, y_s = df_s[feature_cols].values, df_s["label"].values.astype(int)
    Xs_tr, Xs_te, ys_tr, ys_te = train_test_split(X_s, y_s, test_size=0.3, random_state=42, stratify=y_s)
    m_s = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(Xs_tr, ys_tr)
    ps_te = m_s.predict_proba(Xs_te)[:, 1]
    costs_s = [cost_function(ys_te, ps_te, th, 1.0, 5.0) for th in THRESHOLD_GRID]
    opt_th_s = THRESHOLD_GRID[np.argmin(costs_s)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.0))

    # Curve 1: ANATEL
    ax1.plot(THRESHOLD_GRID, costs_a, color='#d62728', linewidth=2.0, label=r'Cost C(P_th)')
    ax1.axvline(x=0.50, color='gray', linestyle='--', linewidth=1.2, label='Static (P_th=0.50)')
    ax1.axvline(x=opt_th_a, color='#2ca02c', linestyle='-', linewidth=1.8, label=f'DAT Optimal ({opt_th_a:.2f})')
    ax1.scatter([opt_th_a], [min(costs_a)], color='#2ca02c', s=70, zorder=5)
    ax1.set_xlabel(r'Probability Threshold ($P_{th}$)')
    ax1.set_ylabel(r'Operational Cost $\mathcal{C}(P_{th})$')
    ax1.set_title('(a) ANATEL Real Network Cost Minimisation', fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Curve 2: Simulated
    ax2.plot(THRESHOLD_GRID, costs_s, color='#1f77b4', linewidth=2.0, label=r'Cost C(P_th)')
    ax2.axvline(x=0.50, color='gray', linestyle='--', linewidth=1.2, label='Static (P_th=0.50)')
    ax2.axvline(x=opt_th_s, color='#2ca02c', linestyle='-', linewidth=1.8, label=f'DAT Optimal ({opt_th_s:.2f})')
    ax2.scatter([opt_th_s], [min(costs_s)], color='#2ca02c', s=70, zorder=5)
    ax2.set_xlabel(r'Probability Threshold ($P_{th}$)')
    ax2.set_ylabel(r'Operational Cost $\mathcal{C}(P_{th})$')
    ax2.set_title('(b) ns-3 Simulated Network Cost Minimisation', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    png_path = os.path.join(PAPER_FIG_DIR, "fig2_dat_cost_curves.png")
    pdf_path = os.path.join(PAPER_FIG_DIR, "fig2_dat_cost_curves.pdf")
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f" Saved -> {png_path}")


# ── 3. FIGURE 3: RLF REDUCTION & TRADEOFF ───────────────────────────────────
def plot_figure_3():
    print("Generating Figure 3: RLF (FN) vs Ping-Pong (FP) Tradeoff...")
    p2_path = os.path.join(RESULTS_DIR, "pillar2_raw_results.json")
    with open(p2_path, "r") as f:
        p2_data = json.load(f)

    anatel_st = p2_data[0]["static"]
    anatel_dat = p2_data[0]["dat"]
    sim_st = p2_data[1]["static"]
    sim_dat = p2_data[1]["dat"]

    labels = ['ANATEL Real Network', 'ns-3 Simulated Network']
    
    rlf_static = [anatel_st["n_rlf_fn"]["mean"], sim_st["n_rlf_fn"]["mean"]]
    rlf_dat = [anatel_dat["n_rlf_fn"]["mean"], sim_dat["n_rlf_fn"]["mean"]]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0))

    # Subplot 1: RLF (FN) Reduction
    rects1 = ax1.bar(x - width/2, rlf_static, width, label='Static (P_th=0.50)', color='#d62728', edgecolor='black')
    rects2 = ax1.bar(x + width/2, rlf_dat, width, label='DAT Optimal', color='#2ca02c', edgecolor='black')
    ax1.set_ylabel('Radio Link Failures (FN Count)')
    ax1.set_title('(a) RLF Drop (81.9% Reduction in ANATEL)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)

    for b in rects1 + rects2:
        h = b.get_height()
        ax1.annotate(f'{h:.1f}', xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    # Subplot 2: Total Cost Reduction
    cost_static = [anatel_st["cost"]["mean"], sim_st["cost"]["mean"]]
    cost_dat = [anatel_dat["cost"]["mean"], sim_dat["cost"]["mean"]]

    rects3 = ax2.bar(x - width/2, cost_static, width, label='Static (P_th=0.50)', color='#7f7f7f', edgecolor='black')
    rects4 = ax2.bar(x + width/2, cost_dat, width, label='DAT Optimal', color='#1f77b4', edgecolor='black')
    ax2.set_ylabel(r'Total Operational Cost $\mathcal{C}(P_{th})$')
    ax2.set_title('(b) Total Cost Reduction (44.7% Reduction in ANATEL)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    for b in rects3 + rects4:
        h = b.get_height()
        ax2.annotate(f'{h:.1f}', xy=(b.get_x() + b.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8.5, fontweight='bold')

    plt.tight_layout()
    png_path = os.path.join(PAPER_FIG_DIR, "fig3_rlf_reduction.png")
    pdf_path = os.path.join(PAPER_FIG_DIR, "fig3_rlf_reduction.pdf")
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f" Saved -> {png_path}")


# ── 4. FIGURE 4: TINYML PARETO FRONTIER ─────────────────────────────────────
def plot_figure_4():
    print("Generating Figure 4: TinyML Edge Pareto Frontier...")
    p3_path = os.path.join(RESULTS_DIR, "pillar3_raw_results.json")
    with open(p3_path, "r") as f:
        p3_data = json.load(f)

    anatel_models = p3_data[0]["models"]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    model_colors = {
        "MLP_FP32": ("#1f77b4", "o", "MLP FP32 (29.6 KB, 88.0%)"),
        "MLP_INT8": ("#2ca02c", "s", "MLP INT8 (12.8 KB, 86.6%, -56.8% RAM)"),
        "RandomForest": ("#d62728", "^", "Random Forest (1.32 MB, 88.7%)"),
        "DecisionTree_Pruned": ("#ff7f0e", "D", "DT Pruned (9.8 KB, 82.9%, 0.096ms)")
    }

    for name, mdata in anatel_models.items():
        color, marker, label = model_colors.get(name, ("black", "o", name))
        size_kb = mdata["size_kb"]
        acc = mdata["accuracy"] * 100

        ax.scatter(size_kb, acc, color=color, marker=marker, s=140, zorder=5, label=label, edgecolor='black', linewidth=0.8)
        
        ax.annotate(f"{name}\n({acc:.1f}%, {size_kb:.1f}KB)",
                    xy=(size_kb, acc), xytext=(size_kb * 1.15, acc - 0.6),
                    fontsize=8.0, fontweight='bold',
                    arrowprops=dict(arrowstyle="->", color=color, lw=0.8))

    ax.set_xscale('log')
    ax.set_xlabel('Model Footprint (KB, log scale)')
    ax.set_ylabel('Classification Accuracy (%)')
    ax.set_title('Near-RT RIC Edge Deployment Pareto Frontier (ANATEL Real Network)', fontweight='bold')
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True)

    ax.axhline(y=85, color='gray', linestyle=':', alpha=0.7)
    ax.text(12, 85.3, 'Target Accuracy Threshold (85%)', fontsize=8, color='gray', fontstyle='italic')

    plt.tight_layout()
    png_path = os.path.join(PAPER_FIG_DIR, "fig4_tinyml_pareto.png")
    pdf_path = os.path.join(PAPER_FIG_DIR, "fig4_tinyml_pareto.pdf")
    plt.savefig(png_path, bbox_inches='tight', dpi=300)
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f" Saved -> {png_path}")


if __name__ == "__main__":
    print("="*60)
    print("Generating All IEEE Paper Publication Figures...")
    print("="*60)
    plot_figure_1()
    plot_figure_2()
    plot_figure_3()
    plot_figure_4()
    print("\nAll figures generated successfully in paper/figures/")
