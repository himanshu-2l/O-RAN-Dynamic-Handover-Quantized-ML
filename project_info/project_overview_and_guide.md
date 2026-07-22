# Project Overview & Execution Guide

Welcome to the **5G O-RAN Handover Optimization** project documentation. This guide provides a high-level overview of the codebase, the 3-Pillar Framework, and how a newcomer can run and understand the entire project.

---

## 📌 The 3-Pillar Framework at a Glance

This project implements a lightweight xApp designed to run inside an O-RAN **Near-Real-Time RAN Intelligent Controller (Near-RT RIC)** to optimize cell handovers:

```
[ 5G NR gNB ] 
      │ (RSRP Sequence: R_t ∈ ℝ⁵⁰ via E2SM-KPM)
      ▼
┌────────────────────────────────────────────────────────┐
│             Near-RT RIC xApp Container                 │
│                                                        │
│  ┌─────────────────────────┐                           │
│  │   PILLAR 1              │                           │
│  │   Kinematic physics-    │                           │
│  │   informed features     │                           │
│  └────────────┬────────────┘                           │
│               │ (Enriched vector X_kin ∈ ℝ⁶⁴)          │
│               ▼                                        │
│  ┌─────────────────────────┐                           │
│  │   PILLAR 3              │                           │
│  │   TinyML INT8 Quantized │                           │
│  │   Neural Network        │                           │
│  └────────────┬────────────┘                           │
│               │ (Handover Probability p̂)               │
│               ▼                                        │
│  ┌─────────────────────────┐                           │
│  │   PILLAR 2              │                           │
│  │   Dynamic Adaptive      │                           │
│  │   Thresholding (DAT)    │                           │
│  └────────────┬────────────┘                           │
│               │ (Optimal Handover Decision)            │
└───────────────┼────────────────────────────────────────┘
                │
                ▼ (E2AP Handover Command)
        [ Target gNB ]
```

---

## 📂 Codebase Directory Tour

Here is a map of the repository files and what they do:

```
.
├── paper/
│   ├── main.tex                         # 6-page IEEE LaTeX source file
│   ├── main.pdf                         # Compiled presentation-ready PDF
│   └── figures/                         # Vector PDF and PNG paper plots
├── notebooks/
│   ├── oran_handover_3pillar_colab.ipynb# Google Colab Notebook (1-click Run All)
│   └── oran_handover_3pillar.ipynb      # Local Jupyter notebook
├── src/
│   ├── prepare_kinematic_data.py        # Preprocessing & CSV dataset loading
│   ├── kinematic_features.py            # Pillar 1 velocity/acceleration extraction
│   ├── evaluate_kinematic_models.py     # Pillar 1 5-fold cross-validation
│   ├── dynamic_thresholding.py          # Pillar 2 cost curve minimization
│   ├── tinyml_quantization.py           # Pillar 3 INT8 quantization & latency profiling
│   ├── generate_arch_diagram.py         # System architecture diagram generator
│   └── generate_paper_plots.py          # Paper publication plot generator
├── data/
│   ├── anatel_classifybase.csv          # Real 5G network drive-test dataset (N=1,458)
│   └── classifybase.csv                 # 3GPP ns-3 simulation dataset (N=3,922)
├── project_info/                        # Presentation, Q&A, and Defense guides
│   ├── project_overview_and_guide.md    # [This File] Core guide
│   ├── accuracy_justification_and_defenses.md # Analysis of accuracy & metrics
│   ├── reviewer_qna_defense.md          # Defense Q&A for supervisor & reviewers
│   └── presentation_slides_content.md   # PPT Presentation outline (Slides 1-10)
└── run_all_pillars.py                   # Master autonomous execution script
```

---

## 🚀 How to Run the Project

### 1. Set Up Python Environment
Ensure you have the required libraries installed:
```bash
pip install torch scikit-learn pandas numpy matplotlib seaborn xgboost
```

### 2. Run the Consolidated Benchmark
To execute the entire pipeline (Pillars 1, 2, and 3) across both datasets and print out a summary table of results:
```bash
python run_all_pillars.py
```

### 3. Generate the Figures
To re-create the publication-quality figures used in the LaTeX paper:
```bash
# Generate Fig. 1 (System Architecture)
python src/generate_arch_diagram.py

# Generate Figs. 2, 3, 4, and 5 (Performance Curves, RLF Drops, Pareto Frontiers)
python src/generate_paper_plots.py
```

### 4. Interactive Development
Open the Jupyter notebooks in `notebooks/` to run step-by-step visualizations or click the **Open in Colab** badge in the notebooks to run them on the web.
