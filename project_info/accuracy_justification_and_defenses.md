# Classification Accuracy Justification & Reviewer Defenses

This document provides the scientific and engineering justifications for the accuracy levels obtained in this project. Use this information to defend your results to your supervisor ("sir") or peer reviewers during the IEEE journal submission process.

---

## 📊 Summary of Obtained Results

### 1. Primary Classification Models (Pillar 1)
Evaluated across **10 runs of 5-fold Stratified Cross-Validation (50 folds total)**:
* **ANATEL (Real Drive-Test Dataset):**
  * **XGBoost (Kinematic):** **$94.43\% \pm 1.27\%$** accuracy, **$0.9886$** ROC-AUC.
  * **Random Forest (Kinematic):** **$94.13\% \pm 1.28\%$** accuracy, **$0.9870$** ROC-AUC.
* **Simulated Dataset (ns-3):**
  * **XGBoost (Kinematic):** **$95.87\% \pm 0.85\%$** accuracy, **$0.9934$** ROC-AUC.
  * **Random Forest (Kinematic):** **$94.90\% \pm 0.80\%$** accuracy, **$0.9899$** ROC-AUC.

### 2. Edge TinyML Models (Pillar 3)
* **MLP FP32 Baseline:** $88.01\%$ accuracy, $29.64\text{ KB}$ footprint, $0.124\text{ ms}$ latency.
* **MLP INT8 (Quantized):** **$86.64\%$** accuracy (**$56.8\%$ RAM compression** to $12.80\text{ KB}$, $0.651\text{ ms}$ latency).
* **Pruned Decision Tree ($depth=8$):** **$82.88\%$** accuracy, **$9.76\text{ KB}$** footprint, **$0.096\text{ ms}$** latency.

---

## 🛡️ Reviewer Defense Arguments

### Defense 1: "Why is the accuracy ~94-96% instead of 99-100%?"
* **Physical Reality of Fading Channels:** RSRP measurements in real-world 5G NR networks are subject to heavy log-normal shadowing and fast multipath fading. Fading fluctuations are inherently stochastic. Expecting 100% accuracy in a high-velocity environment is physically unrealistic.
* **Overfitting / Data Leakage Check:** Many papers claiming $>99\%$ accuracy are flawed because they use *non-causal data* (looking ahead in the window) or overfit to a single trajectory. Our framework uses a strict **sliding-window history (past 50 samples)** to evaluate handover decisions, ensuring no lookahead bias.
* **Statistical Rigor:** Our results are reported as the mean and standard deviation across **50 independent cross-validation folds**. This is far more rigorous than papers reporting accuracy from a single train-test split.

### Defense 2: "Is the ~1.3% accuracy drop in the INT8 quantized model acceptable?"
* **Edge Execution Feasibility:** Near-RT RIC xApps operate under tight resource limits and sub-10 ms latency SLAs. The FP32 model consumes twice as much memory. A minor drop of **$1.37\%$** (ANATEL) and **$0.64\%$** (ns-3) is an exceptional trade-off for a **$56.8\%$ RAM compression** (saving cache space on edge nodes).
* **Quantization Noise:** The minor degradation is expected due to the loss of precision when mapping 32-bit floats to 8-bit integers. However, this is minimized because our MLP contains no batch normalization layers (which are incompatible with dynamic quantization).

### Defense 3: "Why did you select the DAT threshold P_th* instead of optimizing for raw Accuracy?"
* **Asymmetric Failure Costs:** In operational cellular networks, classification errors do not have equal consequences:
  * **False Positive (Ping-Pong HO):** Small signaling overhead, connection is maintained.
  * **False Negative (Radio Link Failure - RLF):** Call drops, catastrophic service disruption.
* **DAT Objective:** Pillar 2 prioritizes connection stability by penalizing False Negatives $5\times$ more heavily than False Positives ($w_2/w_1 = 5.0$). 
* **The Result:** Minimizing this cost function shifts the optimal probability threshold to $P_{\text{th}}^* = 0.35$ (ANATEL). While this shifts accuracy slightly, it **reduces connection drops (RLFs) by 81.9%**.
* **Key Phrase:** *"We intentionally accept a controlled increase in false handovers to achieve an 81.9% reduction in catastrophic link drops."*
