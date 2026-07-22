# PPT Presentation Slides Content

This document contains slide-by-slide text content that you can copy and paste directly into PowerPoint or Google Slides to present your project to your supervisor ("sir") or an academic jury.

---

## 🖥️ Slide 1: Title Slide
* **Title:** Physics-Informed Dynamic Thresholding and TinyML Quantization for Low-Latency 5G O-RAN Handover Optimization
* **Subtitle:** xApp Design for Near-Real-Time RIC Controllers
* **Presenter:** Himanshu Kumar
* **Affiliation:** Department of Electronics & Communication Engineering

---

## 🖥️ Slide 2: The Handover Challenge in 5G
* **Context:** Dense 5G deployments introduce frequent handovers (HO).
* **3GPP A3 Trigger Limits:** Rely on static hysteresis ($H_y$) and Time-to-Trigger ($TTT$) parameters.
* **Failure Modes:**
  * **False Positives:** Ping-Pong handovers (signaling overhead).
  * **False Negatives:** Delayed triggers causing Radio Link Failures (RLF) and dropped calls.
* **Problem:** Static triggers cannot adapt to fast shadowing and user speed fluctuations.

---

## 🖥️ Slide 3: Gaps in Prior Literature
* **Gap 1: Missing Physical Context:** Conventional ML models use raw RSRP sequences, neglecting underlying user kinematics and signal drift velocity.
* **Gap 2: Computational Overhead:** Heavy models (e.g., Random Forest ensembles $>3.7\text{ MB}$) cannot run within the sub-10 ms SLA of Near-Real-Time RAN Intelligent Controllers (Near-RT RIC).
* **Our Solution:** A **3-Pillar Framework** combining Kinematics, Cost-Sensitive Thresholds, and TinyML Quantization.

---

## 🖥️ Slide 4: Pillar 1 – Kinematic Physics Layer
* **Concept:** Extract physics-based derivatives directly from the 50-sample RSRP sliding window ($R_t$).
* **Mathematical Derivations:**
  * **Velocity ($v_t$):** $v_t = \frac{r_t - r_{t-1}}{\Delta t}$ (measures signal drift speed).
  * **Acceleration ($a_t$):** $a_t = r_t - 2r_{t-1} + r_{t-2}$ (isolates fading volatility).
  * **Energy Proxies ($E_v, E_a$):** RMS velocity and acceleration (quantify signal turbulence).
* **Result:** Appends 14 physics-informed features, boosting model capability.

---

## 🖥️ Slide 5: Pillar 2 – Cost-Sensitive DAT
* **Problem:** Standard ML targets symmetric accuracy. In telecom, an RLF (False Negative) is catastrophic, while a Ping-Pong (False Positive) is minor.
* **Cost Function:** $\mathcal{C}(P_{\text{th}}) = w_1 \text{FP}(P_{\text{th}}) + w_2 \text{FN}(P_{\text{th}})$ (set $w_2/w_1 = 5.0$).
* **Dynamic Adaptive Thresholding (DAT):** Automatically profiles predicted probabilities to find optimal $P_{\text{th}}^*$ that minimizes the operational cost.

---

## 🖥️ Slide 6: Pillar 3 – Edge TinyML Engine
* **Goal:** Compress models to execute in $<1\text{ ms}$ on edge xApps.
* **Architectures Evaluated:**
  * **PyTorch MLP FP32:** 64 -> 64 -> 32 -> 16 -> 1 neural network.
  * **PyTorch MLP INT8:** Quantized weights via affine mapping: $q = \text{round}(r/S) + Z$.
  * **Pruned Decision Tree:** Lightweight depth-8 fallback.

---

## 🖥️ Slide 7: Experimental Setup
* **Real-World Data:** ANATEL drive-test dataset ($N = 1,458$ samples).
* **Simulation Data:** ns-3 dual-strip highway mobility dataset ($N = 3,922$ samples).
* **Validation Rigor:** 10 runs of 5-Fold Stratified Cross-Validation (50 independent folds total).

---

## 🖥️ Slide 8: Key Results – Feature Importance & DAT
* **Pillar 1 Findings:** Kinematic energy features (`energy_accel`, `energy_velocity`) rank **#1** in feature importance across classifiers.
* **Pillar 2 Findings:** 
  * Optimal threshold shifts to $P_{\text{th}}^* = 0.35$ (ANATEL) and $0.46$ (Simulation).
  * **Radio Link Failures (RLFs) dropped by 81.9%** ($p < 0.001$).
  * Total operational cost cut by **44.7%**.

---

## 🖥️ Slide 9: Key Results – TinyML Benchmarks
* **INT8 Quantization Impact:**
  * **56.8% RAM Reduction** (compressed from $29.64\text{ KB}$ to $12.80\text{ KB}$).
  * Negligible accuracy loss ($<1.37\%$ drop).
* **Latency Profile:**
  * Pruned Decision Tree: **$0.096\text{ ms}$** (ultra-fast, cache-friendly).
  * MLP INT8: **$0.651\text{ ms}$** (comfortably satisfies sub-10 ms Near-RT RIC SLAs).

---

## 🖥️ Slide 10: Conclusion
* **Scientific Contribution:** Proved that physics-informed features and cost-sensitive thresholds outperform standard static handovers.
* **Practical Value:** Developed an executable xApp codebase that fits L1/L2 memory limits of edge RAN hardware.
* **Future Work:** Multi-cell beamforming optimization and hardware-in-the-loop RIC testing.
