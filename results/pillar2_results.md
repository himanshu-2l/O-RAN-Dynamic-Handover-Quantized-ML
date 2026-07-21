# Pillar 2 Results — Cost-Sensitive Dynamic Adaptive Thresholding (DAT)

## Problem Formulation

Static threshold $P_{th} = 0.5$ treats Ping-Pong handovers and Radio Link Failures (RLF) symmetrically.
In O-RAN operations, these have very different costs:

- **Ping-Pong (FP):** Unnecessary HO trigger → minor overhead, additional signalling
- **Radio Link Failure (FN):** Missed HO trigger → connection drop, re-establishment delay, QoE degradation

**Cost function:**
$$\mathcal{C}(P_{th}) = w_1 \cdot N_{\text{PingPong}}(P_{th}) + w_2 \cdot N_{\text{RLF}}(P_{th})$$

**Weights used:** $w_1 = 1.0$ (Ping-Pong), $w_2 = 5.0$ (RLF — 5× more costly)

**Threshold search:** Grid over $P_{th} \in [0.05, 0.95]$ with 181 points.

**Evaluation:** 5 runs × 5-fold Stratified CV on Random Forest (kinematic-enriched, best Pillar 1 model).

---

## Results: ANATEL (Real Network)

**Optimal Threshold:** $P_{th}^* = 0.3498 \pm 0.0611$

| Metric | Static ($P_{th} = 0.50$) | DAT ($P_{th}^* = 0.35$) | Change |
|:--|:--:|:--:|:--:|
| Accuracy | 94.01% | 92.74% | −1.26% |
| Precision | 93.45% | 88.08% | −5.37% |
| **Recall** | 94.70% | **99.04%** | **+4.34%** |
| F1-Score | 94.04% | 93.20% | −0.84% |
| **Ping-Pong (FP)** | 9.76 | 19.76 | +10.00 (↑ acceptable) |
| **RLF (FN)** | **7.72** | **1.40** | **−6.32 (↓ 81.9%)** |
| **Operational Cost** | **48.36** | **26.76** | **−21.60 (↓ 44.7%)** |

> **Key Result:** DAT reduces Radio Link Failures by **81.9%** (7.72 → 1.40 per fold) at the cost of +10 additional Ping-Pong events per fold. Since $w_2/w_1 = 5$, this tradeoff is highly favourable: each prevented RLF saves 5× the cost of one additional Ping-Pong.

---

## Results: Simulated (ns-3)

**Optimal Threshold:** $P_{th}^* = 0.4574 \pm 0.0549$

| Metric | Static ($P_{th} = 0.50$) | DAT ($P_{th}^* = 0.46$) | Change |
|:--|:--:|:--:|:--:|
| Accuracy | 94.89% | 94.13% | −0.76% |
| Precision | 92.75% | 90.77% | −1.98% |
| **Recall** | 97.42% | **98.39%** | **+0.97%** |
| F1-Score | 95.02% | 94.40% | −0.62% |
| **Ping-Pong (FP)** | 29.96 | 39.72 | +9.76 |
| **RLF (FN)** | **10.12** | **6.32** | **−3.80 (↓ 37.6%)** |
| **Operational Cost** | **80.56** | **71.32** | **−9.24 (↓ 11.5%)** |

> **Key Result:** Simulated data has a smaller threshold shift (0.50 → 0.46) because the ns-3 simulator produces more balanced RLF/Ping-Pong rates than real-world drive tests. DAT still reduces RLF by **37.6%** and total cost by **11.5%**.

---

## Threshold Shift Analysis

| Dataset | Static $P_{th}$ | Optimal $P_{th}^*$ | Shift Direction | Physical Interpretation |
|:--|:--:|:--:|:--|:--|
| ANATEL (Real) | 0.50 | **0.35** | ↓ More aggressive | Real-world channels have bursty RLF; lower threshold triggers HO earlier |
| Simulated (ns-3) | 0.50 | **0.46** | ↓ Moderate | Simulated LTE model has smoother fading; smaller correction needed |

---

## Paper Narrative

The DAT contribution demonstrates that **static 0.5 thresholding is suboptimal** for O-RAN deployments where RLF and Ping-Pong have asymmetric operational costs. By formulating the threshold selection as a cost-minimisation problem:

1. **The optimal threshold shifts below 0.5** on both datasets — validating that conservative early-HO decisions are preferred when RLF costs dominate
2. **RLF reduction of 37–82%** directly translates to improved user QoE and reduced re-establishment signalling overhead
3. **The cost function $\mathcal{C}(P_{th})$ is differentiable in expectation**, enabling future gradient-based optimisation for real-time Near-RT RIC policy updates
4. The result generalises across real (ANATEL) and simulated (ns-3) channel conditions
