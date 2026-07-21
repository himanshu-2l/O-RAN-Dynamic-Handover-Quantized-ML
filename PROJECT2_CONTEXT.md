# PROJECT 2 HANDOFF & CONTEXT DOCUMENT
**Project Title:** Physics-Informed Dynamic Thresholding and TinyML Quantization for Low-Latency 5G O-RAN Handover Optimization

---

## 1. Domain Background & Paper 1 Accomplishments
- **Dataset 1 (Real Network):** ANATEL drive-test measurements from Tier-2 Indian city ($1,458$ balanced samples, 50-sample RSRP sliding windows).
- **Dataset 2 (Simulated Network):** ns-3 LTE module dual-strip mobility logs ($3,922$ balanced samples, 10ms sampling, TTT=64ms, A3 offset=0dB).
- **Paper 1 Results:**
  - Evaluated 9 normalization techniques under 10 runs of 5-fold Stratified Cross-Validation (50 folds).
  - Proven mathematically that Decision Trees are scale-invariant to column-wise monotonic scaling (Min-Max, Z-Score, Mean, MaxAbs, Robust, Decimal Scaling maintain baseline accuracy: 95.14% real, 97.47% simulated).
  - Row-wise L1/L2 normalization degrades accuracy by up to 1.23% due to loss of absolute RSRP signal context.
  - Published LaTeX manuscript in `ieee-paper/paper.tex`.

---

## 2. Paper 2 Objectives & 3-Pillar Framework
We are building a second research paper focusing on a novel 3-Pillar Framework:

1. **Pillar 1: Kinematic Physics-Informed Feature Layer**
   - Extract 1st derivative (RSRP Velocity $v = \frac{d\text{RSRP}}{dt}$) and 2nd derivative (RSRP Acceleration $a = \frac{d^2\text{RSRP}}{dt^2}$).
   - Give the model real-time mobility and fading speed context.

2. **Pillar 2: Cost-Sensitive Dynamic Adaptive Thresholding (DAT)**
   - Replace static $0.5$ thresholding with a dynamic cost-minimization function:
     $$\mathcal{C}(P_{\text{th}}) = w_1 \cdot N_{\text{PingPong}}(P_{\text{th}}) + w_2 \cdot N_{\text{RLF}}(P_{\text{th}})$$
   - Optimize the trade-off between Ping-Pong handovers and Radio Link Failures (RLF).

3. **Pillar 3: Edge TinyML Quantization & Latency Profiling**
   - Apply PyTorch INT8 dynamic quantization and tree pruning.
   - Benchmark inference latency (ms) and RAM usage (KB) for 5G O-RAN Near-RT RIC edge controllers.

---

## 3. Codebase Structure & Data Locations
- Workspace Directory: `C:\Users\Himanshu\Downloads\Anti-- MODEL`
- Real Dataset Path: `lstm-handover/anatel_classifybase.csv`
- Simulated Dataset Path: `lstm-handover/classifybase.csv`
- Previous Paper 1 LaTeX: `ieee-paper/paper.tex`
- Virtual Environment: `venv\Scripts\python.exe`

---

## 4. Initial Prompt to Start New Conversation
Copy and paste the following prompt when opening your new conversation in Antigravity:

> *"Hi Antigravity! We are starting the implementation of our 2nd research paper: 'Physics-Informed Dynamic Thresholding and TinyML Quantization for Low-Latency 5G O-RAN Handover Optimization'. Please read `PROJECT2_CONTEXT.md` in our workspace root to load all the background, datasets, and 3-Pillar framework details. Let's begin by implementing Pillar 1: Kinematic Feature Engineering (RSRP Velocity & Acceleration extraction)."*
