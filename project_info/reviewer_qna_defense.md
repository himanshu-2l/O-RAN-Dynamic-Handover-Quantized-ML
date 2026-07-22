# expected IEEE Reviewer & Advisor Q&A

This document contains expected questions you might receive from your academic advisor ("sir") or IEEE peer reviewers, alongside structured, professional answers.

---

## 🙋‍♂️ Q1: "What is the physical meaning of RSRP Velocity ($v_t$) and Acceleration ($a_t$)?"
* **Answer:** RSRP Velocity ($v_t$) measures the rate of signal change (dBm/step). It directly corresponds to the **signal drift velocity**, indicating whether the user is physically moving closer to (positive velocity) or away from (negative velocity) the serving base station.
* RSRP Acceleration ($a_t$) captures the rate of velocity change. It acts as a proxy for **fading intensity/volatility**. Rapid fluctuations in acceleration indicate multipath fading (like Rayleigh/Rician fading) and scattering, allowing the model to distinguish between steady path-loss changes and temporary signal spikes.

---

## 🙋‍♂️ Q2: "Why did you choose a cost ratio of $5.0$ ($w_2/w_1 = 5.0$) for Pillar 2?"
* **Answer:** This ratio represents standard telecom operator priorities:
  1. A **False Positive (Ping-Pong HO)** forces the core network to exchange control signaling (releasing and allocating radio resources), but the user's connection remains uninterrupted. The penalty is low ($w_1 = 1.0$).
  2. A **False Negative (Radio Link Failure - RLF)** causes a complete connection drop, requiring RRC connection re-establishment, which interrupts user data for several seconds. The penalty is high ($w_2 = 5.0$).
* **Flexible Tuning:** The cost function is highly parameterized; operators can increase the penalty ($w_2 = 10.0$ or $20.0$) in high-reliability networks (e.g., Ultra-Reliable Low-Latency Communications - URLLC).

---

## 🙋‍♂️ Q3: "Why did the optimal threshold $P_{\text{th}}^*$ differ between the ANATEL ($0.35$) and simulated ($0.46$) datasets?"
* **Answer:** The two datasets represent different physical environments:
  * **ANATEL (Real Drive-Test)** features real-world shadowing, terrain blocks, and buildings, leading to highly variable signal drop-offs. The model must trigger handovers earlier ($P_{\text{th}}^* = 0.35$) to prevent sudden link drops.
  * **Simulated (ns-3)** follows a structured dual-strip highway mobility model with standard path-loss equations. The fading is more predictable, allowing the model to wait longer before initiating a handover ($P_{\text{th}}^* = 0.46$).

---

## 🙋‍♂️ Q4: "How does the Near-RT RIC handle the execution of your TinyML model?"
* **Answer:** Near-RT RIC controllers host containerized applications called xApps. Our model is built to run as a **Mobility Management xApp**.
* The xApp subscribes to RSRP telemetry via the **E2 interface** (specifically `E2SM-KPM` service models).
* The gNB streams RSRP measurements asynchronously.
* The xApp runs our quantized INT8 MLP model (taking only $0.651\text{ ms}$) and, if a handover is triggered, sends a control command back to the gNB via the `E2AP` protocol to execute the handover.

---

## 🙋‍♂️ Q5: "Is your framework compatible with 3GPP A3 event triggers?"
* **Answer:** Yes, it can act as a **dynamic parameter override** for 3GPP triggers. Instead of replacing the A3 event, our xApp can dynamically update the A3 parameters ($H_y$ hysteresis and $TTT$ Time-To-Trigger) in real-time based on the user's current velocity and fading acceleration, making standard 3GPP handovers context-aware.

---

## 🙋‍♂️ Q6: "Why did you use PyTorch dynamic quantization instead of static quantization?"
* **Answer:** Dynamic quantization is ideal for Recurrent Neural Networks (RNNs) and Multi-Layer Perceptrons (MLPs) where execution time is dominated by memory bandwidth for loading weights. It quantizes weights to 8-bit integers offline while dynamically calculating activations during inference. This removes the need for a separate calibration dataset, making it easy to deploy across different gNB architectures.
