# RLCD Decision Transformers vs Classical Autoregressive Transformers
## Engineering & Learning Roadmap

This document outlines the complete milestone curriculum for understanding, building, and benchmarking **RLCD (Reinforcement Learning for Calibrated Decisions)** models (like TypeSafe AI's **Jev** and open-source models like **Laya**) against classical autoregressive LLMs.

---

## The 5-Milestone Roadmap

### Milestone 1 — Attention Geometry & The Latency Regime
* **Core Question:** Why does answering a structured question with a generative LLM take 1200 ms and gigabytes of KV cache, while an RLCD decision transformer answers in 25 ms with zero cache?
* **Mechanics:**
  * Causal lower-triangular mask vs. full bidirectional attention matrix.
  * Why autoregressive decode is memory-bandwidth bound ($1.0 \text{ FLOP/byte}$) while single-pass decision is compute-bound.
  * Derivation of KV cache memory complexity $O(L \cdot d)$ vs. zero cache.
* **Metric:** Latency and peak VRAM comparison across batch sizes.

---

### Milestone 2 — Projection Head Manifold Geometry
* **Core Question:** What is the mathematical and architectural difference between vocabulary logit sampling and decision scoring?
* **Mechanics:**
  * Vocabulary projection `lm_head` ($d_{\text{model}} \to 152{,}064$ tokens) vs. Compact Decision Head ($d_{\text{model}} \to K$ choices).
  * Why stochastic sampling (temperature, top-p) causes hallucination and formatting drift.
  * Direct probability simplex projection via calibrated softmax.
* **Metric:** Parameter count, projection FLOPs, and zero formatting error guarantees.

---

### Milestone 3 — Calibration Theory & Proper Scoring Rules
* **Core Question:** Why are classical LLMs dangerously overconfident, and how does RLCD enforce honest confidence?
* **Mechanics:**
  * The failure of binary reward ($+1 / 0$) in RLVR / SFT.
  * Mathematical definition of **Strictly Proper Scoring Rules**:
    * **Brier Score:** $S_{\text{Brier}}(\mathbf{p}, y) = 1 - \frac{1}{2} \sum_{k=1}^K (p_k - \mathbb{I}(y = k))^2$
    * **Logarithmic Score:** $S_{\text{Log}}(\mathbf{p}, y) = \ln(p_y)$
    * **Ranked Probability Score (RPS)** for ordinal scales.
  * Expected Calibration Error (ECE) and Reliability Diagrams.
* **Metric:** Shift in ECE from $>0.20$ (uncalibrated LLM) to $<0.03$ (RLCD model).

---

### Milestone 4 — RLCD Policy Gradient & Training Dynamics
* **Core Question:** How do we train a non-autoregressive encoder with reinforcement learning without crashing gradient variance?
* **Mechanics:**
  * Policy gradient over decision simplex: $J(\theta) = \mathbb{E}[S(\pi_\theta(s), y)]$.
  * Gaussian noise perturbation exploration on logits: $\mathbf{z}_{\text{rollout}} = \mathbf{z} + \epsilon$.
  * Group Relative Policy Optimization (GRPO) normalized against group mean advantage.
* **Metric:** Convergence rate and calibration loss trajectory on uncertain decision datasets.

---

### Milestone 5 — Autonomous Agent Routing & Escalation
* **Core Question:** How does an enterprise agentic system use calibrated probabilities to eliminate 80% of LLM inference costs?
* **Mechanics:**
  * The Three-Tier Governor Pattern:
    * **Tier 1 ($\ge 95\%$):** Execute immediately and autonomously (sub-30ms path).
    * **Tier 2 ($70\% - 95\%$):** Trigger secondary verification or generate a reasoning trace via DeepSeek-R1 / Qwen.
    * **Tier 3 ($< 70\%$):** Safely escalate to human operator with full context.
* **Metric:** End-to-end agentic workflow throughput, latency SLO adherence, and GPU cost reduction.
