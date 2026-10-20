# 8-Day Inference Engineering Progress Tracker

**Current Status:** Day 1 Completed (Milestones 1 & 2 + FlashAttention) | Day 2 Next (Milestone 3: PagedAttention)  
**Last Updated:** October 20, 2026

```
Overall Progress: [██░░░░░░░░] 1 / 8 Days (25% of Roadmap Milestones)
```

---

## Daily Schedule & Progress

### [x] Day 1: The Foundations (M1 Roofline + M2 KV Cache + FlashAttention)
- [x] **Compute vs. Memory Bandwidth:** The Chef vs. The Conveyor Belt mental model.
- [x] **The Ridge Point:** Arithmetic intensity threshold separating memory-bound from compute-bound ($\text{Peak FLOPS} / \text{Bandwidth}$). H100 ridge point is $\approx 295\text{ FLOP/byte}$.
- [x] **The 1.0 FLOP/Byte Bottleneck:** Decode at batch 1 does $2P\text{ FLOPs} / 2P\text{ bytes} = 1.0\text{ FLOP/byte}$, leaving the GPU $99\%$ idle.
- [x] **Prefill vs. Decode:** Prefill is compute-bound ($S\text{ FLOP/byte}$); decode is memory-bound ($1.0\text{ FLOP/byte}$).
- [x] **Batching:** Shares weight reads across $B$ users to scale decode arithmetic intensity.
- [x] **KV Cache Math:** Sizing formula $2 \times L \times n_{kv} \times d_k \times \text{bytes} = 128\text{ KiB/token}$ on Llama-3-8B ($1\text{ GiB}$ at 8K context).
- [x] **GQA (Grouped Query Attention):** Why 8 KV heads vs 32 Q heads slashes cache memory by $4\times$.
- [x] **The 3 KV Cache Gotchas:**
  - Per-layer cursors (`cur_len = [0] * n_layers`) instead of a broken single counter.
  - Live slicing (`self.k[..., :cur, :]`) avoiding reading empty capacity.
  - RoPE position offset calculated once before layer loop.
- [x] **FlashAttention:** Fast SRAM vs. slow VRAM/HBM; Online Softmax tiling trick avoiding $S \times S$ matrix materialization in VRAM.
- [x] **Code & Verification:**
  - `kv_cache.py`: line-by-line understanding of 5D bookshelf tensor layout.
  - `generate_cached()`: bitwise identical to naive autoregressive decode over 256 tokens.
  - `experiments/m2_kvcache.py`: measured $1.30\times \to 2.91\times$ speedup on CPU.
  - Chart generated: `experiments/out/m2_speedup.png`.

---

### [ ] Day 2: PagedAttention (Milestone 3)
- [ ] **The Problem:** Measuring memory waste (internal & external fragmentation) from contiguous allocation.
- [ ] **The Solution:** Building a Block Table (logical $\to$ physical mapping) with 16-token pages.
- [ ] **Copy-on-Write (Forking):** How multiple sequences share identical prefix blocks.
- [ ] **Measurement:** How many extra concurrent sequences fit per GB using paging.
- [ ] **Cycle 1 Recall:** 10-minute verbal walkthrough of Roofline, FlashAttention, and Paging.

---

### [ ] Day 3: Serving & Schedulers (Milestone 4)
- [ ] **Continuous Batching:** Iteration-level scheduling vs. static FCFS batching.
- [ ] **Key Metrics:** TTFT (Time to First Token) vs. TPOT (Time per Output Token).
- [ ] **Chunked Prefill:** Interleaving prefill chunks with decode steps to prevent latency spikes.
- [ ] **Frontier Measurement:** Plotting the throughput vs. latency-SLO trade-off curve.

---

### [ ] Day 4: Reading Fewer Bytes — Quantization (Milestone 5)
- [ ] **The Core Realization:** Quantization is a *bandwidth* optimization (saving memory transfer time), not a compute one.
- [ ] **Weight Quantization:** Round-To-Nearest (RTN) vs. GPTQ (Hessian-based compensation).
- [ ] **KV Cache Quantization:** `fp16` $\to$ `int8` $\to$ `int4` and its effect on maximum concurrency.
- [ ] **Cycle 2 Recall:** 10-minute verbal check-in on Schedulers, TTFT/TPOT, and Quantization.

---

### [ ] Day 5: Generating Fewer Tokens — Speculative Decoding (Milestone 6)
- [ ] **The Core Trick:** Using a tiny 1B draft model to propose $K$ tokens, verified in one batched pass by the 8B target model.
- [ ] **Why it works:** Decode is memory-bound; verifying $K$ tokens costs nearly the same bandwidth as verifying 1 token.
- [ ] **Rejection Sampling:** Exact mathematical distribution matching.
- [ ] **Speedup Model:** Acceptance rate vs. wall-clock speedup measurement.

---

### [ ] Day 6: Scaling Up — Parallelism & Profiling
- [ ] **Tensor Parallelism (TP):** Megatron-LM column and row parallel linear layers across GPUs.
- [ ] **Pipeline Parallelism (PP):** Layer partitioning across GPUs and the 1F1B schedule.
- [ ] **Profiling (M7):** Running `torch.profiler` to identify kernel launch overheads and real hardware roofline gaps.

---

### [ ] Day 7: Modern Frontiers (MoE & DeepSeek MLA)
- [ ] **Mixture of Experts (MoE):** Top-2 routing, active vs. total parameters (e.g. Mixtral 8x7B).
- [ ] **Multi-Head Latent Attention (MLA):** DeepSeek KV cache low-rank compression.
- [ ] **Cycle 3 Recall:** Explaining the two attack directions: doing less work vs. reading fewer bytes.

---

### [ ] Day 8: The Interview Masterclass & Capstone
- [ ] **Drilling the 12-Question Bank:** 2-minute concise verbal answers for each question without notes.
- [ ] **Your Personal Narrative Walkthrough:** Delivering the story of building nanoinfer from scratch.
- [ ] **Final Knowledge Gap Audit:** 100% confidence achieved.

---

## Key Numbers Mastered (Memory Anchors)

| Concept | The Exact Number | Context |
|---|---|---|
| **Decode Intensity at $B=1$** | $1.0\text{ FLOP/byte}$ | $2P\text{ FLOPs} / 2P\text{ bytes}$ for fp16 |
| **H100 Ridge Point** | $295\text{ FLOP/byte}$ | $\text{Peak TFLOPS} (989) / \text{Bandwidth TB/s} (3.35)$ |
| **Llama-3-8B KV Cache per token** | $128\text{ KiB/token}$ | $2 \times 32\text{ layers} \times 8\text{ heads} \times 128\text{ dim} \times 2\text{ bytes}$ |
| **Llama-3-8B Cache at 8K Context** | $1\text{ GiB}$ per user | $8192 \times 128\text{ KiB}$ |
| **GQA Cache Shrinkage** | $4\times$ reduction | 8 KV heads instead of 32 Query heads |
| **Attention FLOPs Crossover** | $\approx 27\text{K}$ tokens | Attention FLOPs only surpass weight FLOPs far past the 8K context window |
| **Measured M2 Speedup (CPU)** | $1.30\times \to 2.91\times$ | Measured across 16 to 256 tokens in `m2_speedup.png` |
