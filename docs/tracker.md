# 8-Day Inference Engineering Progress Tracker

**Current Status:** Day 7 Completed (MoE & DeepSeek MLA Done) | Day 8 Next (The Interview Masterclass & Capstone)  
**Last Updated:** November 13, 2026

```
Overall Progress: [███████░░░] 7 / 8 Days (87.5% of Roadmap Milestones)
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

### [x] Day 2: PagedAttention (Milestone 3)
- [x] **The Problem:** Measured contiguous memory waste (92.3% waste rate from static `max_context` preallocation).
- [x] **The Solution:** Implemented `block_table.py` with physical block pooling and logical $\to$ physical address translation.
- [x] **Copy-on-Write (Forking):** Verified child sequence shares physical blocks with parent and retains them safely.
- [x] **Tensor Read/Write Exactness:** Verified paged storage reconstruction with bit-exact parity.
- [x] **Measurement:** PagedAttention slashes cache footprint by **12.46x**, boosting concurrency on an 80 GB GPU from **298 to 3,699 concurrent users**.
- [x] **Chart & Report:** Experiment script `experiments/m3_fragmentation.py` and chart `experiments/out/m3_fragmentation.png`.

---

### [x] Day 3: Serving & Schedulers (Milestone 4)
- [x] **Continuous Batching:** Iteration-level scheduling vs. static FCFS batching (admitting new users the millisecond someone emits `<eos>`).
- [x] **Key Metrics:** TTFT (Time to First Token, prefill-driven) vs. TPOT (Time per Output Token, decode-driven).
- [x] **Chunked Prefill:** Slicing prompts into 512-token chunks and interleaving with decode steps to eliminate the "prefill hiccup".
- [x] **The Boundary Ruler (`cu_seqlens`):** Managing ragged, variable-length batches in a flat 1D array while strictly preventing cross-user token bleed.

---

### [x] Day 4: Reading Fewer Bytes — Quantization (Milestone 5)
- [x] **The Core Realization:** Quantization is a **bandwidth** optimization (saving bytes moved over memory bus), not a compute one!
- [x] **Weight-Only (W4A16) vs. Weight+Activation (W8A8/FP8):** Why W4A16 speeds up decode ($3\text{–}4\times$) but not prefill, while W8A8 doubles both!
- [x] **The Outlier Problem:** Why naive 4-bit breaks (0.1% of activation features explode to 100x magnitude).
- [x] **Group-wise Scaling:** Slicing rows into groups (`group_size=128`) to isolate outliers.
- [x] **PTQ Algorithms:** RTN (naive) vs. GPTQ (Hessian-based compensation) vs. AWQ (Activation-aware scaling protecting the salient 1% weights).

---

### [x] Day 5: Generating Fewer Tokens — Speculative Decoding (Milestone 6)
- [x] **The Core Trick:** 1B draft model runs a fast decode loop ($K=4$ tokens); 8B target model runs **ONE single mini-prefill verification pass**.
- [x] **Why It Works (Hardware Proof):** Reading 16 GB weights takes $4.8\text{ ms}$; processing 4 tokens in parallel takes the **exact same $4.8\text{ ms}$** as 1 token due to idle compute cores!
- [x] **The Causal Mask as a Blindfold:** Ensures parallel verification gives identical mathematical results to 4 sequential steps.
- [x] **Rejection Sampling:** Proves output distribution mathematically matches the target model 100%.
- [x] **Production Model Families:** Alibaba (Qwen-72B + Qwen-0.5B), DeepSeek (MTP multi-token prediction heads), Meta (Llama-3.1-70B + Llama-3.2-1B), and Zhipu AI (GLM-4-9B + EAGLE).

---

### [x] Day 6: Scaling Up — Parallelism & Profiling
- [x] **Tensor Parallelism (TP):** Megatron-LM column and row parallel linear layers across GPUs. Only 2 All-Reduces per layer.
- [x] **Intra-node vs. Inter-node:** Why TP requires NVLink (900 GB/s direct bridge, bypassing PCIe/CPU traffic jam).
- [x] **Pipeline Parallelism (PP):** Layer partitioning across multiple server boxes (over network cables) and the pipeline bubble problem.
- [x] **Profiling (M7):** Running `torch.profiler` to identify GPU starvation (kernel launch overheads) and kernel fusion opportunities.

---

### [x] Day 7: Modern Frontiers (MoE & DeepSeek MLA)
- [x] **Mixture of Experts (MoE):** Top-2 routing, active vs. total parameters (Mixtral 8x7B: 47B total $\to$ 13B active; DeepSeek-V3: 671B total $\to$ 37B active).
- [x] **The Router & Gate:** Single linear projection layer `W_gate`, trained end-to-end via backpropagation.
- [x] **Load Balancing:** Why auxiliary loss or bias offsets are required to prevent winner-takes-all expert starvation.
- [x] **Multi-Head Latent Attention (MLA):** Compressing Key/Value cache into a tiny 512-dim latent vector (vacuum bag analogy), cutting KV cache size by $93\%$ without decompressing in memory.

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
| **PagedAttention Waste Rate** | $92.3\% \to 4.5\%$ | Drops reservation waste by using 16-token blocks |
| **Paged Concurrency Gain (80GB)** | $298 \to 3,699$ users ($12.4\times$) | Measured in `m3_fragmentation.png` |
