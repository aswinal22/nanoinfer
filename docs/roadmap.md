# Roadmap

Each milestone answers one question, ends with a measurement, and has explicit
acceptance criteria. We do not start a milestone until the previous one is
committed and its numbers are understood.

---

## M1 — Where does inference time actually go? ✅

**Question:** What bounds LLM inference, and by how much?

**Build:** `roofline.py` — an analytical FLOP/byte model for prefill and decode.

**Result:**
- Decode at batch 1 has arithmetic intensity of exactly **1.0 FLOP/byte**
- Prefill becomes compute-bound above ~315 tokens (H100) / ~164 tokens (A100-80GB)
- Batching 1 → 512 gives **51x throughput** at **10x worse per-user latency**
- An 80 GB KV budget holds ~74 full-length Llama-3-8B sequences

**Done.** 32 tests green. Charts in `experiments/out/`.

---

## M2 — The KV cache, for real

**Question:** What does a KV cache actually buy us, in code?

**Build:**
- A minimal decoder-only transformer in raw PyTorch ops — RMSNorm, RoPE,
  SwiGLU, grouped-query attention. No HF dependency.
- Two decode paths: naive full-recompute, and incremental with a KV cache.
- A benchmark comparing them across context length and batch size.
- A correctness test: cached decode must match naive decode exactly.

**Key insight to hit:** the cache converts O(n²) recomputation into O(n) appends.
The speedup *grows with context length*.

**Acceptance:**
- Logits from both paths agree to `< 1e-4` in fp32
- Measured speedup curve is plotted and explained by the M1 model
- `experiments/m2_kvcache.py` produces the comparison

---

## M3 — Paged attention

**Question:** Why does a naive KV allocator waste memory, and how do we fix it?

**Build:**
- `block_table.py`: fixed-size KV blocks, logical→physical mapping, a
  copy-on-write fork path for branching (beam search / speculative prefixes).
- A fragmentation simulator: compare contiguous allocation vs paged.
- Measure: how many extra sequences fit per GB.

**Key insight to hit:** you cannot know a sequence's final length in advance,
so preallocating `max_context` wastes memory for every short request. Paging
allocates on demand and frees on completion.

**Acceptance:**
- Paged allocator fits strictly more sequences than contiguous, measured
- Copy-on-write fork is tested to produce identical logits to the source sequence

---

## M4 — Scheduling a server

**Question:** How should a serving engine order work?

**Build:**
- A discrete-event simulator: Poisson arrivals, prefill/decode mix, KV budget.
- Three policies: FCFS static batching, continuous batching, and continuous
  batching with chunked prefill + priority.
- Metrics: TTFT, TPOT, throughput, and the goodput/latency-SLO frontier.

**Key insight to hit:** static batching makes everyone wait for the slowest
sequence in the batch. Continuous batching admits new requests the instant a
slot frees, which is worth more than any kernel optimization.

**Acceptance:**
- Simulator reproduces the qualitative shape of published vLLM benchmarks
- A plot of the latency/throughput frontier across the three policies

---

## M5 — Reading fewer bytes

**Question:** If we're bandwidth-bound, what do we shrink?

**Build:**
- Round-to-nearest quantization → group-wise quantization → a from-scratch
  GPTQ with Hessian-based error compensation.
- KV cache quantization (int8 / int4) and its effect on concurrency.
- Perplexity-vs-bits measurement on a small real model.

**Key insight to hit:** quantization is a *bandwidth* optimization, not a
compute one. The win is halving bytes, not halving FLOPs. Naive RTN already
gets most of the size win; GPTQ's value is recovering the accuracy.

**Acceptance:**
- Quantized model perplexity within a small margin of fp16 on a toy corpus
- Measured speedup on CPU tracks the predicted bytes reduction

---

## M6 — Generating fewer tokens

**Question:** Can we produce more tokens per byte read?

**Build:**
- Speculative decoding: draft model proposes *k* tokens, target model verifies
  in one batched pass.
- Modified rejection sampling for exact distribution matching.
- Acceptance-rate measurement and the resulting speedup model.

**Key insight to hit:** this works *because* decode is memory-bound — verifying
k tokens in one pass costs nearly the same as one, so accepted tokens are
nearly free. On a compute-bound workload the same trick gains nothing.

**Acceptance:**
- Output distribution provably matches the target model (test with a
  chi-square or total-variation check over many samples)
- Speedup model matches the measured acceptance rate

---

## M7 — Validate on real hardware

M1-M6 are analytical or CPU-only. This is where we check the model against a
real GPU. **Free options, no purchase needed:**

- **Kaggle Notebooks** — 30 h/week of free T4/P100
- **Google Colab** — free T4
- **Lightning AI** — free GPU tier

**Build:**
- Run real vLLM with a small model (Qwen2.5-0.5B) and instrument it.
- Measure real TTFT, TPOT, and throughput-vs-batch-size.
- Overlay measured points on the M1 roofline curve. Report the gap.

**Acceptance:**
- A chart comparing M1's predicted ceiling against measured throughput
- An honest writeup of *why* the gap exists (kernel efficiency, launch
  overhead, memory bandwidth not fully achieved, framework overhead)

---

## Ordering rationale

M1 before everything: without the roofline model, M2-M6 look like a bag of
tricks. With it, each one is a targeted attack on a named bottleneck.

M2 before M3: paging is only motivated once you have a cache and understand
what it costs.

M4 before M5/M6: the scheduler determines *what* gets quantized or speculated
on, so its structure shapes those designs.

M7 last: you cannot tell whether an implementation is good until you know what
good looks like — and you cannot know that without the analytical model first.