# Study Plan — Inference Engineering for Interviews

**Status: Day 2 Complete (Milestone 3: PagedAttention Done). Day 3 (Milestone 4: Continuous Batching) Next.**  
👉 **Daily Progress & Checklist Tracker:** See [docs/tracker.md](tracker.md)

---

## Format

| | |
|---|---|
| Session length | 30 minutes |
| Daily structure | 3 new sessions + 1 recall session |
| Recall session | Explain the 3 sessions' concepts **in words, no code** |
| Code per session | ≤ ~40 lines |

**Passing an interview needs the explanation, not the code.** Every concept
gets written once and recalled once. That is the whole method.

---

## What you have already

| Done | Evidence |
|---|---|
| Roofline cost model | decode at B=1 is 1.0 FLOP/byte; batching 1→512 gives 51× |
| Transformer from scratch | RMSNorm, RoPE, SwiGLU, GQA attention |
| KV cache | per-layer cursors, live slicing, bit-exact |
| Cached decode | bitwise identical to naive over 56 tokens |

**Deep — you own these two files:** `src/nanoinfer/kv_cache.py`,
`src/nanoinfer/transformer.py`

**Reference only, open when stuck:** everything in `docs/`

---

## Cycle 1 — the attention memory problem (Days 1–3)

### Session 1 — Why attention is the bottleneck

The scores tensor is `[B, n_heads, S, S]` and you are materialising it. At
Llama-3-8B, S=2048, that is **1 GB in fp32** for one layer.

- Read: `attention.py` → `causal_attention`, find where `probs` exists
- Task: write ~15 lines computing `scores.nbytes` for your TINY model at
  S = 512 / 2048 / 8192, and for Llama-3-8B analytically
- **Number to be able to state:** the size of the score matrix at 8K context

### Session 2 — FlashAttention

Same math, never materialises `scores`. Tiles Q/K/V into blocks in SRAM,
accumulates in registers. 2–4× faster, O(S) instead of O(S²) memory.

- Task: ~25 lines implementing blocked attention for one head, same output
- Compare its memory to Session 1's number

### Session 3 — PagedAttention (M3)

Your cache wastes memory: `bytes_allocated()` is constant regardless of how
many tokens are live. You cannot know a sequence's length in advance, so
`max_seq_len` is allocated per request regardless.

- Task: measure your waste. Append 8 tokens to a 512-capacity cache. What
  fraction is live? Multiply by M1's concurrency number.
- Then: read `docs/roadmap.md` M3. Block tables, logical→physical mapping.

### Session 4 — Recall

Explain, in words, no code: why attention is memory-bound, what FlashAttention
trades, why a KV cache wastes memory. Include the numbers from sessions 1–3.

---

## Cycle 2 — Throughput (Days 4–6)

| Session | Topic | Task |
|---|---|---|
| 1 | Continuous batching | Simulate FCFS vs continuous. Pure Python, ~40 lines. |
| 2 | Prefill/decode vocabulary | TTFT, TPOT, throughput. Tie back to your prefill fix. |
| 3 | Quantization | RTN on your TINY model. Measure size vs perplexity delta. |
| 4 | Recall | Explain why batching helps, and its latency cost. |

---

## Cycle 3 — Shrinking the work (Days 7–9)

| Session | Topic | Task |
|---|---|---|
| 1 | Speculative decoding | Implement modified rejection sampling. Verify distribution matches. |
| 2 | Tensor / pipeline parallelism | Concepts only. Why one H100 can't hold a 70B model. |
| 3 | CUDA & profiling | Kernel launch overhead, `torch.profiler`. Where time actually goes. |
| 4 | Recall | Explain both attack directions: do less work, or read less. |

---

## Cycle 4 — Integration (Days 10–12)

| Session | Topic |
|---|---|
| 1 | Mixtral / MoE, speculative variants — breadth |
| 2 | Your repo walkthrough as an interview story — you built this, know it cold |
| 3 | Mock interview against the bank below |
| 4 | Recall + gaps list |

---

## The interview question bank

Practice explaining each **in 2 minutes, no code**. The repo answers:

1. Why is decode memory-bound? *(roofline.py, 1.0 FLOP/byte)*
2. Why does a KV cache help? *(kv_cache.py)*
3. How big is the KV cache for Llama-3-8B at 8K? *(128 KiB/token)*
4. What is GQA and why does it exist? *(4× memory)*
5. What is PagedAttention and what does it fix?
6. What is continuous batching and why does it beat static batching?
7. What is TTFT vs TPOT and why do they trade off?
8. What does FlashAttention actually avoid?
9. How does quantization reduce memory — FLOPs or bytes? *(bytes)*
10. What is speculative decoding and why does it only help decode?
11. Why can prefill and decode share one code path?
12. What's the ridge point and why does it matter?

---

## Being honest about the timeline

12 sessions per 3-day cycle, 30 min each. This is a **3–4 month** path to real
interview readiness, not 3 days. What changes the outcome:

- **The recall sessions.** Skipping them converts this back into what you
  already have — "I know it theoretically but can't use it."
- **Being able to say a number.** "The KV cache is 128 KiB per token for
  Llama-3-8B" lands. "It's memory efficient" does not.
- **One story you own.** "I built a transformer and KV cache from scratch and
  measured X" is worth more than having read ten papers.