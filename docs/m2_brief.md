# M2 Brief — The KV Cache

**Your task. Read this first, then implement `src/nanoinfer/kv_cache.py`.**

---

## The problem

`NanoTransformer.generate_greedy` is written the obvious way, and it is
absurdly wasteful:

```python
for _ in range(max_new_tokens):
    logits = self(out)[:, -1, :]     # re-runs the ENTIRE prefix
    out = torch.cat([out, nxt], dim=1)
```

To produce token N, every layer re-computes keys and values for all N-1
previous tokens. Those values are *identical* to what they were one step ago —
the weights did not change, the inputs did not change. We recompute
deterministic results and throw them away.

Work for one step is O(N). Work for a full generation is **O(N²)**.

For a 2000-token generation that is ~2M token-forwards to produce 2000 tokens.
Roughly 1000× more work than necessary.

## The insight

Split what a step computes into two parts:

| Quantity | Depends on | Changes between steps? |
|---|---|---|
| K, V for earlier tokens | earlier inputs | **No** |
| K, V for the new token | the new token | Yes |
| Q for the new token | the new token | Yes |

The first row is the entire motivation. Cache it, and each step becomes O(1)
in the sequence length. Total generation becomes **O(N)**.

---

## What to implement

### 1. `src/nanoinfer/kv_cache.py` — the store

Fill in the `TODO(you)` methods. The interface contract and the five gotchas are
documented in that file's module docstring. Summary:

```
reset()                       -> None
is_empty()                    -> bool
get_seq_len()                 -> int
get_kv(layer_idx)             -> (k, v)
append(layer_idx, k_new, v_new) -> (k_full, v_full)
bytes_allocated()             -> int
```

Preallocate `[n_layers, batch, n_kv_heads, max_seq_len, head_dim]` for K and V,
with **one write pointer shared across layers**.

### 2. `DecoderLayer.forward_cached` — the attention side

```python
def forward_cached(self, x, cos, sin, cache, layer_idx):
```

`x` is `[batch, 1, d_model]`. Steps:

1. Normalise, project to Q/K/V via `project_qkv`
2. `cache.append(layer_idx, k, v)` → full history
3. `causal_attention(q, k_full, v_full)` — note Q has length 1
4. Merge heads, apply `w_o`
5. FFN with residual

`causal_attention` already handles `q_len != kv_len`. You do not need to touch
it.

### 3. `NanoTransformer.forward_step` — the orchestration

```python
def forward_step(self, input_ids, cache) -> logits
```

This must work for **both** cases through one code path:
- prefill: `input_ids` is `[batch, N, d]`, cache empty
- decode: `input_ids` is `[batch, 1, d]`, cache populated

Getting one path to handle both is where offset bugs hide. A separate prefill
fast path will feel faster and will be wrong.

RoPE offset: for prefill use positions `[0, N)`; for decode use position
`cache.get_seq_len()` before appending. `ops.rope_slice` exists for this.

---

## Verifying

```powershell
$PY = "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe"
& $PY -m pytest tests/test_model.py -q -k cache
```

The seven `xfail` tests become your checklist:

| Test | What it protects |
|---|---|
| `test_cached_decode_matches_full_forward` | **The one that matters.** Cached output must be *identical* to naive |
| `test_kv_cache_starts_empty` | Fresh cache is empty |
| `test_kv_cache_append_grows_length` | Length advances correctly |
| `test_kv_cache_reset` | Reusable |
| `test_forward_cached_is_wired_up` | `forward_step` no longer raises |
| `test_cache_stores_kv_heads_not_query_heads` | GQA memory bug |
| `test_cache_capacity_accounting` | Capacity math |

They are `strict=False`, so a pass shows as `XPASS` and a failure shows as
`xfail`. **When `test_cached_decode_matches_full_forward` XPASSes, M2's
correctness goal is met.**

That test compares cached generation against `m(naive)` — the complete prefill
path, which already works. You are not writing an oracle; you are matching an
existing correct implementation.

---

## The five traps

1. **Returning capacity instead of contents.** Slicing to `cur_len` is
   mandatory. Passing the full `[..., :max_seq_len, ...]` buffer lets the model
   read uninitialised memory. No error, just wrong numbers.
2. **RoPE offset.** The new token is at absolute position `cur_len`, not 0.
3. **Cache the right head count.** `n_kv_heads`, not `n_heads`. For `TOY`
   (4:1 GQA) that is a 4× memory difference.
4. **Prefill and decode share a path.** See above.
5. **Batch sharing.** With `batch > 1`, the pointer advances once per layer for
   the *whole batch*. Sequences in a batch must stay in lockstep.

---

## After correctness: the measurement

Correctness first — get `test_cached_decode_matches_full_forward` green before
benchmarking anything.

Then write `experiments/m2_kvcache.py` answering:

1. **How much faster is cached decode, as a function of context length?**
   Predict the shape before measuring. The speedup should grow roughly linearly
   with sequence length, because naive cost per step is O(N) and cached is O(1).
2. **How does the ratio change with batch size?** Hints: at large batch, both
   paths become bandwidth-bound, so the gap narrows.
3. **Does M1 predict it?**

M1 says a decode step reads ~2 bytes per parameter. The cache does not reduce
bytes read — it reduces *work*. So:

- On a real GPU the win is smaller than on CPU, because a GPU is already
  bandwidth-bound and you were not wasting bandwidth in the first place.
- On CPU the naive path also re-does the softmax, the RoPE, the residuals — all
  of which the cache skips.

State this explicitly in your writeup. A measurement without an explanation of
which regime produced it is half a result.

4. **Chart it.** Time vs. sequence length, cached and naive, on a log axis.

---

## Why this matters for everything after

The KV cache is not an optimization. It is the substrate:

- **M3** pages it, because preallocating `max_seq_len` wastes memory for every
  short request. Your `test_cache_capacity_accounting` already measures that gap.
- **M5** quantizes it, because it is the fastest-growing memory in the system.
- **M6** depends on it — speculative decoding verifies *k* draft tokens in one
  batched pass, which is only a win because the pass costs nearly the same as
  one token.

Every one of those is a refinement of what you build here.

---

## Definition of done

- [ ] All seven xfail tests XPASS
- [ ] `experiments/m2_kvcache.py` produces the speedup chart
- [ ] README M2 section written with the measured numbers and the M1 comparison
- [ ] Committed and pushed