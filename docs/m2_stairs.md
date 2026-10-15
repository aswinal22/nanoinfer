# M2 Stairs — one rung at a time

Seven rungs. Each is 5–15 minutes and has its own test. Work down the list and
watch the pass count climb. **Don't look ahead** — each rung depends on the last.

All examples use `TINY`: `n_layers=2, n_heads=4, n_kv_heads=4, head_dim=32,
max_seq_len=512, batch=1`.

Run any single rung's test with:

```powershell
$PY = "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe"
& $PY -m pytest tests/test_kv_cache_steps.py -q -k rung1
```

---

## Rung 1 — Allocate the buffers

**Goal:** constructor creates storage for every layer's K and V, plus a counter.

**Before writing, know this:**

A linear layer holding `n_layers × batch × n_kv_heads × max_seq_len × head_dim`
numbers. For TINY that's `2 × 1 × 4 × 512 × 32`. Two of these — one for K, one for V.

Note the **layer dimension comes first**. That is deliberate: it means one
contiguous tensor holds every layer, so `self.k[0]`, `self.k[1]` are cheap views
rather than a list of separate tensors.

**Think about:**
```python
torch.zeros(...)   # or torch.empty(...) -- see below
```
Which axes does each argument correspond to? `dtype=` and `device=` are keyword
arguments, not positional.

**The cursor:** **one per layer** — a plain Python list of `n_layers` zeros.

**Why not one shared counter?** It looks cleaner, and it is broken. Prefilling
S=3 tokens through a 2-layer model with a shared counter:

```
layer 0: reads counter=0, writes [0:3], counter -> 3
layer 1: reads counter=3, writes [3:6], counter -> 6   <-- WRONG
```

Layer 1 writes at offset 3 instead of 0, so its keys and values no longer line
up with layer 0's, and the length ends at 6 for a 3-token prompt. Single-token
decode breaks the same way. Every layer therefore needs its own cursor.

They stay equal in practice — every layer gets the same tokens in the same
order — which is what lets `get_seq_len()` read just layer 0's and be correct.

**`zeros` vs `empty`:** `zeros` is safer and the difference is irrelevant at this
size. Use `zeros`. Do not worry about it again.

**Verify:** `rung1` — buffer shape is right and the counter starts at 0.

---

## Rung 2 — Pointer accessors

**Goal:** three trivial methods, all reading or writing that one counter.

```
is_empty()    -> is every cursor zero?
get_seq_len() -> the sequence length (read layer 0's cursor)
reset()       -> set every cursor back to zero
```

**The one decision:** does `reset()` reallocate the tensors?

It should **not**. Reallocating would defeat the entire point — the whole reason
for a preallocated cache is that step *N* reuses the memory step *N-1* used.
Just rewind the cursors. Stale values past a cursor are harmless *because
nothing is ever allowed to read past it* — which is rung 3's job.

**Verify:** `rung2` — append 3, check length is 3, reset, check it's 0.

---

## Rung 3 — `get_kv` (the important one)

**Goal:** return the live history for one layer.

```
layer_idx is an int in [0, n_layers)
return (k, v)
```

**The entire content of this method is one slice**, and the slice is the point.

`self.k` has shape `[n_layers, batch, n_kv_heads, max_seq_len, head_dim]`. You
want to hand back `[batch, n_kv_heads, cur_len, head_dim]`.

So: drop the first axis by indexing it with `layer_idx`, then **slice the
sequence axis to that layer's cursor**, leaving the other two axes alone.

Note: to this layer's cursor, not a global one. With per-layer cursors you can
index the layer axis once and then slice with the cursor you read from that
same layer.

**This is trap #1 and it is the one that bites hardest.** If you return the
whole `[..., max_seq_len, ...]` slab, attention will read 512 positions of
which only a handful are real. The rest is uninitialised memory. PyTorch will
not complain. The model will just produce quietly wrong tokens, and you will
chase the bug somewhere else for an hour.

**Verify:** `rung3` — after appending 3 tokens, the returned tensor's
sequence dimension must be **3, not 512**. The test asserts exactly this number
so you can't accidentally pass a wrong implementation.

---

## Rung 4 — `append` (the real work)

**Goal:** write new K/V after the existing history, advance the counter, return
the updated history.

**Inputs:** `k_new`, `v_new`, both `[batch, n_kv_heads, n_new, head_dim]`.
Note **`n_new` is variable** — prefill passes the whole prompt, decode passes 1.
Never hardcode 1.

**Order of operations:**

```
1. would this overflow max_seq_len?  -> raise a clear error
2. write k_new / v_new into the slot starting at THIS LAYER's cursor
3. advance THIS LAYER's cursor by n_new
4. return the live slice (rung 3 again)
```

**Step 2 is a slice *assignment*, not a function call.** You are writing into
an existing tensor. Think `x[...] = value` where `...` is your slot. The
destination slot spans `[cursor : cursor + n_new]` on the sequence axis.

**Watch for:** getting the destination and the cursor inconsistent — writing at
the right place but advancing by the wrong amount, or vice versa. Either
corrupts the cache silently.

**The per-layer cursor is what keeps layers aligned.** In prefill, layer 0
writes its K/V at `[0:S]` and layer 1 writes *its* K/V at `[0:S]` too, because
it reads its own cursor rather than layer 0's. A shared counter would push
layer 1 to `[S:2S]`.

**Verify:** `rung4` — append 3 then append 1, and check **both** that the length
is 4 *and* that the first 3 tokens are unchanged. That second assertion is the
one that catches a write-position bug.

---

## Rung 5 — `DecoderLayer.forward_cached`

**Goal:** the same computation as `forward`, but against a populated cache.

**Start by copying `forward()` and changing three things.** Nothing else. Walk
the two side by side.

| | `forward` (prefill) | `forward_cached` (your task) |
|---|---|---|
| input `x` | `[b, S, d]` | `[b, 1, d]` |
| RoPE positions | start at 0 | start at the current counter |
| K, V | computed fresh | read back from the cache |

**Everything else is identical**: RMSNorm, `project_qkv`, `causal_attention`,
`attn_output_proj`, the FFN, both residuals.

**The two things that are new:**

1. **Where do `cos`/`sin` come from?** You already receive them as arguments —
  `forward_step` computed them. Don't recompute them here.

2. **`causal_attention` already handles `q` being shorter than `k`/`v`.** Its
  signature documents that queries are assumed to be the *last* `q_len`
  positions, and it aligns the mask itself. Pass `q` (length 1) and the full
  history from `append`, and it does the right thing. Do not write your own
  masking.

**The call sequence:**

```
q, k, v = project_qkv(...)          # all length 1
k_full, v_full = cache.append(layer_idx, k, v)   # now length cur+1
out = causal_attention(q, k_full, v_full)        # q=1, k/v=cur+1
```

**Verify:** `rung5` — prefill a prompt through the normal path, then compare a
single cached step against the corresponding slice of the full forward.

---

## Rung 6 — `NanoTransformer.forward_step` (the trap)

**Goal:** drive the layers so that both prefill and decode work, through one code path.

**This rung has the bug that costs everyone an afternoon. Read it twice.**

`cos`/`sin` must be computed **once, before the layer loop**:

```python
offset = cache.get_seq_len()          # ONCE, at the top
c, s = rope_slice(rope_cos, rope_sin, offset, seq_len)
for layer in self.layers:
    x = layer.forward_cached(x, c, s, cache, layer.layer_idx)
```

**Why it must be once:** `append` advances the layer's cursor. So if you
recomputed the offset inside the loop by reading layer 0's cursor:

```
layer 0 reads offset 5, appends -> its cursor becomes 6
layer 1 reads offset 5, appends -> its cursor becomes 6     (fine so far)
```

…that happens to be harmless *only* because each layer has its own cursor. But
if you write `cache.get_seq_len()` while inside the loop **and** the cursors
ever diverge — or if you use a single shared counter — every layer places the
same token at a different position. Output is garbage, no exception is raised,
and the cause is three lines from the symptom.

Compute the offset once at the top regardless. It is clearer, and it makes the
invariant explicit. That is exactly why `forward_cached` takes `cos`/`sin` as
**parameters** — the offset is decided once, by the caller.

**Also:** `rope_slice(cos, sin, offset, length)` — `length` is how many tokens
you're feeding in, which is 1 for decode and the prompt length for prefill.
`cache.get_seq_len()` is **0** for prefill, which is why one code path handles
both.

**Finish with:** final norm, then the tied LM head — identical to the end of
`forward()`.

**Verify:** `rung6` — prefill and decode produce the same logits as the plain
`forward()`.

---

## Rung 7 — The integration test

`tests/test_model.py::test_cached_decode_matches_full_forward`

Generate 6 tokens both ways — naive and cached — and require them to be
**identical**. Not close. Identical. Caching must not change the answer, only
the work done.

If they differ, in order of likelihood:
1. offset recomputed inside the loop (rung 6)
2. `get_kv`/`append` returning full capacity instead of the live slice (rung 3)
3. RoPE offset taken *after* append instead of before (rung 5)

---

## Progress tracker

```powershell
& "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe" -m pytest tests/test_kv_cache_steps.py -q
```

Expected output as you go:

```
rung1 FAIL   rung2 FAIL   rung3 FAIL   rung4 FAIL   rung5 FAIL   rung6 FAIL
```

```
rung1 PASS   rung2 FAIL   rung3 FAIL   rung4 FAIL   rung5 FAIL   rung6 FAIL
```

```
rung1 PASS   rung2 PASS   rung3 PASS   rung4 PASS   rung5 PASS   rung6 PASS
```

Note these are deliberately **not** marked `xfail`. Unlike the acceptance tests
in `test_model.py`, a rung should be visibly red until it's done — that's the
checklist. The `test_model.py` KV tests stay `xfail` because they're the final
integration state, not your progress.

---

## One more time: the whole idea in one paragraph

To generate token N you need attention against tokens 1…N. The K and V for
tokens 1…N-1 were already computed last step, from identical inputs, by
identical weights. Recomputing them is pure waste. Cache them, and each step
drops from O(N) work to O(1). That's the whole of M2. Rungs 1–4 build the
cupboard; rungs 5–6 are the only part that is about attention at all.