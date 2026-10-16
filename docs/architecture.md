# Architecture & Shape Reference

Every tensor shape in the model, verified by `tools/trace_shapes.py` (which runs
the real code, so it cannot drift).

```
& "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe" tools/trace_shapes.py
```

---

## Contents

1. [Symbol legend](#1-symbol-legend) — every letter used below
2. [The whole model](#2-the-whole-model) — top-level flow
3. [Inside one layer](#3-inside-one-layer) — the decoder block
4. [Attention internals](#4-attention-internals) — the part that confuses everyone
5. [Prefill vs decode](#5-prefill-vs-decode) — the only difference that matters
6. [GQA / repeat_kv](#6-gqa--repeat_kv)
7. [RoPE](#7-rope)
8. [KV cache layout](#8-kv-cache-layout)
9. [Worked numbers](#9-worked-numbers) — real values, not symbols

---

## 1. Symbol legend

| Symbol | Meaning | TINY | Llama-3-8B |
|---|---|---|---|
| `B` | batch size | 1 | varies |
| `S` | sequence length (prompt) | up to 512 | up to 8192 |
| `L` | number of layers | 2 | 32 |
| `d_model` | hidden width | 128 | 4096 |
| `n_heads` | **query** heads | 4 | 32 |
| `n_kv_heads` | **key/value** heads | 4 | 8 |
| `head_dim` | size of one head | 32 | 128 |
| `d_ff` | feed-forward width | 512 | 14336 |
| `V` | vocabulary size | 1024 | 128256 |
| `q_len` | queries being processed | `S` or `1` | `S` or `1` |
| `kv_len` | keys available | `S` or `cur+1` | `S` or `cur+1` |
| `cur` | tokens already cached | — | — |

**`n_heads * head_dim == d_model` always.** That identity is why merging heads
gets you back to `d_model`, and why you can change head count freely without
changing parameter count.

---

## 2. The whole model

```
  input_ids                       [B, S]
      │
      │  nn.Embedding(vocab_size, d_model)
      ▼
  x                                [B, S, d_model]
      │
      │  ┌──────────────────────────────────────────┐
      │  │  for layer in range(L):                  │  ← repeat L times
      │  │      x = decoder_layer(x)                │    x never changes size
      │  └──────────────────────────────────────────┘
      ▼
  x                                [B, S, d_model]
      │
      │  rms_norm
      ▼
  x                                [B, S, d_model]
      │
      │  linear with W_embed   (tied: reuses the embedding matrix)
      ▼
  logits                           [B, S, vocab_size]
```

Every layer takes `[B, S, d_model]` and returns `[B, S, d_model]`. **Shape never
changes between layers** — that is why you can loop.

You only care about the **last** position when generating:

```python
next_token = logits[:, -1, :].argmax(dim=-1)     # [B, vocab_size] → [B]
```

---

## 3. Inside one layer

A decoder block is **two sublayers**, each with a residual connection, pre-norm:

```
  x  [B, q_len, d_model]
      │
      ├──► RMSNorm ──► Attention ──┐
      │                             │
      └─────────────────────────────┴──► x = x + attn_out      [B, q_len, d_model]
      │
      ├──► RMSNorm ──► FFN ────────┐
      │                             │
      └─────────────────────────────┴──► x = x + ffn_out       [B, q_len, d_model]
```

### Attention sublayer

```
  h_norm = rms_norm(x, attn_norm)                  [B, q_len, d_model]
      │
      ├──► linear W_q  [n_heads*head_dim, d_model]
      │      └► [B, q_len, n_heads*head_dim]        ← still d_model wide
      │           │ view(B, q_len, n_heads, head_dim).transpose(1, 2)
      │           ▼
      │         q   [B, n_heads, q_len, head_dim]
      │
      ├──► linear W_k  [n_kv_heads*head_dim, d_model]
      │      └► view + transpose
      │           ▼
      │         k   [B, n_kv_heads, q_len, head_dim]
      │
      └──► linear W_v  [n_kv_heads*head_dim, d_model]
             └► view + transpose
                  ▼
                v   [B, n_kv_heads, q_len, head_dim]

      apply_rope to q and k ONLY  (never to v)
      repeat_kv to expand k, v from n_kv_heads → n_heads

      scores = q @ k^T * (1/sqrt(head_dim))         [B, n_heads, q_len, kv_len]
      probs  = softmax(scores, dim=-1)              [B, n_heads, q_len, kv_len]
      attn   = probs @ v                            [B, n_heads, q_len, head_dim]
                  │
                  │ transpose(1,2).reshape(B, q_len, n_heads*head_dim)
                  ▼
                merged                                [B, q_len, d_model]
                  │
                  │ linear W_o  [d_model, n_heads*head_dim]
                  ▼
                proj                                [B, q_len, d_model]

  x = x + proj
```

### FFN sublayer (SwiGLU)

```
  h2 = rms_norm(x, ffn_norm)                        [B, q_len, d_model]
      │
      ├──► linear W_gate  [d_ff, d_model] ──┐
      │      gate = ...                     │ elementwise
      │                                      ├──► swiglu = silu(gate) * up
      └──► linear W_up    [d_ff, d_model] ──┘       [B, q_len, d_ff]
                                                        │
                                                        │ linear W_down [d_model, d_ff]
                                                        ▼
                                                      [B, q_len, d_model]

  x = x + down
```

**Note the d_model → d_ff → d_model detour.** The FFN expands to a wide space,
does its work, and compresses back. It holds ~81% of the parameters.

---

## 4. Attention internals

This is the part people get stuck on. Two ideas.

### Idea A — `[B, S, d_model]` → `[B, n_heads, S, head_dim]`

A 4-D tensor whose axes mean:

```
   B              batch
   n_heads        which head        ← this is the new axis
   S (or q_len)   which token
   head_dim       which feature
```

Why bother? So you can do **one batched matmul** across all heads instead of a
Python loop over heads. The loop would be 32 iterations of small matmuls; the
batched form is one big one.

The code is two operations:

```python
q = q.view(B, q_len, n_heads, head_dim).transpose(1, 2)
#            └ split d_model into (n_heads, head_dim)      └ swap axes 1 and 2
```

`view` splits one axis into two. `transpose` reorders them. Going back:

```python
merged = attn.transpose(1, 2).reshape(B, q_len, n_heads * head_dim)
#           └ swap back                        └ rejoin head_dim and n_heads
```

`n_heads * head_dim == d_model`, so you land back at the original width.

### Idea B — the score matrix

```
                  K^T    [kv_len, head_dim]
        Q             ╲
  [q_len, head_dim]    ╲
              ╲         ▼
               ╲    scores
                ╲  [q_len, kv_len]
                 ╲
                  ▼

  One number per (query, key) pair = "how much should
  token q attend to token k".
```

For prefill (`q_len = kv_len = S`) this is `S × S`, and causal masking removes
the upper triangle:

```
        k →
      ┌───────────────
    q │ ■ ■ ■ □ □ □      ■ = attend (past + self)
    ↓ │ ■ ■ ■ ■ □ □      □ = masked out (-inf)
      │ ■ ■ ■ ■ ■ □
      │ ■ ■ ■ ■ ■ ■
      └───────────────
```

For decode (`q_len = 1`, `kv_len = cur+1`) it is `1 × (cur+1)`, and the single
row is entirely visible:

```
      ┌─────────────────────
    q │ ■ ■ ■ ■ ■ ■ ■ ■ ■
    ↓ └─────────────────────
```

**`causal_attention` computes both automatically.** It assumes queries are the
*last* `q_len` positions of the `kv_len` positions:

```python
q_pos = arange(kv_len - q_len, kv_len)
k_pos = arange(kv_len)
mask  = k_pos[None, :] <= q_pos[:, None]
```

- prefill: `q_pos = [0..S-1]`, `k_pos = [0..S-1]` → triangular
- decode: `q_pos = [cur]`, `k_pos = [0..cur]` → all visible

You never write a mask in your own code.

---

## 5. Prefill vs decode

**Everything is identical except `q_len`, `kv_len`, and where RoPE starts.**

| | Prefill | Decode |
|---|---|---|
| `input_ids` | `[B, S]` | `[B, 1]` |
| RoPE positions | `0 .. S-1` | `cur` |
| `x` | `[B, S, d_model]` | `[B, 1, d_model]` |
| `q` | `[B, n_heads, S, head_dim]` | `[B, n_heads, 1, head_dim]` |
| `k`,`v` (fresh) | `[B, n_kv_heads, S, head_dim]` | `[B, n_kv_heads, 1, head_dim]` |
| `k`,`v` (into attention) | same | `[B, n_kv_heads, cur+1, head_dim]` ← from cache |
| `scores` | `[B, n_heads, S, S]` | `[B, n_heads, 1, cur+1]` |
| `logits` | `[B, S, vocab]` | `[B, 1, vocab]` |
| cache after | length `S` | length `cur+1` |

This is why one code path serves both:

```python
offset = cache.get_seq_len()    # 0 during prefill, cur during decode
c, s   = rope_slice(cos, sin, offset, q_len)
```

Same three lines. The cache being empty or populated is the *only* thing that
differs, and the length arithmetic handles it.

---

## 6. GQA / repeat_kv

K and V are computed with `n_kv_heads`, but attention needs `n_heads`. Bridge
the gap:

```python
k  [B, n_kv_heads, kv_len, head_dim]     # e.g. [1, 8, 2048, 128]
      │ expand each head n_rep times
      ▼
k' [B, n_heads,   kv_len, head_dim]     # e.g. [1, 32, 2048, 128]
```

Query head `j` pairs with KV head `j // n_rep`. With `n_rep = 4`:

```
  kv head:        0    1    2    3    4    5    6    7
                  └────┴────┴────┘    └────┴────┴────┘
  query head:  0  1  2  3    4  5  6  7    8  9 10 11  ...  28 29 30 31
```

**It is `repeat_interleave` semantics, not `repeat`.** Plain `repeat` would give
each KV head a different set of query heads and the model would still train.

`expand` + `reshape` means this is a **view**, not a copy. Real implementations
(vLLM, FlashAttention) skip it entirely and reshape the query instead.

---

## 7. RoPE

RoPE rotates Q and K vectors by an angle that depends on position:

```
  angles[i] = position / base^(2i/head_dim)      base = 10000
  cos, sin  = cos(angles), sin(angles)           cached table [max_pos, head_dim]

  out = x * cos + rotate_half(x) * sin

  rotate_half:  (a, b)  ->  (-b, a)     on the two halves of head_dim
```

Two properties that matter:

- **Position 0 is a no-op** (cos=1, sin=0).
- **Scores depend only on relative distance.** `rot(m)·q` dotted with `rot(n)·k`
  depends on `m-n`, not on `m` and `n` separately. That is why it extrapolates.

During decode you must slice the table at `cur`, not at 0:

```python
c, s = rope_slice(rope_cos, rope_sin, offset=cur, length=1)
```

V is **not** rotated — rotating the values would rotate the output and cancel
the relative-position signal.

---

## 8. KV cache layout

One tensor per tensor type, **layer axis first**:

```
  self.k : [n_layers, batch, n_kv_heads, max_seq_len, head_dim]
  self.v : same

  TINY        →  [2, 1, 4, 512, 32]
  Llama-3-8B  →  [32, B, 8, 8192, 128]
```

The live region for layer `L` is a slice on **axis 2**:

```
  self.k[L, :, :, :cur_len[L], :]     →  [batch, n_kv_heads, cur, head_dim]
             └┬┘
              └── index this axis with the layer number, dropping it
```

Axis map:

```
  axis:      0        1       2            3              4
          n_layers  batch  n_kv_heads   max_seq_len    head_dim
          drop it   keep   SLICE THIS    (capacity)     keep
```

Two things to get right:

1. **Slice axis 2 to the cursor.** Returning the full `max_seq_len` axis hands
   attention uninitialised memory. No error — just wrong numbers.
2. **Store `n_kv_heads`, never `n_heads`.** For Llama-3-8B that is 8 vs 32 —
   a 4× waste that defeats the point of GQA.

---

## 9. Worked numbers

### TINY (what we actually run)

```
  B=1  d_model=128  n_heads=4  n_kv_heads=4  head_dim=32  d_ff=512  vocab=1024  L=2

  input_ids                      [1, 512]
  embedding                      [1, 512, 128]
  q                              [1, 4, 512, 32]
  k, v                           [1, 4, 512, 32]
  scores                         [1, 4, 512, 512]     ← the big one
  attn                           [1, 4, 512, 32]
  merged                         [1, 512, 128]        4*32 = 128 ✓
  gate, up                       [1, 512, 512]
  logits                         [1, 512, 1024]

  cache                          [2, 1, 4, 512, 32]
```

### Llama-3-8B (for scale — you cannot run this, but you can price it)

```
  B=1  d_model=4096  n_heads=32  n_kv_heads=8  head_dim=128  d_ff=14336  L=32

  q                              [1, 32, 2048, 128]
  k, v                           [1,  8, 2048, 128]     8 heads, not 32
  scores                         [1, 32, 2048, 2048]    ← 1 GB in fp32
  merged                         [1, 2048, 4096]        32*128 = 4096 ✓
  gate, up                       [1, 2048, 14336]
  logits                         [1, 2048, 128256]

  cache                          [32, 1, 8, 8192, 128]  = 512 MiB for one
                                                     full-length sequence
```

Two things to notice at scale:

- **`scores` is `S × S`.** At 2048 that is 512 MB per layer in fp32. This
  quadratic blow-up is why FlashAttention exists — it never materialises `probs`.
- **The cache stores 8 heads, not 32.** That single choice is 4× less memory.

---

## Quick self-check

- [ ] Why is `scores` `[B, n_heads, q_len, kv_len]` and not `[B, S, S]`?
- [ ] In decode, what is `kv_len`? Where does that number come from?
- [ ] `n_heads * head_dim == d_model` — what breaks if you set `head_dim` freely?
- [ ] Why does the cache index `n_kv_heads` but the cache *store* `n_kv_heads`?
- [ ] If you forget to slice axis 2, what happens — and why is it silent?

---

## Related

- `docs/m2_stairs.md` — the build task
- `docs/m1_walkthrough.md` — the cost model behind all this
- `tools/trace_shapes.py` — regenerates every shape above