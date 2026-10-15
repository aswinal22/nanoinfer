# M1 Walkthrough — a guided review

Read this alongside the code. It explains *why* each piece is shaped the way it
is, so you can evaluate the design rather than just follow it.

Suggested order:
1. `src/nanoinfer/config.py`
2. `src/nanoinfer/hardware.py`
3. `src/nanoinfer/roofline.py`
4. `tests/test_roofline.py`
5. `experiments/m1_roofline.py`

---

## 0. The structural decision

M1 contains **no model and no tensors**. It only does arithmetic on paper.

That looks odd until you price what the alternative would cost. To measure
Llama-3-8B you would need 16 GB of weights and an H100. By modelling it
analytically, the same file describes a 70B model, a T4, and a batch of 512 —
and runs in under a second on your laptop.

The trade-off: this code tells you the **ceiling**, never the actual. Real
kernels hit 30-60% of roofline, and that gap is M7's problem.

---

## 1. `config.py` — the architecture table

### The core move: parameters as arithmetic

Every property here is a closed-form formula over
`(d_model, n_layers, n_heads, n_kv_heads, d_ff, vocab_size)`. No model object,
no weights. The formulas:

```
attention per layer = d_model·n_heads·head_dim      (q_proj)
                    + 2·d_model·n_kv_heads·head_dim (k_proj, v_proj)
                    + d_model·n_heads·head_dim      (o_proj)

ffn per layer       = 3 · d_model · d_ff            (swiglu: gate, up, down)
                    = 2 · d_model · d_ff            (mlp: c_fc, c_proj)
```

### Three decisions worth interrogating

**(a) Why separate `params_blocks` from `n_params`?**

```python
params_blocks     # matmuls that run for every token
params_lm_head    # only the final position needs this
```

The LM head is `vocab × d_model` — for Llama-3-8B that's 525M params, 7% of the
model, and during decode you compute it for **one** token out of the whole
sequence. Including it in the per-token cost overstates decode work.

`n_params_matmul` is the denominator used for bandwidth. It deliberately
excludes the embedding table because embeddings are **gathers**, not matmuls.

**(b) Why `ffn_type` at all?**

LLaMA/Qwen/Mistral use SwiGLU (3 matrices); GPT-2 uses a classic 2-matrix MLP.
Assuming one and being wrong gives a 50% error on FFN parameter count, which is
most of the block. The test pins both.

**(c) Why `__post_init__` validation?**

`d_model % n_heads` and `n_heads % n_kv_heads` must hold. Silently violating
them produces a shape error deep inside attention, far from the cause.

### Try it yourself

```powershell
$PY = "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe"
& $PY -c "import sys; sys.path.insert(0,'src'); from nanoinfer.config import LLAMA3_8B as c; print(c.describe())"
```

Change `n_kv_heads` from 8 to 32 and re-run `max_batch_at_full_context`. The
concurrency number should drop by 4×. That is GQA, isolated.

---

## 2. `hardware.py` — peaks and ridge points

### The ridge point

```python
ridge_point = peak_flops / bandwidth
```

FLOP/byte at which the compute ceiling and the bandwidth ceiling coincide.
Below it, memory is the limit; above it, compute is.

The important observation from the output: **every GPU has a ridge between 150
and 400 FLOP/byte**, while batch-1 decode operates at 1.0. That 150-400× gap is
the entire optimization opportunity.

### What these numbers are not

Vendor datasheet peaks, sparsity disabled. Real serving reaches a fraction of
them. The docstring says so explicitly — treat them as ceilings, never
forecasts. What *is* trustworthy is the **ratio between phases**, because it
comes from arithmetic no optimization can change.

---

## 3. `roofline.py` — the heart

### The one-line derivation

A dense `[n,k] × [k,m]` matmul costs `2·n·k·m` FLOPs. Summed over a whole
block, that collapses to:

```
FLOPs = 2 · params · n_tokens
```

So decode at batch 1:

```
FLOPs = 2 · P · 1 = 2P
bytes = P · 2     = 2P      (fp16)
      -----------------
intensity = 1.0 FLOP/byte
```

Exactly one, regardless of model size. A 7B and a 70B model have the same
problem. The test `test_decode_at_batch_one_is_about_one_flop_per_byte` pins it.

### Why batching is the only lever

Look at `decode_workload`:

```python
weight_bytes = cfg.n_params_matmul * dtype_bytes    # constant, whatever batch
kv_bytes     = cfg.kv_bytes(ctx_len, batch, ...)    # grows
flops        = 2 * params * batch                   # grows
```

Weights are read **once for the entire batch**. FLOPs scale with batch; weight
bytes do not. So intensity climbs toward `2 × dtype_bytes` = 4 FLOP/byte for
fp16 — and even at batch 512 you reach only 54, still 5× below the H100 ridge.

**Batch size raises intensity, but you need an enormous batch to reach the
compute roof.** That is why vLLM's continuous batching matters and why nobody
runs batch 4096.

### Attention FLOPs — and the subtlety that bit us

```python
per_layer = 4.0 * n_queries * context_len * cfg.d_model
```

The `4` is `2` for QK^T plus `2` for AV. Summing over heads collapses the head
dimension into `d_model`.

Note the shape: **linear in `context_len`, quadratic in `n_queries`.** So:

- **Decode** (1 query, N keys) is O(N) — linear
- **Prefill** (S queries, S keys) is O(S²) — quadratic

I originally wrote a test asserting attention FLOPs are quadratic in context.
It failed. That test encoded a real misconception and now there are two tests
pinning the correct behaviour.

### The `attn-only` workload

`attention_only_workload` prices QK^T+AV with **no weights**. Its intensity at
batch 32 is only 4.0 FLOP/byte, versus 21.8 for the full decode step. Attention
is the most bandwidth-hungry part of the model, which is exactly why Flash,
paging, and KV quantization target it.

### `roofline()`

```python
achieved = min(peak_tflops, intensity * bandwidth)
```

Straightforward once you accept that hardware cannot exceed either limit.
`tests/test_roofline.py::test_ridge_intensity_is_exactly_the_boundary` checks the
switch happens precisely at the ridge.

---

## 4. `tests/test_roofline.py` — read this one carefully

The tests are not paperwork. **Three of them failed during M1, and all three
were wrong assumptions in the tests, not bugs in the model.** The implementation
was right; I had assumed wrong. That is the process working.

### The tests that protect the important claims

| Test | Protects |
|---|---|
| `test_param_counts_match_published_values` | The architecture table against Llama-3-8B=8.03B, 70B=70.6B, Mistral=7.24B, GPT-2-Small=124M |
| `test_decode_at_batch_one_is_about_one_flop_per_byte` | The headline |
| `test_attention_flops_linear_in_context_when_decoding` | Linear, not quadratic, for 1 query |
| `test_attention_flops_quadratic_when_prefilling` | Quadratic when query count grows |
| `test_attention_is_a_minority_of_flops_inside_the_context_window` | **The corrected misconception** |
| `test_kv_bytes_scale_with_context_while_weight_bytes_do_not` | The memory problem is real, just not a FLOP problem |
| `test_flop_split_adds_up_to_the_decode_total` | No double-counting |
| `test_gqa_shrinks_kv_cache_by_the_group_ratio` | GQA does what it claims |

### Parametrized tests

```python
@pytest.mark.parametrize("cfg,expected", [
    (cfgmod.LLAMA3_8B, 8.03e9), ...
], ids=lambda v: getattr(v, "name", str(v)))
```

`ids=` makes failures readable (`test_param_counts[Llama-3-8B]`) instead of
`test_param_counts[0]`. Small thing, saves real time when something breaks.

---

## 5. `experiments/m1_roofline.py`

Structured as six numbered parts, each making one point. Read it for the
narrative — the module docstrings cover the mechanics.

`charts()` uses `matplotlib.use("Agg")` so it works headless, and wraps the
import in try/except so a missing matplotlib never breaks the numeric report.

---

## 6. Things you should be able to answer now

Check yourself. If any is shaky, that's what to ask about.

1. Why is decode at batch 1 exactly 1.0 FLOP/byte? Where does the `2` in
   `dtype_bytes` come from?
2. If intensity at batch 512 is 54 and the H100 ridge is 295, would batch 2048
   reach compute-bound? Work it out before answering.
3. Why does `n_params_matmul` exclude the embedding table?
4. Attention FLOPs: why linear in `context_len` for decode but quadratic for
   prefill?
5. An 80 GB budget holds 74 Llama-3-8B sequences. What three techniques would
   raise that, and by roughly how much each?
6. Why does a 200-token prompt run mostly idle on an A100-80GB?

### The answers, briefly

2. No. Intensity asymptotes at `2 × dtype_bytes` = 4 for fp16 once KV bytes
   stop dominating; you observed 54 at batch 512 because KV traffic still
   mattered. It saturates well under 295. This is why **no one solves decode by
   batching alone** — it needs quantization or speculation.
5. GQA (4×, already applied), paged allocation (~2-4× at realistic load),
   KV quantization (2× at int8, 4× at int4).

---

## 7. Where to poke at it

Things to try that will change how you think:

- Set `n_kv_heads=1` (MQA) on Llama-3-8B and recompute concurrency. Explains why
  MQA exists even though GQA is the modern compromise.
- Add a `Hardware` entry for your own CPU and find its ridge. Then explain why
  M2-M6 measurements on CPU will not match GPU behaviour.
- Change `prefill_workload` to charge only the causal triangle (halve the
  attention FLOPs) and re-derive the compute-bound threshold. Why is the naive
  implementation wrong to do this, and why do optimized kernels still not fix it?
- Break `dtype_bytes` to 1 (int8) and watch decode intensity double.

---

Once you're comfortable, M2 is where you write code — see
`docs/m2_brief.md`. The task is marked clearly in `src/nanoinfer/kv_cache.py`.