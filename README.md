# nanoinfer

A LLM inference engine built from scratch, one milestone at a time.

The goal is not to reimplement vLLM. The goal is to build the **intuition that
lets you read vLLM, diagnose a slow deployment, and know which optimization to
reach for and why**. Every milestone answers exactly one engineering question and
produces a measurement you can check.

Built on CPU. No GPU required for M1-M6.

---

## The one idea everything follows from

Autoregressive decoding at batch size 1 performs **~1 FLOP per byte moved**.

```
FLOPs = 2 * P          bytes = 2 * P          intensity = 1.0 FLOP/byte
```

A GPU can do ~300 FLOP/byte (H100). So batch-1 decode uses roughly
**0.1% of the available compute**. The tensor cores sit idle waiting on memory.

Everything that matters in LLM inference is a consequence:

| Observation | Consequence |
|---|---|
| Decode is memory-bound | Batch more sequences together — weights are read once for the whole batch |
| Batching trades latency for throughput | A *scheduler* is needed to make that trade per-request, not per-engine |
| Prefill is compute-bound | Don't mix huge prefills into decode steps; chunk them |
| KV cache grows without bound | Cache memory caps concurrency; page it and quantize it |
| Wall time is dominated by bytes | Compression and speculative decoding both attack this |

---

## Setup

Your `python` on PATH may point at an interpreter without PyTorch. Check first:

```powershell
python -c "import torch; print(torch.__version__)"
```

If that fails, use the full interpreter path:

```powershell
$PY = "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe"
& $PY -m pip install pytest matplotlib
```

Then run:

```powershell
& $PY -m pytest tests/ -q          # 32 tests
& $PY experiments/m1_roofline.py   # full report + charts
```

No install step needed — scripts add `src/` to the path themselves.

---

## Milestones

See [docs/roadmap.md](docs/roadmap.md) for the full plan with acceptance criteria.

| # | Question | Status |
|---|---|---|
| M1 | What are we actually bound by? | **Done** |
| M2 | Why is a KV cache worth having? | Next |
| M3 | Why does the KV cache waste memory, and how do we page it? | |
| M4 | How should a server schedule requests? | |
| M5 | How do we shrink the bytes we must read? | |
| M6 | How do we generate fewer tokens? | |
| M7 | Validate everything on a real GPU | |

---

## Milestone 1 — Where does inference time go?

Pure analysis, no execution. Answers:

- **Prefill is compute-bound, decode is memory-bound** — and by how much
- **Batch size is the only real lever on decode.** Llama-3-8B on one H100:
  total throughput rises **51x** from batch 1 → 512, while per-user latency
  gets **10x worse**. That trade is the entire reason serving engines exist.
- **Prefill goes compute-bound above a prompt-length threshold** — ~315 tokens
  on H100, ~164 on A100-80GB. This is *why chunked prefill exists*.
- **KV cache caps concurrency** — an 80 GB budget holds only ~74 full-length
  Llama-3-8B sequences, against 15 GB of weights. The cache, not the model,
  is the scarce resource.

### Results

![Roofline](experiments/out/m1_roofline.png)

![Batch scaling](experiments/out/m1_batch_scaling.png)

### A correction worth recording

The intuitive claim "attention dominates long-context inference" is **wrong**,
and the model says so. Attention FLOPs overtake weight FLOPs at

```
S_crossover = params / (2 * d_model * n_layers)
```

which is **~27K tokens for Llama-3-8B** — outside its 8K window. Inside a
realistic context, the weights dominate the arithmetic.

What *is* true: KV **bytes** overtake weight bytes at long context, because
weight traffic is constant per token while cache traffic is linear in context.
That is a memory problem, not a FLOP problem — which is exactly why paged and
quantized KV caches help, and why a compute-focused optimization would not.

Both facts are pinned by tests so the misconception cannot creep back in.

---

## Layout

```
nanoinfer/
  src/nanoinfer/
    config.py        Model architecture descriptions
    hardware.py      Accelerator specs and ridge points
    roofline.py      FLOPs, bytes, intensity, ceilings
  experiments/
    m1_roofline.py   The M1 report
  tests/
    test_roofline.py 32 tests
  docs/
    roadmap.md
```

---

## Ground rules

1. **Measure before optimizing.** No change lands without a before/after number.
2. **Test the math before trusting it.** Two of the three M1 test failures were
   wrong assumptions in the tests, not bugs in the code. That is the process
   working.
3. **A model you can't run is still a model you can price.** Most of this
   repo's reasoning is analytical.
4. **CPU-only is a feature.** If an idea only makes sense on a GPU, it probably
   isn't understood well enough yet.