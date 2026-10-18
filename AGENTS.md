# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this project is

**nanoinfer** — a from-scratch LLM inference engine built one milestone at a
time, on CPU. The goal is *intuition*, not a vLLM reimplementation. Every
milestone answers exactly one engineering question and ends with a measurement
you can check.

Read `README.md` and `docs/roadmap.md` before making changes. The roadmap's
milestones (M1 roofline ✅, M2 KV cache next) define what code belongs here
right now. Do not get ahead of the roadmap.

## Environment (this machine, Windows)

Your `python` on PATH may lack PyTorch. First run:

```powershell
python -c "import torch; print(torch.__version__)"
```

If it fails, use the full interpreter path:

```powershell
$PY = "C:\Users\HP\AppData\Local\Programs\Python\Python312\python.exe"
& $PY -m pip install pytest matplotlib
```

No install step is needed for the package itself — tests and experiment
scripts add `src/` to the path themselves (and `pyproject.toml` sets
`pythonpath = ["src"]` for pytest).

## Commands

```powershell
& $PY -m pytest tests/ -q       # run the test suite
& $PY experiments/<name>.py     # run an experiment report + charts
```

Run the tests after every change. Never leave the suite red.

## Layout

```
nanoinfer/
  src/nanoinfer/    Library code (config, hardware, roofline, ops, ...)
  experiments/      Per-milestone report scripts; charts land in out/
  tests/            Pytest, grouped per milestone
  docs/             roadmap.md, milestone briefs and walkthroughs
```

Keep new code inside that structure. Experiments produce numbers and charts;
the library provides models the experiments analyze.

## Ground rules (enforced)

1. **Measure before optimizing.** No change lands without a before/after
   number. Analytical results count as measurements here.
2. **Test the math before trusting it.** When a test fails, check whether the
   test's assumptions are wrong before blaming the code — that has happened.
3. **CPU-only is a feature.** If an idea only makes sense on a GPU, it
   probably isn't understood well enough yet.
4. **Pin facts with tests.** When a measurement overturns an intuition, write
   a test so the misconception cannot creep back in (see the attention FLOPs
   crossover in the README).

## Style

- Small, plain PyTorch/NumPy code. No heavyweight dependencies beyond
  `torch`, `numpy`, `pytest`, `matplotlib`. No `transformers`/HF dependency.
- No install step. Do not add one unless the milestone requires it.
- Every milestone brief lives in `docs/`; document each milestone's result
  in `README.md` the way M1 does.
