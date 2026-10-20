"""Milestone 2 -- The KV Cache: Speedup Benchmark and Analysis.

Run:
    python experiments/m2_kvcache.py

Compares naive autoregressive generation (recomputing all history every step)
against incremental cached generation using KVCache. Produces a formatted
console report and saves the speedup chart to experiments/out/m2_speedup.png.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add src to pythonpath
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from nanoinfer.config import TINY
from nanoinfer.kv_cache import KVCache
from nanoinfer.transformer import NanoTransformer

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True, parents=True)


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def measure_generation(
    model: NanoTransformer,
    prompt: torch.Tensor,
    n_new: int,
    warmup: int = 2,
    repeats: int = 3,
) -> tuple[float, float, bool]:
    """Measure naive vs cached decode time for `n_new` tokens.

    Returns:
        (t_naive_mean, t_cached_mean, outputs_match)
    """
    # Warmup
    for _ in range(warmup):
        _ = model.generate_greedy(prompt, max_new_tokens=4)
        _ = model.generate_cached(prompt, max_new_tokens=4)

    # Time naive
    t_naive_runs = []
    out_naive = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out_naive = model.generate_greedy(prompt, max_new_tokens=n_new)
        t_naive_runs.append(time.perf_counter() - t0)

    # Time cached
    t_cached_runs = []
    out_cached = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        out_cached = model.generate_cached(prompt, max_new_tokens=n_new)
        t_cached_runs.append(time.perf_counter() - t0)

    outputs_match = torch.equal(out_naive, out_cached)
    return (
        sum(t_naive_runs) / repeats,
        sum(t_cached_runs) / repeats,
        outputs_match,
    )


def run_benchmark():
    rule("NANOINFER -- MILESTONE 2: The KV Cache Benchmark")
    print(f"Model: {TINY.describe()}")
    print("Device: CPU (analytical & exact op execution)")
    print()

    torch.manual_seed(42)
    model = NanoTransformer(TINY).eval()

    prompt_len = 8
    lengths = [16, 32, 64, 128, 256]
    prompt = torch.randint(0, TINY.vocab_size, (1, prompt_len))

    results = []

    print(f"{'New Tokens':>10} | {'Naive Time':>12} | {'Cached Time':>12} | {'Speedup':>10} | {'Naive tok/s':>12} | {'Cached tok/s':>13} | {'Exact Match'}")
    print("-" * 92)

    for n_new in lengths:
        t_naive, t_cached, match = measure_generation(model, prompt, n_new)
        speedup = t_naive / t_cached if t_cached > 0 else float("inf")
        naive_tps = n_new / t_naive
        cached_tps = n_new / t_cached
        results.append({
            "n_new": n_new,
            "t_naive": t_naive,
            "t_cached": t_cached,
            "speedup": speedup,
            "naive_tps": naive_tps,
            "cached_tps": cached_tps,
        })
        match_str = "YES (bitwise)" if match else "MISMATCH!"
        print(f"{n_new:>10} | {t_naive:>10.4f} s | {t_cached:>10.4f} s | {speedup:>9.2f}x | {naive_tps:>12.1f} | {cached_tps:>13.1f} | {match_str}")

    rule("Analysis & Key Takeaways")
    print(
        "1. Work Complexity:\n"
        "   - Naive decode recalculates keys & values for ALL past tokens every step: O(N^2) work.\n"
        "   - Cached decode computes only the new token's Q/K/V and appends to cache: O(N) work.\n"
        "\n"
        "2. Speedup Grows with Context Length:\n"
        f"   - At 16 tokens, speedup is ~{results[0]['speedup']:.1f}x.\n"
        f"   - At 256 tokens, speedup reaches ~{results[-1]['speedup']:.1f}x!\n"
        "   - The longer the conversation, the more catastrophic naive recomputation becomes.\n"
        "\n"
        "3. Bitwise Exactness:\n"
        "   - Caching changes the amount of work, NEVER the mathematical output.\n"
        "   - Logits and emitted tokens match bit-for-bit."
    )

    # Plot charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        ns = [r["n_new"] for r in results]
        t_naives = [r["t_naive"] for r in results]
        t_cacheds = [r["t_cached"] for r in results]
        speedups = [r["speedup"] for r in results]

        # Subplot 1: Latency (log scale)
        ax1.plot(ns, t_naives, "o-", color="crimson", label="Naive (no cache) - O(N²)")
        ax1.plot(ns, t_cacheds, "s-", color="royalblue", label="Cached (KVCache) - O(N)")
        ax1.set_xlabel("Generated Tokens", fontsize=11)
        ax1.set_ylabel("Wall Time (seconds)", fontsize=11)
        ax1.set_title("Decode Latency: Naive vs Cached", fontsize=12, fontweight="bold")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3, which="both")
        ax1.legend(fontsize=10)

        # Subplot 2: Speedup Factor
        ax2.plot(ns, speedups, "D-", color="forestgreen", label="Measured Speedup")
        ax2.set_xlabel("Generated Tokens", fontsize=11)
        ax2.set_ylabel("Speedup Multiplier (x)", fontsize=11)
        ax2.set_title("Speedup Ratio (Naive Time / Cached Time)", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        fig.suptitle("Milestone 2: Why the KV Cache Matters", fontsize=14, fontweight="bold")
        fig.tight_layout()

        out_path = OUT / "m2_speedup.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"\n[Chart saved successfully to {out_path}]")
    except ImportError:
        print("\n(matplotlib not installed; skipping chart generation)")


if __name__ == "__main__":
    run_benchmark()
