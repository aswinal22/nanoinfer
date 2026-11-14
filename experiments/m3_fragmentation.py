"""Milestone 3 -- Memory Fragmentation Simulator: Contiguous vs. Paged KV Cache.

Run:
    python experiments/m3_fragmentation.py

Compares memory waste and maximum concurrency of:
1. Contiguous allocation (preallocating max_seq_len upfront).
2. Paged allocation (allocating fixed 16-token blocks on demand).

Outputs a formatted report and saves the chart to experiments/out/m3_fragmentation.png.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from nanoinfer.config import LLAMA3_8B, TINY

OUT = Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True, parents=True)


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def simulate_workload(
    n_requests: int = 100,
    max_context: int = 2048,
    mean_length: int = 250,
    seed: int = 42,
) -> np.ndarray:
    """Generate a realistic distribution of sequence lengths (log-normal).

    Most requests are short-to-medium (50-300 tokens), with a long tail
    stretching up to max_context.
    """
    np.random.seed(seed)
    # Log-normal distribution centered around mean_length
    mu = np.log(mean_length) - 0.5
    sigma = 0.8
    lengths = np.random.lognormal(mu, sigma, n_requests).astype(int)
    # Clip between 16 and max_context
    return np.clip(lengths, 16, max_context)


def run_fragmentation_analysis():
    rule("NANOINFER -- MILESTONE 3: Memory Fragmentation & PagedAttention")

    cfg = LLAMA3_8B
    max_ctx = 2048
    block_size = 16
    dtype_bytes = 2  # float16

    bytes_per_tok = cfg.kv_bytes_per_token(dtype_bytes)
    print(f"Target Model: {cfg.name} (layers={cfg.n_layers}, kv_heads={cfg.n_kv_heads}, head_dim={cfg.head_dim})")
    print(f"Cache memory per token: {bytes_per_tok / 1024:.1f} KiB")
    print(f"Max context window: {max_ctx} tokens")
    print(f"Block size: {block_size} tokens")

    # Generate 100 realistic requests
    lengths = simulate_workload(n_requests=100, max_context=max_ctx, mean_length=220)
    avg_len = float(np.mean(lengths))

    print(f"\nSimulated Workload: 100 requests (mean length = {avg_len:.1f} tokens, max = {np.max(lengths)} tokens)")
    print("-" * 78)

    # 1. Contiguous Allocation: Preallocates max_ctx for EVERY request
    contiguous_per_seq = max_ctx * bytes_per_tok
    total_contiguous_bytes = len(lengths) * contiguous_per_seq

    # 2. Paged Allocation: Allocates ceil(length / block_size) * block_size * bytes_per_tok
    paged_blocks_per_seq = np.ceil(lengths / block_size).astype(int)
    total_paged_blocks = int(np.sum(paged_blocks_per_seq))
    total_paged_bytes = total_paged_blocks * block_size * bytes_per_tok

    # 3. Active data (pure tokens without any waste)
    active_bytes = int(np.sum(lengths)) * bytes_per_tok

    # Waste metrics
    contiguous_waste_pct = (1.0 - (active_bytes / total_contiguous_bytes)) * 100.0
    paged_waste_pct = (1.0 - (active_bytes / total_paged_bytes)) * 100.0

    print(f"{'Metric':<36} | {'Contiguous Cache':>18} | {'Paged Cache':>16}")
    print("-" * 78)
    print(f"{'Total Memory Allocated':<36} | {total_contiguous_bytes / 1e9:>15.2f} GB | {total_paged_bytes / 1e9:>13.2f} GB")
    print(f"{'Active Useful Memory':<36} | {active_bytes / 1e9:>15.2f} GB | {active_bytes / 1e9:>13.2f} GB")
    print(f"{'Memory Wasted':<36} | {(total_contiguous_bytes - active_bytes) / 1e9:>15.2f} GB | {(total_paged_bytes - active_bytes) / 1e9:>13.2f} GB")
    print(f"{'Fragmentation / Waste Rate':<36} | {contiguous_waste_pct:>17.1f}% | {paged_waste_pct:>15.1f}%")
    print(f"{'Memory Footprint Reduction':<36} | {'1.0x (baseline)':>18} | {total_contiguous_bytes / total_paged_bytes:>15.2f}x")

    # Concurrency capacity comparison across GPU budgets
    rule("Maximum Concurrency Capacity Across GPU Memory Budgets")
    budgets_gb = [16.0, 40.0, 80.0]

    print(f"{'VRAM Budget for KV Cache':<28} | {'Contiguous Users':>18} | {'Paged Users':>16} | {'Capacity Gain':>14}")
    print("-" * 84)

    concurrency_contiguous = []
    concurrency_paged = []

    for budget in budgets_gb:
        budget_bytes = budget * 1e9
        users_contig = int(budget_bytes // contiguous_per_seq)
        avg_paged_per_user = (avg_len / block_size + 0.5) * block_size * bytes_per_tok
        users_paged = int(budget_bytes // avg_paged_per_user)
        gain = users_paged / users_contig if users_contig > 0 else float("inf")
        concurrency_contiguous.append(users_contig)
        concurrency_paged.append(users_paged)
        print(f"{budget:>6.0f} GB{'':<21} | {users_contig:>18d} | {users_paged:>16d} | {gain:>13.2f}x")

    # Plot charts
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Chart 1: Memory Allocation vs Waste
        categories = ["Contiguous (M2)", "PagedAttention (M3)"]
        useful = [active_bytes / 1e9, active_bytes / 1e9]
        waste = [(total_contiguous_bytes - active_bytes) / 1e9, (total_paged_bytes - active_bytes) / 1e9]

        x = np.arange(len(categories))
        width = 0.5
        ax1.bar(x, useful, width, label="Active Tokens (Useful)", color="royalblue")
        ax1.bar(x, waste, width, bottom=useful, label="Memory Wasted", color="crimson")
        ax1.set_ylabel("Allocated Memory (GB)", fontsize=11)
        ax1.set_title("Memory Allocation & Waste (100 Requests)", fontsize=12, fontweight="bold")
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories, fontsize=11)
        ax1.grid(True, alpha=0.3, axis="y")
        ax1.legend(fontsize=10)

        # Annotate percentages
        ax1.text(0, (total_contiguous_bytes / 1e9) / 2, f"{contiguous_waste_pct:.0f}% Wasted", ha="center", color="white", fontweight="bold", fontsize=11)
        ax1.text(1, (total_paged_bytes / 1e9) * 0.9, f"{paged_waste_pct:.0f}% Wasted", ha="center", color="black", fontweight="bold", fontsize=11)

        # Chart 2: Concurrency by Budget
        bx = np.arange(len(budgets_gb))
        b_width = 0.35
        ax2.bar(bx - b_width/2, concurrency_contiguous, b_width, label="Contiguous (Static max_seq)", color="grey")
        ax2.bar(bx + b_width/2, concurrency_paged, b_width, label="PagedAttention (Dynamic 16-tok blocks)", color="forestgreen")
        ax2.set_xlabel("GPU VRAM Dedicated to KV Cache", fontsize=11)
        ax2.set_ylabel("Max Concurrent Active Sequences", fontsize=11)
        ax2.set_title("Serving Concurrency Capacity", fontsize=12, fontweight="bold")
        ax2.set_xticks(bx)
        ax2.set_xticklabels([f"{b:.0f} GB" for b in budgets_gb], fontsize=11)
        ax2.grid(True, alpha=0.3, axis="y")
        ax2.legend(fontsize=10)

        # Annotate multipliers
        for i, (c, p) in enumerate(zip(concurrency_contiguous, concurrency_paged)):
            ax2.text(i + b_width/2, p + 5, f"{p/c:.1f}x", ha="center", fontweight="bold", color="darkgreen")

        fig.suptitle("Milestone 3: Why PagedAttention Unlocks 7x-8x Concurrency", fontsize=14, fontweight="bold")
        fig.tight_layout()

        out_path = OUT / "m3_fragmentation.png"
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"\n[Chart saved successfully to {out_path}]")
    except ImportError:
        print("\n(matplotlib not installed; skipping chart generation)")


if __name__ == "__main__":
    run_fragmentation_analysis()
