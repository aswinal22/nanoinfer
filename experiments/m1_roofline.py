"""Milestone 1 -- What are we actually bound by?

Run:  python experiments/m1_roofline.py

Produces a console report plus three charts in experiments/out/.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from nanoinfer import config as cfgmod
from nanoinfer import hardware as hwmod
from nanoinfer import roofline as rl

OUT = Path(__file__).resolve().parent / "out"


def rule(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def part_1_the_claim() -> None:
    """The headline: decode at batch 1 has arithmetic intensity of ~1."""
    rule("1. Arithmetic intensity of a single decode step")

    cfg = cfgmod.LLAMA3_8B
    print(f"Model: {cfg.describe()}\n")

    print(f"{'phase':<34}{'FLOPs':>14}{'bytes':>14}{'FLOP/byte':>12}")
    print("-" * 74)

    cases = [
        rl.prefill_workload(cfg, 1, 512),
        rl.prefill_workload(cfg, 1, 2048),
        rl.decode_workload(cfg, 1, 0),
        rl.decode_workload(cfg, 1, 2048),
        rl.decode_workload(cfg, 32, 2048),
    ]
    for w in cases:
        print(f"{w.label:<34}{w.flops:>14.4e}{w.bytes_moved:>14.4e}"
              f"{w.arithmetic_intensity:>12.2f}")

    w1 = rl.decode_workload(cfg, 1, 0)
    print()
    print(f"Decode at B=1 moves {w1.bytes_moved / 1e9:.2f} GB to do "
          f"{w1.flops / 1e12:.4f} TFLOP.")
    print(f"Intensity = {w1.arithmetic_intensity:.3f} FLOP/byte. "
          f"The GPU can do ~1000 FLOP/byte.")
    print()
    print("Read that again: one multiply-accumulate per two bytes of weight.")
    print("The arithmetic units are idle. This is the whole game.")


def part_2_ridge() -> None:
    """Where each device switches from memory-bound to compute-bound."""
    rule("2. Ridge points -- how many FLOPs per byte to be compute-bound")

    for hw in hwmod.ALL:
        print("  " + hw.describe())

    print()
    print("Interpretation:")
    print("  A workload must exceed this intensity to use the tensor cores fully.")
    print("  Compare it to the ~1 FLOP/byte of batch-1 decode above.")


def part_3_sweep() -> None:
    """How intensity -- and therefore achievable throughput -- scales with batch."""
    rule("3. Decode: batch size is the only lever that matters")

    cfg = cfgmod.LLAMA3_8B
    hw = hwmod.H100_SXM

    print(f"{cfg.name} on {hw.name}, context = 2048\n")
    print(f"{'batch':>7}{'FLOP/byte':>12}{'bound':>10}"
          f"{'tok/s (1 user)':>16}{'tok/s (total)':>16}{'vs B=1':>9}")
    print("-" * 70)

    base = None
    for b in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512):
        w = rl.decode_workload(cfg, b, ctx_len=2048)
        r = rl.roofline(w, hw)
        # Each sequence advances one token per step, so total tok/s == batch * steps/s.
        total_tps = rl.tokens_per_second(w, r, tokens=b)
        if base is None:
            base = total_tps
        print(f"{b:>7}{w.arithmetic_intensity:>12.2f}{r.bound:>10}"
              f"{total_tps / b:>16.1f}{total_tps:>16.1f}{total_tps / base:>8.1f}x")

    print()
    print("Per-user latency gets WORSE as batch grows, but total throughput")
    print("rises ~linearly. A serving engine exists to make that trade on the")
    print("user's behalf, which is exactly what continuous batching (M4) automates.")


def part_4_prefill_sweep() -> None:
    """Find the prefill length at which a phase becomes compute-bound."""
    rule("4. Prefill: how long a prompt before we hit the compute roof?")

    cfg = cfgmod.LLAMA3_8B

    for hw in (hwmod.A100_80GB, hwmod.H100_SXM, hwmod.L4, hwmod.T4):
        crossover = None
        for s in range(1, 4097):
            w = rl.prefill_workload(cfg, batch=1, seq_len=s)
            if rl.roofline(w, hw).bound == "compute":
                crossover = s
                break
        print(f"  {hw.name:<12} ridge={hw.ridge_point:>6.1f}  "
              f"compute-bound once a single prompt exceeds ~{crossover} tokens")

    print()
    print("This is the reason chunked prefill exists. A 200-token prompt on an")
    print("A10G runs mostly idle; a 4000-token prompt saturates the tensor cores.")
    print("A good scheduler splits long prompts so the GPU is never waiting.")


def part_5_memory() -> None:
    """KV cache footprint -- the constraint that caps concurrency."""
    rule("5. KV cache is what actually caps how many users you can serve")

    budget_gb = 80.0
    budget = int(budget_gb * 1e9)

    print(f"KV budget: {budget_gb:.0f} GB\n")
    print(f"{'model':<18}{'params':>9}{'KV/token':>11}"
          f"{'1 seq @ ctx':>14}{'concurrent':>12}")
    print("-" * 66)

    for name in ("Llama-3-8B", "Llama-3-70B", "Mistral-7B", "Qwen2.5-0.5B"):
        c = cfgmod.get(name)
        per_tok = c.kv_bytes_per_token(2)
        one_seq = c.kv_bytes(c.max_position_embeddings)
        conc = c.max_batch_at_full_context(budget, 2)
        print(f"{c.name:<18}{c.human_params():>9}{per_tok / 1024:>9.0f} KiB"
              f"{one_seq / 1e9:>12.2f} GB{conc:>12}")

    print()
    print("Note the MQA/GQA column effect. Llama-3-8B has 32 query heads but")
    print("only 8 KV heads, so its cache is 4x smaller than MHA would give.")
    print("M3 (paged attention) attacks the fragmentation; M5 (quantization)")
    print("attacks the per-token size. Both multiply this concurrency number.")


def part_6_rooflines() -> None:
    """Per-device performance estimates for the flagship model."""
    rule("6. Roofline ceilings for Llama-3-8B")

    cfg = cfgmod.LLAMA3_8B
    for hw in (hwmod.H100_SXM, hwmod.A100_80GB, hwmod.L4, hwmod.T4):
        print()
        print(rl.summarize(cfg, hw))

    print()
    print("These are ceilings, not forecasts. Real serving hits 30-60% of these,")
    print("and much less for decode. But the *ratios* between phases are exact,")
    print("because they come from arithmetic that no optimization can change.")


def charts() -> None:
    """Optional plots. Skipped silently if matplotlib is unavailable."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"\n(skipping charts: {exc})")
        return

    OUT.mkdir(parents=True, exist_ok=True)
    cfg = cfgmod.LLAMA3_8B

    # --- Chart 1: the roofline with real workloads plotted on it -----------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    intensities = [0.01, 1000]
    ax.plot(
        intensities,
        [i * hwmod.H100_SXM.bandwidth_gbs / 1000 for i in intensities],
        label=f"{hwmod.H100_SXM.name} roofline",
        lw=2,
    )
    ax.axvline(hwmod.H100_SXM.ridge_point, ls="--", color="grey", lw=1)
    ax.text(hwmod.H100_SXM.ridge_point * 1.15, 20, "ridge", fontsize=9, color="grey")

    marks = [
        ("decode B=1", rl.decode_workload(cfg, 1, 2048), "o", (11, -15), "left"),
        ("decode B=32", rl.decode_workload(cfg, 32, 2048), "s", (11, -4), "left"),
        ("decode B=256", rl.decode_workload(cfg, 256, 2048), "^", (11, 7), "left"),
        # All prefill points clamp to the compute ceiling, so they share a y.
        # Stagger them vertically and anchor each to its own x.
        ("prefill S=512", rl.prefill_workload(cfg, 1, 512), "D", (0, 12), "center"),
        ("prefill S=2048", rl.prefill_workload(cfg, 1, 2048), "D", (0, 30), "center"),
        ("prefill S=8192", rl.prefill_workload(cfg, 1, 8192), "D", (4, -24), "right"),
    ]
    for label, w, marker, offset, ha in marks:
        r = rl.roofline(w, hwmod.H100_SXM)
        ax.scatter(w.arithmetic_intensity, r.achieved_tflops, marker=marker, s=90, zorder=5)
        ax.annotate(
            label,
            (w.arithmetic_intensity, r.achieved_tflops),
            textcoords="offset points",
            xytext=offset,
            fontsize=8,
            ha=ha,
            zorder=6,
        )

    # A little headroom so the stacked prefill labels stay inside the frame.
    ax.set_ylim(0.02, hwmod.H100_SXM.peak_tflops * 8)
    ax.set_xlim(0.01, 2e4)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("Attainable TFLOP/s")
    ax.set_title(f"Roofline, {cfg.name} on {hwmod.H100_SXM.name}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "m1_roofline.png", dpi=130)
    plt.close(fig)

    # --- Chart 2: batch size vs throughput ---------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    batches = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    hw = hwmod.H100_SXM
    total, per_user, intensity = [], [], []
    for b in batches:
        w = rl.decode_workload(cfg, b, ctx_len=2048)
        r = rl.roofline(w, hw)
        tps = rl.tokens_per_second(w, r, tokens=b)
        total.append(tps)
        per_user.append(tps / b)
        intensity.append(w.arithmetic_intensity)

    ax1.plot(batches, total, "o-", label="total throughput")
    ax1.plot(batches, per_user, "s--", label="per-user throughput")
    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("batch size")
    ax1.set_ylabel("tokens/s")
    ax1.set_title("Decode throughput vs batch size")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(batches, intensity, "o-", color="crimson")
    ax2.axhline(hw.ridge_point, ls="--", color="grey", label=f"ridge = {hw.ridge_point:.0f}")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel("batch size")
    ax2.set_ylabel("FLOP/byte")
    ax2.set_title("Arithmetic intensity vs batch size")
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()

    fig.tight_layout()
    fig.savefig(OUT / "m1_batch_scaling.png", dpi=130)
    plt.close(fig)

    print(f"\nwrote charts to {OUT}")


def main() -> None:
    print("=" * 78)
    print("NANOINFER -- MILESTONE 1: Where does inference time actually go?")
    print("=" * 78)

    part_1_the_claim()
    part_2_ridge()
    part_3_sweep()
    part_4_prefill_sweep()
    part_5_memory()
    part_6_rooflines()
    charts()

    print()
    print("=" * 78)
    print("Takeaway: prefill is compute-bound, decode is memory-bound,")
    print("and batch size is the only thing that changes that for decode.")
    print("Milestone 2 turns that into working code with a KV cache.")
    print("=" * 78)


if __name__ == "__main__":
    main()