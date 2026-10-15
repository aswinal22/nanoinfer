"""Analytical cost model for transformer inference.

This module answers, without running a single kernel:

  * How many FLOPs does prefill / decode cost?
  * How many bytes must be moved?
  * What is the arithmetic intensity, and therefore is the phase compute-bound
    or bandwidth-bound?
  * Given a device, what is the speed ceiling and what limits us?

The headline result, which everything else in this repo follows from:

    Autoregressive decoding at batch size 1 has arithmetic intensity ~1 FLOP/byte.

One multiply-accumulate per weight byte read. The GPU reads ~2 TB/s and can do
~1000 TFLOP/s, so it idles at roughly 0.1% of its compute. That is why decode
throughput scales with *batch size* and barely at all with FLOPs, and it is the
economic argument for every serving optimization that follows in this repo.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import TransformerConfig
from .hardware import Hardware

BYTES_PER_DTYPE = {
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int4": 0.5,
}


@dataclass(frozen=True)
class Workload:
    """A unit of work, priced in FLOPs and bytes.

    Attributes:
        label: Human-readable description.
        flops: Floating-point operations required.
        bytes_moved: Bytes that must cross the memory hierarchy.
    """

    label: str
    flops: float
    bytes_moved: float

    @property
    def arithmetic_intensity(self) -> float:
        """FLOP performed per byte moved. The x-axis of the roofline plot."""
        if self.bytes_moved == 0:
            return float("inf")
        return self.flops / self.bytes_moved

    @property
    def ideal_seconds(self) -> float:
        """Lower bound on wall time for a perfect device: bytes / bandwidth."""
        return 0.0 if self.bytes_moved == 0 else float("nan")


@dataclass(frozen=True)
class RooflineResult:
    """Where a workload lands relative to the device's roof."""

    achieved_tflops: float
    bound: str  # "compute" | "memory"
    hardware: Hardware
    seconds: float

    def describe(self) -> str:
        return (
            f"{self.bound.upper():>7}-bound  "
            f"{self.achieved_tflops:>7.1f} TFLOP/s  "
            f"{self.seconds * 1e3:>8.3f} ms  ({self.hardware.name})"
        )


# ---------------------------------------------------------------------------
# FLOP counting
# ---------------------------------------------------------------------------


def linear_flops(cfg: TransformerConfig, n_tokens: int, include_lm_head: bool = True) -> float:
    """FLOPs for all per-token weight matmuls.

    A dense matmul of an [n, k] by [k, m] matrix costs 2*n*k*m FLOPs (one
    multiply + one accumulate per output element). Summed over a block, that
    collapses to 2 * (parameters) * n_tokens.
    """
    params = cfg.params_blocks + (cfg.params_lm_head if include_lm_head else 0)
    return 2.0 * params * n_tokens


def attention_flops(cfg: TransformerConfig, n_queries: int, context_len: int, batch: int = 1) -> float:
    """FLOPs for QK^T and (softmax)V.

    Per head, QK^T is [S, d] @ [d, S] = 2*S*S*d FLOPs. Summing over heads
    collapses the head dimension to `d_model`. AV costs the same again, giving
    4 * S^2 * d_model per layer.

    This charges the *full* S x S score matrix. Causal attention only needs the
    lower triangle, so roughly half of this is wasted arithmetic in a naive
    implementation. Blockwise kernels that skip fully-masked blocks recover it;
    running the math in half precision instead saves much more.
    """
    if context_len <= 0:
        return 0.0
    per_layer = 4.0 * n_queries * context_len * cfg.d_model
    return per_layer * cfg.n_layers * batch


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


def prefill_workload(
    cfg: TransformerConfig,
    batch: int = 1,
    seq_len: int = 512,
    dtype_bytes: int = 2,
    include_lm_head: bool = False,
) -> Workload:
    """Cost of processing `seq_len` prompt tokens across `batch` sequences.

    In prefill every weight is read exactly once and reused across all
    batch*seq_len tokens. Arithmetic intensity therefore scales *linearly with
    the token count* -- which is why a 512-token prompt is compute-bound while
    a 1-token generation is not.
    """
    n_tokens = batch * seq_len

    flops = linear_flops(cfg, n_tokens, include_lm_head)
    flops += attention_flops(cfg, seq_len, seq_len, batch)

    # Weights read once. This is the dominant byte term.
    weight_bytes = cfg.n_params_matmul * dtype_bytes
    # KV cache written out for the tokens we just processed.
    kv_bytes = cfg.kv_bytes(seq_len, batch, dtype_bytes)
    # Activations: log2(depth) checkpointing is negligible next to the above,
    # and we deliberately ignore it rather than guess.
    act_bytes = 0.0

    total_bytes = weight_bytes + kv_bytes + act_bytes

    return Workload(
        label=f"prefill B={batch} S={seq_len}",
        flops=flops,
        bytes_moved=total_bytes,
    )


def decode_workload(
    cfg: TransformerConfig,
    batch: int = 1,
    ctx_len: int = 0,
    dtype_bytes: int = 2,
    include_lm_head: bool = True,
) -> Workload:
    """Cost of generating one token for `batch` sequences at context `ctx_len`.

    Two properties drive everything:

    1. Weights are read once *for the whole batch*, so the weight-byte term is
       fixed no matter how many sequences are in flight.
    2. FLOPs scale with `batch`. So intensity climbs toward
       `2 * dtype_bytes` (2 FLOP/byte for fp16) as batch grows -- and on real
       hardware that is still far below the ridge point.
    """
    flops = linear_flops(cfg, batch, include_lm_head)
    flops += attention_flops(cfg, 1, ctx_len, batch)

    weight_bytes = cfg.n_params_matmul * dtype_bytes
    kv_bytes = cfg.kv_bytes(ctx_len, batch, dtype_bytes)

    return Workload(
        label=f"decode  B={batch} ctx={ctx_len}",
        flops=flops,
        bytes_moved=weight_bytes + kv_bytes,
    )


def attention_only_workload(
    cfg: TransformerConfig,
    batch: int = 1,
    ctx_len: int = 2048,
    dtype_bytes: int = 2,
) -> Workload:
    """Cost of just the QK^T + AV matmuls, with no weights.

    This is the piece that scaling optimizations (FlashAttention, PagedAttention,
    KV cache quantization) actually target, so it is worth pricing separately.
    """
    flops = attention_flops(cfg, 1, ctx_len, batch)
    # We read K and V once each; we do not read the weights at all here.
    kv_bytes = cfg.kv_bytes(ctx_len, batch, dtype_bytes)
    return Workload(
        label=f"attn-only B={batch} ctx={ctx_len}",
        flops=flops,
        bytes_moved=kv_bytes,
    )


# ---------------------------------------------------------------------------
# The roofline itself
# ---------------------------------------------------------------------------


def roofline(workload: Workload, hw: Hardware) -> RooflineResult:
    """Upper bound on achieved throughput for `workload` on `hw`.

    The attainable performance is the lesser of what the compute units can
    deliver and what the memory system can feed them:

        perf = min(peak_flops, intensity * peak_bandwidth)

    Above the ridge point the workload is compute-bound; below it, memory-bound.
    """
    intensity = workload.arithmetic_intensity
    compute_ceiling = hw.peak_tflops
    memory_ceiling = intensity * (hw.bandwidth_gbs / 1000.0)  # TFLOP/s

    achieved = min(compute_ceiling, memory_ceiling)
    seconds = workload.flops / (achieved * 1e12) if achieved > 0 else float("inf")

    return RooflineResult(
        achieved_tflops=achieved,
        bound="compute" if memory_ceiling >= compute_ceiling else "memory",
        hardware=hw,
        seconds=seconds,
    )


def tokens_per_second(workload: Workload, result: RooflineResult, tokens: int) -> float:
    """Throughput implied by the roofline ceiling, in tokens/s."""
    if result.seconds <= 0 or tokens <= 0:
        return float("inf")
    return tokens / result.seconds


def summarize(cfg: TransformerConfig, hw: Hardware) -> str:
    """A compact human-readable report for one model on one device."""
    lines: list[str] = []
    lines.append(f"{cfg.name} on {hw.name}")
    lines.append("-" * 78)
    lines.append(f"  ridge point: {hw.ridge_point:.1f} FLOP/byte")
    lines.append(f"  weights:     {cfg.n_params_matmul / 1e9:.2f}B params "
                 f"= {cfg.n_params_matmul * 2 / 1e9:.2f} GB @fp16")
    kv = cfg.kv_bytes_per_token(2)
    lines.append(f"  KV/token:    {kv / 1024:.1f} KiB  "
                 f"(full {cfg.max_position_embeddings}-token seq: "
                 f"{cfg.kv_bytes(cfg.max_position_embeddings) / 1e9:.2f} GB)")
    lines.append("")

    header = f"  {'workload':<28}{'TFLOP':>9}{'GB':>9}{'FLOP/B':>10}{'bound':>10}{'ms':>9}"
    lines.append(header)
    lines.append("  " + "-" * 74)

    workloads = [
        prefill_workload(cfg, batch=1, seq_len=512),
        prefill_workload(cfg, batch=1, seq_len=2048),
        decode_workload(cfg, batch=1, ctx_len=0),
        decode_workload(cfg, batch=1, ctx_len=2048),
        decode_workload(cfg, batch=8, ctx_len=2048),
        decode_workload(cfg, batch=32, ctx_len=2048),
        decode_workload(cfg, batch=128, ctx_len=2048),
        attention_only_workload(cfg, batch=32, ctx_len=2048),
    ]

    for w in workloads:
        r = roofline(w, hw)
        lines.append(
            f"  {w.label:<28}"
            f"{w.flops / 1e12:>9.3f}"
            f"{w.bytes_moved / 1e9:>9.3f}"
            f"{w.arithmetic_intensity:>10.2f}"
            f"{r.bound:>10}"
            f"{r.seconds * 1e3:>9.3f}"
        )
    return "\n".join(lines)