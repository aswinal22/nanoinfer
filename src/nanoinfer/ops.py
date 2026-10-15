"""Primitive layers, written out explicitly rather than pulled from a library.

Every one of these is a few lines. The point is not to be convenient -- it is
that when something produces wrong numbers at M2 you will know exactly which
of these five operations is wrong.
"""

from __future__ import annotations

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def rms_norm(x: Tensor, weight: Tensor, eps: float = 1e-6) -> Tensor:
    """Root-mean-square layer norm (Zhang & Sennrich, 2019).

    LLaMA-style models use RMSNorm, not LayerNorm: no mean subtraction and no
    bias, which is both cheaper and empirically equivalent.

    Args:
        x: [..., d_model] activations.
        weight: [d_model] learned gain.
        eps: Numerical floor inside the sqrt.

    Returns:
        Normalised and scaled [..., d_model].
    """
    # Accumulate in fp32 regardless of storage dtype: squaring a fp16 value of
    # magnitude ~10 overflows fp16's 65504 limit.
    x32 = x.float()
    inv_rms = torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (x32 * inv_rms).to(x.dtype) * weight


# ---------------------------------------------------------------------------
# Rotary position embeddings
# ---------------------------------------------------------------------------


def build_rope_cache(
    head_dim: int,
    max_position: int,
    base: float = 10000.0,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[Tensor, Tensor]:
    """Precompute cos/sin tables for RoPE.

    There are two conventions for pairing rotation components:
      - interleaved: (0,1), (2,3), ...        -- original GPT-2
      - half-split:  (i, i+head_dim/2), ...  -- GPT-NeoX and LLaMA

    We use **half-split**, because that is what modern open models use. Getting
    this wrong produces plausible-looking garbage, so it is worth being sure.

    Args:
        head_dim: Must be even.
        max_position: Longest context the table covers.
        base: Geometric decay; 10000 is the standard value.
        device: Where to allocate.
        dtype: Precision of the returned tables.

    Returns:
        (cos, sin), each [max_position, head_dim].
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even for RoPE, got {head_dim}")

    # inv_freq[i] = base^(-2i/head_dim), i in [0, head_dim/2)
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    positions = torch.arange(max_position, device=device).float()
    freqs = torch.outer(positions, inv_freq)  # [max_position, head_dim/2]

    # Duplicate so the table lines up with the half-split rotation below.
    emb = torch.cat((freqs, freqs), dim=-1)  # [max_position, head_dim]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: Tensor) -> Tensor:
    """Rotate the two halves of the last dimension: (a, b) -> (-b, a)."""
    a, b = x.chunk(2, dim=-1)
    return torch.cat((-b, a), dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply rotary embeddings to queries or keys.

    RoPE encodes absolute position as a *rotation* of the query/key vector
    rather than an additive offset. The useful property is that the dot product
    between a rotated q and k depends only on their relative distance, which is
    what makes the encoding extrapolate.

    Args:
        x: [batch, n_heads, seq, head_dim]
        cos: [seq, head_dim] slice covering the positions being processed.
        sin: [seq, head_dim], same slice.

    Returns:
        Rotated tensor, same shape as `x`.
    """
    # cos/sin broadcast over the batch and head dimensions.
    return x * cos + rotate_half(x) * sin


def rope_slice(
    cos: Tensor, sin: Tensor, offset: int, length: int
) -> tuple[Tensor, Tensor]:
    """Extract the table rows for positions [offset, offset+length).

    **This is the function that matters during cached decode.** Prefill starts
    at position 0; decode starts wherever the cache currently ends. Passing the
    wrong offset produces silently wrong attention with no error raised.
    """
    return cos[offset : offset + length], sin[offset : offset + length]


# ---------------------------------------------------------------------------
# Feed-forward
# ---------------------------------------------------------------------------


def silu(x: Tensor) -> torch.Tensor:
    """SiLU / swish: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """Gated linear unit with SiLU activation (Shazeer, 2020).

    Uses 3 matrices rather than 2 for better quality at equal parameter count.
    This is why LLaMA-style `d_ff` is so much larger than GPT-2's for a similar
    hidden width.
    """
    return silu(gate) * up


def gelu(x: Tensor) -> Tensor:
    """Exact GELU, for the classic 2-matrix FFN (GPT-2)."""
    return torch.nn.functional.gelu(x, approximate="none")