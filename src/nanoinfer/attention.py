"""Attention, written out so the shape logic is visible.

The one thing to internalise here: **queries and keys are allowed to have
different lengths.** Prefill has q_len == kv_len. Cached decode has q_len == 1
and kv_len == everything generated so far. Nearly every M2 bug is a failure to
handle that second case correctly.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .ops import apply_rope


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    """Expand KV heads to match query heads for grouped-query attention.

    Args:
        x: [batch, n_kv_heads, seq, head_dim]
        n_rep: How many query heads share each KV head (the GQA ratio).

    Returns:
        [batch, n_kv_heads * n_rep, seq, head_dim]

    GQA stores fewer KV heads than query heads because K and V are much less
    expressive than Q: several query heads can share one KV head with minimal
    quality loss, which shrinks the cache by the GQA ratio.
    """
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    # repeat_interleave -- not repeat -- so head i*g..(i+1)*g-1 all map to head i.
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


def causal_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    scale: float | None = None,
) -> Tensor:
    """Scaled dot-product attention with causal masking.

    Args:
        q: [batch, n_heads, q_len, head_dim]
        k: [batch, n_kv_heads, kv_len, head_dim]
        v: [batch, n_kv_heads, kv_len, head_dim]
        scale: Defaults to 1/sqrt(head_dim).

    Returns:
        [batch, n_heads, q_len, head_dim]

    Requires kv_len >= q_len. The queries are assumed to be the **last**
    q_len positions of the kv_len positions -- which is what makes this one
    function correct for both prefill and cached decode.
    """
    q_len, kv_len, head_dim = q.shape[-2], k.shape[-2], q.shape[-1]
    if kv_len < q_len:
        raise ValueError(f"kv_len ({kv_len}) must be >= q_len ({q_len})")
    if scale is None:
        scale = 1.0 / (head_dim ** 0.5)

    n_rep = q.shape[1] // k.shape[1]
    k = repeat_kv(k, n_rep)
    v = repeat_kv(v, n_rep)

    scores = (q @ k.transpose(-2, -1)) * scale  # [B, H, q_len, kv_len]

    # --- the masking detail that matters ---------------------------------
    # Query at local index i sits at absolute position (kv_len - q_len + i),
    # and may attend to every key at or before that position.
    #
    # For prefill (q_len == kv_len) this collapses to the familiar triangular
    # mask. For decode (q_len == 1) the single query sits at kv_len-1 and sees
    # everything -- so the mask is all-true. Getting this offset wrong silently
    # lets early tokens attend to later ones.
    device = q.device
    q_pos = torch.arange(kv_len - q_len, kv_len, device=device)
    k_pos = torch.arange(kv_len, device=device)
    mask = k_pos[None, :] <= q_pos[:, None]  # [q_len, kv_len]

    scores = scores.masked_fill(~mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return probs @ v  # [B, H, q_len, head_dim]


def project_qkv(
    x: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    w_v: Tensor,
    n_heads: int,
    n_kv_heads: int,
    cos: Tensor,
    sin: Tensor,
    b_q: Tensor | None = None,
    b_k: Tensor | None = None,
    b_v: Tensor | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Project hidden states to Q/K/V, apply RoPE, and split into heads.

    Args:
        x: [batch, seq, d_model]
        w_q/w_k/w_v: [out_features, in_features] projection weights.
        n_heads: Number of query heads.
        n_kv_heads: Number of key/value heads.
        cos, sin: [seq, head_dim] RoPE tables for the positions in `x`.
        b_q/b_k/b_v: Optional biases.

    Returns:
        q: [batch, n_heads, seq, head_dim]
        k: [batch, n_kv_heads, seq, head_dim]
        v: [batch, n_kv_heads, seq, head_dim]
    """
    b, s, _ = x.shape
    head_dim = w_q.shape[0] // n_heads

    q = torch.nn.functional.linear(x, w_q, b_q)
    k = torch.nn.functional.linear(x, w_k, b_k)
    v = torch.nn.functional.linear(x, w_v, b_v)

    q = q.view(b, s, n_heads, head_dim).transpose(1, 2)
    k = k.view(b, s, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(b, s, n_kv_heads, head_dim).transpose(1, 2)

    # RoPE is applied to Q and K only. V is left alone -- rotating the values
    # would rotate the output and cancel the relative-position signal.
    q = apply_rope(q, cos, sin)
    k = apply_rope(k, cos, sin)

    return q, k, v


def attn_output_proj(attn: Tensor, w_o: Tensor, b_o: Tensor | None = None) -> Tensor:
    """Merge heads and project back to `d_model`.

    Args:
        attn: [batch, n_heads, q_len, head_dim]
        w_o: [d_model, n_heads * head_dim]
        b_o: Optional bias.

    Returns:
        [batch, q_len, d_model]
    """
    b, h, s, d = attn.shape
    merged = attn.transpose(1, 2).reshape(b, s, h * d)
    return torch.nn.functional.linear(merged, w_o, b_o)