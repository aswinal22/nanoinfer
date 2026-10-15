"""Tests for the M2 model components.

Two groups:

  * Tests of the provided code (ops, attention, prefill). These pass now.
  * Tests of the KV cache. Marked `xfail` -- they will start passing when
    `src/nanoinfer/kv_cache.py` is implemented. That is the definition of done.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from nanoinfer import config as cfgmod
from nanoinfer.attention import causal_attention, repeat_kv
from nanoinfer.config import TINY, TOY
from nanoinfer.kv_cache import KVCache
from nanoinfer.ops import (
    apply_rope,
    build_rope_cache,
    rms_norm,
    rotate_half,
    rope_slice,
    silu,
    swiglu,
)
from nanoinfer.transformer import NanoTransformer


@pytest.fixture(scope="module")
def model() -> NanoTransformer:
    torch.manual_seed(0)
    return NanoTransformer(TINY).eval()


# ===========================================================================
# RMSNorm
# ===========================================================================


def test_rms_norm_produces_unit_rms():
    x = torch.randn(4, 8, 128) * 10.0
    out = rms_norm(x, torch.ones(128))
    rms = out.pow(2).mean(-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)


def test_rms_norm_is_scale_invariant():
    """Normalisation removes scale; only the gain survives."""
    x = torch.randn(2, 5, 128)
    w = torch.randn(128)
    assert torch.allclose(rms_norm(x, w), rms_norm(x * 7.0, w), atol=1e-5)


def test_rms_norm_applies_gain():
    x = torch.randn(3, 128)
    w = torch.full((128,), 2.5)
    assert torch.allclose(rms_norm(x, w), rms_norm(x, torch.ones(128)) * 2.5, atol=1e-5)


def test_rms_norm_survives_large_inputs_in_float16():
    """The fp32 upcast is load-bearing: fp16 maxes out at 65504."""
    x = torch.full((1, 128), 500.0, dtype=torch.float16)
    out = rms_norm(x, torch.ones(128, dtype=torch.float16))
    assert torch.isfinite(out).all()


# ===========================================================================
# RoPE
# ===========================================================================


def test_rope_cache_shape():
    cos, sin = build_rope_cache(head_dim=32, max_position=128)
    assert cos.shape == (128, 32)
    assert sin.shape == (128, 32)


def test_rope_requires_even_head_dim():
    with pytest.raises(ValueError, match="even"):
        build_rope_cache(head_dim=7, max_position=4)


def test_rotate_half_swaps_and_negates():
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    out = rotate_half(x)
    assert torch.allclose(out, torch.tensor([[-3.0, -4.0, 1.0, 2.0]]))


def test_rope_preserves_vector_norms():
    """RoPE is a rotation, so it must not change any head's magnitude."""
    cos, sin = build_rope_cache(head_dim=32, max_position=16)
    q = torch.randn(2, 4, 16, 32)
    out = apply_rope(q, cos, sin)
    assert torch.allclose(q.norm(dim=-1), out.norm(dim=-1), atol=1e-4)


def test_rope_position_zero_is_identity():
    """Angle 0 means cos=1, sin=0, so RoPE is a no-op at position zero."""
    cos, sin = build_rope_cache(head_dim=32, max_position=8)
    c, s = rope_slice(cos, sin, 0, 1)  # position 0 only
    x = torch.randn(1, 2, 1, 32)
    assert torch.allclose(apply_rope(x, c, s), x, atol=1e-5)


def test_rope_is_not_identity_at_later_positions():
    """Guard against `apply_rope` accidentally being a no-op everywhere."""
    cos, sin = build_rope_cache(head_dim=32, max_position=8)
    c, s = rope_slice(cos, sin, 4, 1)
    x = torch.randn(1, 2, 1, 32)
    assert not torch.allclose(apply_rope(x, c, s), x, atol=1e-3)


def test_rope_scores_depend_only_on_relative_distance():
    """The defining property: attention logits depend on m - n, not m and n.

    Two (q, k) pairs with the same gap but different absolute positions must
    give the same dot product. This is why RoPE extrapolates past its training
    context.
    """
    cos, sin = build_rope_cache(head_dim=16, max_position=64)
    q = torch.randn(1, 1, 1, 16)
    k = torch.randn(1, 1, 1, 16)

    def score(m: int, n: int) -> float:
        cq, sq = rope_slice(cos, sin, m, 1)
        ck, sk = rope_slice(cos, sin, n, 1)
        return float((apply_rope(q, cq, sq) * apply_rope(k, ck, sk)).sum())

    for gap in (1, 5, 17):
        a = score(gap + 3, 3)
        b = score(gap + 40, 40)
        assert a == pytest.approx(b, abs=1e-4), f"gap {gap}"


def test_rope_slice_offsets_correctly():
    cos, sin = build_rope_cache(head_dim=8, max_position=16)
    c_full, _ = rope_slice(cos, sin, 0, 16)
    c_part, _ = rope_slice(cos, sin, 5, 4)
    assert torch.allclose(c_part, c_full[5:9])


# ===========================================================================
# GQA expansion
# ===========================================================================


def test_repeat_kv_expands_heads_in_groups():
    """Head i*g..(i+1)*g-1 must all map to KV head i, not be interleaved."""
    x = torch.arange(2 * 2 * 3 * 4, dtype=torch.float32).reshape(2, 2, 3, 4)
    out = repeat_kv(x, n_rep=2)
    assert out.shape == (2, 4, 3, 4)
    assert torch.allclose(out[:, 0], out[:, 1])
    assert torch.allclose(out[:, 2], out[:, 3])
    assert not torch.allclose(out[:, 0], out[:, 2])


def test_repeat_kv_identity():
    x = torch.randn(1, 4, 2, 8)
    assert torch.equal(repeat_kv(x, 1), x)


# ===========================================================================
# Causal attention
# ===========================================================================


def naive_attention(q: Tensor, k: Tensor, v: Tensor) -> Tensor:
    """Explicit-loop reference. Obviously correct, far too slow."""
    b, h, q_len, d = q.shape
    kv_len = k.shape[2]
    out = torch.zeros(b, h, q_len, d)
    scale = 1.0 / math.sqrt(d)
    for bi in range(b):
        for hi in range(h):
            for i in range(q_len):
                pos = kv_len - q_len + i  # absolute position of this query
                scores = []
                for j in range(kv_len):
                    if j <= pos:
                        scores.append(float((q[bi, hi, i] * k[bi, hi, j]).sum()) * scale)
                    else:
                        scores.append(float("-inf"))
                p = torch.softmax(torch.tensor(scores), dim=0)
                for j in range(kv_len):
                    out[bi, hi, i] += p[j] * v[bi, hi, j]
    return out


def test_causal_attention_matches_naive_reference():
    torch.manual_seed(1)
    q = torch.randn(2, 2, 5, 8)
    k = torch.randn(2, 2, 5, 8)
    v = torch.randn(2, 2, 5, 8)
    assert torch.allclose(causal_attention(q, k, v), naive_attention(q, k, v), atol=1e-5)


def test_future_tokens_cannot_influence_the_past():
    """Change v at position 4; positions 0-3 must not move."""
    torch.manual_seed(2)
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    v2 = v.clone()
    v2[:, :, 4, :] = torch.randn(2, 8) * 50

    a = causal_attention(q, k, v)
    b = causal_attention(q, k, v2)
    assert torch.allclose(a[:, :, :4], b[:, :, :4], atol=1e-6)
    assert not torch.allclose(a[:, :, 4], b[:, :, 4], atol=1e-3)


def test_single_query_against_full_history():
    """The decode case: q_len=1, kv_len=N. The last query sees everything."""
    torch.manual_seed(3)
    n = 7
    q = torch.randn(1, 2, 1, 8)
    k = torch.randn(1, 2, n, 8)
    v = torch.randn(1, 2, n, 8)
    out = causal_attention(q, k, v)
    assert out.shape == (1, 2, 1, 8)
    assert torch.allclose(out, naive_attention(q, k, v), atol=1e-5)


def test_single_query_is_unmasked():
    """A lone query must see every key -- nothing for it to be masked against."""
    torch.manual_seed(4)
    q = torch.randn(1, 1, 1, 4)
    k = torch.randn(1, 1, 5, 4)
    v = torch.randn(1, 1, 5, 4)
    v2 = v.clone()
    v2[:, :, 0, :] += 100.0
    # If any key were masked out, this perturbation would be ignored.
    assert not torch.allclose(causal_attention(q, k, v), causal_attention(q, k, v2))


def test_decoding_last_token_matches_prefill_last_token():
    """Prefill S tokens then decode, vs. prefill S+1 tokens.

    If the mask alignment in the q_len=1 path were wrong, these would diverge.
    This is the same property the KV cache test relies on, with a hand-built
    cache standing in for yours.
    """
    torch.manual_seed(5)
    s = 6
    q = torch.randn(1, 2, s + 1, 8)
    k = torch.randn(1, 2, s + 1, 8)
    v = torch.randn(1, 2, s + 1, 8)

    full = causal_attention(q, k, v)[:, :, -1, :]
    step = causal_attention(q[:, :, -1:, :], k, v)[:, :, -1, :]
    assert torch.allclose(full, step, atol=1e-5)


def test_rejects_impossible_shapes():
    q = torch.randn(1, 2, 5, 8)
    k = torch.randn(1, 2, 3, 8)  # kv_len < q_len
    with pytest.raises(ValueError, match="kv_len"):
        causal_attention(q, k, k)


# ===========================================================================
# Feed-forward primitives
# ===========================================================================


def test_silu_matches_definition():
    x = torch.randn(50)
    assert torch.allclose(silu(x), x * torch.sigmoid(x))


def test_swiglu_shape_and_range():
    g, u = torch.randn(3, 7), torch.randn(3, 7)
    assert swiglu(g, u).shape == (3, 7)


# ===========================================================================
# The model (prefill path -- provided)
# ===========================================================================


def test_forward_produces_logits(model: NanoTransformer):
    ids = torch.randint(0, TINY.vocab_size, (2, 7))
    with torch.no_grad():
        logits = model(ids)
    assert logits.shape == (2, 7, TINY.vocab_size)
    assert torch.isfinite(logits).all()


def test_forward_is_causal():
    """Changing the last input token must not alter earlier logits."""
    torch.manual_seed(6)
    m = NanoTransformer(TINY).eval()
    ids = torch.randint(0, TINY.vocab_size, (1, 6))
    ids2 = ids.clone()
    ids2[0, -1] = (ids[0, -1] + 1) % TINY.vocab_size

    with torch.no_grad():
        a, b = m(ids), m(ids2)
    assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)


def test_parameter_count_matches_analytical_model(model: NanoTransformer):
    """The M1 cost model and the real module must agree.

    A mismatch means `config.py` and `transformer.py` have drifted apart, and
    every FLOP prediction in the repo becomes suspect. This test is what caught
    `n_params` omitting RMSNorm gains.
    """
    d, hd, kvh, L = TINY.d_model, TINY.head_dim, TINY.n_kv_heads, TINY.n_layers

    q_and_o = 2 * d * TINY.n_heads * hd  # w_q and w_o
    k_and_v = 2 * d * kvh * hd  # w_k and w_v
    ffn = 3 * d * TINY.d_ff  # gate, up, down
    norms = 2 * d  # attn_norm + ffn_norm
    embed = TINY.vocab_size * d

    expected = embed + L * (q_and_o + k_and_v + ffn + norms) + d  # + final norm
    assert model.n_params() == expected

    # And the analytical model in config.py agrees with both.
    assert TINY.n_params == expected


def test_greedy_generation_without_cache(model: NanoTransformer):
    """The naive baseline your cache must beat."""
    ids = torch.randint(0, TINY.vocab_size, (1, 4))
    with torch.no_grad():
        out = model.generate_greedy(ids, max_new_tokens=5)
    assert out.shape == (1, 9)


def test_rejects_context_overflow(model: NanoTransformer):
    ids = torch.zeros(1, TINY.max_position_embeddings + 1, dtype=torch.long)
    with pytest.raises(ValueError, match="exceeds context"):
        model(ids)


def test_toy_config_runs_and_has_gqa():
    torch.manual_seed(7)
    m = NanoTransformer(TOY).eval()
    ids = torch.randint(0, TOY.vocab_size, (2, 12))
    with torch.no_grad():
        logits = m(ids)
    assert logits.shape == (2, 12, TOY.vocab_size)
    assert TOY.gqa_ratio == 4
    assert torch.isfinite(logits).all()


# ===========================================================================
# M2 TASK -- KV cache. xfail until implemented.
# ===========================================================================


@pytest.mark.xfail(strict=False, reason="M2 task: implement KVCache")
def test_kv_cache_starts_empty():
    cache = KVCache.from_config(TINY, batch=2)
    assert cache.is_empty()
    assert cache.get_seq_len() == 0


@pytest.mark.xfail(strict=False, reason="M2 task: implement KVCache")
def test_kv_cache_append_grows_length():
    cache = KVCache.from_config(TINY, batch=2)
    d = TINY.n_kv_heads * TINY.head_dim
    k = torch.randn(2, TINY.n_kv_heads, 3, TINY.head_dim)
    v = torch.randn(2, TINY.n_kv_heads, 3, TINY.head_dim)
    assert d  # head geometry is sane

    kf, vf = cache.append(0, k, v)
    assert cache.get_seq_len() == 3
    assert kf.shape == (2, TINY.n_kv_heads, 3, TINY.head_dim)


@pytest.mark.xfail(strict=False, reason="M2 task: implement KVCache")
def test_kv_cache_reset():
    cache = KVCache.from_config(TINY, batch=1)
    cache.append(0, torch.randn(1, TINY.n_kv_heads, 4, TINY.head_dim),
                 torch.randn(1, TINY.n_kv_heads, 4, TINY.head_dim))
    cache.reset()
    assert cache.is_empty()


@pytest.mark.xfail(strict=False, reason="M2 task: implement KVCache")
def test_cached_decode_matches_full_forward():
    """THE acceptance test.

    Cached generation must be numerically identical to re-running the full
    forward every step. Any difference is a bug -- caching must not change the
    answer, only the amount of work done.
    """
    torch.manual_seed(8)
    m = NanoTransformer(TINY).eval()
    ids = torch.randint(0, TINY.vocab_size, (1, 6))

    cache = KVCache.from_config(TINY, batch=1, dtype=torch.float32)

    with torch.no_grad():
        # --- naive: recompute everything, every step ---
        naive = ids
        for _ in range(6):
            nxt = m(naive)[:, -1, :].argmax(dim=-1, keepdim=True)
            naive = torch.cat([naive, nxt], dim=1)

        # --- cached: one forward for the prompt, then one token at a time ---
        cur = ids
        logits = m.forward_step(cur, cache)
        nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        cached = torch.cat([ids, nxt], dim=1)
        for _ in range(5):
            logits = m.forward_step(nxt, cache)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cached = torch.cat([cached, nxt], dim=1)

    assert torch.equal(cached, naive), (
        f"cached {cached.tolist()} != naive {naive.tolist()}"
    )


@pytest.mark.xfail(strict=False, reason="M2 task: implement KVCache")
def test_forward_cached_is_wired_up():
    """forward_step exists, so the layer-level path should no longer raise."""
    torch.manual_seed(9)
    m = NanoTransformer(TINY).eval()
    cache = KVCache.from_config(TINY, batch=1)
    ids = torch.randint(0, TINY.vocab_size, (1, 4))
    with torch.no_grad():
        out = m.forward_step(ids, cache)
    assert out.shape == (1, 4, TINY.vocab_size)


@pytest.mark.xfail(strict=False, reason="M2 task: implement KVCache")
def test_cache_stores_kv_heads_not_query_heads():
    """Guards the GQA memory bug: 4x waste if you cache n_heads instead."""
    cache = KVCache.from_config(TOY, batch=1)  # TOY has 4:1 GQA
    k = torch.randn(1, TOY.n_kv_heads, 2, TOY.head_dim)
    v = torch.randn(1, TOY.n_kv_heads, 2, TOY.head_dim)
    kf, _ = cache.append(0, k, v)
    assert kf.shape[1] == TOY.n_kv_heads
    assert kf.shape[1] != TOY.n_heads


@pytest.mark.xfail(strict=False, reason="M2 task: implement KVCache")
def test_cache_capacity_accounting():
    """Allocated capacity vs live data -- the gap motivates M3."""
    cache = KVCache.from_config(TINY, batch=1)
    cache.append(0, torch.randn(1, TINY.n_kv_heads, 8, TINY.head_dim),
                 torch.randn(1, TINY.n_kv_heads, 8, TINY.head_dim))
    allocated = cache.bytes_allocated()
    per_token = TINY.kv_bytes_per_token(4)
    assert allocated == per_token * TINY.max_position_embeddings
    assert allocated > per_token * cache.get_seq_len()