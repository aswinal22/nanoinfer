"""Granular progress tracker for M2.

One test per rung from `docs/m2_stairs.md`. Deliberately **not** marked `xfail`:
a rung should be visibly red until it is done, so the pass count is your
progress bar.

    & $PY -m pytest tests/test_kv_cache_steps.py -q

Expect 6 failures now. Each rung turns one green.
"""

from __future__ import annotations

import pytest
import torch

from nanoinfer.attention import project_qkv
from nanoinfer.config import TINY
from nanoinfer.kv_cache import KVCache
from nanoinfer.ops import build_rope_cache, rms_norm
from nanoinfer.transformer import DecoderLayer, NanoTransformer

B, PROMPT_LEN = 1, 5


@pytest.fixture(scope="module")
def cos_sin():
    return build_rope_cache(TINY.head_dim, TINY.max_position_embeddings)


@pytest.fixture
def cache() -> KVCache:
    return KVCache.from_config(TINY, batch=B)


def rand_kv(n: int, batch: int = B) -> torch.Tensor:
    return torch.randn(batch, TINY.n_kv_heads, n, TINY.head_dim)


# ===========================================================================
# RUNG 1 -- allocate the buffers
# ===========================================================================


def test_rung1_cache_allocates_one_buffer_per_layer_and_tensor():
    """k and v exist, with shape [n_layers, batch, n_kv_heads, max_seq, head_dim].

    The layer axis comes FIRST so a single tensor holds every layer.
    """
    c = KVCache.from_config(TINY, batch=B)
    expected = (
        TINY.n_layers,
        B,
        TINY.n_kv_heads,
        TINY.max_position_embeddings,
        TINY.head_dim,
    )
    assert torch.is_tensor(c.k), "attribute 'k' should hold the key buffer"
    assert torch.is_tensor(c.v), "attribute 'v' should hold the value buffer"
    assert tuple(c.k.shape) == expected, f"k is {tuple(c.k.shape)}, want {expected}"
    assert tuple(c.v.shape) == expected


def test_rung1_length_starts_at_zero():
    c = KVCache.from_config(TINY, batch=B)
    assert c.get_seq_len() == 0
    assert c.is_empty() is True


# ===========================================================================
# RUNG 2 -- pointer accessors
# ===========================================================================


def test_rung2_length_tracks_appends():
    c = KVCache.from_config(TINY, batch=B)
    for n in (1, 3, 6):
        c.append(0, rand_kv(n), rand_kv(n))
        assert c.get_seq_len() == n, f"after appending {n}"
        assert c.is_empty() is False


def test_rung2_reset_rewinds_without_reallocating():
    """reset() must NOT reallocate -- reusing the memory is the entire point."""
    c = KVCache.from_config(TINY, batch=B)
    c.append(0, rand_kv(4), rand_kv(4))
    addr_before = c.k.data_ptr()
    c.reset()
    assert c.get_seq_len() == 0
    assert c.is_empty() is True
    assert c.k.data_ptr() == addr_before, "reset reallocated; it should rewind only"


# ===========================================================================
# RUNG 3 -- get_kv returns LIVE length, not capacity
# ===========================================================================


def test_rung3_get_kv_returns_only_the_live_length():
    """THE trap. Appending 3 tokens must yield a length-3 view, not 512.

    max_position_embeddings is 512 and max_seq_len is 512, so a wrong
    implementation returns something 170x too large and silently feeds
    uninitialised memory into attention.
    """
    c = KVCache.from_config(TINY, batch=B)
    c.append(0, rand_kv(3), rand_kv(3))

    k, v = c.get_kv(0)
    assert k.shape[-2] == 3, (
        f"get_kv returned length {k.shape[-2]}, expected 3. "
        f"You are probably returning the full max_seq_len buffer."
    )
    assert v.shape[-2] == 3
    assert k.shape == (B, TINY.n_kv_heads, 3, TINY.head_dim)


def test_rung3_get_kv_is_empty_before_any_append():
    c = KVCache.from_config(TINY, batch=B)
    k, v = c.get_kv(0)
    assert k.shape[-2] == 0
    assert v.shape[-2] == 0


def test_rung3_layers_hold_independent_data():
    c = KVCache.from_config(TINY, batch=B)
    c.append(0, rand_kv(2), rand_kv(2))
    c.append(1, rand_kv(5), rand_kv(5))

    assert c.get_kv(0)[0].shape[-2] == 2
    assert c.get_kv(1)[0].shape[-2] == 5


# ===========================================================================
# RUNG 4 -- append
# ===========================================================================


def test_rung4_append_preserves_earlier_tokens():
    """A write-position bug corrupts old data. This is what catches it."""
    c = KVCache.from_config(TINY, batch=B)

    k1, v1 = rand_kv(3), rand_kv(3)
    c.append(0, k1, v1)
    k_out, _ = c.get_kv(0)
    assert torch.equal(k_out[:, :, :3], k1), "first append did not store correctly"

    k2, v2 = rand_kv(2), rand_kv(2)
    k_full, v_full = c.append(0, k2, v2)

    assert k_full.shape[-2] == 5, "length should be 3 + 2"
    assert v_full.shape[-2] == 5
    assert torch.equal(k_full[:, :, :3], k1), "append overwrote earlier tokens"
    assert torch.equal(k_full[:, :, 3:], k2), "append stored at the wrong offset"


def test_rung4_append_of_single_token():
    c = KVCache.from_config(TINY, batch=B)
    for i in range(7):
        kf, vf = c.append(0, rand_kv(1), rand_kv(1))
        assert kf.shape[-2] == i + 1
        assert vf.shape[-2] == i + 1


def test_rung4_pointer_is_shared_across_layers():
    """One counter for all layers -- that is what keeps them aligned."""
    c = KVCache.from_config(TINY, batch=B)
    c.append(0, rand_kv(4), rand_kv(4))
    c.append(1, rand_kv(4), rand_kv(4))
    assert c.get_seq_len() == 4, "layers must share one counter"
    assert c.get_kv(0)[0].shape[-2] == 4
    assert c.get_kv(1)[0].shape[-2] == 4


def test_rung4_raises_on_overflow():
    c = KVCache.from_config(TINY, batch=B)
    c.append(0, rand_kv(TINY.max_position_embeddings), rand_kv(TINY.max_position_embeddings))
    with pytest.raises((ValueError, IndexError, RuntimeError, torch.OutOfMemoryError)):
        c.append(0, rand_kv(1), rand_kv(1))


def test_rung4_bytes_allocated_is_capacity_not_live():
    """Capacity is constant. The gap vs live data is what M3 (paging) fixes."""
    c = KVCache.from_config(TINY, batch=B)
    assert c.bytes_allocated() == TINY.kv_bytes(TINY.max_position_embeddings, B, 4)

    c.append(0, rand_kv(8), rand_kv(8))
    assert c.bytes_allocated() == TINY.kv_bytes(TINY.max_position_embeddings, B, 4)
    assert TINY.kv_bytes(8, B, 4) < c.bytes_allocated()


# ===========================================================================
# RUNG 5 -- DecoderLayer.forward_cached
# ===========================================================================


def test_rung5_cached_layer_step_matches_full_forward(cos_sin):
    """Prefill 5 tokens fully, then take the 5th step from cache.

    The cache is filled by hand here so this test does not depend on rung 6.
    """
    torch.manual_seed(11)
    cfg = TINY
    cos, sin = cos_sin
    layer = DecoderLayer(cfg, 0).eval()

    x = torch.randn(B, PROMPT_LEN, cfg.d_model)

    # Reference: the complete prefill path, which already works.
    with torch.no_grad():
        expected = layer(x, cos, sin)

    # Fill the cache with the first 4 tokens, computing K/V exactly as the
    # layer would, then take step 5 through forward_cached.
    cache = KVCache.from_config(cfg, batch=B)
    prefix = x[:, : PROMPT_LEN - 1]
    with torch.no_grad():
        h = rms_norm(prefix, layer.attn_norm)
        _, k, v = project_qkv(
            h, layer.w_q, layer.w_k, layer.w_v,
            cfg.n_heads, cfg.n_kv_heads, cos[: PROMPT_LEN - 1], sin[: PROMPT_LEN - 1],
        )
        cache.append(0, k, v)
        last = PROMPT_LEN - 1
        got = layer.forward_cached(
            x[:, last : last + 1], cos[last : last + 1], sin[last : last + 1], cache, 0
        )

    assert got.shape == (B, 1, cfg.d_model)
    assert torch.allclose(got, expected[:, last : last + 1], atol=1e-4), (
        f"max diff {(got - expected[:, last:last+1]).abs().max().item():.3e}"
    )


def test_rung5_cache_stores_kv_heads_not_query_heads():
    """TOY has 4:1 GQA. Caching n_heads would waste 4x the memory."""
    from nanoinfer.config import TOY

    c = KVCache.from_config(TOY, batch=1)
    k = torch.randn(1, TOY.n_kv_heads, 2, TOY.head_dim)
    kf, _ = c.append(0, k, k)
    assert kf.shape[1] == TOY.n_kv_heads, (
        f"cached {kf.shape[1]} heads, but n_kv_heads is {TOY.n_kv_heads}"
    )
    assert kf.shape[1] != TOY.n_heads


# ===========================================================================
# RUNG 6 -- NanoTransformer.forward_step
# ===========================================================================


def test_rung6_forward_step_prefill_matches_forward():
    """Prefilling THROUGH THE CACHE must equal the plain forward.

    This is where the shared-pointer bug appears: if the RoPE offset is
    recomputed inside the layer loop, each layer places the token at a
    different position and this test fails with no error raised.
    """
    torch.manual_seed(12)
    m = NanoTransformer(TINY).eval()
    ids = torch.randint(0, TINY.vocab_size, (B, PROMPT_LEN))

    with torch.no_grad():
        expected = m(ids)
        cache = KVCache.from_config(TINY, batch=B)
        got = m.forward_step(ids, cache)

    assert got.shape == expected.shape
    assert torch.allclose(got, expected, atol=1e-4), (
        f"max diff {(got - expected).abs().max().item():.3e}"
    )
    assert cache.get_seq_len() == PROMPT_LEN


def test_rung6_forward_step_decode_matches_forward_slice():
    """One decode step after a prefill matches the matching slice of a full forward."""
    torch.manual_seed(13)
    m = NanoTransformer(TINY).eval()
    ids = torch.randint(0, TINY.vocab_size, (B, PROMPT_LEN))

    with torch.no_grad():
        expected = m(ids)

        cache = KVCache.from_config(TINY, batch=B)
        m.forward_step(ids[:, :PROMPT_LEN - 1], cache)
        assert cache.get_seq_len() == PROMPT_LEN - 1
        got = m.forward_step(ids[:, PROMPT_LEN - 1 :], cache)

    assert cache.get_seq_len() == PROMPT_LEN
    assert torch.allclose(got, expected[:, PROMPT_LEN - 1 :], atol=1e-4), (
        f"max diff {(got - expected[:, PROMPT_LEN - 1:]).abs().max().item():.3e}"
    )


def test_rung6_reset_allows_reuse_for_a_second_sequence():
    torch.manual_seed(14)
    m = NanoTransformer(TINY).eval()
    cache = KVCache.from_config(TINY, batch=B)

    a = torch.randint(0, TINY.vocab_size, (B, PROMPT_LEN))
    b = torch.randint(0, TINY.vocab_size, (B, PROMPT_LEN))

    with torch.no_grad():
        first = m.forward_step(a, cache)
        cache.reset()
        second = m.forward_step(b, cache)

    with torch.no_grad():
        assert torch.allclose(first, m(a), atol=1e-4)
        assert torch.allclose(second, m(b), atol=1e-4)