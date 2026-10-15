"""Correctness tests for the analytical cost model.

These are the tests that matter early: if the parameter counts or FLOP formulas
are wrong, every conclusion drawn from them is wrong, and nothing downstream
will catch it.
"""

from __future__ import annotations

import pytest

from nanoinfer import config as cfgmod
from nanoinfer import hardware as hwmod
from nanoinfer import roofline as rl


# --- parameter accounting --------------------------------------------------


@pytest.mark.parametrize(
    "cfg,expected",
    [
        (cfgmod.LLAMA3_8B, 8.03e9),
        (cfgmod.LLAMA3_70B, 70.6e9),
        (cfgmod.MISTRAL_7B, 7.24e9),
        (cfgmod.GPT2_SMALL, 124.4e6),
    ],
    ids=lambda v: getattr(v, "name", str(v)),
)
def test_param_counts_match_published_values(cfg, expected):
    """Guards the architecture table against typos.

    Tolerances are ~1%: published totals often exclude or include the LM head
    and any post-training adjustments differently.
    """
    assert cfg.n_params == pytest.approx(expected, rel=0.01)


def test_gqa_ratio():
    assert cfgmod.LLAMA3_8B.gqa_ratio == 4
    assert cfgmod.MISTRAL_7B.gqa_ratio == 4
    assert cfgmod.QWEN2_5_0_5B.gqa_ratio == 7
    assert cfgmod.GPT2_SMALL.gqa_ratio == 1


def test_ffn_shapes_differ():
    swiglu = cfgmod.LLAMA3_8B
    mlp = cfgmod.GPT2_SMALL
    assert swiglu.params_ffn_per_layer == 3 * swiglu.d_model * swiglu.d_ff
    assert mlp.params_ffn_per_layer == 2 * mlp.d_model * mlp.d_ff


def test_tied_embeddings_have_no_lm_head():
    assert cfgmod.GPT2_SMALL.params_lm_head == 0
    assert cfgmod.LLAMA3_8B.params_lm_head == cfgmod.LLAMA3_8B.vocab_size * 4096


def test_rejects_bad_geometry():
    with pytest.raises(ValueError, match="divisible"):
        cfgmod.TransformerConfig("bad", 100, 2, 3, 3, 64, 1000)
    with pytest.raises(ValueError, match="divisible"):
        cfgmod.TransformerConfig("bad", 128, 2, 12, 5, 64, 1000)


# --- KV cache --------------------------------------------------------------


def test_kv_bytes_per_token_formula():
    cfg = cfgmod.LLAMA3_8B
    expected = 2 * cfg.n_kv_heads * cfg.head_dim * 2 * cfg.n_layers
    assert cfg.kv_bytes_per_token(2) == expected
    # 8 KV heads * 128 dim * 2 bytes * 2 (K and V) * 32 layers = 128 KiB
    assert cfg.kv_bytes_per_token(2) == 128 * 1024


def test_gqa_shrinks_kv_cache_by_the_group_ratio():
    """8 KV heads instead of 32 means 4x less cache. This is the point of GQA."""
    llama3 = cfgmod.LLAMA3_8B
    mha_equivalent = cfgmod.TransformerConfig(
        "mha", llama3.d_model, llama3.n_layers, llama3.n_heads,
        llama3.n_heads, llama3.d_ff, llama3.vocab_size,
    )
    ratio = mha_equivalent.kv_bytes_per_token(2) / llama3.kv_bytes_per_token(2)
    assert ratio == llama3.gqa_ratio


def test_kv_quantization_halves_the_cache():
    cfg = cfgmod.LLAMA3_8B
    assert cfg.kv_bytes_per_token(1) == cfg.kv_bytes_per_token(2) / 2


def test_concurrency_scales_with_budget():
    cfg = cfgmod.QWEN2_5_0_5B
    one = cfg.max_batch_at_full_context(1_000_000_000, 2)
    two = cfg.max_batch_at_full_context(2_000_000_000, 2)
    assert two == pytest.approx(one * 2, rel=0.02)


# --- FLOP formulas ---------------------------------------------------------


def test_linear_flops_is_two_per_parameter_per_token():
    cfg = cfgmod.TINY
    w = rl.linear_flops(cfg, 7, include_lm_head=True)
    assert w == 2 * cfg.params_blocks * 7  # tied embeddings -> no lm_head params


def test_attention_flops_linear_in_context_when_decoding():
    """One query against N keys is O(N), not O(N^2)."""
    cfg = cfgmod.TINY
    a = rl.attention_flops(cfg, 1, 512)
    b = rl.attention_flops(cfg, 1, 1024)
    assert b == pytest.approx(a * 2, rel=1e-9)


def test_attention_flops_quadratic_when_prefilling():
    """S queries against S keys is O(S^2). Doubling context costs 4x."""
    cfg = cfgmod.TINY
    a = rl.attention_flops(cfg, 512, 512)
    b = rl.attention_flops(cfg, 1024, 1024)
    assert b == pytest.approx(a * 4, rel=1e-9)


def test_attention_is_a_minority_of_flops_inside_the_context_window():
    """A common misconception, corrected by the model itself.

    Intuitively attention feels like the expensive part. FLOP-wise it is not:
    weights dominate until roughly S = params / (2 * d_model * n_layers), which
    for Llama-3-8B is ~27K tokens -- well outside its 8K window.
    """
    cfg = cfgmod.LLAMA3_8B
    s = cfg.max_position_embeddings
    linear = rl.linear_flops(cfg, s, include_lm_head=False)
    attn = rl.attention_flops(cfg, s, s)
    assert attn < linear

    crossover = cfg.params_blocks / (2 * cfg.d_model * cfg.n_layers)
    assert 20_000 < crossover < 35_000
    assert crossover > cfg.max_position_embeddings


def test_kv_bytes_scale_with_context_while_weight_bytes_do_not():
    """The reason long context hurts even when FLOPs look fine.

    Weight traffic is constant per token. KV traffic grows without bound, so at
    some context length it overtakes weights entirely and decode becomes
    dominated by cache reads. That is the regime where KV quantization (M5)
    and paging (M3) pay for themselves.
    """
    cfg = cfgmod.LLAMA3_8B
    weight_bytes = cfg.n_params_matmul * 2

    assert cfg.kv_bytes(1024) < weight_bytes
    # Crossover happens far out, well past the 8K training window.
    assert cfg.kv_bytes(131072) > weight_bytes

    # And it is strictly linear in context.
    assert cfg.kv_bytes(4096) == pytest.approx(4 * cfg.kv_bytes(1024), rel=1e-9)


def test_flop_split_adds_up_to_the_decode_total():
    """Guards against double-counting or dropping a term."""
    cfg = cfgmod.LLAMA3_8B
    b, ctx = 8, 2048
    w = rl.decode_workload(cfg, b, ctx)
    parts = (
        rl.linear_flops(cfg, b, include_lm_head=True)
        + rl.attention_flops(cfg, 1, ctx, b)
    )
    assert w.flops == pytest.approx(parts, rel=1e-12)


def test_attention_flops_zero_without_context():
    assert rl.attention_flops(cfgmod.TINY, 1, 0) == 0.0


# --- the headline claim ----------------------------------------------------


def test_decode_at_batch_one_is_about_one_flop_per_byte():
    """The single most important number in this repo."""
    cfg = cfgmod.LLAMA3_8B
    w = rl.decode_workload(cfg, batch=1, ctx_len=0, dtype_bytes=2)
    # 2*P FLOPs against P*2 bytes -> exactly 1.0 FLOP/byte.
    assert w.arithmetic_intensity == pytest.approx(1.0, rel=1e-6)


def test_decode_intensity_rises_with_batch():
    cfg = cfgmod.LLAMA3_8B
    intensities = [
        rl.decode_workload(cfg, b, ctx_len=2048).arithmetic_intensity
        for b in (1, 8, 64, 512)
    ]
    assert intensities == sorted(intensities)


def test_batch_one_decode_is_memory_bound_on_every_real_gpu():
    cfg = cfgmod.LLAMA3_8B
    for hw in hwmod.ALL:
        w = rl.decode_workload(cfg, batch=1, ctx_len=2048)
        assert rl.roofline(w, hw).bound == "memory", hw.name


def test_prefill_becomes_compute_bound_as_prompts_grow():
    cfg = cfgmod.LLAMA3_8B
    hw = hwmod.H100_SXM
    short = rl.prefill_workload(cfg, 1, 64)
    long = rl.prefill_workload(cfg, 1, 8192)
    assert rl.roofline(short, hw).bound == "memory"
    assert rl.roofline(long, hw).bound == "compute"


def test_prefill_intensity_scales_linearly_with_seq_len():
    """Amortised weights mean intensity ~ seq_len FLOP/byte."""
    cfg = cfgmod.LLAMA3_8B
    small = rl.prefill_workload(cfg, 1, 256).arithmetic_intensity
    large = rl.prefill_workload(cfg, 1, 256 * 4).arithmetic_intensity
    assert large == pytest.approx(small * 4, rel=0.05)


# --- roofline --------------------------------------------------------------


def test_ridge_point_matches_definition():
    hw = hwmod.H100_SXM
    assert hw.ridge_point == pytest.approx(989e12 / 3350e9)


def test_roofline_respects_both_ceilings():
    hw = hwmod.H100_SXM
    heavy = rl.Workload("heavy", 1e15, 1e9)  # intensity 1e6 -> compute-bound
    assert rl.roofline(heavy, hw).achieved_tflops == hw.peak_tflops
    assert rl.roofline(heavy, hw).bound == "compute"

    # Achievable throughput is intensity * bandwidth, floored at the ridge point.
    for intensity in (0.1, 1.0, 10.0):
        w = rl.Workload("x", intensity * 1e9, 1e9)
        expected = min(hw.peak_tflops, intensity * hw.bandwidth_gbs / 1000)
        assert rl.roofline(w, hw).achieved_tflops == pytest.approx(expected)


def test_memory_bound_workload_below_ridge_is_scaled_by_intensity():
    """At intensity 1 the device delivers exactly its bandwidth, in FLOP/s."""
    hw = hwmod.H100_SXM
    w = rl.Workload("decode-like", 1e9, 1e9)  # intensity exactly 1.0
    r = rl.roofline(w, hw)
    assert r.bound == "memory"
    assert r.achieved_tflops == pytest.approx(hw.bandwidth_gbs / 1000)


def test_decode_of_a_real_model_hits_fraction_of_compute_peak():
    """Sanity anchor: batch-1 decode uses a tiny sliver of the tensor cores.

    A useful number to carry in your head. If a change does not move this ratio,
    it is not addressing the real bottleneck.
    """
    hw = hwmod.H100_SXM
    w = rl.decode_workload(cfgmod.LLAMA3_8B, batch=1, ctx_len=2048)
    r = rl.roofline(w, hw)
    assert r.bound == "memory"
    assert 0.001 < r.achieved_tflops / hw.peak_tflops < 0.05


def test_ridge_intensity_is_exactly_the_boundary():
    hw = hwmod.A100_80GB
    at_ridge = rl.Workload("at", hw.ridge_point * 1e9, 1e9)
    assert rl.roofline(at_ridge, hw).bound == "compute"
    just_below = rl.Workload("below", (hw.ridge_point * 0.99) * 1e9, 1e9)
    assert rl.roofline(just_below, hw).bound == "memory"


def test_seconds_is_flops_over_achieved_throughput():
    hw = hwmod.L4
    w = rl.decode_workload(cfgmod.MISTRAL_7B, batch=32, ctx_len=2048)
    r = rl.roofline(w, hw)
    assert r.seconds == pytest.approx(w.flops / (r.achieved_tflops * 1e12))


def test_hardware_lookup_is_forgiving():
    assert hwmod.get("h100 sxm") is hwmod.H100_SXM
    assert hwmod.get("A100-80GB") is hwmod.A100_80GB
    with pytest.raises(KeyError):
        hwmod.get("nonexistent-gpu")


def test_model_lookup_is_case_insensitive():
    assert cfgmod.get("llama-3-8b") is cfgmod.LLAMA3_8B
    with pytest.raises(KeyError):
        cfgmod.get("not-a-model")