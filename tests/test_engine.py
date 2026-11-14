"""
tests/test_engine.py - Tests for the custom LLM inference engine.
Verifies all building blocks:
- RMSNorm
- RoPE
- GQA with KV Cache
- SwiGLU
- Full Transformer forward pass (prefill & decode)
- Sampling & Generation
"""

import pytest
import torch
from nanoinfer.engine import (
    RMSNorm,
    precompute_rope_freqs,
    apply_rope,
    GroupedQueryAttention,
    SwiGLU,
    DecoderLayer,
    SmolLMTransformer,
    sample_next_token,
    generate_autoregressive,
)


def test_rmsnorm():
    norm = RMSNorm(dim=64)
    x = torch.randn(2, 4, 64)
    out = norm(x)
    assert out.shape == x.shape
    # Check that scale matches parameter
    assert norm.weight.shape == (64,)


def test_rope_rotation():
    cos, sin = precompute_rope_freqs(head_dim=32, max_seq_len=64)
    q = torch.randn(1, 4, 8, 32)  # [batch, heads, seq, head_dim]
    q_rot = apply_rope(q, cos, sin, start_pos=0)
    assert q_rot.shape == q.shape
    # RoPE should preserve vector norms approximately
    norm_before = q.norm(dim=-1)
    norm_after = q_rot.norm(dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-5)


def test_gqa_kv_cache():
    attn = GroupedQueryAttention(hidden_size=64, num_heads=4, num_kv_heads=2)
    cos, sin = precompute_rope_freqs(head_dim=16, max_seq_len=64)
    
    # Prefill step
    x_prompt = torch.randn(1, 5, 64)
    out, cache = attn(x_prompt, cos, sin, start_pos=0, kv_cache=None)
    assert out.shape == (1, 5, 64)
    assert cache[0].shape == (1, 2, 5, 16) # [batch, kv_heads, seq, head_dim]
    
    # Decode step (1 token)
    x_next = torch.randn(1, 1, 64)
    out_next, cache_next = attn(x_next, cos, sin, start_pos=5, kv_cache=cache)
    assert out_next.shape == (1, 1, 64)
    assert cache_next[0].shape == (1, 2, 6, 16) # seq length incremented to 6!


def test_swiglu():
    mlp = SwiGLU(hidden_size=64, intermediate_size=128)
    x = torch.randn(2, 3, 64)
    out = mlp(x)
    assert out.shape == (2, 3, 64)


def test_full_model_prefill_and_decode():
    model = SmolLMTransformer(
        vocab_size=200,
        hidden_size=64,
        intermediate_size=128,
        num_layers=2,
        num_heads=4,
        num_kv_heads=2,
        max_seq_len=64
    )
    prompt = torch.randint(0, 200, (1, 6))
    
    # Prefill
    logits, kv_caches = model(prompt, start_pos=0)
    assert logits.shape == (1, 6, 200)
    assert len(kv_caches) == 2
    assert kv_caches[0][0].shape[-2] == 6
    
    # Decode 1 token
    token = torch.tensor([[42]])
    next_logits, next_caches = model(token, start_pos=6, kv_caches=kv_caches)
    assert next_logits.shape == (1, 1, 200)
    assert next_caches[0][0].shape[-2] == 7


def test_sampling_and_generation():
    model = SmolLMTransformer(
        vocab_size=100,
        hidden_size=32,
        intermediate_size=64,
        num_layers=2,
        num_heads=2,
        num_kv_heads=1,
        max_seq_len=32
    )
    prompt = torch.tensor([[10, 20, 30]])
    
    # Generate 5 new tokens with greedy decode
    tokens = generate_autoregressive(
        model,
        prompt,
        max_new_tokens=5,
        temperature=0.0
    )
    assert len(tokens) == 3 + 5 # 3 prompt tokens + 5 generated
