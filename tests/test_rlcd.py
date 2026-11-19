"""Unit tests for Milestone 1 of RLCD Architecture: Attention Geometry and Single-Pass Evaluation."""

import torch
import pytest
from experiments.bench_rlcd_vs_causal import BidirectionalDecisionModel, CausalTransformer


def test_rlcd_single_pass_shape():
    """Verify that an RLCD decision model produces calibrated probabilities over classes."""
    vocab_size = 1000
    dim = 64
    n_heads = 4
    n_layers = 2
    num_classes = 3
    batch_size = 4
    seq_len = 32

    model = BidirectionalDecisionModel(
        vocab_size=vocab_size,
        dim=dim,
        n_heads=n_heads,
        n_layers=n_layers,
        num_classes=num_classes,
    )

    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    probs = model(x)

    # 1. Output shape must be [Batch, num_classes]
    assert probs.shape == (batch_size, num_classes)
    
    # 2. Output must be a valid probability simplex (sum to 1.0 across classes)
    sums = probs.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5)
    
    # 3. All probabilities must be in range [0, 1]
    assert (probs >= 0.0).all() and (probs <= 1.0).all()


def test_rlcd_vs_causal_invariants():
    """Test that RLCD models require 0 KV cache while causal models require dynamic KV cache."""
    vocab_size = 500
    dim = 64
    n_heads = 4
    n_layers = 2

    causal = CausalTransformer(vocab_size, dim, n_heads, n_layers)
    rlcd = BidirectionalDecisionModel(vocab_size, dim, n_heads, n_layers, num_classes=2)

    prompt = torch.randint(0, vocab_size, (1, 16))

    # Causal step returns a non-empty KV cache
    _, kv_cache = causal.forward_step(prompt)
    assert kv_cache is not None
    assert len(kv_cache) == n_layers
    k, v = kv_cache[0]
    assert k.numel() > 0 and v.numel() > 0

    # RLCD returns pure probability tensor without cache state
    probs = rlcd(prompt)
    assert isinstance(probs, torch.Tensor)
    assert probs.shape == (1, 2)
