"""Benchmark: Causal Autoregressive LLM vs. Bidirectional RLCD Decision Model.

Answers Milestone 1 of the RLCD Architecture Roadmap:
"Why does answering a structured question with a generative LLM take 10x longer
and require gigabytes of KV cache, while an RLCD decision transformer answers
in a single pass with zero cache?"

Measures:
1. Number of forward passes (N sequential steps vs. 1 single-shot pass).
2. KV Cache memory overhead (bytes allocated vs. 0 bytes).
3. Wall-clock latency (ms) on CPU.
"""

from __future__ import annotations

import time
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# 1. Classical Causal Decoder (Autoregressive Generation)
# -----------------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, S, _ = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)

        q = q.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            k = torch.cat([prev_k, k], dim=2)
            v = torch.cat([prev_v, v], dim=2)

        new_cache = (k, v)

        # Causal triangular mask
        total_len = k.size(2)
        mask = torch.triu(torch.full((S, total_len), float("-inf"), device=x.device), diagonal=total_len - S + 1)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        attn_weights = F.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)

        out = out.transpose(1, 2).contiguous().view(B, S, self.dim)
        return self.out_proj(out), new_cache


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_heads: int, n_layers: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([CausalSelfAttention(dim, n_heads) for _ in range(n_layers)])
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

    def forward_step(
        self,
        token_ids: torch.Tensor,
        past_caches: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        x = self.embed(token_ids)
        new_caches = []
        for i, layer in enumerate(self.layers):
            c = past_caches[i] if past_caches is not None else None
            x, updated_c = layer(x, kv_cache=c)
            new_caches.append(updated_c)
        logits = self.lm_head(x[:, -1:, :])
        return logits, new_caches


# -----------------------------------------------------------------------------
# 2. Bidirectional RLCD Decision Model (Jev / Laya Style)
# -----------------------------------------------------------------------------
class BidirectionalDecisionModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int, n_heads: int, n_layers: int, num_classes: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        # Standard unmasked bidirectional attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=n_heads,
            dim_feedforward=dim * 4,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Compact decision head: dim -> num_classes (e.g. 2 for YES/NO)
        self.decision_head = nn.Linear(dim, num_classes, bias=True)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # 1 single forward pass across full context
        h = self.embed(input_ids)
        # Bidirectional cross-attention (no causal mask!)
        encoded = self.encoder(h)
        # Mean pooling across sequence tokens
        pooled = encoded.mean(dim=1)
        # Project directly to decision simplex
        logits = self.decision_head(pooled)
        probs = F.softmax(logits, dim=-1)
        return probs


# -----------------------------------------------------------------------------
# 3. Measurement & Benchmark Execution
# -----------------------------------------------------------------------------
def run_benchmark():
    torch.manual_seed(42)
    device = "cpu"

    vocab_size = 32000
    dim = 512
    n_heads = 8
    n_layers = 6
    seq_len = 128
    decode_tokens = 16  # Tokens needed to generate a structured text answer

    print("=" * 80)
    print("RLCD MILESTONE 1: ATTENTION GEOMETRY & LATENCY MEASUREMENT")
    print("=" * 80)
    print(f"Configuration: Layers={n_layers}, Dim={dim}, Heads={n_heads}")
    print(f"Context Length: {seq_len} tokens | Target Answer: {decode_tokens} tokens\n")

    # Instantiate both models
    causal_model = CausalTransformer(vocab_size, dim, n_heads, n_layers).to(device).eval()
    rlcd_model = BidirectionalDecisionModel(vocab_size, dim, n_heads, n_layers, num_classes=2).to(device).eval()

    # Synthetic prompt tokens
    prompt_tokens = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # ---------------------------------------------------------
    # Path A: Causal Autoregressive Generation
    # ---------------------------------------------------------
    t0 = time.perf_counter()
    # 1. Prefill step
    logits, kv_cache = causal_model.forward_step(prompt_tokens)
    causal_passes = 1
    
    # 2. Sequential decode steps
    curr_token = torch.argmax(logits, dim=-1)
    for _ in range(decode_tokens - 1):
        logits, kv_cache = causal_model.forward_step(curr_token, past_caches=kv_cache)
        curr_token = torch.argmax(logits, dim=-1)
        causal_passes += 1
    t_causal = (time.perf_counter() - t0) * 1000

    # Calculate KV cache memory footprint
    k_sample, v_sample = kv_cache[0]
    bytes_per_layer = (k_sample.element_size() * k_sample.nelement()) + (v_sample.element_size() * v_sample.nelement())
    total_kv_bytes = bytes_per_layer * n_layers

    # ---------------------------------------------------------
    # Path B: RLCD Bidirectional Decision Pass
    # ---------------------------------------------------------
    t1 = time.perf_counter()
    probs = rlcd_model(prompt_tokens)
    t_rlcd = (time.perf_counter() - t1) * 1000
    rlcd_passes = 1
    rlcd_kv_bytes = 0

    # ---------------------------------------------------------
    # Results & Insights
    # ---------------------------------------------------------
    speedup = t_causal / t_rlcd

    print("-" * 80)
    print(f"{'Metric':<30} | {'Causal Generative LLM':<22} | {'RLCD Decision Model':<22}")
    print("-" * 80)
    print(f"{'Forward Passes':<30} | {causal_passes:<22} | {rlcd_passes:<22}")
    print(f"{'KV Cache Memory':<30} | {f'{total_kv_bytes / 1024:.2f} KB':<22} | {f'{rlcd_kv_bytes} Bytes':<22}")
    print(f"{'Latency (ms)':<30} | {f'{t_causal:.2f} ms':<22} | {f'{t_rlcd:.2f} ms':<22}")
    print(f"{'Speedup':<30} | {'1.00x (Baseline)':<22} | {f'{speedup:.2f}x Faster!':<22}")
    print("-" * 80)
    print(f"\nFinal Calibrated Output: P(True) = {probs[0, 1].item():.4f}, P(False) = {probs[0, 0].item():.4f}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
