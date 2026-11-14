"""
src/nanoinfer/engine.py - Production-grade, zero-black-box LLM inference engine.
Implements the exact LLaMA-3 / SmolLM-135M architecture:
1. RMSNorm
2. Rotary Position Embeddings (RoPE)
3. Grouped-Query Attention (GQA) with dynamic KV Cache
4. SwiGLU Feed-Forward Network
5. Zero-copy Safetensors loader (handling tied word embeddings)
6. Dual-phase generation loop (Prefill + Autoregressive Decode)
"""

import math
import os
import json
import time
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


# ============================================================================
# BLOCK 1: RMSNorm (Root Mean Square Layer Normalization)
# ============================================================================
class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).
    Used in LLaMA-3, SmolLM, Mistral, and DeepSeek.
    Formula: y = (x / RMS(x)) * weight
    Where RMS(x) = sqrt(mean(x^2) + eps)
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_f32 = x.to(torch.float32)
        variance = x_f32.pow(2).mean(dim=-1, keepdim=True)
        normed = x_f32 * torch.rsqrt(variance + self.eps)
        return (normed.to(input_dtype)) * self.weight


# ============================================================================
# BLOCK 2: RoPE (Rotary Position Embeddings)
# ============================================================================
def precompute_rope_freqs(head_dim: int, max_seq_len: int = 2048, theta: float = 10000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precomputes cos and sin rotation tensors for RoPE.
    theta_i = 1.0 / (theta ** (2i / head_dim)) for i in [0, 1, ..., head_dim/2 - 1]
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE pairwise rotation"
    i = torch.arange(0, head_dim, 2).float()
    freqs = 1.0 / (theta ** (i / head_dim))
    t = torch.arange(max_seq_len).float()
    angles = torch.outer(t, freqs)
    
    # Repeat interleave so each freq matches both (x_2i, x_2i+1)
    cos = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates pairs: [-x1, x0, -x3, x2, ...]"""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
    """
    Applies 2D rotation to Query or Key.
    x shape: [batch, n_heads, seq_len, head_dim]
    """
    seq_len = x.shape[-2]
    cos_slice = cos[start_pos : start_pos + seq_len, :].to(x.device, dtype=x.dtype)
    sin_slice = sin[start_pos : start_pos + seq_len, :].to(x.device, dtype=x.dtype)
    
    # Broadcast across [batch, n_heads, seq_len, head_dim]
    cos_slice = cos_slice.unsqueeze(0).unsqueeze(0)
    sin_slice = sin_slice.unsqueeze(0).unsqueeze(0)
    
    return (x * cos_slice) + (rotate_half(x) * sin_slice)


# ============================================================================
# BLOCK 3: Grouped-Query Attention (GQA) with Dynamic KV Cache
# ============================================================================
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) with dynamic Key-Value Cache.
    In SmolLM-135M:
      num_heads = 9
      num_kv_heads = 3  (3 Query heads share 1 KV head)
      head_dim = 64
      hidden_size = 576
    """
    def __init__(self, hidden_size: int = 576, num_heads: int = 9, num_kv_heads: int = 3):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads
        self.num_kv_groups = num_heads // num_kv_heads

        # Weight Projections
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Step 1: Linear projections
        # Q: [batch, seq_len, n_heads * head_dim] -> [batch, n_heads, seq_len, head_dim]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V: [batch, seq_len, n_kv_heads * head_dim] -> [batch, n_kv_heads, seq_len, head_dim]
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Step 2: Apply RoPE to Q and K (Value is NEVER rotated!)
        q = apply_rope(q, cos, sin, start_pos=start_pos)
        k = apply_rope(k, cos, sin, start_pos=start_pos)

        # Step 3: Dynamic KV Cache management
        if kv_cache is not None:
            k_past, v_past = kv_cache
            # Concatenate past keys/values with current ones along sequence dimension
            k = torch.cat([k_past, k], dim=-2)
            v = torch.cat([v_past, v], dim=-2)
        new_kv_cache = (k, v)

        # Step 4: GQA expansion - repeat KV heads to match Q head count
        # shape: [batch, n_heads, total_seq_len, head_dim]
        k_expanded = k.repeat_interleave(self.num_kv_groups, dim=1)
        v_expanded = v.repeat_interleave(self.num_kv_groups, dim=1)

        # Step 5: Scaled Dot-Product Attention: (Q @ K^T) / sqrt(d_k)
        # Q: [batch, heads, seq_len, head_dim]
        # K^T: [batch, heads, head_dim, total_seq_len]
        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Step 6: Causal Masking (only during prefill when seq_len > 1)
        total_seq_len = k.shape[-2]
        if seq_len > 1:
            # Mask out upper triangle: token i cannot see token j where j > i
            # diagonal offset matches the start_pos
            mask = torch.full((seq_len, total_seq_len), float("-inf"), device=x.device)
            mask = torch.triu(mask, diagonal=1 + (total_seq_len - seq_len))
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        # Softmax over key sequence dimension
        attn_probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

        # Step 7: Attention output @ V
        # output shape: [batch, n_heads, seq_len, head_dim]
        attn_out = torch.matmul(attn_probs, v_expanded)

        # Step 8: Reshape back to [batch, seq_len, hidden_size] and project with W_o
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_out), new_kv_cache


# ============================================================================
# BLOCK 4: SwiGLU (Swish-Gated Linear Unit) MLP
# ============================================================================
class SwiGLU(nn.Module):
    """
    SwiGLU Feed-Forward Network used in LLaMA and SmolLM.
    Formula: y = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """
    def __init__(self, hidden_size: int = 576, intermediate_size: int = 1536):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Element-wise product of SiLU activation and linear projection
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# BLOCK 5: Decoder Layer
# ============================================================================
class DecoderLayer(nn.Module):
    """
    Full Transformer Decoder Layer combining:
    1. Input RMSNorm + Grouped-Query Attention with residual
    2. Post-Attention RMSNorm + SwiGLU MLP with residual
    """
    def __init__(self, hidden_size: int = 576, num_heads: int = 9, num_kv_heads: int = 3, intermediate_size: int = 1536, eps: float = 1e-5):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.self_attn = GroupedQueryAttention(hidden_size, num_heads, num_kv_heads)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        start_pos: int = 0,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Residual branch 1: Attention
        normed_x = self.input_layernorm(x)
        attn_out, new_kv_cache = self.self_attn(normed_x, cos, sin, start_pos=start_pos, kv_cache=kv_cache)
        x = x + attn_out

        # Residual branch 2: MLP
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_kv_cache


# ============================================================================
# BLOCK 6: Full Transformer Model
# ============================================================================
class SmolLMTransformer(nn.Module):
    """
    Complete LLaMA-3 / SmolLM-135M Transformer Architecture.
    """
    def __init__(
        self,
        vocab_size: int = 49152,
        hidden_size: int = 576,
        intermediate_size: int = 1536,
        num_layers: int = 30,
        num_heads: int = 9,
        num_kv_heads: int = 3,
        rms_norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_seq_len: int = 2048,
        tie_word_embeddings: bool = True
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.tie_word_embeddings = tie_word_embeddings

        # 1. Token Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # 2. Precomputed RoPE frequencies
        cos, sin = precompute_rope_freqs(hidden_size // num_heads, max_seq_len, rope_theta)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # 3. Stack of 30 Decoder Layers
        self.layers = nn.ModuleList([
            DecoderLayer(hidden_size, num_heads, num_kv_heads, intermediate_size, eps=rms_norm_eps)
            for _ in range(num_layers)
        ])

        # 4. Final RMSNorm
        self.norm = RMSNorm(hidden_size, eps=rms_norm_eps)

        # 5. Output Projection Head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        if tie_word_embeddings:
            # Re-use the embedding weights directly to save VRAM!
            self.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        start_pos: int = 0,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        # input_ids: [batch, seq_len]
        x = self.embed_tokens(input_ids)

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = layer(x, self.cos, self.sin, start_pos=start_pos, kv_cache=layer_cache)
            new_kv_caches.append(new_cache)

        # Final norm and logit projection
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_kv_caches


# ============================================================================
# BLOCK 7: Zero-Copy Safetensors Weight Loader
# ============================================================================
def load_weights_from_safetensors(model: SmolLMTransformer, safetensors_path: str, device: str = "cpu"):
    """
    Loads raw binary weights from Hugging Face .safetensors directly into our model.
    Handles tied word embeddings automatically if lm_head.weight is absent from disk.
    """
    print(f"[Loader] Opening safetensors file: {safetensors_path}")
    t0 = time.time()
    loaded_keys = 0

    with safe_open(safetensors_path, framework="pt", device=device) as f:
        # Load embedding weights
        if "model.embed_tokens.weight" in f.keys():
            model.embed_tokens.weight.data.copy_(f.get_tensor("model.embed_tokens.weight"))
            loaded_keys += 1

        # Load each of the 30 decoder layers
        for i in range(model.num_layers):
            prefix = f"model.layers.{i}."
            layer = model.layers[i]

            # Attention weights
            layer.input_layernorm.weight.data.copy_(f.get_tensor(f"{prefix}input_layernorm.weight"))
            layer.self_attn.q_proj.weight.data.copy_(f.get_tensor(f"{prefix}self_attn.q_proj.weight"))
            layer.self_attn.k_proj.weight.data.copy_(f.get_tensor(f"{prefix}self_attn.k_proj.weight"))
            layer.self_attn.v_proj.weight.data.copy_(f.get_tensor(f"{prefix}self_attn.v_proj.weight"))
            layer.self_attn.o_proj.weight.data.copy_(f.get_tensor(f"{prefix}self_attn.o_proj.weight"))

            # MLP weights
            layer.post_attention_layernorm.weight.data.copy_(f.get_tensor(f"{prefix}post_attention_layernorm.weight"))
            layer.mlp.gate_proj.weight.data.copy_(f.get_tensor(f"{prefix}mlp.gate_proj.weight"))
            layer.mlp.up_proj.weight.data.copy_(f.get_tensor(f"{prefix}mlp.up_proj.weight"))
            layer.mlp.down_proj.weight.data.copy_(f.get_tensor(f"{prefix}mlp.down_proj.weight"))
            loaded_keys += 9

        # Load final norm
        if "model.norm.weight" in f.keys():
            model.norm.weight.data.copy_(f.get_tensor("model.norm.weight"))
            loaded_keys += 1

        # Check for lm_head.weight (Tied Embedding scenario!)
        if "lm_head.weight" in f.keys():
            model.lm_head.weight.data.copy_(f.get_tensor("lm_head.weight"))
            loaded_keys += 1
            print("[Loader] Loaded explicit 'lm_head.weight'.")
        else:
            print("[Loader] 'lm_head.weight' not in file -> Tying lm_head.weight to embed_tokens.weight (Standard SmolLM behavior).")
            model.lm_head.weight = model.embed_tokens.weight

    elapsed = time.time() - t0
    print(f"[Loader] Successfully loaded {loaded_keys} weight tensors in {elapsed:.2f} seconds.")
    return model


# ============================================================================
# BLOCK 8: Generation Runtime & Sampling Engine
# ============================================================================
def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 50
) -> int:
    """
    Samples the next token from output logits:
    - temperature == 0.0: Pure Greedy (argmax)
    - temperature > 0.0: Temperature scaling + Top-K + Top-P (Nucleus) filtering
    """
    # Grab the last token logits: shape [vocab_size]
    last_logits = logits[0, -1, :].clone()

    # Case 1: Greedy Decoding
    if temperature == 0.0:
        return int(torch.argmax(last_logits).item())

    # Case 2: Temperature Scaling
    last_logits = last_logits / max(temperature, 1e-5)

    # Top-K Filtering
    if top_k > 0:
        values, indices = torch.topk(last_logits, min(top_k, last_logits.size(-1)))
        mask = torch.full_like(last_logits, float("-inf"))
        mask.scatter_(-1, indices, values)
        last_logits = mask

    # Top-P (Nucleus) Filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(last_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right by 1 to keep at least the first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        last_logits[indices_to_remove] = float("-inf")

    # Sample from the filtered probability distribution
    probs = F.softmax(last_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return int(next_token.item())


def generate_autoregressive(
    model: SmolLMTransformer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
    top_p: float = 0.9,
    eos_token_id: Optional[int] = None,
    stream_callback = None
) -> List[int]:
    """
    Dual-phase autoregressive generation:
    1. Phase A (Prefill): Compute prompt in parallel, populate KV cache.
    2. Phase B (Decode): One token per step, passing past KV cache.
    """
    model.eval()
    generated_tokens = input_ids.tolist()[0]
    device = input_ids.device

    with torch.no_grad():
        # PHASE A: PREFILL
        # Process the entire prompt simultaneously
        t_prefill_start = time.time()
        logits, kv_caches = model(input_ids, start_pos=0, kv_caches=None)
        ttft = time.time() - t_prefill_start

        # Sample the first new token
        next_token = sample_next_token(logits, temperature=temperature, top_p=top_p)
        generated_tokens.append(next_token)
        if stream_callback:
            stream_callback(next_token)

        if eos_token_id is not None and next_token == eos_token_id:
            return generated_tokens

        # PHASE B: DECODE
        # Process one token at a time, reusing the KV cache
        cur_pos = input_ids.shape[-1]
        decode_latencies = []

        for step in range(1, max_new_tokens):
            t_step_start = time.time()
            # Input is ONLY the single newest token! shape: [1, 1]
            token_tensor = torch.tensor([[next_token]], dtype=torch.long, device=device)

            logits, kv_caches = model(token_tensor, start_pos=cur_pos, kv_caches=kv_caches)
            next_token = sample_next_token(logits, temperature=temperature, top_p=top_p)

            step_time = time.time() - t_step_start
            decode_latencies.append(step_time)
            cur_pos += 1

            generated_tokens.append(next_token)
            if stream_callback:
                stream_callback(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

    return generated_tokens
