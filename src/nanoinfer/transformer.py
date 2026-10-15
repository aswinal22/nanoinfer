"""A complete decoder-only transformer in raw PyTorch ops.

Deliberately no `transformers` dependency. The M2 goal is to be able to point at
any line and explain what it does, and that is impossible while a framework is
hiding the implementation.

Two paths are provided:

  * **Prefill** -- the full forward over a prompt. Complete and working.
  * **Incremental decode** -- one token at a time against a KV cache.
    `forward_cached` is left for you to implement (see docs/m2_brief.md).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import attn_output_proj, causal_attention, project_qkv
from .config import TransformerConfig
from .ops import build_rope_cache, rms_norm, rope_slice, silu, swiglu


class DecoderLayer(nn.Module):
    """One pre-norm transformer block.

    Pre-norm means the residual stream runs clean and each sublayer sees a
    freshly normalised input:

        x = x + Attn(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    The alternative (post-norm, original Transformer) needs a warmup schedule
    to train stably. Pre-norm removed the warmup requirement, which is one
    reason it won.
    """

    def __init__(self, cfg: TransformerConfig, layer_idx: int = 0):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx

        d, h, kvh = cfg.d_model, cfg.n_heads, cfg.n_kv_heads
        head_dim = cfg.head_dim

        # All projection weights are [out_features, in_features], matching
        # nn.Linear's convention, so torch.nn.functional.linear applies them
        # directly without a transpose.
        self.attn_norm = nn.Parameter(torch.ones(d))
        self.w_q = nn.Parameter(torch.empty(h * head_dim, d))
        self.w_k = nn.Parameter(torch.empty(kvh * head_dim, d))
        self.w_v = nn.Parameter(torch.empty(kvh * head_dim, d))
        self.w_o = nn.Parameter(torch.empty(d, h * head_dim))

        self.ffn_norm = nn.Parameter(torch.ones(d))
        if cfg.ffn_type == "swiglu":
            self.w_gate = nn.Parameter(torch.empty(cfg.d_ff, d))
            self.w_up = nn.Parameter(torch.empty(cfg.d_ff, d))
            self.w_down = nn.Parameter(torch.empty(d, cfg.d_ff))
        else:
            self.w_in = nn.Parameter(torch.empty(cfg.d_ff, d))
            self.w_out = nn.Parameter(torch.empty(d, cfg.d_ff))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Scaled init: small residuals, no need for a warmup schedule."""
        d, head_dim = self.cfg.d_model, self.cfg.head_dim

        def normal_(t: Tensor, std: float) -> None:
            with torch.no_grad():
                t.normal_(0.0, std)

        # 1/sqrt(2 * n_layers) keeps the residual stream variance stable
        # across depth.
        res_std = (2.0 * self.cfg.n_layers) ** -0.5
        normal_(self.w_q, (2.0 / (d + head_dim)) ** 0.5)
        normal_(self.w_k, (2.0 / (d + head_dim)) ** 0.5)
        normal_(self.w_v, (2.0 / (d + head_dim)) ** 0.5)
        normal_(self.w_o, res_std)

        if self.cfg.ffn_type == "swiglu":
            normal_(self.w_gate, res_std)
            normal_(self.w_up, res_std)
            normal_(self.w_down, res_std)
        else:
            normal_(self.w_in, (2.0 / (d + self.cfg.d_ff)) ** 0.5)
            normal_(self.w_out, res_std)

    def _ffn(self, x: Tensor) -> Tensor:
        if self.cfg.ffn_type == "swiglu":
            gate = torch.nn.functional.linear(x, self.w_gate)
            up = torch.nn.functional.linear(x, self.w_up)
            return torch.nn.functional.linear(swiglu(gate, up), self.w_down)
        hidden = torch.nn.functional.linear(silu(x), self.w_in)
        return torch.nn.functional.linear(hidden, self.w_out)

    # --- prefill: the complete reference implementation -------------------

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Full forward over `x`. O(seq^2) attention, no caching.

        This is the ground truth. Your cached path must reproduce it exactly.
        """
        h = self.cfg
        b, s, _ = x.shape
        pos = torch.arange(s, device=x.device)
        c, n = cos[pos], sin[pos]  # [s, head_dim]

        h_norm = rms_norm(x, self.attn_norm)
        q, k, v = project_qkv(
            h_norm, self.w_q, self.w_k, self.w_v, h.n_heads, h.n_kv_heads, c, n
        )
        attn = causal_attention(q, k, v)
        x = x + attn_output_proj(attn, self.w_o)

        x = x + self._ffn(rms_norm(x, self.ffn_norm))
        return x

    # --- cached decode: YOUR TASK -----------------------------------------

    def forward_cached(
        self, x: Tensor, cos: Tensor, sin: Tensor, cache: "KVCache", layer_idx: int
    ) -> Tensor:
        """One incremental step against a populated KV cache.

        x is [batch, 1, d_model] -- a single new token. The cache holds
        everything before it.

        Two differences from `forward`, both of which will bite you:

        1. **Q has length 1, K and V have length `past + 1`.** The mask and
           the softmax must be computed over the full K/V length.
        2. **RoPE positions start at `past`, not 0.** The new token lives at
           absolute position `past`.

        See docs/m2_brief.md and src/nanoinfer/kv_cache.py.
        """
        raise NotImplementedError(
            "M2 task: implement cached decoding. "
            "See src/nanoinfer/kv_cache.py and docs/m2_brief.md"
        )


class NanoTransformer(nn.Module):
    """Embedding, N decoder blocks, final norm, tied LM head."""

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, i) for i in range(cfg.n_layers)]
        )
        self.final_norm = nn.Parameter(torch.ones(cfg.d_model))

        cos, sin = build_rope_cache(cfg.head_dim, cfg.max_position_embeddings)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.apply(self._init_weights)
        # Embeddings want a smaller std than residual projections.
        with torch.no_grad():
            self.embed.weight.normal_(0.0, 0.02)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # --- prefill -----------------------------------------------------------

    def forward(self, input_ids: Tensor) -> Tensor:
        """Full forward over a prompt.

        Args:
            input_ids: [batch, seq] token ids.

        Returns:
            [batch, seq, vocab] logits at every position.
        """
        s = input_ids.shape[-1]
        if s > self.cfg.max_position_embeddings:
            raise ValueError(
                f"sequence length {s} exceeds context "
                f"{self.cfg.max_position_embeddings}"
            )
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin)
        x = rms_norm(x, self.final_norm)
        return torch.nn.functional.linear(x, self.embed.weight)  # tied head

    # --- cached generation: YOUR TASK -------------------------------------

    def forward_step(self, input_ids: Tensor, cache: "KVCache") -> Tensor:
        """Advance the cache by `input_ids` and return logits.

        This should work for both a multi-token prefill (into an empty cache)
        and a single-token decode (into a populated one). One code path, not
        two -- handling both in a single branch is where the offset bugs hide.

        Returns:
            [batch, seq, vocab]
        """
        raise NotImplementedError(
            "M2 task: implement forward_step. See docs/m2_brief.md"
        )

    # --- convenience -------------------------------------------------------

    def n_params(self) -> int:
        seen: set[int] = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total

    @torch.no_grad()
    def generate_greedy(self, input_ids: Tensor, max_new_tokens: int = 16) -> Tensor:
        """Greedy generation with NO cache -- the naive baseline for M2.

        This is what you are optimising against. Note that every step runs a
        full O(n) forward over the whole prefix, so total work is O(n^2).

        Returns:
            [batch, prompt_len + max_new_tokens] including the prompt.
        """
        out = input_ids
        for _ in range(max_new_tokens):
            logits = self(out)[:, -1, :]
            nxt = logits.argmax(dim=-1, keepdim=True)
            out = torch.cat([out, nxt], dim=1)
        return out