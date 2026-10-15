"""Model architecture descriptions.

A `TransformerConfig` is enough to answer *analytical* questions about a model:
how many parameters it has, how much arithmetic a forward pass performs, and how
much memory its KV cache occupies. No weights required.

That is deliberate. For the first several milestones we reason about models we
could not possibly run on this laptop; being able to price a 70B model's decode
step on paper is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

FFN_TYPES = ("swiglu", "mlp")


@dataclass(frozen=True)
class TransformerConfig:
    """Architecture of a decoder-only transformer.

    Attributes:
        name: Identifier for the architecture.
        d_model: Hidden width (aka `hidden_size`).
        n_layers: Number of decoder blocks.
        n_heads: Number of attention query heads.
        n_kv_heads: Number of key/value heads. Equal to `n_heads` for MHA,
            smaller for GQA/MQA. This is the knob that shrinks the KV cache.
        d_ff: Feed-forward intermediate width.
        vocab_size: Tokenizer vocabulary.
        ffn_type: `"swiglu"` for 3-matrix FFNs (LLaMA, Qwen, Mistral) or
            `"mlp"` for classic 2-matrix FFNs (GPT-2).
        tie_embeddings: Whether lm_head shares weights with the token embedding.
        max_position_embeddings: Maximum supported context length.
    """

    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    vocab_size: int
    ffn_type: str = "swiglu"
    tie_embeddings: bool = False
    max_position_embeddings: int = 8192

    def __post_init__(self) -> None:
        if self.ffn_type not in FFN_TYPES:
            raise ValueError(f"ffn_type must be one of {FFN_TYPES}, got {self.ffn_type!r}")
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} not divisible by n_heads={self.n_heads}"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads={self.n_heads} must be divisible by n_kv_heads={self.n_kv_heads}"
            )
        if not 0 < self.n_kv_heads <= self.n_heads:
            raise ValueError("n_kv_heads must be in (0, n_heads]")

    # --- derived geometry ---------------------------------------------------

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def gqa_ratio(self) -> int:
        """How many query heads share each KV head. 1 = MHA, 8 = Llama-3-8B."""
        return self.n_heads // self.n_kv_heads

    # --- parameter accounting ----------------------------------------------

    @property
    def params_attention_per_layer(self) -> int:
        """Q, K, V and output projections for one layer."""
        q = self.d_model * self.n_heads * self.head_dim
        kv = self.d_model * self.n_kv_heads * self.head_dim * 2  # K and V
        o = (self.n_heads * self.head_dim) * self.d_model
        return q + kv + o

    @property
    def params_ffn_per_layer(self) -> int:
        """Feed-forward matrices for one layer."""
        if self.ffn_type == "swiglu":
            # gate_proj, up_proj, down_proj
            return 3 * self.d_model * self.d_ff
        # c_fc, c_proj
        return 2 * self.d_model * self.d_ff

    @property
    def params_per_layer(self) -> int:
        """Transformer-block parameters, excluding embeddings and the head."""
        return self.params_attention_per_layer + self.params_ffn_per_layer

    @property
    def params_blocks(self) -> int:
        """Parameters that participate in per-token matmuls."""
        return self.n_layers * self.params_per_layer

    @property
    def params_embedding(self) -> int:
        """Token embedding table."""
        return self.vocab_size * self.d_model

    @property
    def params_lm_head(self) -> int:
        """Output projection. Tied models pay nothing here."""
        return 0 if self.tie_embeddings else self.vocab_size * self.d_model

    @property
    def n_params(self) -> int:
        """Total parameters, including embeddings and the LM head."""
        return self.params_blocks + self.params_embedding + self.params_lm_head

    @property
    def n_params_matmul(self) -> int:
        """Parameters touched by a GEMM per token.

        Embeddings are gathers, not matmuls; the LM head is usually computed only
        for the final position during decode. This is the denominator that matters
        for the "decode is memory-bound" argument.
        """
        return self.params_blocks + self.params_lm_head

    def human_params(self) -> str:
        n = self.n_params
        if n >= 1e9:
            return f"{n / 1e9:.2f}B"
        return f"{n / 1e6:.1f}M"

    # --- KV cache -----------------------------------------------------------

    def kv_bytes_per_token(self, dtype_bytes: int) -> int:
        """Bytes of KV cache held by one token, across all layers.

        Per layer per token we store K and V for each KV head:
            2 (K,V) * n_kv_heads * head_dim * dtype_bytes
        """
        per_layer = 2 * self.n_kv_heads * self.head_dim * dtype_bytes
        return per_layer * self.n_layers

    def kv_bytes(self, seq_len: int, batch: int = 1, dtype_bytes: int = 2) -> int:
        """Total KV cache for `batch` sequences of `seq_len` tokens."""
        return self.kv_bytes_per_token(dtype_bytes) * seq_len * batch

    def kv_bytes_per_token_per_seq(self, dtype_bytes: int = 2) -> int:
        return self.kv_bytes_per_token(dtype_bytes)

    def max_batch_at_full_context(self, kv_budget_bytes: int, dtype_bytes: int = 2) -> int:
        """How many sequences at `max_position_embeddings` fit in a KV budget.

        This is the single most important serving capacity number, and it is the
        reason KV cache quantization exists: it directly multiplies concurrency.
        """
        per_seq = self.kv_bytes(self.max_position_embeddings, 1, dtype_bytes)
        return max(1, kv_budget_bytes // per_seq)

    def describe(self) -> str:
        return (
            f"{self.name}\n"
            f"  d_model={self.d_model}  layers={self.n_layers}  heads={self.n_heads}"
            f"  kv_heads={self.n_kv_heads} (GQA {self.gqa_ratio}x)\n"
            f"  head_dim={self.head_dim}  d_ff={self.d_ff}  ffn={self.ffn_type}\n"
            f"  vocab={self.vocab_size:,}  ctx={self.max_position_embeddings}\n"
            f"  params={self.human_params()}  (blocks {self.params_blocks / 1e9:.2f}B)"
        )


# ---------------------------------------------------------------------------
# Real architectures. Parameter counts are checked in tests/ against published
# values, so a mistake in this table fails loudly rather than silently.
# ---------------------------------------------------------------------------

LLAMA3_8B = TransformerConfig(
    name="Llama-3-8B",
    d_model=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    d_ff=14336,
    vocab_size=128256,
    ffn_type="swiglu",
    tie_embeddings=False,
    max_position_embeddings=8192,
)

LLAMA3_70B = TransformerConfig(
    name="Llama-3-70B",
    d_model=8192,
    n_layers=80,
    n_heads=64,
    n_kv_heads=8,
    d_ff=28672,
    vocab_size=128256,
    ffn_type="swiglu",
    tie_embeddings=False,
    max_position_embeddings=8192,
)

MISTRAL_7B = TransformerConfig(
    name="Mistral-7B",
    d_model=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    d_ff=14336,
    vocab_size=32000,
    ffn_type="swiglu",
    tie_embeddings=False,
    max_position_embeddings=32768,
)

QWEN2_5_0_5B = TransformerConfig(
    name="Qwen2.5-0.5B",
    d_model=896,
    n_layers=24,
    n_heads=14,
    n_kv_heads=2,
    d_ff=4864,
    vocab_size=151936,
    ffn_type="swiglu",
    tie_embeddings=True,
    max_position_embeddings=32768,
)

GPT2_SMALL = TransformerConfig(
    name="GPT-2-Small",
    d_model=768,
    n_layers=12,
    n_heads=12,
    n_kv_heads=12,
    d_ff=3072,
    vocab_size=50257,
    ffn_type="mlp",
    tie_embeddings=True,
    max_position_embeddings=1024,
)

TINY = TransformerConfig(
    name="nano-tiny",
    d_model=128,
    n_layers=2,
    n_heads=4,
    n_kv_heads=4,
    d_ff=512,
    vocab_size=1024,
    ffn_type="swiglu",
    tie_embeddings=True,
    max_position_embeddings=512,
)

TOY = TransformerConfig(
    name="nano-toy",
    d_model=256,
    n_layers=4,
    n_heads=8,
    n_kv_heads=2,
    d_ff=1024,
    vocab_size=4096,
    ffn_type="swiglu",
    tie_embeddings=True,
    max_position_embeddings=1024,
)

REGISTRY: dict[str, TransformerConfig] = {
    c.name: c
    for c in (
        LLAMA3_8B,
        LLAMA3_70B,
        MISTRAL_7B,
        QWEN2_5_0_5B,
        GPT2_SMALL,
        TINY,
        TOY,
    )
}


def get(name: str) -> TransformerConfig:
    key = name.strip().lower()
    for cfg in REGISTRY.values():
        if cfg.name.lower() == key:
            return cfg
    raise KeyError(f"Unknown model {name!r}. Known: {sorted(REGISTRY)}")