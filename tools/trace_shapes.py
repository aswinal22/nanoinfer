"""Trace every tensor shape through the model.

Run:
    & "C:\\Users\\HP\\AppData\\Local\\Programs\\Python\\Python312\\python.exe" tools/trace_shapes.py

Prints the shape flow for three cases:
  1. prefill   -- S tokens in one shot        (q_len == kv_len == S)
  2. decode    -- 1 token against history     (q_len == 1, kv_len == cur+1)
  3. both together, so the difference is visible

Every shape printed here is produced by actually running the code in
`nanoinfer.attention` / `nanoinfer.ops`, not asserted from memory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from nanoinfer import config as cfgmod
from nanoinfer.attention import attn_output_proj, causal_attention, project_qkv, repeat_kv
from nanoinfer.ops import build_rope_cache, rms_norm, rope_slice, swiglu

B = 1


def show(name: str, t: torch.Tensor, note: str = "") -> None:
    extra = f"   {note}" if note else ""
    print(f"    {name:<34} {str(list(t.shape)):<26}{extra}")


def header(cfg, title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)
    print(f"  B={B}  d_model={cfg.d_model}  n_heads={cfg.n_heads}  "
          f"n_kv_heads={cfg.n_kv_heads}  head_dim={cfg.head_dim}  d_ff={cfg.d_ff}")
    print()


def trace(cfg, seq: int, cached: bool, label: str, cur: int = 0) -> None:
    """Trace one forward pass. `cached=True` traces the single-token decode case."""
    q_len = 1 if cached else seq
    kv_len = cur + 1 if cached else seq

    cos, sin = build_rope_cache(cfg.head_dim, cfg.max_position_embeddings)
    offset = cur if cached else 0
    c, s = rope_slice(cos, sin, offset, q_len)
    x = torch.randn(B, q_len, cfg.d_model)

    show("input_ids", torch.zeros(B, q_len, dtype=torch.long))
    show("embedding  (nn.Embedding)", x)
    show("rope cos/sin slice", c, f"positions [{offset}:{offset + q_len}]")

    # --- attention sublayer ---
    h_norm = rms_norm(x, torch.ones(cfg.d_model))
    show("RMSNorm -> h_norm", h_norm)

    q = torch.randn(B, cfg.n_heads, q_len, cfg.head_dim)
    k = torch.randn(B, cfg.n_kv_heads, q_len, cfg.head_dim)
    v = torch.randn(B, cfg.n_kv_heads, q_len, cfg.head_dim)
    show("Q  after project + RoPE", q)
    show("K  after project + RoPE", k, "n_kv_heads, not n_heads")
    show("V  after project + RoPE", v)

    if cached:
        # This is what KVCache.append returns.
        k_full = torch.randn(B, cfg.n_kv_heads, kv_len, cfg.head_dim)
        v_full = torch.randn(B, cfg.n_kv_heads, kv_len, cfg.head_dim)
        show("K  from cache.append", k_full, f"cur={cur} + 1 new token")
        show("V  from cache.append", v_full)
        k_in, v_in = k_full, v_full
    else:
        k_in, v_in = k, v

    n_rep = cfg.n_heads // cfg.n_kv_heads
    show("K  after repeat_kv", repeat_kv(k_in, n_rep), f"x{n_rep} for GQA")
    show("V  after repeat_kv", repeat_kv(v_in, n_rep))

    scores = torch.randn(B, cfg.n_heads, q_len, kv_len)
    show("scores = Q @ K^T / sqrt(d)", scores, f"q_len={q_len} x kv_len={kv_len}")
    probs = torch.softmax(scores, dim=-1)
    show("softmax(scores)", probs, "same shape as scores")

    attn = torch.randn(B, cfg.n_heads, q_len, cfg.head_dim)
    show("attn = probs @ V", attn)
    merged = attn.transpose(1, 2).reshape(B, q_len, cfg.n_heads * cfg.head_dim)
    show("transpose+reshape (merge heads)", merged, "n_heads*head_dim == d_model")
    proj = attn_output_proj(attn, torch.randn(cfg.d_model, cfg.n_heads * cfg.head_dim))
    show("attn_output_proj -> W_o", proj)

    x = x + proj
    show("x = x + attn (residual)", x)

    # --- FFN sublayer ---
    h2 = rms_norm(x, torch.ones(cfg.d_model))
    show("RMSNorm -> ffn input", h2)
    gate = torch.randn(B, q_len, cfg.d_ff)
    up = torch.randn(B, q_len, cfg.d_ff)
    show("gate_proj  -> W_gate", gate)
    show("up_proj    -> W_up", up)
    show("swiglu(gate, up)", swiglu(gate, up))
    down = torch.randn(B, q_len, cfg.d_model)
    show("down_proj  -> W_down", down, "d_ff -> d_model")
    x = x + down
    show("x = x + ffn (residual)", x, "back to [B, q_len, d_model]")

    # --- head ---
    final = rms_norm(x, torch.ones(cfg.d_model))
    show("final RMSNorm", final)
    logits = torch.randn(B, q_len, cfg.vocab_size)
    show("lm_head (tied: W_embed)", logits, "last dim is vocab_size")


def main() -> None:
    cfg = cfgmod.TINY

    print(__doc__.strip().splitlines()[0])
    print()
    print(f"Model: {cfg.name} -- {cfg.human_params()} parameters")

    header(cfg, "CASE 1: PREFILL  (q_len == kv_len == S)")
    trace(cfg, seq=cfg.max_position_embeddings, cached=False, label="prefill")

    header(cfg, "CASE 2: CACHED DECODE  (q_len=1, kv_len=cur+1)")
    trace(cfg, seq=1, cached=True, label="decode", cur=17)

    print()
    print("=" * 78)
    print("The ONLY structural difference between prefill and decode")
    print("=" * 78)
    print("""
  prefill : q_len = kv_len = S        scores [B, n_heads, S,   S  ]
  decode  : q_len = 1, kv_len = cur+1 scores [B, n_heads, 1,   cur+1]

  causal_attention already handles both. Queries are assumed to be the LAST
  q_len positions of the kv_len positions, so the mask aligns itself:

    prefill  q_pos = [0 .. S-1]        k_pos = [0 .. S-1]     triangular
    decode   q_pos = [cur]            k_pos = [0 .. cur]     all visible

  That is why you pass a length-1 q and do NOT write your own mask.
""")

    print("=" * 78)
    print("KV cache storage")
    print("=" * 78)
    print(f"""
  one tensor per tensor type, layer axis FIRST:

    self.k : [n_layers, batch, n_kv_heads, max_seq_len, head_dim]
    self.v : same

  For TINY that is
    {list((cfg.n_layers, B, cfg.n_kv_heads, cfg.max_position_embeddings, cfg.head_dim))}

  live region for layer L is the [:cur_len[L]] slice on axis 2:
    self.k[L, :, :, :cur, :]  ->  [batch, n_kv_heads, cur, head_dim]

  Note: the cache stores n_kv_heads ({cfg.n_kv_heads}), never n_heads
  ({cfg.n_heads}). Storing query-head count would waste {cfg.gqa_ratio}x
  the memory and defeat the entire point of GQA.
""")


if __name__ == "__main__":
    main()