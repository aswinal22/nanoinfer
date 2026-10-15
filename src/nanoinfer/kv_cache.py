"""TODO(you): KV cache.

=============================================================================
THIS IS YOUR M2 TASK. Everything else here is scaffolding.
=============================================================================

Read `docs/m2_brief.md` first, then fill in the methods marked `TODO`.

## What you are building

Autoregressive generation is wasteful. To produce token N, `NanoTransformer.
forward` re-runs the entire prefix through every layer -- O(N) work -- even
though the prefix's keys and values have not changed since they were computed
for token N-1.

The KV cache stores them. Each step then computes only the new token's Q, K, V,
appends K and V to the cache, and attends against the full history. Work drops
from O(N) to O(1) per step, so generation goes from O(N^2) to O(N).

## The interface you must satisfy

Shapes assume `[batch, n_heads, seq, head_dim]` throughout.

    reset()          -> None
        Drop all history. The cache must be reusable for a new sequence.

    get_seq_len()    -> int
        Current number of tokens in the cache. Identical across layers,
        because every layer consumes the same tokens.

    get_kv(layer)    -> (k, v)
        The live history for `layer`, each [batch, n_kv_heads, seq_len, head_dim].
        Read-only view; must not expose the unused capacity in `max_seq_len`.

    append(layer, k_new, v_new) -> (k_full, v_full)
        Write `k_new`/`v_new` ([batch, n_kv_heads, n_new, head_dim]) after the
        existing history, grow the length by `n_new`, and return the full
        updated k and v.

        `n_new` may exceed 1 -- prefill appends a whole prompt in one call.
        That single-call generality is what lets prefill and decode share one
        code path.

    is_empty()       -> bool
        True before the first append.

## Names to use

    Tests in `tests/test_kv_cache_steps.py` reach into attributes directly, so
    use these exact names:

        self.k       key buffer
        self.v       value buffer
        self.cur_len the shared write pointer (an int)

## Layout to use

Preallocate one contiguous buffer per tensor, with the layer axis FIRST:

    [n_layers, batch, n_kv_heads, max_seq_len, head_dim]

One contiguous tensor per tensor type means `self.k[0]` is a cheap view rather
than a Python list of separate allocations.

Plus a single write pointer `self.cur_len` shared across layers. `append`
writes at the pointer and advances it.

Allocating once and reusing the memory is the whole point: per-step allocation
would reintroduce the overhead you are trying to remove.

Why a single shared pointer? Every layer sees the same tokens, so every layer
is always at the same length. One counter, not `n_layers` of them.

## Work in order

Rungs 1-4 (this file) are pure bookkeeping and contain no new ideas. Rungs 5-6
are in transformer.py and are where the actual lesson lives.

`docs/m2_stairs.md` walks through all six with hints and one test per rung.

## Gotchas, in the order people hit them

1. **Returning capacity instead of contents.** Slicing `[:cur_len]` is required.
   If you pass the full `[..., :max_seq_len, ...]` tensor to attention, the
   model reads uninitialised memory. No error is raised; the output is just
   wrong.

2. **RoPE position offset.** The new token at index `cur_len` sits at absolute
   position `cur_len`, not 0. See `ops.rope_slice`.

3. **Mask alignment.** `causal_attention` assumes queries are the *last* q_len
   positions. It handles the alignment itself -- you do not need to pass an
   offset, but you must pass the full k/v including the newly appended token.

4. **Prefill must go through the same path.** If you special-case prefill with
   a separate "fast path" you will build two code paths and debug both.

5. **GQA head count.** The cache holds `n_kv_heads`, not `n_heads`. Storing
   query-head-count worth of KV quadruples memory for Llama-3-8B and is the
   single easiest way to lose the entire point of GQA.

## Success criterion

`tests/test_model.py::test_cached_decode_matches_full_forward` goes green, and
it is currently marked `xfail`. That test is your oracle: it compares cached
generation against the complete prefill path, which already works.

=============================================================================
"""

from __future__ import annotations

import torch
from torch import Tensor

from .config import TransformerConfig


class KVCache:
    """Fixed-capacity preallocated key/value cache.

    Storage layout per layer: [batch, n_kv_heads, max_seq_len, head_dim].
    One write pointer is shared by all layers.

    This is the naive-but-correct design. It wastes the tail of `max_seq_len`
    for short sequences -- which is exactly the problem M3 (paged attention)
    exists to solve. Build it this way first so you understand what paging fixes.
    """

    def __init__(
        self,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        max_seq_len: int,
        batch: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.batch = batch

        # TODO(you): allocate the K and V buffers.
        #
        #   self.k, self.v
        #   shape: [n_layers, batch, n_kv_heads, max_seq_len, head_dim]
        #
        # torch.zeros(...) is fine. torch.empty(...) is faster and also
        # correct, provided you never read beyond `cur_len` -- see gotcha #1.

        # TODO(you): initialise the write pointer.
        #
        #   self.cur_len = 0

    @classmethod
    def from_config(
        cls,
        cfg: TransformerConfig,
        batch: int = 1,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> "KVCache":
        """Build a cache sized for `cfg`. Convenience constructor."""
        return cls(
            n_layers=cfg.n_layers,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_position_embeddings,
            batch=batch,
            dtype=dtype,
            device=device,
        )

    # --- state -------------------------------------------------------------

    def is_empty(self) -> bool:
        """True before the first append."""
        # TODO(you)
        raise NotImplementedError("M2 task")

    def get_seq_len(self) -> int:
        """Number of tokens currently cached. Same for every layer."""
        # TODO(you)
        raise NotImplementedError("M2 task")

    def reset(self) -> None:
        """Drop all history so the cache can serve a new sequence."""
        # TODO(you): set the pointer back to 0. You do not need to zero the
        # buffers -- nothing may read past the pointer.
        raise NotImplementedError("M2 task")

    # --- data --------------------------------------------------------------

    def get_kv(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Live history for one layer.

        Returns:
            (k, v), each [batch, n_kv_heads, cur_len, head_dim]
        """
        # TODO(you): slice to the live length only.
        raise NotImplementedError("M2 task")

    def append(
        self, layer_idx: int, k_new: Tensor, v_new: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Write new keys/values and return the updated history.

        Args:
            layer_idx: Which decoder block is calling.
            k_new: [batch, n_kv_heads, n_new, head_dim]
            v_new: [batch, n_kv_heads, n_new, head_dim]

        Returns:
            (k_full, v_full), each [batch, n_kv_heads, cur_len + n_new, head_dim]
        """
        # TODO(you)
        #
        # Order of operations:
        #   1. bounds-check against max_seq_len (raise a clear error on overflow)
        #   2. write k_new/v_new at [:, :, cur:cur+n_new, :]
        #   3. advance the pointer by n_new
        #   4. return the live slice
        #
        # Note the pointer is shared across layers, so layer 0 advancing it in
        # prefill is what makes layer 1's slice start at the right place.
        raise NotImplementedError("M2 task")

    # --- diagnostics -------------------------------------------------------

    def bytes_allocated(self) -> int:
        """Bytes held by the buffers, including unused capacity."""
        # TODO(you), or leave as: sum of numel * element_size over self.k, self.v
        raise NotImplementedError("M2 task")

    def __repr__(self) -> str:
        try:
            n = self.get_seq_len()
        except NotImplementedError:
            n = -1
        return (
            f"KVCache(layers={self.n_layers}, batch={self.batch}, "
            f"kv_heads={self.n_kv_heads}, head_dim={self.head_dim}, "
            f"len={n}/{self.max_seq_len})"
        )