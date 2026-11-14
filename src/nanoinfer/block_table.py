"""Milestone 3 -- Paged Attention: Block Tables & Memory Management.

Implements OS-style virtual memory paging for LLM KV caches:
- Fixed-size physical blocks (e.g., 16 tokens).
- Decouples logical sequence length from physical VRAM allocation.
- Copy-on-Write (CoW) branching for prefix sharing and parallel sampling.
- Eliminates reservation waste and external fragmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import Tensor

from .config import TransformerConfig


@dataclass
class PhysicalBlock:
    """A single fixed-size physical block in GPU memory."""

    block_id: int
    block_size: int = 16
    ref_count: int = 0

    @property
    def is_free(self) -> bool:
        """True if no sequence is currently referencing this block."""
        return self.ref_count == 0

    def retain(self) -> None:
        """Increment reference count when a sequence references this block."""
        self.ref_count += 1

    def release(self) -> None:
        """Decrement reference count when a sequence releases this block."""
        if self.ref_count <= 0:
            raise ValueError(f"Block {self.block_id} double-free error!")
        self.ref_count -= 1


class BlockTable:
    """Logical-to-physical block mapping for one sequence."""

    def __init__(self, block_size: int = 16):
        self.block_size = block_size
        self.physical_block_ids: list[int] = []
        self.num_tokens: int = 0

    def get_physical_slot(self, logical_pos: int) -> tuple[int, int]:
        """Translate logical token index to (physical_block_id, slot_idx)."""
        if logical_pos < 0 or logical_pos >= self.num_tokens:
            raise IndexError(
                f"Position {logical_pos} out of range (length {self.num_tokens})"
            )
        logical_block_idx = logical_pos // self.block_size
        slot_idx = logical_pos % self.block_size
        physical_block_id = self.physical_block_ids[logical_block_idx]
        return physical_block_id, slot_idx

    @property
    def num_blocks(self) -> int:
        return len(self.physical_block_ids)

    def is_last_block_full(self) -> bool:
        """True if the sequence has no remaining space in its allocated blocks."""
        if not self.physical_block_ids:
            return True
        return (self.num_tokens % self.block_size) == 0


class BlockManager:
    """Global manager allocating and reclaiming physical blocks from a pool."""

    def __init__(self, num_blocks: int, block_size: int = 16):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.blocks = [PhysicalBlock(i, block_size) for i in range(num_blocks)]
        self.free_blocks: list[int] = list(range(num_blocks))

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    def allocate(self) -> int:
        """Pop a free physical block from the pool and retain it."""
        if not self.free_blocks:
            raise MemoryError("Out of memory: No free physical blocks available!")
        block_id = self.free_blocks.pop(0)
        self.blocks[block_id].retain()
        return block_id

    def free_block(self, block_id: int) -> None:
        """Release a block. If reference count reaches 0, return to free list."""
        block = self.blocks[block_id]
        block.release()
        if block.is_free:
            self.free_blocks.append(block_id)

    def append_slot(self, table: BlockTable) -> tuple[int, int]:
        """Prepare slot for the next incoming token, allocating a new block if needed."""
        if table.is_last_block_full():
            new_block_id = self.allocate()
            table.physical_block_ids.append(new_block_id)

        logical_pos = table.num_tokens
        table.num_tokens += 1
        return table.get_physical_slot(logical_pos)

    def fork_sequence(self, source_table: BlockTable) -> BlockTable:
        """Fork a sequence sharing all existing physical blocks (Copy-on-Write)."""
        child_table = BlockTable(self.block_size)
        child_table.num_tokens = source_table.num_tokens
        child_table.physical_block_ids = list(source_table.physical_block_ids)

        # All shared blocks increment their reference count
        for block_id in child_table.physical_block_ids:
            self.blocks[block_id].retain()

        return child_table

    def free_sequence(self, table: BlockTable) -> None:
        """Release all blocks held by a sequence when it finishes."""
        for block_id in table.physical_block_ids:
            self.free_block(block_id)
        table.physical_block_ids.clear()
        table.num_tokens = 0


class PagedKVCache:
    """Tensor-backed KV storage managed by BlockManager."""

    def __init__(
        self,
        num_blocks: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ):
        self.manager = BlockManager(num_blocks=num_blocks, block_size=block_size)
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size

        # Physical storage in VRAM: [num_blocks, n_layers, n_kv_heads, block_size, head_dim]
        shape = (num_blocks, n_layers, n_kv_heads, block_size, head_dim)
        self.k_cache = torch.empty(shape, dtype=dtype, device=device)
        self.v_cache = torch.empty(shape, dtype=dtype, device=device)

    @classmethod
    def from_config(
        cls,
        cfg: TransformerConfig,
        num_blocks: int = 256,
        block_size: int = 16,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> "PagedKVCache":
        return cls(
            num_blocks=num_blocks,
            n_layers=cfg.n_layers,
            n_kv_heads=cfg.n_kv_heads,
            head_dim=cfg.head_dim,
            block_size=block_size,
            dtype=dtype,
            device=device,
        )

    def write_token_kv(
        self,
        table: BlockTable,
        layer_idx: int,
        k_vec: Tensor,
        v_vec: Tensor,
    ) -> None:
        """Write 1 token's KV vector into the paged storage at the current slot.

        k_vec, v_vec: [1, n_kv_heads, 1, head_dim] or [n_kv_heads, head_dim]
        """
        # Squeeze batch & token dimensions if present
        if k_vec.ndim == 4:
            k_vec = k_vec.squeeze(0).squeeze(1)  # [n_kv_heads, head_dim]
            v_vec = v_vec.squeeze(0).squeeze(1)

        # Get slot for the latest token
        block_id, slot_idx = table.get_physical_slot(table.num_tokens - 1)
        self.k_cache[block_id, layer_idx, :, slot_idx, :] = k_vec
        self.v_cache[block_id, layer_idx, :, slot_idx, :] = v_vec

    def read_sequence_kv(
        self,
        table: BlockTable,
        layer_idx: int,
    ) -> tuple[Tensor, Tensor]:
        """Read all live tokens for one sequence as contiguous [1, n_kv_heads, cur_len, head_dim]."""
        if table.num_tokens == 0:
            empty = torch.empty(
                (1, self.n_kv_heads, 0, self.head_dim),
                dtype=self.k_cache.dtype,
                device=self.k_cache.device,
            )
            return empty, empty

        # Gather keys and values across the sequence's blocks
        k_blocks = []
        v_blocks = []
        tokens_gathered = 0

        for block_id in table.physical_block_ids:
            tokens_in_this_block = min(
                self.block_size, table.num_tokens - tokens_gathered
            )
            k_blocks.append(
                self.k_cache[block_id, layer_idx, :, :tokens_in_this_block, :]
            )
            v_blocks.append(
                self.v_cache[block_id, layer_idx, :, :tokens_in_this_block, :]
            )
            tokens_gathered += tokens_in_this_block

        # Concatenate into standard [1, n_kv_heads, cur_len, head_dim]
        k_full = torch.cat(k_blocks, dim=1).unsqueeze(0)
        v_full = torch.cat(v_blocks, dim=1).unsqueeze(0)
        return k_full, v_full
