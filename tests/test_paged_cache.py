"""Unit tests for Milestone 3 -- Paged Attention and Block Tables."""

import pytest
import torch

from nanoinfer.block_table import BlockManager, BlockTable, PagedKVCache, PhysicalBlock
from nanoinfer.config import TINY


def test_physical_block_ref_counting():
    """Verify retain, release, and double-free protection."""
    block = PhysicalBlock(block_id=0, block_size=16)
    assert block.is_free is True
    assert block.ref_count == 0

    block.retain()
    assert block.ref_count == 1
    assert block.is_free is False

    block.retain()
    assert block.ref_count == 2

    block.release()
    assert block.ref_count == 1

    block.release()
    assert block.ref_count == 0
    assert block.is_free is True

    # Double free must raise an error
    with pytest.raises(ValueError, match="double-free"):
        block.release()


def test_block_table_address_translation():
    """Verify integer division (//) and modulo (%) translation."""
    table = BlockTable(block_size=16)
    table.physical_block_ids = [7, 2, 9]  # 3 physical blocks allocated
    table.num_tokens = 36  # Tokens 0 to 35

    # Token 0 -> Block 7, Slot 0
    assert table.get_physical_slot(0) == (7, 0)

    # Token 15 -> Block 7, Slot 15 (last slot of 1st block)
    assert table.get_physical_slot(15) == (7, 15)

    # Token 16 -> Block 2, Slot 0 (first slot of 2nd block)
    assert table.get_physical_slot(16) == (2, 0)

    # Token 35 -> Block 9, Slot 3
    assert table.get_physical_slot(35) == (9, 3)

    # Token 36 is out of range
    with pytest.raises(IndexError):
        table.get_physical_slot(36)


def test_block_manager_allocation_and_boundary_crossing():
    """Verify manager dynamically allocates new blocks as tokens arrive."""
    manager = BlockManager(num_blocks=10, block_size=16)
    assert manager.num_free_blocks == 10

    table = BlockTable(block_size=16)

    # Add 16 tokens (should allocate exactly 1 block: Block 0)
    for _ in range(16):
        block_id, slot_idx = manager.append_slot(table)

    assert table.num_blocks == 1
    assert table.physical_block_ids == [0]
    assert manager.num_free_blocks == 9
    assert manager.blocks[0].ref_count == 1

    # Add token 17 (should cross boundary and allocate Block 1)
    block_id, slot_idx = manager.append_slot(table)
    assert block_id == 1
    assert slot_idx == 0
    assert table.num_blocks == 2
    assert table.physical_block_ids == [0, 1]
    assert manager.num_free_blocks == 8


def test_copy_on_write_forking():
    """Verify forked sequence shares blocks and manages ref_counts correctly."""
    manager = BlockManager(num_blocks=10, block_size=16)
    parent_table = BlockTable(block_size=16)

    # Parent has 20 tokens (Blocks 0 and 1)
    for _ in range(20):
        manager.append_slot(parent_table)

    assert parent_table.physical_block_ids == [0, 1]
    assert manager.blocks[0].ref_count == 1
    assert manager.blocks[1].ref_count == 1

    # Fork child sequence
    child_table = manager.fork_sequence(parent_table)
    assert child_table.physical_block_ids == [0, 1]
    assert child_table.num_tokens == 20

    # Both blocks should now have ref_count == 2 (shared!)
    assert manager.blocks[0].ref_count == 2
    assert manager.blocks[1].ref_count == 2

    # Free parent: blocks 0 and 1 must NOT return to free pool because child still uses them!
    manager.free_sequence(parent_table)
    assert manager.blocks[0].ref_count == 1
    assert manager.blocks[1].ref_count == 1
    assert manager.num_free_blocks == 8  # 8 free, 2 held by child

    # Free child: blocks 0 and 1 now reach ref_count == 0 and return to free pool!
    manager.free_sequence(child_table)
    assert manager.blocks[0].ref_count == 0
    assert manager.blocks[1].ref_count == 0
    assert manager.num_free_blocks == 10  # All 10 free!


def test_paged_kv_cache_read_write_exactness():
    """Verify tensors stored in paged blocks are reconstructed with 100% fidelity."""
    torch.manual_seed(42)
    cache = PagedKVCache.from_config(TINY, num_blocks=16, block_size=16)
    table = BlockTable(block_size=16)

    n_tokens = 35  # Across 3 physical blocks (16 + 16 + 3)
    k_expected = []
    v_expected = []

    for _ in range(n_tokens):
        cache.manager.append_slot(table)
        k_vec = torch.randn(TINY.n_kv_heads, TINY.head_dim)
        v_vec = torch.randn(TINY.n_kv_heads, TINY.head_dim)
        k_expected.append(k_vec)
        v_expected.append(v_vec)

        # Write to layer 0
        cache.write_token_kv(table, layer_idx=0, k_vec=k_vec, v_vec=v_vec)

    # Read back all 35 tokens from layer 0
    k_full, v_full = cache.read_sequence_kv(table, layer_idx=0)

    # Stack expected vectors: [1, n_kv_heads, 35, head_dim]
    k_expected_tensor = torch.stack(k_expected, dim=1).unsqueeze(0)
    v_expected_tensor = torch.stack(v_expected, dim=1).unsqueeze(0)

    assert torch.equal(k_full, k_expected_tensor), "Paged K tensor does not match expected!"
    assert torch.equal(v_full, v_expected_tensor), "Paged V tensor does not match expected!"
