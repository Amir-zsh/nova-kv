"""SWA pairing stays consistent when the sliding-window tier is exhausted.

``alloc_swa_for`` releases whatever SWA is still mapped to the incoming full
ids *before* asking the tier for new slots (release-on-reuse). If that request
then fails, the rows must not keep pointing at the slots just released: their
next owner would be aliased, and a later ``free_swa`` over the same ids would
release them twice.

Run: python tests/test_swa_alloc_exhaustion.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from sglang.srt.mem_cache.swa_memory_pool import (  # noqa: E402
    SWATokenToKVPoolAllocator,
)

SWA_SIZE = 64


class _FakeSwaAllocator:
    def __init__(self, size):
        self.size = size
        self.free_slots = list(range(1, size + 1))  # 0 is the unmapped sentinel

    def available_size(self):
        return len(self.free_slots)

    def alloc(self, n):
        if n > len(self.free_slots):
            return None
        out, self.free_slots = self.free_slots[:n], self.free_slots[n:]
        return torch.tensor(out, dtype=torch.int64)

    def free(self, indices):
        self.free_slots.extend(int(i) for i in indices.tolist())


def _make_allocator():
    """A SWATokenToKVPoolAllocator wired to fakes, bypassing the heavy __init__."""
    alloc = object.__new__(SWATokenToKVPoolAllocator)
    alloc.page_size = 8
    alloc.device = "cpu"
    alloc.swa_attn_allocator = _FakeSwaAllocator(SWA_SIZE)
    alloc.full_to_swa_index_mapping = torch.zeros(256, dtype=torch.int64)
    alloc._kvcache = SimpleNamespace()
    return alloc


def test_exhaustion_leaves_the_allocator_consistent():
    alloc = _make_allocator()
    ids = torch.arange(10, 16, dtype=torch.int64)  # 6 full ids

    assert alloc.alloc_swa_for(ids[:4]) is not None
    alloc.swa_attn_allocator.alloc(SWA_SIZE - 4)  # drain the rest of the tier

    # Needs 6; releasing the 4 stale slots first is not enough.
    assert alloc.alloc_swa_for(ids) is None
    assert int(alloc.full_to_swa_index_mapping[ids].sum()) == 0, (
        "rows still point at slots that were returned to the free list"
    )
    assert alloc.swa_attn_allocator.available_size() == 4
    free = alloc.swa_attn_allocator.free_slots
    assert len(free) == len(set(free)), "a slot was added to the free list twice"

    # With the rows cleared, sweeping the same ids frees nothing extra.
    alloc.free_swa(ids)
    assert alloc.swa_attn_allocator.available_size() == 4


def test_success_path_is_unchanged():
    alloc = _make_allocator()
    ids = torch.arange(10, 14, dtype=torch.int64)
    got = alloc.alloc_swa_for(ids)
    assert got is not None and got.numel() == 4
    assert torch.equal(alloc.full_to_swa_index_mapping[ids], got)
    assert alloc.swa_attn_allocator.available_size() == SWA_SIZE - 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
