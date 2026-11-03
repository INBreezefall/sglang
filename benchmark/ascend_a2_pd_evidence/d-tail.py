import argparse
import json
import time
from array import array
from contextlib import nullcontext
from functools import partial
from unittest.mock import patch

import torch
import torch_npu
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture, match_prefix

p = argparse.ArgumentParser()
p.add_argument("variant", choices=["main", "opt"])
p.add_argument("--paged-pressure", action="store_true")
a = p.parse_args()
torch.npu.set_device(1)
for needed in [1280, 8192, 32768]:
    for repeat in range(3):
        use_paged = patch(
            "sglang.test.kits.unified_radix_cache.fixtures.TokenToKVPoolAllocator",
            partial(NPUPagedTokenToKVPoolAllocator, page_size=128),
        ) if a.paged_pressure else nullcontext()
        with use_paged:
            cache, allocator, _ = build_fixture(
                CacheConfig(page_size=128, kv_size=65536, max_context_len=65536,
                            num_layers=1, head_num=1, head_dim=64),
                tree_core_backend="python",
            )
        tokens = list(range(32768))
        slots = allocator.alloc(len(tokens))
        assert slots.device.type == "npu", slots.device
        torch.npu.synchronize()
        cache.insert(InsertParams(key=RadixKey(array("q", tokens)), value=slots))
        if a.paged_pressure:
            occupied = allocator.alloc(allocator.available_size())
            assert occupied is not None and allocator.alloc(needed) is None
            data = allocator.get_kvcache().get_key_buffer(0)
            expected = torch.arange(len(tokens), device=slots.device).remainder(113).to(data.dtype)
            data[slots, 0, 0] = expected
        free = allocator.available_size()
        t0 = time.perf_counter()
        count = cache.evict_for_alloc(EvictParams(num_tokens=needed)).num_tokens_evicted
        torch.npu.synchronize()
        elapsed = 1000 * (time.perf_counter() - t0)
        match = match_prefix(cache, tokens)
        retained = match.device_prefix_len
        if retained:
            actual = cache.tree_core.collect_full_device_indices(match.last_device_node, cache.root_node_handle())
            assert torch.equal(actual, slots[:retained])
        assert allocator.available_size() - free == count and retained + count == len(tokens)
        if a.paged_pressure:
            reused = allocator.alloc(needed)
            assert reused is not None
            data[reused, 0, 0] = -7.0
            if retained:
                assert torch.equal(data[actual, 0, 0], expected[:retained])
        cache.sanity_check()
        print(json.dumps(dict(variant=a.variant, repeat=repeat, npu=str(slots.device),
                              leaf=len(tokens), shortfall=needed, freed=count,
                              allocator=type(allocator).__name__, free_before=free,
                              next_turn_matched=retained, walk_ms=elapsed)), flush=True)
