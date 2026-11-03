import argparse
from array import array
import json
import random
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu
from sglang.srt.managers import data_parallel_controller as dispatch
from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import EvictParams, InsertParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.kits.unified_radix_cache.config import CacheConfig
from sglang.test.kits.unified_radix_cache.fixtures import build_fixture, match_prefix

p = argparse.ArgumentParser()
p.add_argument("arm", choices=["main", "opt"])
p.add_argument("--sessions", type=int, choices=[8, 24], default=8)
p.add_argument("--seed", type=int, default=11)
a = p.parse_args()
initial = [4096, 6144, 2048, 8192, 7168, 3072, 1024, 5120]
normal = list(range(8))
rotated = [normal, [7, 5, 2, 1, 4, 6, 3, 0], [4, 1, 7, 0, 3, 6, 5, 2]]
cap = 65536
if a.sessions == 24:
    initial = [16384, 12288, 8192, 10240, 6144, 4096, 2048, 1024] * 3
    normal = list(range(24))
    rng = random.Random(a.seed)
    rng.shuffle(normal)
    rotated = [normal, rng.sample(normal, len(normal)), rng.sample(normal, len(normal))]
    cap = 32768
for order, arrival in [("fixed", [normal] * 3), ("permuted", rotated)]:
    ctrl = dispatch.DataParallelController.__new__(dispatch.DataParallelController)
    ctrl.workers = list(range(8))
    ctrl._active_workers = list(range(8))
    ctrl.dp_budget = dispatch.DPBudget(8)
    if a.arm == "opt":
        ctrl.prefill_sessions = dispatch.PrefillSessionPlacement(8, slots_per_rank=3)
    trees = []
    for rank in range(8):
        torch.npu.set_device(rank)
        cfg = CacheConfig(page_size=128, kv_size=cap, max_context_len=32768,
                          head_num=1, head_dim=64, num_layers=1)
        if a.sessions == 24:
            with patch("sglang.test.kits.unified_radix_cache.fixtures.TokenToKVPoolAllocator",
                       partial(NPUPagedTokenToKVPoolAllocator, page_size=128)):
                trees.append(build_fixture(cfg, tree_core_backend="python")[:2])
        else:
            trees.append(build_fixture(cfg, tree_core_backend="python")[:2])
    rows = []
    for turn, sessions in enumerate(arrival):
        # Each wave starts after the prior one drains. No active requests or locked
        # tokens remain, and previously inserted prefixes are still evictable.
        ctrl.dp_budget.update_budget([SimpleNamespace(dp_rank=r, timestamp=turn+1,
                num_running_reqs=0, num_waiting_reqs=0, num_total_tokens=0,
                max_total_num_tokens=cap) for r in range(8)])
        for sid in sessions:
            length = initial[sid] + turn * 512
            tokens = array("q", [sid + 1000] * length)
            req = TokenizedGenerateReqInput(rid=f"{turn}-{sid}", input_text=None,
                    input_ids=tokens, input_embeds=None, mm_inputs=None, token_type_ids=None,
                    sampling_params=SamplingParams(max_new_tokens=128),
                    return_logprob=False, logprob_start_len=-1, top_logprobs_num=0,
                    token_ids_logprob=None, stream=False, session_id=str(sid))
            chosen = []
            with patch.object(dispatch, "sock_send", side_effect=lambda w, _: chosen.append(w)):
                if a.arm == "opt":
                    ctrl.prefill_session_tokens_scheduler(req)
                else:
                    ctrl.total_tokens_scheduler(req)
            assert len(chosen) == 1
            rank = chosen[0]
            torch.npu.set_device(rank)
            cache, allocator = trees[rank]
            before = match_prefix(cache, tokens)
            hit = before.device_prefix_len
            missing = length - hit
            shortfall = max(0, missing - allocator.available_size())
            held = cache.inc_lock_ref(before.last_device_node).to_dec_params()
            evicted = 0
            if shortfall:
                evicted = cache.evict_for_alloc(EvictParams(num_tokens=shortfall)).num_tokens_evicted
            slots = allocator.alloc(missing)
            assert slots is not None and str(slots.device) == f"npu:{rank}"
            if hit:
                previous = cache.tree_core.collect_full_device_indices(
                    before.last_device_node, cache.root_node_handle())
                slots = torch.cat([previous, slots])
            cache.insert(InsertParams(key=RadixKey(tokens), value=slots))
            cache.dec_lock_ref(before.last_device_node, held)
            assert match_prefix(cache, tokens).device_prefix_len == length
            rows.append(dict(turn=turn+1, session=sid, rank=rank, input=length,
                             matched=hit, missing=missing, evicted=evicted))
    for rank, (cache, _) in enumerate(trees):
        torch.npu.set_device(rank)
        cache.sanity_check()
    print(json.dumps(dict(arm=a.arm, order=order, seed=a.seed, capacity=cap,
                          sessions=a.sessions, rows=rows)), flush=True)
    del trees
