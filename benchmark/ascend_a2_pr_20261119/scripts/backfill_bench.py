"""Replay bounded KV pressure through the actual decode queue and NPU page allocator."""

import importlib.util, json, os, sys, time
from pathlib import Path
import torch, torch_npu
from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator
from sglang.srt.managers.scheduler_components.request_list import move

torch.npu.set_device(0)
torch.set_num_threads(1)
fixture = Path(
    "/bench/src/49103/test/registered/unit/disaggregation/test_decode_queue_cleanup.py"
)
spec = importlib.util.spec_from_file_location("queue_fixture", fixture)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
enabled = bool(int(sys.argv[2]))
results = []
for rep in range(12):
    case = m.TestDecodePreallocBackfill()
    case.setUp()
    head = case.make_request("large", 96)
    queue, pool = case.make_queue([head], free=128)
    alloc = NPUPagedTokenToKVPoolAllocator(128, 1, torch.bfloat16, "npu:0", None, False)
    occupancy = alloc.alloc(80)
    content = torch.full((130,), -1, dtype=torch.int32, device="npu")
    queue.token_to_kv_pool_allocator = alloc
    queue.enable_prealloc_backfill = enabled
    ids = {}

    def allocate(req, prefix_indices, prefix_len, total_prefix_len):
        count = len(req.origin_input_ids) - prefix_len
        slots = alloc.alloc(count)
        assert slots is not None and slots.device.type == "npu"
        req.kv.req_pool_idx = len(ids)
        ids[req.rid] = (slots, req.kv.req_pool_idx + 1)
        content.index_fill_(0, slots.long(), req.kv.req_pool_idx + 1)
        return slots

    queue._pre_alloc = allocate
    for _ in range(8):
        assert queue.move_preallocated(queue.transfer_queue.queue) == ([], [])
    small = [case.make_request("small" + str(i), 4) for i in range(6)]
    move(small, queue.queue)
    admitted = []
    for _ in range(5):
        admitted.extend(
            x.req.rid for x in queue.move_preallocated(queue.transfer_queue.queue)[0]
        )
    early = list(admitted)
    alloc.free(occupancy)
    for _ in range(4):
        admitted.extend(
            x.req.rid for x in queue.move_preallocated(queue.transfer_queue.queue)[0]
        )
    assert len(set(admitted)) == 7 and len(admitted) == 7, admitted
    assert len(early) == (4 if enabled else 0), early
    for key, (slots, v) in ids.items():
        assert (content.index_select(0, slots.long()) == v).all().item()
    torch.npu.synchronize()
    results.append(
        dict(
            early=early,
            full=admitted,
            kv_used=sum(v[0].numel() for v in ids.values()),
            remaining=alloc.available_size(),
        )
    )
    case.doCleanups()
Path(sys.argv[1]).write_text(
    json.dumps(
        dict(
            sha=os.environ["ARM_SHA"],
            device=torch.npu.get_device_name(),
            enabled=enabled,
            results=results,
        ),
        indent=2,
    )
    + "\n"
)
