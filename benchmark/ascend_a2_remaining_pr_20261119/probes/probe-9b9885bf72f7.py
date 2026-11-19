"""Allocate the three BF16 target mappings at the measured GLM-5.2 host-pool geometry."""

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import torch
import torch_npu
from sglang.srt.mem_cache.pool_host.common import alloc_with_pin_memory_npu

feature, flag, output = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
assert feature in (48543, 48575)
torch.npu.set_device(0)
torch.set_num_threads(1)
os.environ.pop("SGLANG_HICACHE_HOST_MEM_BACKEND", None)
if feature == 48543:
    os.environ["SGLANG_NPU_HICACHE_HOST_ALLOCATOR"] = "aclrt" if flag else "torch"
if feature == 48575:
    os.environ["SGLANG_NPU_PREWARM_PINNED_HOST"] = str(flag)
config = json.loads(Path("/model/config.json").read_text())
config = config.get("text_config", config)
layers = config["num_hidden_layers"]
widths = [config[key] for key in ("kv_lora_rank", "qk_rope_head_dim", "index_head_dim")]
pages = 385
page_size = 128
assert [layers, *widths] == [78, 512, 64, 128]


def timed(fn):
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    value = fn()
    torch.npu.synchronize()
    return value, (time.perf_counter_ns() - start) / 1e6


warm = lambda: None
if feature == 48575 and flag:
    from sglang.srt.hardware_backend.npu.pinned_host import maybe_prewarm_pinned_host

    def warm():
        return maybe_prewarm_pinned_host(device="npu", is_draft_worker=False)


_, startup = timed(warm)
views = []
mapping = []
for label, width in zip(("latent_k", "rope", "index_k"), widths):
    shape = (pages, layers, page_size, width)
    host, elapsed = timed(
        lambda: alloc_with_pin_memory_npu(shape, torch.bfloat16, "cpu", True, None)
    )
    mapping.append(
        dict(
            label=label,
            shape=shape,
            payload_bytes=host.numel() * host.element_size(),
            allocation_ms=elapsed,
        )
    )
    for page, layer in ((0, 0), (pages // 2, layers // 2), (pages - 1, layers - 1)):
        part = host[page, layer]
        part.fill_(len(views) + 7)
        views.append(part)
    del host
hashes = []
copies = []
for view in views:
    npu, to_npu = timed(lambda: view.to("npu", non_blocking=True))
    cpu, to_cpu = timed(lambda: npu.to("cpu", non_blocking=True))
    assert torch.equal(cpu, view)
    hashes.append(hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest())
    copies.append([to_npu, to_cpu])
output.write_text(
    json.dumps(
        dict(
            feature=feature,
            flag=flag,
            sha=os.environ["ARM_SHA"],
            device=torch.npu.get_device_name(0),
            logical_tokens=pages * page_size,
            mappings=mapping,
            hook_ms=startup,
            transfer_ms=copies,
            roundtrip_sha256=hashes,
        ),
        indent=2,
    )
    + "\n"
)
