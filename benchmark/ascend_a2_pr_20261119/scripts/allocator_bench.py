"""Time the production paged allocator and validate every emitted device slot."""

import hashlib, inspect, json, os, sys, time
from pathlib import Path
import torch, torch_npu
from sglang.srt.hardware_backend.npu.allocator_npu import NPUPagedTokenToKVPoolAllocator

torch.npu.set_device(0)
torch.set_num_threads(1)
dev = "npu:0"
page = 128
alloc = NPUPagedTokenToKVPoolAllocator(65536, page, torch.bfloat16, dev, None, False)
free = alloc.free_pages.clone()
lengths = [1, 17, 65, 128, 257, 513, 1025, 2047, 2048, 2049, 4097]
inputs = {}
for size in lengths:
    prefix = torch.tensor([0], dtype=torch.int64)
    seq = torch.tensor([size], dtype=torch.int64)
    inputs[size] = (
        prefix.to(dev),
        prefix,
        seq.to(dev),
        seq,
        torch.tensor([-1], device=dev, dtype=torch.int64),
    )


def run(size):
    alloc.free_pages = free
    torch.npu.synchronize()
    t = time.perf_counter_ns()
    got = alloc.alloc_extend(
        *inputs[size], size, num_new_pages=(size + page - 1) // page
    )
    torch.npu.synchronize()
    dt = (time.perf_counter_ns() - t) / 1e6
    return dt, got


rows = []
for size in lengths:
    elapsed, got = run(size)
    assert got.dtype == torch.int32 and got.numel() == size
    torch.testing.assert_close(
        got.cpu(), torch.arange(page, page + size, dtype=torch.int32), rtol=0, atol=0
    )
    rows.append(dict(length=size, first_ms=elapsed, warm_ms=[]))
for i in range(12):
    for row in rows if i % 2 == 0 else reversed(rows):
        elapsed, _ = run(row["length"])
        row["warm_ms"].append(elapsed)
root = Path(os.environ["TRITON_CACHE_DIR"])
arts = [
    str(p.relative_to(root))
    for p in sorted(root.rglob("*"))
    if p.is_file() and "alloc_extend_kernel" in p.name
]
path = Path(inspect.getfile(NPUPagedTokenToKVPoolAllocator))
out = dict(
    rows=rows,
    artifacts=arts,
    device=torch.npu.get_device_name(),
    torch=torch.__version__,
    torch_npu=torch_npu.__version__,
    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    sha=(path.parents[5] / "SHA").read_text().strip()
    if (path.parents[5] / "SHA").is_file()
    else os.environ["ARM_SHA"],
)
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + "\n")
print(
    json.dumps(
        {"samples": len(rows), "artifact_count": len(arts), "device": out["device"]}
    )
)
