import json
import os
import sys
from pathlib import Path
import time

node = int(Path("/sys/bus/pci/devices/0000:c1:00.0/numa_node").read_text())
cpu_range = Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip()
local_cpus = set()
for part in cpu_range.split(","):
    ends = [int(x) for x in part.split("-")]
    local_cpus.update(range(ends[0], ends[-1] + 1))
os.sched_setaffinity(0, local_cpus.intersection(os.sched_getaffinity(0)))

import torch
import torch_npu

torch.set_num_threads(4)
torch.npu.set_device(0)
for variant in ["main", "pr"]:
    torch.ops.load_library(f"/bench/paired-{variant}/lib/libsgl_kernel_npu.so")
ops = {"main": (torch.ops.evidence_main.transfer_kv_dim_exchange, 2),
       "off": (torch.ops.evidence_pr.transfer_kv_dim_exchange, 2),
       "on": (torch.ops.evidence_pr.transfer_kv_dim_exchange, 6)}

def pinned_placement(buf):
    address = buf.data_ptr()
    for area in Path("/proc/self/maps").read_text().splitlines():
        lo, hi = (int(x, 16) for x in area.split(" ", 1)[0].split("-"))
        if lo <= address < hi:
            for row in Path("/proc/self/numa_maps").read_text().splitlines():
                fields = row.split()
                if int(fields[0], 16) == lo:
                    return [f for f in fields if f.startswith("N") and "=" in f]
    return []

shapes = [(128, 656, "run"), (128, 656, "gaps"), (256, 128, "run"),
          (128, 576, "run"), (128, 576, "gaps"), (32, 128, "run"), (256, 4, "run")]
page_size, layers = 128, 78
if "--model-layout" in sys.argv:
    # The target has 78 MLA layers and 21 physical indexer layers. The MTP draft
    # has its own pool. Generate synthetic bytes; no checkpoint data enters this test.
    for rep in range(2):
        layouts = [(32, "run"), (128, "run"), (128, "gaps"), (256, "run")]
        for pages, mapping in layouts[::(1 if rep == 0 else -1)]:
            cap = pages + 2
            dst = torch.arange(1, pages + 1)
            src = dst if mapping == "run" else torch.cat((dst[::2], dst[1::2]))
            di = (dst[:, None] * page_size + torch.arange(page_size)).flatten()
            hi = (src[:, None] * page_size + torch.arange(page_size)).flatten()
            parts = []
            byte_count = 0
            for n, width, dtype in [(78, 656, torch.int8), (21, 128, torch.int8),
                                    (21, 1, torch.float16)]:
                cpu = torch.empty((cap, n, page_size, 1, width), dtype=dtype, pin_memory=True)
                cpu.copy_(((torch.arange(cap)[:, None] * 17 + torch.arange(n)[None, :] * 23) % 113)[:, :, None, None, None])
                npu = torch.empty((n, cap, page_size, 1, width), dtype=dtype, device="npu")
                parts.append((npu, cpu, torch.empty(0, 0, 0, 0, width, dtype=dtype)))
                byte_count += n * pages * page_size * width * cpu.element_size()
            samples = []
            for cycle in range(8):
                seq = ["main", "off", "on"] if cycle % 2 == 0 else ["on", "off", "main"]
                shift = (cycle // 2) % len(seq)
                for arm in seq[shift:] + seq[:shift]:
                    op, flag = ops[arm]
                    if cycle == 0:
                        for npu, _, _ in parts:
                            npu.fill_(-1)
                    torch.npu.synchronize()
                    begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                    begin.record()
                    t0 = time.perf_counter()
                    for npu, cpu, empty in parts:
                        op(npu, cpu, empty, empty, di, hi, page_size, 1, flag)
                    submit = 1000 * (time.perf_counter() - t0)
                    end.record()
                    end.synchronize()
                    wall = 1000 * (time.perf_counter() - t0)
                    if cycle == 0:
                        for npu, cpu, _ in parts:
                            for layer in range(npu.shape[0]):
                                actual = npu[layer].cpu()
                                assert actual[0].eq(-1).all() and actual[-1].eq(-1).all()
                                assert torch.equal(actual[1:-1], cpu[:, layer].index_select(0, src))
                    else:
                        samples.append(dict(arm=arm, cycle=cycle, submit_ms=submit,
                                            stream_ms=begin.elapsed_time(end), wall_ms=wall))
            print(json.dumps(dict(rep=rep, components=[[78,656,"int8"],[21,128,"int8"],[21,1,"fp16"]],
                                  bytes=byte_count, pages=pages, mapping=mapping, correct=True,
                                  numa_node=node, rows=samples)), flush=True)
            del parts, npu, cpu
    sys.exit(0)
for rep in range(2):
    for pages, width, mapping in shapes[::(1 if rep == 0 else -1)]:
        cap = pages + 2
        host = torch.empty((cap, layers, page_size, 1, width), dtype=torch.int8,
                           pin_memory=True)
        pattern = (torch.arange(cap)[:, None] * 17 + torch.arange(layers)[None, :] * 23) % 113
        host.copy_(pattern[:, :, None, None, None])
        dev = torch.empty((layers, cap, page_size, 1, width), dtype=torch.int8, device="npu")
        empty = torch.empty(0, 0, 0, 0, width)
        chosen = torch.arange(1, pages + 1)
        source = chosen if mapping == "run" else torch.cat((chosen[::2], chosen[1::2]))
        di = (chosen[:, None] * page_size + torch.arange(page_size)).flatten()
        hi = (source[:, None] * page_size + torch.arange(page_size)).flatten()
        rows = []
        for cycle in range(8):
            seq = ["main", "off", "on"] if cycle % 2 == 0 else ["on", "off", "main"]
            shift = (cycle // 2) % 3
            for arm in seq[shift:] + seq[:shift]:
                op, flag = ops[arm]
                if cycle == 0:
                    dev.fill_(-1)
                torch.npu.synchronize()
                begin, end = (torch.npu.Event(enable_timing=True) for _ in range(2))
                begin.record()
                t0 = time.perf_counter()
                op(dev, host, empty, empty, di, hi, page_size, 1, flag)
                submit_ms = 1000 * (time.perf_counter() - t0)
                end.record()
                end.synchronize()
                wall_ms = 1000 * (time.perf_counter() - t0)
                if cycle == 0:
                    for layer in range(layers):
                        result = dev[layer].cpu()
                        assert result[0].eq(-1).all() and result[-1].eq(-1).all()
                        assert torch.equal(result[1:-1], host[:, layer].index_select(0, source))
                else:
                    rows.append(dict(arm=arm, cycle=cycle, submit_ms=submit_ms,
                                     stream_ms=begin.elapsed_time(end), wall_ms=wall_ms))
        print(json.dumps(dict(rep=rep, layers=layers, pages=pages, width=width, mapping=mapping,
                              correct=True, numa_node=node, host_vma=pinned_placement(host),
                              bytes=pages*layers*page_size*width, rows=rows)), flush=True)
        del host, dev
