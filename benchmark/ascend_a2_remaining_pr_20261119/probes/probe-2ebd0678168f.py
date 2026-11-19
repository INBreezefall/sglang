"""Real Ascend device/host allocation, checkpoint, MoE, and hot/cold copy comparisons."""

import contextlib, gc, hashlib, json, os, statistics, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace as NS
import torch, torch_npu

feature, flag, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
torch.npu.set_device(0)
torch.set_num_threads(1)
torch.manual_seed(31601)


def sync():
    torch.npu.synchronize()


def measured(fn):
    sync()
    start = time.perf_counter_ns()
    v = fn()
    sync()
    return v, (time.perf_counter_ns() - start) / 1e6


def digest(x):
    return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def result(**d):
    out.write_text(
        json.dumps(
            dict(
                feature=feature,
                flag=flag,
                sha=os.environ["ARM_SHA"],
                device=torch.npu.get_device_name(0),
                **d,
            ),
            indent=2,
        )
        + "\n"
    )


def host_stats():
    fn = getattr(torch.npu, "host_memory_stats", None)
    if fn is None:
        return {}
    return {
        str(k): int(v)
        for k, v in fn().items()
        if isinstance(v, (int, float))
        and (
            "bytes" in k and "current" in k or k in ("num_host_alloc", "num_host_free")
        )
    }


if feature == 48543:
    from sglang.srt.mem_cache.pool_host.common import alloc_with_pin_memory_npu

    os.environ.pop("SGLANG_HICACHE_HOST_MEM_BACKEND", None)
    os.environ["SGLANG_NPU_HICACHE_HOST_ALLOCATOR"] = "aclrt" if flag else "torch"
    sizes = [3 * 1024**2 + 513, 8 * 1024**2 + 129, 17 * 1024**2 + 7]
    rows = []
    for k in range(9):
        size = sizes[k % len(sizes)]
        before = host_stats()
        host, ms = measured(
            lambda: alloc_with_pin_memory_npu((size,), torch.uint8, "cpu", True, None)
        )
        committed = host_stats()
        view = host[3:-2]
        del host
        gc.collect()
        # Only this view owns the original allocation throughout both DMA copies.
        view.fill_(17 + k)
        dev, dma = measured(lambda: view.to("npu", non_blocking=True))
        read, d2h = measured(lambda: dev.to("cpu", non_blocking=True))
        assert torch.equal(read, view)
        rows.append(
            dict(
                bytes=size,
                view_bytes=view.numel(),
                allocation_ms=ms,
                h2d_ms=dma,
                d2h_ms=d2h,
                hash=digest(read),
                before=before,
                after=committed,
            )
        )
        del view, dev, read
        gc.collect()
    result(rows=rows)
elif feature == 48575:
    os.environ["SGLANG_NPU_PREWARM_PINNED_HOST"] = "1" if flag else "0"

    def warm():
        if flag:
            from sglang.srt.hardware_backend.npu.pinned_host import (
                maybe_prewarm_pinned_host,
            )

            maybe_prewarm_pinned_host(device="npu", is_draft_worker=False)

    # Import before the timed startup hook; the baseline has no hook.
    if flag:
        from sglang.srt.hardware_backend.npu.pinned_host import (
            maybe_prewarm_pinned_host,
        )
    before = host_stats()
    _, hook = measured(warm)
    after = host_stats()
    rows = []
    for k in range(9):
        h, dt = measured(
            lambda: torch.empty(
                4 * 1024**2 + 127, dtype=torch.uint8, device="cpu", pin_memory=True
            )
        )
        h.fill_(k + 1)
        v, ms = measured(lambda: h.to("npu", non_blocking=True))
        check = v.cpu()
        assert torch.equal(h, check)
        rows.append(
            dict(allocate_ms=dt, h2d_ms=ms, pinned=h.is_pinned(), hash=digest(check))
        )
        del h, v, check
        gc.collect()
    result(reserve_mb=64, before=before, after_warmup=after, warmup_ms=hook, rows=rows)
elif feature == 48540:
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    if flag:
        from sglang.srt.layers.moe.prefill_moe_tile import tiled_experts
    # Four fixed experts with a top-1 route; every expert executes two native NPU matmuls.
    # One activation per input row; weights and inputs use exactly representable binary fractions.
    h, inner, experts_n = 256, 1024, 4
    generator = torch.Generator().manual_seed(48540)

    def matrix(start, end, shape):
        return (
            torch.randint(start, end, shape, generator=generator).to(torch.float32) / 8
        ).to("npu", torch.bfloat16)

    up = matrix(-4, 5, (experts_n, h, inner))
    down = matrix(-2, 3, (experts_n, inner, h))
    rows = []
    for count, tile in [(1536, 256), (3073, 256), (4096, 512)]:
        x = matrix(-4, 5, (count, h))
        ids = (
            torch.arange(count, dtype=torch.int32, device="npu")
            .remainder(experts_n)
            .unsqueeze(1)
        )
        routes = StandardTopKOutput(
            torch.ones((count, 1), device="npu", dtype=torch.float32), ids, None
        )
        # Tokens are already interleaved by expert in this fixed route, so each slice has a known lane.
        calls = []

        def expert(inputs, topk_output):
            # Avoid reusing the full activation temporary across tiles.
            outbuf = torch.empty_like(inputs)
            token_count = inputs.shape[0]
            calls.append(token_count)
            for e in range(experts_n):
                v = inputs[e::experts_n]
                z = torch.nn.functional.silu(v @ up[e])
                outbuf[e::experts_n] = (z @ down[e]) * topk_output.topk_weights[
                    e::experts_n
                ]
            return outbuf

        oracle = expert(x, routes)
        sync()
        res = []
        for k in range(10):
            calls.clear()
            torch.npu.reset_peak_memory_stats()
            initial = torch.npu.memory_allocated()
            actual, tm = measured(
                lambda: (
                    tiled_experts(expert, x, routes, tile)
                    if flag
                    else expert(x, routes)
                )
            )
            peak = torch.npu.max_memory_allocated() - initial
            assert torch.equal(actual, oracle)
            res.append(
                dict(
                    ms=tm,
                    peak_extra_bytes=peak,
                    max_input_rows=max(calls),
                    expert_calls=len(calls),
                )
            )
            del actual
        rows.append(
            dict(tokens=count, tile=tile, hash=digest(oracle.float()), trials=res[2:])
        )
    result(
        rows=rows,
        scope="Real matmul/SiLU expert fixture with precomputed top-1 routing; the full-model comparison exercises ModelSlim separately.",
    )
elif feature == 49029:
    from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMHATokenToKVPool
    from torch.utils._python_dispatch import TorchDispatchMode

    holder = NS(cpu_offloading_chunk_size=128)

    # Identical full NPU buffers in both arms; inject only the first pinned-copy failure.
    class CopyTrace(TorchDispatchMode):
        def __init__(self, inject):
            self.inject = inject
            self.flags = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if (
                func == torch.ops.aten._to_copy.default
                and args[0].device.type == "npu"
                and getattr(kwargs.get("device"), "type", None) == "cpu"
            ):
                nonblock = bool(kwargs.get("non_blocking", False))
                self.flags.append(nonblock)
                if self.inject and nonblock and len(self.flags) == 1:
                    raise torch.OutOfMemoryError(
                        "bounded injection at the pinned host copy boundary"
                    )
            return func(*args, **kwargs)

    source = torch.arange(1024 * 128, dtype=torch.int32).reshape(1024, 1, 128)
    layers = [[source.add(10 * i + j).to("npu") for j in range(2)] for i in range(4)]
    ind = torch.arange(1024, device="npu", dtype=torch.int64)
    rows = []
    for fault in (False, True):
        for k in range(7):
            trace = CopyTrace(fault)
            err = None
            hashes = []
            with trace:
                try:
                    v, dt = measured(
                        lambda: NPUMHATokenToKVPool._chunk_copy_npu_to_cpu(
                            holder, layers, ind
                        )
                    )
                except torch.OutOfMemoryError as e:
                    err = type(e).__name__
                    dt = None
            if not err:
                for i, layer in enumerate(v):
                    for j in range(2):
                        joined = torch.cat([part[j] for part in layer])
                        assert torch.equal(joined, source + 10 * i + j)
                        hashes.append(digest(joined))
            rows.append(
                dict(
                    injected=fault,
                    completed=err is None,
                    ms=dt,
                    copy_flags=trace.flags,
                    hashes=hashes,
                    error=err,
                )
            )
    result(
        rows=rows,
        scope="Healthy asynchronous backup and deterministic first-copy OOM injection followed by real pageable NPU-to-CPU copies.",
    )
elif feature == 48631:
    from sglang.srt.model_loader.loader import PreshardedModelLoader as L
    from safetensors.torch import load_file, save_file

    torch.npu.config.allow_internal_format = True
    # The real W4A8 producer packs eight signed 4-bit weights into an int32 word.
    raw = torch.randint(-7, 8, (256, 1024), dtype=torch.int32).to("npu")
    packed = torch_npu.npu_convert_weight_to_int4pack(raw)
    if torch_npu.get_npu_format(packed) != 29:
        # Use the vendor's byte NZ storage before reinterpreting packed words if the pack returned ND.
        packed = torch_npu.npu_format_cast(packed.view(torch.int8), 29).view(
            torch.int32
        )
    fmt = int(torch_npu.get_npu_format(packed))
    assert fmt == 29 and packed.dtype == torch.int32, (fmt, packed.dtype)
    rows = []
    for k in range(6):
        try:
            h, hash_ms = measured(lambda: L._hash_tensor(packed))
            cpu, export_ms = measured(
                lambda: (
                    L._to_cpu_bits(packed, copy=True)
                    if flag
                    else packed.detach().to("cpu", copy=True).contiguous()
                )
            )
            with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
                file = str(Path(tmp) / "weight.safetensors")
                save_file({"packed": cpu}, file)
                stored = load_file(file)["packed"]
                target = torch_npu.npu_format_cast(
                    torch.zeros_like(packed.view(torch.int8)), 29
                ).view(torch.int32)
                pointer = target.data_ptr()
                _, restore_ms = measured(
                    lambda: (
                        L._copy_packed_nz(target, stored)
                        if flag
                        else target.copy_(stored)
                    )
                )
                actual = L._to_cpu_bits(target, copy=True) if flag else target.cpu()
                assert torch.equal(actual, stored) and target.data_ptr() == pointer
            rows.append(
                dict(
                    ok=True,
                    hash=h,
                    stored=digest(stored),
                    hash_ms=hash_ms,
                    export_ms=export_ms,
                    restore_ms=restore_ms,
                )
            )
        except (RuntimeError, ValueError) as e:
            rows.append(
                dict(
                    ok=False,
                    error=type(e).__name__,
                    message=str(e).splitlines()[0][:300],
                )
            )
            break
    result(format=fmt, words=list(packed.shape), rows=rows)
else:
    raise ValueError(feature)
