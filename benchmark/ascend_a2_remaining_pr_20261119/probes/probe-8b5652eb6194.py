"""Eight physical Ascend ranks exercise SGLang's normal and low-latency dtype policy."""

import datetime, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path

feature, flag, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
if "LOCAL_RANK" not in os.environ:
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        env = dict(
            os.environ,
            DEEPEP_OUT=tmp,
            DEEP_NORMAL_MODE_USE_INT8_QUANT="0",
            SGLANG_NPU_DEEPEP_NORMAL_BF16=str(flag),
            HCCL_CONNECT_TIMEOUT="180",
            HCCL_EXEC_TIMEOUT="180",
            HCCL_BUFFSIZE="1536",
            GLOO_SOCKET_IFNAME="lo",
            HCCL_SOCKET_IFNAME="bond0",
            HCCL_INTRA_PCIE_ENABLE="1",
            HCCL_INTRA_ROCE_ENABLE="0",
        )
        vendor = next(
            Path("/usr/local").glob(
                "python*/lib/python*/site-packages/deep_ep/vendors/hwcomputing/op_api/lib"
            )
        )
        lib = vendor / "libcust_opapi.so"
        env["LD_LIBRARY_PATH"] = str(vendor) + ":" + env.get("LD_LIBRARY_PATH", "")
        env["ASCEND_CUSTOM_OPP_PATH"] = (
            str(vendor.parents[1]) + ":" + env.get("ASCEND_CUSTOM_OPP_PATH", "")
        )
        if lib.exists():
            env["LD_PRELOAD"] = str(lib) + (
                ":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else ""
            )
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nnodes=1",
                "--nproc-per-node=8",
                __file__,
                str(feature),
                str(flag),
                str(out),
            ],
            env=env,
        )
        if proc.returncode:
            sys.exit(proc.returncode)
        ranks = [
            json.loads((Path(tmp) / f"rank{i}.json").read_text()) for i in range(8)
        ]
        out.write_text(
            json.dumps(
                dict(
                    feature=feature, flag=flag, sha=os.environ["ARM_SHA"], ranks=ranks
                ),
                indent=2,
            )
            + "\n"
        )
    sys.exit()
import torch, torch_npu, torch.distributed as dist
from deep_ep import Buffer
from sglang.srt.runtime_context import get_context
from sglang.srt.layers.moe.token_dispatcher import deepep

rank = int(os.environ["LOCAL_RANK"])
size = int(os.environ["WORLD_SIZE"])
assert size == 8
torch.npu.set_device(rank)
torch.set_num_threads(1)
dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=180))
opts = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
opts.hccl_config = {"hccl_buffer_size": 1536}
group = dist.new_group(
    list(range(size)),
    backend="hccl",
    timeout=datetime.timedelta(seconds=180),
    pg_options=opts,
)
dist.barrier(group=group)
h = 6144
experts = 256
top = 8
cap = 32
dconf = Buffer.get_dispatch_config(size)
cconf = Buffer.get_combine_config(size)
nvl = max(c.get_nvl_buffer_size_hint(h * 2, size) for c in (dconf, cconf))
rdma = max(
    Buffer.get_low_latency_rdma_size_hint(cap, h, size, experts),
    *(c.get_rdma_buffer_size_hint(h * 2, size) for c in (dconf, cconf)),
)
buf = Buffer(
    group,
    num_nvl_bytes=nvl,
    num_rdma_bytes=rdma,
    low_latency_mode=True,
    num_qps_per_rank=experts // size,
)


def dequant(v):
    if not isinstance(v, tuple):
        return v, None
    q, scale = v
    assert q.dtype == torch.int8 and scale.dtype == torch.float32, (
        q.dtype,
        scale.dtype,
    )
    group_len = q.numel() // max(1, scale.numel())
    z = (
        (q.float().reshape(-1, group_len) * scale.reshape(-1, 1))
        .reshape(q.shape)
        .to(torch.bfloat16)
    )
    return z, list(scale.shape)


def policy(cls):
    obj = cls.__new__(cls)
    obj.quant_config = {"dispatcher_output_dtype": "int8"}
    obj.set_deepep_dispatcher_dtype()
    return obj


rows = []
with get_context().override_server_args(
    device="npu", deepep_dispatcher_output_dtype="int8"
):
    normal = policy(deepep._DeepEPDispatcherImplNormal)
    low = policy(deepep._DeepEPDispatcherImplLowLatency)
    for mode, obj in [("normal", normal), ("low_latency", low)]:
        for tokens in (8, 32):
            g = torch.Generator().manual_seed(48740 + rank + tokens)
            x = (torch.randn((tokens, h), generator=g) * 0.25).to("npu", torch.bfloat16)
            idx = (
                (
                    torch.arange(tokens, device="npu")[:, None] * 13
                    + torch.arange(top, device="npu")[None, :] * 31
                    + rank * 17
                )
                % experts
            ).long()
            weights = torch.full(
                (tokens, top), 1 / top, device="npu", dtype=torch.float32
            )
            for k in range(6):
                dist.barrier(group=group)
                torch.npu.synchronize()
                t = time.perf_counter_ns()
                if mode == "normal":
                    nr, rd, ne, mask, _ = buf.get_dispatch_layout(idx, experts)
                    received, local_ids, w, counts, handle, _ = buf.dispatch(
                        x=x,
                        num_tokens_per_rank=nr,
                        num_tokens_per_rdma_rank=rd,
                        is_token_in_rank=mask,
                        num_tokens_per_expert=ne,
                        config=dconf,
                        topk_idx=idx,
                        topk_weights=weights,
                        **normal._get_quantization_kwargs(buf),
                    )
                    torch.npu.synchronize()
                    split = time.perf_counter_ns()
                    z, scales = dequant(received)
                    cw = (
                        handle["topk_weights"]
                        if isinstance(handle, dict)
                        else handle[7]
                    )
                    y, _, _ = buf.combine(
                        x=z,
                        handle=handle,
                        config=cconf,
                        topk_weights=cw,
                        async_finish=False,
                    )
                else:
                    received, counts, handle, _, _ = buf.low_latency_dispatch(
                        x,
                        idx,
                        cap,
                        experts,
                        use_fp8=low.use_fp8,
                        round_scale=False,
                        async_finish=False,
                        return_recv_hook=False,
                        topk_weights=weights,
                    )
                    torch.npu.synchronize()
                    split = time.perf_counter_ns()
                    z, scales = dequant(received)
                    y, _, _ = buf.low_latency_combine(
                        z,
                        idx,
                        weights,
                        handle,
                        async_finish=False,
                        return_recv_hook=False,
                    )
                torch.npu.synchronize()
                end = time.perf_counter_ns()
                assert y.shape == x.shape and torch.isfinite(y).all().item()
                diff = float(
                    (
                        (y.float() - x.float()).square().mean()
                        / x.float().square().mean()
                    )
                    .sqrt()
                    .item()
                )
                if k >= 2:
                    rows.append(
                        dict(
                            mode=mode,
                            tokens=tokens,
                            selected=obj.deepep_output_dtype.value,
                            wire_dtype=str(
                                (
                                    received[0]
                                    if isinstance(received, tuple)
                                    else received
                                ).dtype
                            ),
                            scale_shape=scales,
                            dispatch_ms=(split - t) / 1e6,
                            roundtrip_ms=(end - t) / 1e6,
                            relative_rms=diff,
                            output_hash=hashlib.sha256(
                                y.float().cpu().numpy().tobytes()
                            ).hexdigest(),
                        )
                    )
(Path(os.environ["DEEPEP_OUT"]) / f"rank{rank}.json").write_text(
    json.dumps(rows) + "\n"
)
dist.barrier()
dist.destroy_process_group()
