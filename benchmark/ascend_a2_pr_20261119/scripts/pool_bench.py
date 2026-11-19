"""Exercise the production A2 pool builder, PD buffer descriptors and native HiCache IO."""

from types import SimpleNamespace as N
import hashlib, json, os, sys, time
from pathlib import Path
import torch, torch_npu
from sglang.srt.mem_cache import kv_cache_configurator as cfg
from sglang.srt.mem_cache.pool_host.mla import MLATokenToKVPoolHost
from sglang.srt.model_executor.pool_configurator import DefaultPoolConfigurator
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.srt.disaggregation import utils

os.environ["SGLANG_HICACHE_HOST_MEM_BACKEND"] = "acl"
torch.npu.set_device(0)
torch.set_num_threads(1)
out = Path(sys.argv[1])
mode = sys.argv[2]
ns = 78
sz = 1024
page = 128
if mode == "compact":
    os.environ["SGLANG_NPU_COMPACT_DSA_INDEXER"] = "1"
config = json.loads(Path("/model/config.json").read_text())
hf = N(**config)
dtype = torch.int8 if mode in ("packed", "indexer") else torch.bfloat16
args = dict(
    page_size=page,
    disaggregation_mode="prefill",
    enable_hierarchical_cache=True,
    kv_cache_dtype="int8" if dtype == torch.int8 else "bf16",
    hicache_mem_layout="page_first_kv_split",
    enable_hisparse=False,
    enable_unified_cache_external_linker=False,
)
if mode == "indexer":
    args["dsa_indexer_kv_dtype"] = "int8"
co = N(
    modelslim_cache_layout=None,
    model_config=N(
        hf_config=hf, kv_lora_rank=512, qk_rope_head_dim=64, index_head_dim=128
    ),
    layer_info=N(start_layer=0, end_layer=ns, num_effective_layers=ns),
    server_args=N(enable_hisparse=False, indexer_kv_cache_dtype=None),
    mambaish_config=None,
    is_draft_worker=False,
    kv_cache_dtype=dtype,
    device="npu:0",
    use_combined_kv_buffer=False,
)
with (
    get_context().override_server_args(**args),
    get_parallel().override(
        pp_size=1, attn_cp_size=1, attn_tp_size=1, attn_dcp_size=1, attn_dcp_rank=0
    ),
):
    pool = cfg.KVCacheConfigurator._build_ascend_mla_kv_pool(
        co, max_total_num_tokens=sz, max_running_requests=8, is_dsa_model=True
    )
    lengths = pool.get_contiguous_buf_infos()[1]
    layers = pool.get_kv_layer_ids()
    tensors = [
        t
        for t in (
            pool.k_buffer,
            pool.v_buffer,
            pool.index_k_buffer,
            pool.index_k_scale_buffer,
        )
        if t is not None and t.numel()
    ]
    assert sum(lengths) == sum(t.nbytes for t in tensors) and len(layers) == len(
        lengths
    )
    rec = dict(
        sha=os.environ["ARM_SHA"],
        device=torch.npu.get_device_name(),
        mode=mode,
        physical_indexer_layers=pool.num_indexer_layers,
        layers=ns,
        token_rows=sz + page,
        registered_bytes=sum(lengths),
        indexer_bytes=pool.index_k_buffer.nbytes
        + (
            pool.index_k_scale_buffer.nbytes
            if pool.index_k_scale_buffer is not None
            else 0
        ),
        main_bytes=pool.k_buffer.nbytes + pool.v_buffer.nbytes,
        k_dtype=str(pool.k_buffer.dtype),
        index_dtype=str(pool.index_k_buffer.dtype),
        registered_entries=len(layers),
    )
    out.with_name("allocation.json").write_text(json.dumps(rec, indent=2))
    host = MLATokenToKVPoolHost(
        pool._mla_buffer_info(),
        host_to_device_ratio=2,
        host_size=0,
        page_size=page,
        layout="page_first_kv_split",
        pin_memory=True,
        device="cpu",
        device_capacity_tokens=sz,
        start_layer=0,
        end_layer=ns,
        target_layer_num=ns,
    )
    cpuids = (
        torch.cat([torch.arange(384, 512), torch.arange(128, 256)]).long().to("npu")
    )
    gpuids = torch.arange(256, dtype=torch.long, device="npu")
    # The ordinary kernel_ascend path accepts device and host indices; the binding performs its own transfer ordering.
    samples = []
    for iteration in range(9):
        originals = []
        for j, t in enumerate(tensors):
            vals = (
                torch.arange(
                    t.shape[0] * t.shape[1], dtype=torch.int32, device="npu"
                ).reshape(t.shape[0], t.shape[1])
                % 47
                + j
                + iteration
                + 1
            )
            shape = (t.shape[0], t.shape[1]) + ((1,) * (t.ndim - 2))
            t.copy_(vals.reshape(shape).expand_as(t).to(t.dtype))
            originals.append(vals.cpu())
        torch.npu.synchronize()
        t0 = time.perf_counter_ns()
        host.backup_from_device_all_layer(pool, cpuids, gpuids, "kernel_ascend")
        torch.npu.synchronize()
        t1 = time.perf_counter_ns()
        for t in tensors:
            t[:, :2].zero_()
        torch.npu.synchronize()
        t2 = time.perf_counter_ns()
        for layer in range(ns):
            host.load_to_device_per_layer(pool, cpuids, gpuids, layer, "kernel_ascend")
        torch.npu.synchronize()
        t3 = time.perf_counter_ns()
        for t, vals in zip(tensors, originals):
            shape = (t.shape[0], t.shape[1]) + ((1,) * (t.ndim - 2))
            expect = vals.to("npu", dtype=t.dtype).reshape(shape)
            assert bool((t == expect).all().item()), (mode, t.shape)
        if iteration >= 2:
            samples.append(dict(d2h_ms=(t1 - t0) / 1e6, h2d_ms=(t3 - t2) / 1e6))
    rec.update(
        roundtrips=9,
        bytes_per_direction=sum(t[:, 0:2].nbytes for t in tensors),
        samples=samples,
        host_budget_bytes_per_token=host.get_size_per_token(),
    )
    out.write_text(json.dumps(rec, indent=2) + "\n")
