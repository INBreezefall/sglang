"""Compare actual BF16 and packed INT8 pool writes and sparse attention with a floating-point oracle."""

import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace as N
import torch, torch_npu
from sglang.srt.hardware_backend.npu.memory_pool_npu import NPUMLATokenToKVPool
from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    ForwardMetadata,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_context, get_parallel

torch.npu.set_device(0)
torch.set_num_threads(1)
dest = Path(sys.argv[1])
mode = sys.argv[2]
packed = mode == "int8"
bs = 2
page = 128
heads = 16
lengths = [96, 115]
gen = torch.Generator().manual_seed(48560)


def gen_bf(*shape):
    return (torch.randn(shape, generator=gen) * 0.25).bfloat16()


latent = gen_bf(256, 1, 512)
rope = gen_bf(256, 1, 64)
q = gen_bf(bs, heads, 512)
qr = gen_bf(bs, heads, 64)
kl = latent.to("npu")
kr = rope.to("npu")
qd = q.to("npu")
qrd = qr.to("npu")
ids = torch.arange(256, dtype=torch.int32, device="npu")
with (
    get_context().override_server_args(page_size=page),
    get_parallel().override(
        attn_tp_size=1, attn_cp_size=1, attn_dcp_size=1, attn_dcp_rank=0
    ),
):
    kw = dict(
        size=256,
        page_size=page,
        dtype=torch.int8 if packed else torch.bfloat16,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        layer_num=1,
        device="npu:0",
        enable_memory_saver=False,
        index_head_dim=128,
        kv_cache_dim=656 if packed else None,
    )
    if packed:
        from sglang.srt.hardware_backend.npu.kv_capability import (
            resolve_npu_kv_capability,
        )

        kw["kv_capability"] = resolve_npu_kv_capability(torch.int8)
    pool = NPUMLATokenToKVPool(**kw)
    layer = N(layer_id=0, scaling=576**-0.5)
    pool.set_kv_buffer(layer, ids, kl, kr)
    backend = object.__new__(AscendAttnBackend)
    backend.dsa_dcp_target = False
    backend.kv_cache_dtype = kw["dtype"]
    backend.qk_rope_head_dim = 64
    if hasattr(pool, "kv_capability"):
        backend.kv_capability = pool.kv_capability
    backend.token_to_kv_pool = pool
    lens = torch.tensor(lengths, dtype=torch.int32, device="npu")
    backend.forward_metadata = ForwardMetadata(
        seq_lens=lens,
        seq_lens_cpu_int=lens.cpu(),
        actual_seq_lengths_q=torch.arange(1, bs + 1, dtype=torch.int32, device="npu"),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32, device="npu"),
    )
    sparse = torch.full((bs, 2048), -1, dtype=torch.int32, device="npu")
    batch = N(
        forward_mode=ForwardMode.DECODE,
        attn_cp_metadata=None,
        npu_dsa_token_shard_plan=None,
    )
    oracle = []
    for b, n in enumerate(lengths):
        sparse[b, :n] = torch.arange(n, device="npu")
        kval = latent[b * page : b * page + n, 0].float()
        rval = rope[b * page : b * page + n, 0].float()
        scores = (q[b].float() @ kval.T + qr[b].float() @ rval.T) * layer.scaling
        oracle.append(scores.softmax(-1) @ kval)
    gold = torch.stack(oracle)
    ms = []
    for i in range(38):
        torch.npu.synchronize()
        t = time.perf_counter_ns()
        val = backend.forward_sparse(
            qd,
            None,
            None,
            layer,
            batch,
            save_kv_cache=False,
            q_rope=qrd,
            topk_indices=sparse,
        )
        torch.npu.synchronize()
        if i >= 6:
            ms.append((time.perf_counter_ns() - t) / 1e6)
    val = val.cpu().float().reshape_as(gold)
    err = val - gold
    rel = (torch.linalg.vector_norm(err) / torch.linalg.vector_norm(gold)).item()
    assert torch.isfinite(val).all() and rel < 0.04, rel
    res = dict(
        sha=os.environ["ARM_SHA"],
        device=torch.npu.get_device_name(),
        mode=mode,
        lengths=lengths,
        heads=heads,
        layers=1,
        main_bytes=pool.k_buffer.nbytes + pool.v_buffer.nbytes,
        row_bytes=(pool.k_buffer.nbytes + pool.v_buffer.nbytes) // (kw["size"] + page),
        relative_l2=rel,
        max_abs_err=err.abs().max().item(),
        ms=ms,
    )
    dest.write_text(json.dumps(res, indent=2) + "\n")
