"""Compare the real, eager Ascend sparse attention entry point on identical tensors."""

import hashlib, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace as NS
import torch, torch_npu
from sglang.srt.hardware_backend.npu.attention.ascend_backend import (
    AscendAttnBackend,
    ForwardMetadata,
)
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.srt.runtime_context import get_parallel

torch.npu.set_device(0)
torch.set_num_threads(1)
gen = torch.Generator().manual_seed(49117)


def data(*shape):
    return (torch.randn(shape, generator=gen) * 0.1).bfloat16().to("npu")


class HostCopies:
    def __init__(self, ls):
        self.x = torch.tensor(ls, dtype=torch.int32)
        self.count = 0

    def to(self, *a, **kw):
        self.count += 1
        return self.x.to(*a, **kw)


backend = object.__new__(AscendAttnBackend)
backend.dsa_dcp_target = False
backend.kv_cache_dtype = torch.bfloat16
backend.qk_rope_head_dim = 64
kv = data(2, 128, 1, 512)
rope = data(2, 128, 1, 64)
backend.token_to_kv_pool = NS(get_kv_buffer=lambda layer: (kv, rope))
layer = NS(layer_id=0, scaling=576**-0.5)
table = torch.tensor([[0], [1]], device="npu", dtype=torch.int32)
rows = []
with get_parallel().override(attn_cp_size=1, attn_tp_size=1):
    for mode, qlens, lens in [
        (ForwardMode.EXTEND, [2, 3], [48, 96]),
        (ForwardMode.DECODE, [1, 1], [40, 80]),
        (ForwardMode.TARGET_VERIFY, [3, 3], [43, 83]),
        (ForwardMode.DRAFT_EXTEND_V2, [3, 3], [41, 81]),
    ]:
        q = data(sum(qlens), 16, 512)
        qr = data(sum(qlens), 16, 64)
        sparse = torch.full((sum(qlens), 2048), -1, dtype=torch.int32, device="npu")
        offset = 0
        for a, b in zip(qlens, lens):
            sparse[offset : offset + a, :b] = torch.arange(b, device="npu")
            offset += a
        batch = NS(
            forward_mode=mode, attn_cp_metadata=None, npu_dsa_token_shard_plan=None
        )
        qn = torch.tensor(qlens, dtype=torch.int32, device="npu").cumsum(0)
        dn = torch.tensor([1, 1], dtype=torch.int32, device="npu")
        ms = []
        copies = []
        digest = None
        for rep in range(18):
            host = HostCopies(lens)
            backend.forward_metadata = ForwardMetadata(
                seq_lens_cpu_int=host,
                seq_lens=dn,
                actual_seq_lengths_q=qn,
                block_tables=table,
            )
            torch.npu.synchronize()
            t = time.perf_counter_ns()
            for _ in range(12):
                out = backend.forward_sparse(
                    q,
                    None,
                    None,
                    layer,
                    batch,
                    save_kv_cache=False,
                    q_rope=qr,
                    topk_indices=sparse,
                )
            torch.npu.synchronize()
            dt = (time.perf_counter_ns() - t) / 1e6
            val = out.cpu()
            assert torch.isfinite(val).all()
            h = hashlib.sha256(val.view(torch.uint8).numpy().tobytes()).hexdigest()
            if digest is not None:
                assert h == digest
            digest = h
            if rep >= 3:
                ms.append(dt)
                copies.append(host.count)
        rows.append(
            dict(
                mode=mode.name,
                lengths=lens,
                ms=ms,
                uploads=copies,
                output_sha256=digest,
                shape=list(out.shape),
                layers=12,
            )
        )
Path(sys.argv[1]).write_text(
    json.dumps(
        dict(sha=os.environ["ARM_SHA"], device=torch.npu.get_device_name(), rows=rows),
        indent=2,
    )
    + "\n"
)
