"""Dump and restore full checkpoint expert matrices with the eight-rank presharded loader."""

import collections
import datetime
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

pr, flag, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
if "LOCAL_RANK" not in os.environ:
    with tempfile.TemporaryDirectory(dir=out.parent) as scratch:
        env = dict(
            os.environ,
            PRESHARD_TEST=scratch,
            HCCL_SOCKET_IFNAME="bond0",
            GLOO_SOCKET_IFNAME="lo",
            HCCL_INTRA_PCIE_ENABLE="1",
            HCCL_INTRA_ROCE_ENABLE="0",
            HCCL_BUFFSIZE="128",
            HCCL_CONNECT_TIMEOUT="180",
            HCCL_EXEC_TIMEOUT="180",
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
                str(pr),
                str(flag),
                str(out),
            ],
            env=env,
        )
        if proc.returncode:
            sys.exit(proc.returncode)
        rows = [
            json.loads((Path(scratch) / f"rank-{r}.json").read_text()) for r in range(8)
        ]
        out.write_text(
            json.dumps(
                dict(feature=pr, flag=flag, sha=os.environ["ARM_SHA"], ranks=rows),
                indent=2,
            )
            + "\n"
        )
    sys.exit()
import torch
import torch_npu
import torch.distributed as dist
from safetensors import safe_open
from sglang.srt.model_loader.loader import PreshardedModelLoader as Loader

rank = int(os.environ["LOCAL_RANK"])
world = int(os.environ["WORLD_SIZE"])
assert world == 8
torch.npu.set_device(rank)
torch.npu.config.allow_internal_format = True
dist.init_process_group("hccl", timeout=datetime.timedelta(seconds=200))
folder = Path(os.environ["PRESHARD_TEST"])
dest = folder / "checkpoint"
root = Path("/model")
index = json.loads((root / "quant_model_weights.safetensors.index.json").read_text())[
    "weight_map"
]
reference = {}
for projection in ("gate_proj", "up_proj", "down_proj"):
    name = f"model.layers.10.mlp.experts.{rank}.{projection}.weight"
    with safe_open(root / index[name], framework="pt", device="cpu") as f:
        original = f.get_tensor(name).contiguous()
    assert (
        original.dtype == torch.int8
        and original.ndim == 2
        and original.shape[-1] % 4 == 0
    )
    reference[projection] = original


def nz(x):
    packed = torch_npu.npu_format_cast(x.to("npu"), 29).view(torch.int32)
    assert (
        int(torch_npu.get_npu_format(packed)) == 29 and packed.dtype == torch.int32
    ), "could not allocate native INT32 FRACTAL_NZ"
    return packed


def digest(x):
    return hashlib.sha256(
        x.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def module(zero):
    m = torch.nn.Module()

    def tensor(k):
        x = reference[k]
        return nz(torch.zeros_like(x) if zero else x)

    for k in ("gate_proj", "up_proj"):
        m.register_parameter(k, torch.nn.Parameter(tensor(k), requires_grad=False))
    m.down_proj = tensor("down_proj")
    return m


# Standalone process group supplies the same rank and barrier interface as the server world group.
load = Loader.__new__(Loader)
load._world_rank_and_size = lambda: (rank, world)
load._world_barrier = dist.barrier
load._hash_num_threads = 1
load._max_file_bytes = 32 * 1024**2
load._verify_on_load = True
source = module(False)
extras = load._collect_extra_tensors(source)
assert set(extras) == {"down_proj"}
result = dict(
    rank=rank,
    required_bytes=sum(x.numel() * x.element_size() for x in reference.values()),
    shapes={k: list(x.shape) for k, x in reference.items()},
    reference_sha256={k: digest(x) for k, x in reference.items()},
)
try:
    load._ensure_presharded_dir_writable(str(dest))
    torch.npu.synchronize()
    t = time.perf_counter_ns()
    load._dump_state_to_disk(
        dict(source.state_dict()), extras, str(dest), dict(world_size=world, layer=10)
    )
    result["dump_ms"] = (time.perf_counter_ns() - t) / 1e6
    assert (dest / load.READY_FILENAME).is_file()
    target = module(True)
    before = {k: getattr(target, k).data_ptr() for k in reference}
    state = dict(target.state_dict())
    plan = json.loads((dest / load.CHECKSUM_FILENAME).read_text())
    grouped = collections.defaultdict(list)
    for entry in plan["rank_to_reads"][str(rank)]:
        grouped[entry["filename"]].append(entry)
    keys = set()
    hashes = []
    t = time.perf_counter_ns()
    for filename, items in grouped.items():
        stored = list(dict.fromkeys(x["stored_key"] for x in items))
        cache = load._read_presharded_file(str(dest / filename), stored)
        load._apply_presharded_file(
            items=items,
            cached=cache,
            model=target,
            state_dict=state,
            target_device=torch.device("npu", rank),
            loaded_param_keys=keys,
            verify_hashes=hashes,
        )
    load._verify_rank_checksum(hashes, plan, rank, str(dest))
    torch.npu.synchronize()
    result["read_verify_apply_ms"] = (time.perf_counter_ns() - t) / 1e6
    captured = {}
    for k, ref in reference.items():
        retrieved = getattr(target, k)
        words = (
            Loader._to_cpu_bits(retrieved, copy=True)
            if flag
            else retrieved.detach().to("cpu").contiguous()
        )
        assert torch.equal(words, ref.view(torch.int32)), (
            "checkpoint payload mismatch in " + k
        )
        assert retrieved.data_ptr() == before[k], "storage pointer changed for " + k
        assert int(torch_npu.get_npu_format(retrieved)) == 29, (
            "restored format changed for " + k
        )
        captured[k] = digest(words)
    assert keys == {"gate_proj", "up_proj"} and len(hashes) == 3
    result.update(
        ok=True,
        file_count=len(plan["files"]),
        verified_names=len(hashes),
        stored_sha256=captured,
        stable_pointers=3,
    )
except (RuntimeError, ValueError, AssertionError) as e:
    result.update(
        ok=False,
        stage="native checkpoint dump or reload",
        error=type(e).__name__,
        message=str(e).splitlines()[0][:180],
    )
(folder / f"rank-{rank}.json").write_text(json.dumps(result) + "\n")
dist.barrier()
dist.destroy_process_group()
