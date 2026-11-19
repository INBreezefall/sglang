"""Exercise isolated Mooncake changes across two physical Ascend hosts."""

import concurrent.futures
import contextlib
import hashlib
import json
import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu
from mooncake.engine import TransferEngine
from sglang.srt.disaggregation.mooncake import conn
from sglang.srt.distributed.device_communicators import mooncake_transfer_engine as mc
from sglang.srt.utils.network import NetworkAddress, get_free_port

pr, flag, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
role = int(os.environ["CROSS_RANK"])
host = os.environ["LOCAL_ADDR"]
peer_host = os.environ["PEER_ADDR"]
port = int(os.environ["RENDEZVOUS_PORT"])
device = 1 if role == 0 else 2
torch.npu.set_device(device)
torch.set_num_threads(1)
os.environ["ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE"] = "1"
os.environ["SGLANG_MOONCAKE_ASCEND_REGISTER_ALLOCATION"] = (
    "1" if flag and pr == 48537 else "0"
)
os.environ["SGLANG_MOONCAKE_ASCEND_USE_DEVICE_SUFFIX"] = (
    "0" if flag and pr == 49077 else "1"
)
width = 1024**2
offset = 128 * 1024
count = 8
data = torch.zeros(offset + (count + 1) * width, device="npu", dtype=torch.uint8)
views = [data[offset + i * width : offset + (i + 1) * width] for i in range(count)]
for i, v in enumerate(views):
    v.fill_(31 + i if role == 0 else 0)
torch.npu.synchronize()
args = SimpleNamespace(
    kv_data_ptrs=[x.data_ptr() for x in views],
    kv_data_lens=[width] * count,
    aux_data_ptrs=[],
    aux_data_lens=[],
    state_data_ptrs=[],
    state_data_lens=[],
)
regions = (
    conn.MooncakeKVManager._registerable_regions(SimpleNamespace(kv_args=args))
    if pr == 48537
    else [(data.data_ptr(), data.numel())]
)


def connect():
    if not role:
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(1)
            listener.settimeout(80)
            c, peer = listener.accept()
            assert peer[0] == peer_host
            return c
    deadline = time.monotonic() + 75
    while True:
        try:
            return socket.create_connection((peer_host, port), timeout=4)
        except OSError:
            if time.monotonic() > deadline:
                raise
            time.sleep(0.25)


def send(value):
    stream.write((json.dumps(value) + "\n").encode())
    stream.flush()


def recv():
    raw = stream.readline(65536)
    if not raw:
        raise EOFError("peer exited")
    return json.loads(raw)


def fixed_endpoint():
    native = TransferEngine()
    rc = native.initialize(
        host + ":" + str(get_free_port()), "P2PHANDSHAKE", "ascend", ""
    )
    if rc:
        raise RuntimeError("native initialization rc " + str(rc))
    peer_id = NetworkAddress(host, native.get_rpc_port()).to_host_port_str()
    wrapper = mc.MooncakeTransferEngine.__new__(mc.MooncakeTransferEngine)
    wrapper.engine = native
    return native, peer_id, wrapper


def named_endpoint():
    original = mc.run_with_deadline

    def bound(fn, **kw):
        def fn_on_device():
            torch.npu.set_device(device)
            return fn()

        return original(fn_on_device, **kw)

    with patch.object(mc, "run_with_deadline", bound):
        wrapper = mc.MooncakeTransferEngine(
            hostname=host, gpu_id=device, ib_device=None
        )
    return wrapper.engine, wrapper.get_session_id(), wrapper


def transfer_on_worker(wrapper, peer, stack):
    def common_init(m, args, mode, *unused, **kwargs):
        m.kv_args = args
        m.disaggregation_mode = mode

    for key, x in [
        ("SGLANG_DISAGG_STAGING_BUFFER", False),
        ("SGLANG_ENABLE_FAILED_SESSION_PROBE", False),
        ("SGLANG_DISAGGREGATION_THREAD_POOL_SIZE", 1),
        ("SGLANG_DISAGGREGATION_QUEUE_SIZE", 1),
    ]:
        stack.enter_context(getattr(conn.envs, key).override(x))
    stack.enter_context(patch.object(conn.CommonKVManager, "__init__", common_init))
    stack.enter_context(
        patch.object(
            conn.MooncakeKVManager,
            "init_engine",
            lambda m: setattr(m, "engine", wrapper),
        )
    )
    for name in ("register_buffer_to_engine", "start_prefill_thread"):
        stack.enter_context(patch.object(conn.MooncakeKVManager, name, lambda *a: None))
    stack.enter_context(
        patch.object(
            conn, "check_mooncake_custom_mem_pool_enabled", return_value=(False, None)
        )
    )
    stack.enter_context(
        patch.object(
            conn, "get_observability", return_value=SimpleNamespace(enable_trace=False)
        )
    )
    m = conn.MooncakeKVManager(
        SimpleNamespace(gpu_id=device), conn.DisaggregationMode.PREFILL, None
    )
    try:

        def run_transfer():
            blocks = [
                (v.data_ptr(), peer["addresses"][i], width) for i, v in enumerate(views)
            ]
            begin = time.perf_counter_ns()
            rc = m._transfer_data(peer["sid"], blocks)
            return dict(
                rc=int(rc),
                batch_slices=len(blocks),
                ms=(time.perf_counter_ns() - begin) / 1e6,
                executor_device=torch.npu.current_device(),
            )

        return [m.executors[0].submit(run_transfer).result(timeout=70)]
    finally:
        m.transfer_queues[0].put(None)
        m._worker_threads[0].join(8)
        m.executors[0].shutdown(wait=True)


stats = dict(
    feature=pr,
    flag=flag,
    sha=os.environ["ARM_SHA"],
    role=role,
    physical_device=device,
    views=count,
    regions=len(regions),
    planned_bytes=sum(n for _, n in regions),
)
with connect() as channel, contextlib.ExitStack() as stack:
    channel.settimeout(100)
    with channel.makefile("rwb") as stream:
        send([pr, flag, os.environ["CROSS_ROUND"]])
        assert recv() == [pr, flag, os.environ["CROSS_ROUND"]]
        native = None
        codes = []
        start = time.perf_counter_ns()
        try:
            native, sid, wrapper = named_endpoint() if pr == 49077 else fixed_endpoint()
            for ptr, n in regions:
                codes.append(int(native.register_memory(ptr, n)))
            ready = bool(codes) and all(x == 0 for x in codes)
        except Exception as err:
            stats["initialization_error"] = type(err).__name__
            sid = ""
            ready = False
        stats.update(
            registration_ms=(time.perf_counter_ns() - start) / 1e6,
            registration_rc=codes,
        )
        send(dict(ok=ready, sid=sid, addresses=[v.data_ptr() for v in views]))
        peer = recv()
        if role == 0:
            results = []
            if ready and peer["ok"]:
                try:
                    stats["probe_rc"] = int(native.send_probe(peer["sid"]))
                    if pr == 48632:
                        results = transfer_on_worker(wrapper, peer, stack)
                    else:
                        for i, v in enumerate(views):
                            t = time.perf_counter_ns()
                            rc = int(
                                native.transfer_sync_write(
                                    peer["sid"],
                                    v.data_ptr(),
                                    peer["addresses"][i],
                                    width,
                                )
                            )
                            results.append(
                                dict(rc=rc, ms=(time.perf_counter_ns() - t) / 1e6)
                            )
                            if rc < 0:
                                break
                except Exception as err:
                    stats["transfer_error"] = type(err).__name__
            stats["writes"] = results
            send(dict(done=True))
            stats["destination"] = recv()
        else:
            assert recv()["done"]
            torch.npu.synchronize()
            checks = [v.cpu() for v in views]
            verified = dict(
                correct_slices=sum(
                    bool((v == 31 + i).all()) for i, v in enumerate(checks)
                ),
                count=count,
                data_sha256=hashlib.sha256(
                    b"".join(v.numpy().tobytes() for v in checks)
                ).hexdigest(),
            )
            stats.update(verified)
            send(verified)
        send("captured")
        assert recv() == "captured"
        if native is not None:
            for (ptr, _), rc in zip(regions, codes):
                if rc == 0:
                    native.unregister_memory(ptr)
out.write_text(json.dumps(stats, indent=2) + "\n")
