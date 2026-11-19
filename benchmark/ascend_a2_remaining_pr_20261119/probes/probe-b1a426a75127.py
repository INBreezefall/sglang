"""Exercise the native Ascend Mooncake endpoint and registered slice geometry on two NPUs."""

import base64, hashlib, json, os, subprocess, sys, tempfile, time
from pathlib import Path
from types import SimpleNamespace as N

feature, flag, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
os.environ["ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE"] = "1"
os.environ["SGLANG_MOONCAKE_ASCEND_USE_DEVICE_SUFFIX"] = (
    "0" if flag and feature == 49077 else "1"
)
os.environ["SGLANG_MOONCAKE_ASCEND_REGISTER_ALLOCATION"] = (
    "1" if flag and feature == 48537 else "0"
)


def write(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def wait(path, limit=45):
    d = time.monotonic() + limit
    while not path.exists():
        if time.monotonic() > d:
            raise TimeoutError("local peer handshake")
        time.sleep(0.025)
    return json.loads(path.read_text())


def child(root, rank):
    import torch, torch_npu
    from mooncake.engine import TransferEngine
    from sglang.srt.distributed.device_communicators import (
        mooncake_transfer_engine as mc,
    )
    from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
    from sglang.srt.utils.network import get_free_port

    torch.npu.set_device(rank)
    host = os.environ["TEST_HOST_ADDR"]
    width = 1024 * 1024
    offset = 128 * 1024
    nviews = 8
    size = offset + (nviews + 1) * width
    data = torch.zeros(size, device="npu:" + str(rank), dtype=torch.uint8)
    sections = [
        data[offset + i * width : offset + (i + 1) * width] for i in range(nviews)
    ]
    for i, v in enumerate(sections):
        v.fill_((17 + i) if rank == 0 else 0)
    torch.npu.synchronize()
    args = N(
        kv_data_ptrs=[v.data_ptr() for v in sections],
        kv_data_lens=[width] * nviews,
        aux_data_ptrs=[],
        aux_data_lens=[],
        state_data_ptrs=[],
        state_data_lens=[],
    )
    if feature == 48537:
        regions = MooncakeKVManager._registerable_regions(N(kv_args=args))
    else:
        regions = [(data.data_ptr(), data.numel())]
    stats = dict(
        rank=rank,
        views=nviews,
        regions=len(regions),
        registered_bytes=sum(s for _, s in regions),
        npu=rank,
    )
    engine = None
    mc_engine = None
    try:
        if feature == 48537:
            from sglang.srt.utils.network import NetworkAddress

            # Fix the endpoint format on both arms, isolating the registered regions.
            engine = TransferEngine()
            port = get_free_port()
            rc = engine.initialize(host + ":" + str(port), "P2PHANDSHAKE", "ascend", "")
            if rc:
                raise RuntimeError("native initialize returned " + str(rc))
            sid = NetworkAddress(host, engine.get_rpc_port()).to_host_port_str()
        else:
            # The CANN device is thread-local. Both arms bind it in SGLang's launch-deadline thread.
            orig = mc.run_with_deadline

            def bound(fn, **kw):
                def call():
                    torch.npu.set_device(rank)
                    return fn()

                return orig(call, **kw)

            mc.run_with_deadline = bound
            mc_engine = mc.MooncakeTransferEngine(
                hostname=host, gpu_id=rank, ib_device=None
            )
            engine = mc_engine.engine
            sid = mc_engine.get_session_id()
        t = time.perf_counter_ns()
        codes = []
        for address, n in regions:
            codes.append(int(engine.register_memory(address, n)))
        stats.update(register_rc=codes, register_ms=(time.perf_counter_ns() - t) / 1e6)
        ready = all(v == 0 for v in codes)
        write(
            root / f"ready{rank}.json",
            dict(ok=ready, sid=sid, addresses=[v.data_ptr() for v in sections]),
        )
        peer = wait(root / f"ready{1 - rank}.json")
        if rank == 0:
            sent = []
            probe = None
            lookup = None
            if ready and peer.get("ok"):
                try:
                    probe = int(engine.send_probe(peer["sid"]))
                except Exception:
                    pass
                try:
                    lookup = bool(engine.get_first_buffer_address(peer["sid"]))
                except Exception:
                    pass
                for i, v in enumerate(sections):
                    ts = time.perf_counter_ns()
                    rc = int(
                        engine.transfer_sync_write(
                            peer["sid"], v.data_ptr(), peer["addresses"][i], v.numel()
                        )
                    )
                    sent.append(dict(rc=rc, ms=(time.perf_counter_ns() - ts) / 1e6))
                    if rc < 0:
                        break
            stats.update(
                send_probe=probe, peer_first_address_resolved=lookup, transfers=sent
            )
            write(
                root / "sent.json",
                dict(complete=len(sent) == nviews and all(x["rc"] >= 0 for x in sent)),
            )
            stats.update(receiver=wait(root / "checked.json"))
        else:
            tx = wait(root / "sent.json", 85)
            torch.npu.synchronize()
            bits = [v.cpu() for v in sections]
            checks = [bool((v == 17 + i).all()) for i, v in enumerate(bits)]
            check = dict(
                correct_slices=sum(checks),
                total=nviews,
                completed_write=tx["complete"],
                data_sha256=hashlib.sha256(
                    b"".join(v.numpy().tobytes() for v in bits)
                ).hexdigest(),
            )
            stats.update(check)
            write(root / "checked.json", check)
    except Exception as e:
        # Keep endpoints and machine addresses in the local logs only.
        print(type(e).__name__, str(e)[:450], file=sys.stderr, flush=True)
        stats["error"] = type(e).__name__
        stats["stage"] = (
            "initialization"
            if not (root / f"ready{rank}.json").exists()
            else "transfer"
        )
        if not (root / f"ready{rank}.json").exists():
            write(root / f"ready{rank}.json", dict(ok=False))
        if rank == 0 and not (root / "sent.json").exists():
            write(root / "sent.json", dict(complete=False))
        if rank == 1 and not (root / "checked.json").exists():
            write(root / "checked.json", dict(correct_slices=0, total=nviews))
    write(root / f"stats{rank}.json", stats)
    if engine is not None:
        for address, _ in regions:
            try:
                engine.unregister_memory(address)
            except Exception:
                pass


if len(sys.argv) > 4:
    child(Path(sys.argv[4]), int(sys.argv[5]))
    sys.exit()
with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
    root = Path(tmp)
    ps = []
    handles = []
    for n in range(2):
        h = (out.parent / (out.stem + f"-rank{n}.log")).open("w")
        handles.append(h)
        ps.append(
            subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    str(feature),
                    str(flag),
                    str(out),
                    tmp,
                    str(n),
                ],
                stdout=h,
                stderr=subprocess.STDOUT,
            )
        )
    limit = time.monotonic() + 140
    for p in ps:
        try:
            p.wait(timeout=max(1, limit - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()
    for h in handles:
        h.close()
    rows = [
        json.loads((root / f"stats{n}.json").read_text())
        if (root / f"stats{n}.json").exists()
        else dict(
            rank=n, exit_code=ps[n].returncode, error="native process did not complete"
        )
        for n in range(2)
    ]
    out.write_text(
        json.dumps(
            dict(
                feature=feature,
                flag=flag,
                sha=os.environ["ARM_SHA"],
                rows=rows,
                device_count=2,
            ),
            indent=2,
        )
        + "\n"
    )
