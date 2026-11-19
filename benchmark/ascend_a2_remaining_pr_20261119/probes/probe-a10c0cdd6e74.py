"""Measure local service recovery, native NPU thread binding, and real multi-rank SHM writers."""

import contextlib, datetime, hashlib, json, multiprocessing, os, socket, sys, tempfile, threading, time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch

feature, flag, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
rows = []
if feature == 48597:
    import requests
    from sglang.srt.disaggregation.common.conn import CommonKVManager
    from sglang.srt.disaggregation.base.conn import KVPoll

    visits = []
    responses = []
    timeouts = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            visits.append(payload["rid"])
            body = b'{"ok":true}'
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    class RealSession(requests.Session):
        def post(self, *args, **kw):
            timeouts.append(kw["timeout"])
            reply = super().post(*args, **kw)
            responses.append(reply.status_code)
            return reply

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_port
    host = "127.0.0.1:" + str(port)
    for mode in [
        "valid",
        "stale-rank",
        "negative-rank",
        "bool-rank",
        "closed-executor",
    ] * 5:
        room = len(rows) + 1
        manager = CommonKVManager.__new__(CommonKVManager)
        manager.attn_tp_rank = manager.attn_cp_rank = manager.pp_rank = 0
        manager.waiting_timeout = 12
        manager.defer_decode_allocation = False
        manager.prefill_info_table = {host: N(dp_size=2, prefill_http_port=port)}
        reason = []
        state = {room: KVPoll.Transferring}
        manager.request_status = state
        manager.record_failure = lambda room, why: reason.append(why)
        rx = N(
            bootstrap_addr=host,
            bootstrap_room=room,
            _lifecycle_lock=threading.RLock(),
            abort=lambda: state.update({room: KVPoll.Failed}),
        )
        with RealSession() as sess, ThreadPoolExecutor(max_workers=1) as pool:
            manager._ensure_prefill_recompute_executor = lambda: pool
            manager._get_prefill_recompute_session = lambda: sess
            if mode == "closed-executor":
                pool.shutdown()
            start = time.perf_counter_ns()
            err = None
            posts = len(visits)
            n = len(timeouts)
            sn = len(responses)
            payload = {
                "rid": "probe-" + str(room),
                "input_ids": [1, 2, 3],
                "routed_dp_rank": {
                    "stale-rank": 9,
                    "negative-rank": -1,
                    "bool-rank": True,
                }.get(mode, 0),
            }
            try:
                manager.submit_prefill_recompute(rx, payload)
            except Exception as e:
                err = type(e).__name__
            pool.shutdown(wait=True)
            rows.append(
                dict(
                    case=mode,
                    posts=len(visits) - posts,
                    http_status=responses[sn:],
                    failed=state[room] == KVPoll.Failed,
                    reason_recorded=bool(reason),
                    raised=err,
                    configured_timeouts=timeouts[n:],
                    elapsed_ms=(time.perf_counter_ns() - start) / 1e6,
                )
            )
    server.shutdown()
    server.server_close()
elif feature == 48632:
    import torch, torch_npu
    from sglang.srt.disaggregation.mooncake import conn
    from sglang.srt.platforms import current_platform

    class States(dict):
        def collect(self, room):
            return [self[room], self[room]]

    def init_common(m, args, mode, *rest, **kw):
        m.kv_args = args
        m.disaggregation_mode = mode
        m.request_status = States({7: conn.KVPoll.Transferring})

    for valid in (True, False, True, False):
        device = 0 if valid else torch.npu.device_count() + 1
        errs = []
        with contextlib.ExitStack() as stack:
            for key, x in [
                ("SGLANG_DISAGG_STAGING_BUFFER", False),
                ("SGLANG_ENABLE_FAILED_SESSION_PROBE", False),
                ("SGLANG_DISAGGREGATION_THREAD_POOL_SIZE", 1),
                ("SGLANG_DISAGGREGATION_QUEUE_SIZE", 1),
            ]:
                stack.enter_context(getattr(conn.envs, key).override(x))
            stack.enter_context(
                patch.object(conn.CommonKVManager, "__init__", init_common)
            )
            for method in (
                "init_engine",
                "register_buffer_to_engine",
                "start_prefill_thread",
            ):
                stack.enter_context(
                    patch.object(conn.MooncakeKVManager, method, lambda *args: None)
                )
            stack.enter_context(
                patch.object(
                    conn,
                    "check_mooncake_custom_mem_pool_enabled",
                    return_value=(False, None),
                )
            )
            stack.enter_context(
                patch.object(
                    conn, "get_observability", return_value=N(enable_trace=False)
                )
            )
            stack.enter_context(
                patch.object(
                    threading,
                    "excepthook",
                    lambda x: errs.append(type(x.exc_value).__name__),
                )
            )
            m = conn.MooncakeKVManager(
                N(gpu_id=device), conn.DisaggregationMode.PREFILL, None
            )
            m.transfer_queues[0].put(None)
            m._worker_threads[0].join(8)
            assert not m._worker_threads[0].is_alive()
            worker_data = None
            pool_error = None
            try:
                worker_data = (
                    m.executors[0]
                    .submit(lambda: int(torch.arange(256, device="npu").sum().item()))
                    .result(timeout=12)
                )
            except Exception as e:
                pool_error = type(e).__name__
            m.executors[0].shutdown(wait=True)
            sender = conn.MooncakeKVSender.__new__(conn.MooncakeKVSender)
            sender.kv_mgr = m
            sender.bootstrap_room = 7
            sender.conclude_state = None
            events = []
            for fn in (sender.poll, sender.poll_pp_consensus):
                t = time.perf_counter_ns()
                try:
                    v = fn()
                    events.append(
                        dict(
                            state=str(v),
                            visible_failure=False,
                            us=(time.perf_counter_ns() - t) / 1000,
                        )
                    )
                except Exception as e:
                    events.append(
                        dict(
                            state=type(e).__name__,
                            cause=type(e.__cause__).__name__,
                            visible_failure=True,
                            us=(time.perf_counter_ns() - t) / 1000,
                        )
                    )
            if valid:
                assert worker_data == 32640 and not errs
            else:
                assert errs or pool_error
            rows.append(
                dict(
                    valid_device=valid,
                    bind_errors=errs,
                    executor_error=pool_error,
                    healthy_device_sum=worker_data,
                    polls=events,
                )
            )
elif feature == 48843:
    import torch, torch.distributed as dist, zmq
    from sglang.srt.distributed.device_communicators import shm_broadcast as shm

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    world = 4
    writers_per_rank = 3
    block = 32
    os.environ["TORCH_GLOO_LAZY_INIT"] = "1"
    torch.set_num_threads(1)

    # Reserve no other user's port. Check our candidate loopback ranges before the timed run.
    def can_bind(base):
        bound = []
        try:
            for i in range(world):
                for off in range(8):
                    s = socket.socket()
                    bound.append(s)
                    s.bind(("127.0.0.1", base + i * block + off))
            return True
        except OSError:
            return False
        finally:
            for s in bound:
                s.close()

    base = next(x for x in range(22531, 29500, 419) if can_bind(x))
    os.environ["SGLANG_PORT"] = str(base)
    if flag:
        os.environ["SGLANG_SHM_PORTS_PER_RANK"] = str(block)

    def child(rank, root, q, barrier, turns):
        dist.init_process_group(
            "gloo",
            init_method="file://" + str(root / "store"),
            rank=rank,
            world_size=world,
            timeout=datetime.timedelta(seconds=75),
        )
        barrier.wait()
        held = []
        results = []
        assert turns[rank].wait(60)
        for index in range(writers_per_rank):
            begin = time.perf_counter_ns()
            err = None
            delivered = 0
            port = None
            try:
                writer = shm.MessageQueue(
                    n_reader=1, n_local_reader=1, max_chunk_bytes=256, max_chunks=2
                )
                held.append(writer)
                h = writer.export_handle()
                port = h.local_subscribe_port
                reader = shm.MessageQueue.create_from_handle(h, rank=0)
                held.append(reader)
                for x in (writer, reader):
                    x.local_socket.setsockopt(zmq.RCVTIMEO, 3000)
                    x.local_socket.setsockopt(zmq.SNDTIMEO, 3000)
                writer.wait_until_ready()
                reader.wait_until_ready()
                for j in range(16):
                    msg = f"probe-{rank}-{index}-{j}".encode()
                    writer.local_socket.send(msg)
                    assert reader.local_socket.recv() == msg
                    delivered += 1
            except Exception as e:
                err = type(e).__name__
            results.append(
                dict(
                    rank=rank,
                    writer=index,
                    port_offset=None if port is None else port - base,
                    delivered=delivered,
                    error=err,
                    ms=(time.perf_counter_ns() - begin) / 1e6,
                )
            )
        # Keep all successful writers bound while other ranks are still acquiring their windows.
        if rank + 1 < world:
            turns[rank + 1].set()
        barrier.wait()
        q.put(results)
        contexts = {x.local_socket.context for x in held}
        for x in held:
            x.local_socket.close(linger=0)
        for ctx0 in contexts:
            ctx0.term()
        dist.destroy_process_group()

    ctx = multiprocessing.get_context("fork")
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        q = ctx.Queue()
        barrier = ctx.Barrier(world, timeout=80)
        turns = [ctx.Event() for _ in range(world)]
        turns[0].set()
        ps = [
            ctx.Process(target=child, args=(i, Path(tmp), q, barrier, turns))
            for i in range(world)
        ]
        for p in ps:
            p.start()
        for i in range(world):
            rows.extend(q.get(timeout=95))
        for p in ps:
            p.join(15)
            assert p.exitcode == 0, p.exitcode
    rows.sort(key=lambda v: v["rank"])
else:
    raise ValueError(feature)
out.write_text(
    json.dumps(
        dict(feature=feature, flag=flag, sha=os.environ["ARM_SHA"], rows=rows), indent=2
    )
    + "\n"
)
