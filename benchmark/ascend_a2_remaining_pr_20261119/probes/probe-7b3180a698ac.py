"""Capture production attribution and scheduling signals around bounded real hardware work."""

import contextlib, hashlib, json, logging, os, queue, sys, tempfile, threading, time
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import Mock
import torch, torch_npu

feature, flag, out = int(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
rows = []
torch.npu.set_device(0)
torch.set_num_threads(1)


def digest(x):
    return hashlib.sha256(
        x.detach().to("cpu").contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


class Records(logging.Handler):
    def __init__(self):
        super().__init__()
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())


rec = Records()
if feature == 48576:
    from sglang.srt.managers.cache_controller import HiCacheController
    from sglang.srt.mem_cache import hicache_storage as hs

    os.environ["SGLANG_HICACHE_PREFETCH_TIMING_TRACE"] = "1" if flag else "0"
    os.environ["SGLANG_HICACHE_TRACE_RUN_ID"] = "fixed-inputs"
    logging.getLogger(hs.__name__).addHandler(rec)
    logging.getLogger(hs.__name__).setLevel(logging.INFO)
    ctl = HiCacheController.__new__(HiCacheController)
    ctl.storage_stop_event = threading.Event()
    ctl.page_size = 128
    for attr in (
        "prefetch_queue",
        "prefetch_hit_queue",
        "prefetch_buffer",
        "prefetch_sync_queue",
        "ack_prefetch_queue",
    ):
        setattr(ctl, attr, queue.Queue())
    ctl.prefetch_hits_sync_groups = ctl.prefetch_completion_sync_groups = None
    # A one-rank host group still executes the production MIN operation.
    import torch.distributed as dist

    tmpd = tempfile.TemporaryDirectory(dir=out.parent)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(Path(tmpd.name) / "gloo"),
        rank=0,
        world_size=1,
    )
    ctl._all_reduce = lambda t, op, groups: dist.all_reduce(t, op=op)
    payload = torch.arange(2 * 1024**2 // 4, dtype=torch.int32)
    file = Path(tmpd.name) / "kv-pages"
    file.write_bytes(payload.numpy().tobytes())
    pinned = torch.empty_like(payload, pin_memory=True)

    def lookup(op):
        assert file.stat().st_size == payload.numel() * 4
        return ["page-" + str(i) for i in range(4)], 512

    def read(op):
        with file.open("rb", buffering=0) as f:
            assert (
                f.readinto(memoryview(pinned.numpy()).cast("B")) == payload.numel() * 4
            )

    ctl._storage_hit_query = lookup
    ctl._page_transfer = read
    workers = [
        threading.Thread(target=getattr(ctl, x), daemon=True)
        for x in (
            "prefetch_thread_func",
            "prefetch_io_aux_func",
            "prefetch_sync_thread_func",
        )
    ]
    ctl.prefetch_io_aux_thread = workers[1]
    for t in workers:
        t.start()
    try:
        for k in range(16):
            rid = "work-" + str(k)
            start = time.perf_counter_ns()
            begin = len(rec.lines)
            op = ctl.prefetch(rid, list(range(512)))
            hit = ctl.prefetch_hit_queue.get(timeout=5)
            assert hit is op and hit.storage_hit_count == 512
            op.host_indices = torch.arange(512)
            ctl.prefetch_buffer.put(op)
            ack = ctl.ack_prefetch_queue.get(timeout=5)
            assert ack.rid == rid and ack.completed_req
            if flag:
                hs.log_hicache_prefetch_consume(op)
            kv = pinned.to("npu", non_blocking=True)
            torch.npu.synchronize()
            if flag:
                hs.log_hicache_prefetch_forward_entries([N(rid=rid)])
            tm = (time.perf_counter_ns() - start) / 1e6
            assert torch.equal(kv.cpu(), payload)
            events = [
                json.loads(x.split(" ", 1)[1])
                for x in rec.lines[begin:]
                if x.startswith("HICACHE_PREFETCH_TIMING ")
            ]
            consume = next((v for v in events if v["event"] == "consume"), {})
            # Publish event names and stage latencies only; the logger's host and PID stay private.
            rows.append(
                dict(
                    ms=tm,
                    bytes=payload.numel() * 4,
                    events=[v["event"] for v in events],
                    stage_ns={
                        key: v
                        for key, v in consume.items()
                        if key.endswith("_ns") and key != "timestamp_monotonic_ns"
                    },
                    hash=digest(kv),
                )
            )
    finally:
        ctl.storage_stop_event.set()
        for q in (ctl.prefetch_queue, ctl.prefetch_buffer, ctl.prefetch_sync_queue):
            q.put(None)
        for t in workers:
            t.join(4)
        dist.destroy_process_group()
        tmpd.cleanup()
elif feature == 48577:
    from prometheus_client import REGISTRY
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
    from sglang.srt.observability.metrics_collector import StorageMetricsCollector

    labels = dict(
        storage_backend="local-fd",
        tp_rank="0",
        dp_rank="0",
        pp_rank="0",
        pp_size="1",
        attn_cp_rank="0",
        attn_cp_size="1",
    )
    collector = StorageMetricsCollector(labels)
    c = UnifiedRadixCache.__new__(UnifiedRadixCache)
    c.tree_core = N(enable_storage=True, page_size=128)
    c.enable_storage_metrics = True
    c.storage_metrics_collector = collector
    c.host_memory_mode = "cache"
    c.buffer_pipeline = None
    c.ongoing_backup = {}
    c.ongoing_write_through = {}
    c._backup_requested_tokens = {}
    c.hicache_storage_pass_prefix_keys = False
    c.inc_host_lock_ref = lambda node: N(to_dec_params=lambda: None)
    c.dec_host_lock_ref = lambda *args: None
    c._all_reduce_storage_drain = lambda x: None
    c._build_sidecar_transfers = lambda *args: []
    cc = N(
        backup_queue=queue.Queue(),
        ack_backup_queue=queue.Queue(),
        ack_write_queue=[],
        prefetch_hit_queue=queue.Queue(),
        ack_prefetch_queue=queue.Queue(),
        host_mem_release_queue=queue.Queue(),
        extra_pool_host_mem_release_queues={},
        backup_skip=False,
        page_size=128,
    )
    c.cache_controller = cc
    counter = iter(range(64))
    qty = 512
    total = 8
    dev = torch.arange(qty * 64, dtype=torch.int32, device="npu").reshape(qty, 64)
    host = torch.empty((qty, 64), dtype=torch.int32, pin_memory=True)
    host.copy_(dev, non_blocking=True)
    torch.npu.synchronize()
    assert torch.equal(host, dev.cpu())
    byte_row = host[0].numel() * host.element_size()
    event = threading.Event()
    ready = threading.Event()
    tmp = tempfile.TemporaryDirectory(dir=out.parent)

    def write(indices, ids, keys, prefix, extra_pools=None):
        oid = next(counter)
        cc.backup_queue.put(N(id=oid))
        return oid

    cc.write_storage = write
    c.tree_core.build_storage_backup_spec = lambda nid, passed: N(
        host_value=torch.arange(qty),
        token_ids=list(range(qty)),
        hash_value=["a", "b", "c", "d"],
        prefix_keys=None,
        comp_xfers={},
    )

    def worker():
        ready.set()
        event.wait(10)
        for i in range(total):
            op = cc.backup_queue.get(timeout=8)
            op.failed = op.id in (2, 7)
            op.unwritten_pages = 1 if op.failed else 0
            op.sidecar_unwritten_by_pool = {}
            op.completed_tokens = qty - op.unwritten_pages * 128
            path = Path(tmp.name) / str(op.id)
            with path.open("wb") as h:
                h.write(
                    memoryview(host.numpy()).cast("B")[: op.completed_tokens * byte_row]
                )
                h.flush()
                os.fsync(h.fileno())
            cc.ack_backup_queue.put(op)

    cc.backup_thread = threading.Thread(target=worker)
    cc.backup_thread.start()
    ready.wait(3)

    def snapshot(name):
        if hasattr(c, "_update_hicache_gauges"):
            c._update_hicache_gauges()
        samples = {}
        for metric in REGISTRY.collect():
            for s in metric.samples:
                if (
                    s.name.startswith("sglang:hicache_")
                    and not s.name.endswith("_created")
                    and all(s.labels.get(k) == v for k, v in labels.items())
                    and "le" not in s.labels
                ):
                    if (
                        "backup" in s.name
                        or "pending" in s.name
                        or "queue_depth" in s.name
                    ):
                        samples[s.name] = s.value
        rows.append(
            dict(
                phase=name,
                queued=cc.backup_queue.qsize(),
                acked=cc.ack_backup_queue.qsize(),
                ongoing=len(c.ongoing_backup),
                metrics=samples,
            )
        )

    t = time.perf_counter_ns()
    for i in range(total):
        c.write_backup_storage(i)
    snapshot("queued")
    event.set()
    cc.backup_thread.join(20)
    assert cc.ack_backup_queue.qsize() == 8
    snapshot("written")
    c._drain_storage_control_queues_impl(0, 0, 8, 0, {}, log_metrics=True)
    snapshot("drained")
    bytes_saved = sum(f.stat().st_size for f in Path(tmp.name).iterdir())
    assert bytes_saved == (total * qty - 256) * byte_row
    rows[-1].update(
        saved_bytes=bytes_saved,
        requested_tokens=total * qty,
        actually_persisted_tokens=bytes_saved // byte_row,
        partial_operations=2,
        elapsed_ms=(time.perf_counter_ns() - t) / 1e6,
    )
    tmp.cleanup()
elif feature == 48630:
    from sglang.srt.runtime_context import get_context
    from sglang.srt.managers import scheduler as sm
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_mode import ForwardMode

    for lname in (
        sm.__name__,
        "sglang.srt.managers.scheduler_components.prefill_admission_trace",
    ):
        log = logging.getLogger(lname)
        log.setLevel(logging.INFO)
        log.addHandler(rec)
    with get_context().override_server_args(page_size=128):
        s = sm.Scheduler.__new__(sm.Scheduler)
        s.dp_round_trace = bool(flag)
        s.forward_ct = 0
        s.prefill_delayer = None
        s.metrics_reporter = s.profiler_manager = s.invariant_checker = Mock()
        s._join_prefill_launched = lambda *args: None
        s.scripted_scheduler_hook = None
        s._sched_idled = False
        s.processed_tokens_counter = 0
        data = (
            torch.arange(512 * 128, dtype=torch.float32, device="npu")
            .view(512, 128)
            .remainder(7)
        )
        weights = torch.eye(128, device="npu")
        for k in range(18):
            before = len(rec.lines)
            batch = ScheduleBatch(
                reqs=[], forward_mode=ForwardMode.EXTEND, extend_num_tokens=512
            )
            produces = k % 3 != 0

            def select(**args):
                tr = args.get("trace")
                if tr:
                    tr.queued = 4
                    tr.visited = 2
                    tr.admitted = int(produces)
                    tr.stop = "queue_exhausted" if produces else "pass_budget"
                return batch if produces else None, args["running_batch"]

            s._get_new_batch_prefill_raw = select
            start = time.perf_counter_ns()
            plan = s.get_new_batch_prefill(batch)
            if produces:
                assert plan.batch_to_run is batch
                batch.dp_balance_stats = N(
                    local_tokens=512,
                    max_tokens=512,
                    sync_wait_seconds=0.0,
                    local_attention_pairs=512,
                    max_attention_pairs=512,
                )
                s._on_step_launch(batch)
                v = data @ weights
                torch.npu.synchronize()
                assert torch.equal(v, data)
            logs = [
                x
                for x in rec.lines[before:]
                if x.startswith(("DP_ADMISSION", "DP_ROUND"))
            ]
            rows.append(
                dict(
                    scan=k + 1,
                    issued=produces,
                    events=logs,
                    elapsed_ms=(time.perf_counter_ns() - start) / 1e6,
                )
            )
else:
    raise ValueError(feature)
out.write_text(
    json.dumps(
        dict(
            feature=feature,
            flag=flag,
            sha=os.environ["ARM_SHA"],
            device=torch.npu.get_device_name(0),
            rows=rows,
        ),
        indent=2,
    )
    + "\n"
)
