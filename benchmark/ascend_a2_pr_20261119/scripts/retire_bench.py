"""One scheduler iteration, a real pinned H2D completion, then fixed NPU forward work."""

import json, os, sys, time
from pathlib import Path
from types import SimpleNamespace as N
from unittest.mock import patch
import torch, torch_npu
from sglang.srt.disaggregation import prefill
from sglang.srt.runtime_context import get_context
from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin

torch.npu.set_device(0)
torch.set_num_threads(1)
enabled = bool(int(sys.argv[2]))
os.environ["SGLANG_DISAGG_PREFILL_POLL_TRANSFERS_BEFORE_FORWARD"] = str(int(enabled))
host = torch.full((8 << 20,), 7, dtype=torch.uint8, pin_memory=True)
buf = torch.empty_like(host, device="npu")
x = torch.full((1024, 1024), 0.125, dtype=torch.float16, device="npu")
w = torch.full_like(x, 0.125)
stream = torch.npu.Stream()
rows = []
with (
    get_context().override_server_args(pp_size=1),
    patch.object(prefill, "set_schedule_time_batch"),
):
    for i in range(22):
        sched = SchedulerDisaggregationPrefillMixin()
        sched.gracefully_exit = False
        sched.increment_scheduler_iterations = lambda: None
        sched.ingest_requests = lambda: False
        sched.pause_state = N(paused=False)
        sched.waiting_queue = N()
        sched.disagg_prefill_bootstrap_queue = N(move_bootstrapped=lambda _: None)
        sched.running_batch = N(batch_is_full=False)
        sched.last_batch = None
        sched.chunked_req = None
        sched.enable_staging = False
        batch = N()
        sched.get_next_disagg_prefill_batch_to_run = lambda **kw: N(
            running_batch=sched.running_batch, batch_to_run=batch
        )
        ctx = {}
        event = torch.npu.Event()

        def prepare(value, **kwargs):
            with torch.npu.stream(stream):
                buf.copy_(host, non_blocking=True)
                event.record()
            event.synchronize()
            ctx["ready"] = time.perf_counter_ns()
            return value

        def poll():
            if "ready" in ctx and "retired" not in ctx and event.query():
                ctx["retired"] = time.perf_counter_ns()

        def forward(_):
            ctx["before_forward"] = "retired" in ctx
            for _ in range(18):
                res = torch.mm(x, w)
            torch.npu.synchronize()
            ctx["forward_end"] = time.perf_counter_ns()
            return res

        def done(*_):
            sched.gracefully_exit = True

        sched.ngram_embedding_manager = N(prepare_for_forward=prepare)
        sched.process_disagg_prefill_transfer_queue = poll
        sched.run_batch = forward
        sched.process_batch_result = done
        sched.invariant_checker = N(run=lambda *args: None)
        sched.event_loop_normal_disagg_prefill()
        assert int(buf[0]) == 7 and ctx["before_forward"] == enabled
        if i >= 4:
            rows.append(
                dict(
                    retired_before_forward=ctx["before_forward"],
                    completion_to_retirement_ms=(ctx["retired"] - ctx["ready"]) / 1e6,
                    completion_to_forward_end_ms=(ctx["forward_end"] - ctx["ready"])
                    / 1e6,
                )
            )
Path(sys.argv[1]).write_text(
    json.dumps(
        dict(
            sha=os.environ["ARM_SHA"],
            device=torch.npu.get_device_name(),
            enabled=enabled,
            copy_bytes=host.numel(),
            rows=rows,
        ),
        indent=2,
    )
    + "\n"
)
