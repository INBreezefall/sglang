# GLM-5.2 PD workload

Each deployment used four 8-device Ascend 910B4-1 hosts. Two formed one TP16 prefill service and two formed one TP16 decode service. Both used attention DP8, CANN 9.1, ModelSlim W4A8C8 weights, BF16 KV, 128-token pages, eager execution and total-token routing. Prefill enabled HiCache. Decode enabled its device radix cache. Individual profile pages list any changes such as DeepEP, NEXTN, storage and token capacity. The anonymous role and rank launcher files retain the actual command arguments and log and Prometheus counts. A model source SHA is recorded for each role.

The client flushed the caches, issued eight warmups, flushed again and started the formal clock. It issued three waves of 24 concurrent sessions. Each session reused a deterministic random token prefix and appended 128 input tokens per turn. Output length was fixed at 24 tokens, temperature zero, ignoring EOS. The client sent a matched native request to both PD roles. It measured the first positive decode stream event for TTFT and divided all 1,728 decoded tokens by the elapsed formal wall time for throughput. The per-request decode metric uses the 23 tokens after the first event. The workload uses synthetic token IDs.

Each comparison verifies that all 72 request hashes and input lengths agree, that every request returned 24 tokens and that the digest covers all 24 actual token IDs. Exact output agreement is reported per deployment. Variation across independent main launches can be material on these quantized, throughput-oriented deployments. Timing differences from a single pair are descriptive. All traces and storage counters in the launcher files span initialization, warmup and formal requests across the recorded ranks. Only client samples delimit the formal timed cohort. The earlier client and the later client differ only by recording wall-clock timestamps at the start and end of the formal cohort. Workload construction, request hashes and metric calculations are identical.

| Deployment | Client data | Measured client | Launcher and counters |
| --- | --- | --- | --- |
| base1 | [72 requests](base1.json) | [client](../probes/client-306866803a4a.py) | [launcher](base1-launcher.json) |
| base2 | [72 requests](base2.json) | [client](../probes/client-8b768e8a88c4.py) | [launcher](base2-launcher.json) |
| evict-control | [72 requests](evict-control.json) | [client](../probes/client-8b768e8a88c4.py) | [launcher](evict-control-launcher.json) |
| evict-tail | [72 requests](evict-tail.json) | [client](../probes/client-8b768e8a88c4.py) | [launcher](evict-tail-launcher.json) |
| moe-tile | [72 requests](moe-tile.json) | [client](../probes/client-306866803a4a.py) | [launcher](moe-tile-launcher.json) |
| scan-trace | [72 requests](scan-trace.json) | [client](../probes/client-8b768e8a88c4.py) | [launcher](scan-trace-launcher.json) |

The launchers below are the original measured scripts with node IP addresses replaced by PREFILL_HEAD, DECODE_HEAD and NODE_IP environment variables. Set those three values for each host. Mount identical model weights at /model and the checkouts at /bench/src/<arm>. The shell takes role, two-node rank, source arm, node index, case name, profile and rendezvous port. The source arm sets the feature switch for that PR. [Launcher checksums](../launchers/manifest.json)

| Deployment | Role launcher | Four short fixed-answer checks after timing |
| --- | --- | --- |
| base1 | [launcher](../launchers/launch-5104d3b0bb8f.sh) | Not collected |
| base2 | [launcher](../launchers/launch-6db341033f5e.sh) | Not collected |
| evict-control | [launcher](../launchers/launch-55ca9f29dfa4.sh) | [answers](evict-control-answer-checks.json) |
| evict-tail | [launcher](../launchers/launch-55ca9f29dfa4.sh) | [answers](evict-tail-answer-checks.json) |
| moe-tile | [launcher](../launchers/launch-988764d3c5d7.sh) | Not collected |
| scan-trace | [launcher](../launchers/launch-55ca9f29dfa4.sh) | [answers](scan-trace-answer-checks.json) |

[Full checkpoint manifest](../runtime/checkpoint.json) · [Per-deployment source and raw checksums](clients.json)
