# GLM-5.2 PD workload

Each deployment used four 8-device Ascend 910B4-1 hosts. Two formed one TP16 prefill service and two formed one TP16 decode service. Both used attention DP8, CANN 9.1, ModelSlim W4A8C8 weights, BF16 KV, 128-token pages, eager execution and total-token routing. Prefill enabled HiCache. Decode enabled its device radix cache. Individual profile pages list any changes such as DeepEP, NEXTN, storage and token capacity. The anonymous role and rank launcher files retain the actual command arguments and log and Prometheus counts. A model source SHA is recorded for each role.

The client flushed the caches, issued eight warmups, flushed again and started the formal clock. It issued three waves of 24 concurrent sessions. Each session reused a deterministic random token prefix and appended 128 input tokens per turn. Output length was fixed at 24 tokens, temperature zero, ignoring EOS. The client sent a matched native request to both PD roles. It measured the first positive decode stream event for TTFT and divided all 1,728 decoded tokens by the elapsed formal wall time for throughput. The per-request decode metric uses the 23 tokens after the first event. No tokenizer, prompt or output strings from user data were used.

Each comparison verifies that all 72 request hashes and input lengths agree, that every request returned 24 tokens and that the digest covers all 24 actual token IDs. Exact output agreement is reported per deployment. Variation across independent main launches can be material on these quantized, throughput-oriented deployments. Timing differences from a single pair are descriptive. All traces and storage counters in the launcher files span initialization, warmup and formal requests across the recorded ranks. Only client samples delimit the formal timed cohort.

| Deployment | Client data | Measured client | Launcher and counters |
| --- | --- | --- | --- |
| base1 | [72 requests](base1.json) | [client](../probes/client-306866803a4a.py) | [launcher](base1-launcher.json) |
| moe-tile | [72 requests](moe-tile.json) | [client](../probes/client-306866803a4a.py) | [launcher](moe-tile-launcher.json) |

[Full checkpoint manifest](../runtime/checkpoint.json) · [Per-deployment source and raw checksums](clients.json)
