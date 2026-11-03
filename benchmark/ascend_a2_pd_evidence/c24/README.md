# GLM-5.2 W4A8C8 C24 closed-loop PD replay on Ascend 910B4

Four hosts, eight 64-GiB Ascend 910B4-1 NPUs on each. The model ran on two prefill hosts and two decode hosts with the same model files. A two-turn request sent through the real PD gateway returned 323 for 17 × 19 and 324 when asked to add one to its previous answer. This checks the text path and conversation handoff. No broad model-quality suite was run.

## Controlled comparison

The original 300-second first-token control warmed 24/24 sessions, then attempted 50/72 formal requests: 38 succeeded and 12 failed. The stop-on-error policy prevents interpreting a survivors-only mean as a complete-replay performance result. All 12 timeouts changed prefill shard relative to that session's previous turn, including its warm turn. Among the successful requests, 17/38 also changed prefill shard. On decode, 11/12 timeouts moved. These are associations in the control.

The SGLang baseline is `40a64e1b5ba00c73b7f8a123a903a2123b7d467c` plus a common A2/C8 compatibility patch with SHA-256 `95eab8a2a6fa981b94ca76e1370aa2b45d994e98b72d4290099acff4653bdb17`. The patch is present in every arm. It carries packed A2 INT8 DSA/indexer support, target/draft pool layout handling, W4 tensor-parallel bias slicing, worker-thread NPU device binding for the installed Mooncake backend, and the same instrumentation. The unmodified pinned main rejects the A2 C8 configuration. The results isolate each PR on that fixed baseline.

The reference Ascend kernel source is `bc2b60032f99bf36dd098babc914e9c42bbe7cea`. For the H2D comparison, the exact native transfer operator and its high-level Python wrapper from each reference/candidate source were installed over the same operator runtime and checked via the PrivateUse1 dispatcher. Both Python wrappers pass the same five-dimensional empty V descriptor to the unchanged native signature for sparse-index and scale copies. Both directions were verified byte-for-byte on device. The remaining native operators used the identical installed wheel. Software versions: CANN 9.1.0, torch 2.10.0+cpu, torch-npu 2.10.0.post6, Mooncake NPU 0.3.13.post1, transformers 5.19.0.

* Each role used TP16 and attention DP8. P used EP1 and L2 HiCache, with 253,952 HBM tokens and 988,160 host tokens per rank. P prefill chunks were 8,192 global and 1,024 per DP rank. D used EP16, low-latency DeepEP and 253,952 HBM KV slots per shard; graph sizes were 1, 2, 4, and 8. The target used C8 KV plus an INT8 indexer, and the MTP draft used BF16. Mooncake transferred KV between roles. L3 was off.
* Every replay flushed both engine caches, warmed the same 24 sessions once, kept their prefix state, and waited 15 seconds. All sessions then ran three formal closed-loop turns, with the real previous generated reply and recorded next trigger. Generation used temperature 0, top-p 1, a 256-token output cap, and zero retries. For the full-window A/B runs, the observation deadline for the **first** semantic token was 1,200 seconds. The post-token stream idle deadline remained 300 seconds and the overall timeout remained 1,500 seconds. The original 300-second first-token run is reported separately, and every full-window row reports the number of requests over 300 seconds. The two PD transport wait budgets were 1,500 seconds for every arm. No external P rank was pinned. The fixed workload selection hash is `13853989ef8c845f685a01f6ac0a3da0d6ccfbfe6699a3c57c44c4c856302b10`. The conversation data is private.
* TTFT is the arithmetic mean of the 72 client first-token latencies. Mean per-request decode is the mean of the original client's typing speeds: output tokens minus one, divided by the time between the first and last semantic streaming events. The contributing count is in the JSON. The PD relay waits for prefill completion before exposing decode bytes to the client. Short or buffered semantic event spans and requests missing a valid span can change the client mean. D ITL inverse is the increase in D token-interval count divided by its increase in elapsed token-interval seconds. An additional reciprocal of mean native per-request TPOT is in the JSON. Output TPS is the sum of actual output tokens divided by the formal wall window. Each row in the summary weights complete replays equally. The restored control is shown separately to expose run-order variation.
* Per-request/rank samples are provided with ordinalized session IDs and without conversation text. P and D routing are measured at scheduler receipt. Same-rank counts refer to adjacent **formal** turns only. Native copy bytes are summed over physical rank operands. The trace's host elapsed time measures submission. It does not synchronize the device.

| Arm | Complete replays | Mean TTFT, s ↓ | Mean per-request decode, tok/s ↑ | D ITL inverse, tok/s ↑ | Output TPS ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shared-fix control | 1 | 296.74 | 94.71 | 36.41 | 9.47 |
| #48096 D eviction | 1 | 241.30 | 87.57 | 38.67 | 8.96 |
| #48098 P session routing | 1 | 28.47 | 36.18 | 15.07 | 81.34 |
| #947 P H2D | 1 | 257.24 | 77.64 | 44.41 | 9.63 |
| Restored control | 1 | 272.47 | 119.61 | 45.99 | 9.31 |

Equal-replay means relative to the mean of both independent controls. Observed ranges are recorded in the JSON; two runs do not establish statistical significance.

| Candidate | Replays | TTFT change | Client per-request decode change | Native D ITL inverse change | Output throughput change |
| --- | ---: | ---: | ---: | ---: | ---: |
| #48096 D eviction | 1 | -15.2% | -18.3% | -6.1% | -4.6% |
| #48098 P session routing | 1 | -90.0% | -66.2% | -63.4% | +766.4% |
| #947 P H2D | 1 | -9.6% | -27.6% | +7.8% | +2.6% |

| Replay | Success | TTFT mean / p95 / worst, s | TTFT >300s | Client decode, tok/s | D ITL inverse, tok/s | Output tok/s | Output tokens / window, s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base-tail | 72/72 | 296.74 / 620.25 / 755.92 | 31 | 94.71 | 36.41 | 9.47 | 11040 / 1165.82 |
| d-tail | 72/72 | 241.30 / 710.53 / 859.64 | 17 | 87.57 | 38.67 | 8.96 | 11301 / 1261.34 |
| p-tail | 72/72 | 28.47 / 59.80 / 69.88 | 0 | 36.18 | 15.07 | 81.34 | 11255 / 138.37 |
| k-tail | 72/72 | 257.24 / 593.59 / 650.55 | 30 | 77.64 | 44.41 | 9.63 | 11646 / 1208.91 |
| anchor-tail | 72/72 | 272.47 / 580.21 / 873.78 | 26 | 119.61 | 45.99 | 9.31 | 11235 / 1207.19 |

| Replay | P same rank / formal continuations | D same rank / formal continuations | D evicted leaf tokens | P H2D calls / GiB |
| --- | ---: | ---: | ---: | ---: |
| base-tail | 8/48 | 24/48 | 6,367,104 | 100 / 174.74 |
| d-tail | 9/48 | 21/48 | 4,525,824 | 76 / 109.69 |
| p-tail | 48/48 | 5/48 | 6,462,336 | 180 / 319.64 |
| k-tail | 13/48 | 21/48 | 5,302,144 | 60 / 91.38 |
| anchor-tail | 12/48 | 22/48 | 6,282,112 | 60 / 103.44 |

The lower P cache cap and chunk used in this comparison were chosen before collecting any formal samples. With 327,680 P tokens and 2,048 tokens per local chunk, the W4 MoE forward ran out of NPU memory during cold warm-up. A D cap of 291,840 had also exhausted memory during MTP graph capture. Those preflights are excluded.

Individual samples, counters, and ranges are in [c24-results.json](c24-results.json). The mechanism microbenchmarks remain in the [parent evidence directory](..).
