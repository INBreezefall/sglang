# Four-host GLM-5.2 serving replay

The 32 Ascend 910B4-1 devices served the full GLM-5.2 W4A8C8 checkpoint. Two hosts ran Prefill with TP16 and attention DP8. Two hosts ran Decode with the same parallelism. Each row used the native `/generate` path on both roles, Ascend device-RDMA transfer, eager attention, 128-token cache pages and a shared 32,768-token limit per rank. Prefill used HiCache, Decode used radix caching. Each row links to the raw per-request measurements.

| Deployment | Mean TTFT (s) | p95 TTFT (s) | Decode (tok/s/request) | Output (tok/s) | P same rank | D same rank | Uncached P tokens on turns 2–3 | Exact token outputs vs main |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [main](base-c24.json) | 24.67 | 35.41 | 1.55 | 11.13 | 7/48 | 8/48 | 91,008 | 72/72 |
| [#48098](p48098.json) | 10.99 | 13.57 | 1.73 | 19.91 | 48/48 | 13/48 | 6,144 | 15/72 |
| [#48541](d48541.json) | 8.71 | 12.98 | 1.73 | 21.24 | 2/48 | 48/48 | 84,864 | 19/72 |

## Workload and timing

Each cohort flushed both caches, ran eight fixed warm-up requests and flushed both caches again. Twenty-four sessions then ran three gated waves with a deterministic shuffle per wave. Each session began with 768–4,096 deterministic input token IDs and kept its entire prefix when adding 128 tokens per turn. Each request sampled greedily with a fixed 24-token output, ignoring EOS. All workload and output comparisons use SHA-256 of the JSON-encoded request or token IDs. Transport IDs were excluded from the input hash. Input IDs are synthetic. Output agreement on this test measures reproducibility for these prompts.

Mean and p95 TTFT use the time from client dispatch to the first streamed event containing generated tokens. Client decode rate is the mean across requests of 23 tokens divided by the elapsed first-to-last streamed span, expressed as tokens per second. Output throughput is 1,728 tokens divided by the complete three-wave wall time. Network buffering and event coalescing can affect client decode rate. Rank reuse counts the 48 adjacent turns per deployment. Uncached P tokens are prompt tokens minus the reported Prefill cached tokens over turns two and three.

The initial main deployment also ran a one-request, four-token connectivity check before its formal cohort. The fixture clears prefix caches before each cohort and after its warmup. Driver and filesystem caches persist between deployments. The table reports one cohort per named deployment. Use the raw data to inspect countervailing effects and the run order. All four hosts independently read and SHA-256 hashed every byte of each of the 96 weight shards (404,794,116,064 bytes per host). The individual hashes, configuration, tokenizer and index agree across the four copies. [Actual anonymized server arguments](service-argv.json) · [Model manifest](model-manifest.json) · [Replay client](pd_client.py)

Cohort launch order was base-c24 (04:38 UTC), p48098 (04:55 UTC), d48541 (05:06 UTC).
