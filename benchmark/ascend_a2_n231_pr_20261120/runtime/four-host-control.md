# Shared full-model control on four Ascend hosts

The full GLM-5.2 W4A8C8 model used two TP16 roles, each spanning two eight-NPU hosts. Both roles ran upstream main `68c262b30617731b7e1761169aceb6b17feb918a`. The prefill device pool was capped at 8,192 tokens and the host ratio was four. The two roles used Mooncake Ascend Direct 0.3.14 with the same allocator registration fixture as the subsequent matched comparisons.

| Fixed control measurement | Result |
| --- | ---: |
| Completed requests at 24 concurrent sessions × three turns | 72/72 |
| Generated output tokens | 1,728 |
| Mean / P95 TTFT | 12.147 / 19.570 s |
| Aggregate output throughput | 17.746 tokens/s |
| Mean client decode speed | 1.897 tokens/s/request |
| Fixed natural-language answers | 32/32 |
| Public ARC-Challenge | 248/256 |

Three mixed-prefix waves reported 41,472 host-hit tokens. A separate audited restore checked 72/72 sampled device rows against the owning host data. The scored retrieval cohort had 14,080 host-hit tokens, 8/8 correct answers after restoration and 8/8 correct after flushing. All 8/8 generated token sequences matched the cold reference.

Eight uncached inputs provided 96 measured rank-forwards. The original prompt lengths were 3,073 through 7,169 tokens. With attention DP8, the configured 8,192-token chunk became 1,024 per rank. The raw file records actual per-forward allocator peaks and token counts. For the single ARC choice that changed in the isolated TP8 compact-index study, six independent control repetitions yielded {'D': 5, 'B': 1}. The repeated cohort is reported separately from the 256-question score.

[Input hashes, rank observations, model scores, forward peaks and individual requests](../samples/four-host-pressure-main-f2.json) · [Shared transport controls](transport-checks.md).
