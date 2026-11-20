# Four-host NEXTN and HiCache control

Both TP16 roles loaded the real GLM-5.2 target and NEXTN draft on upstream main `ad96481482fe9d400195d12286f66a6359c56bb3`. Sixteen forced-rank requests first exercised every attention-DP rank. In the subsequent matched C24 control all 72 fixed requests completed and produced 1,728 tokens. Mean TTFT was 11.478 s, P95 was 19.308 s, aggregate output throughput was 20.548 tokens/s and mean per-request decode speed was 2.733 tokens/s.

The earlier [keep-alive client attempt](../samples/spec-control-f2-incomplete-http.json) completed 24 of 72 requests before the decode client reported a disconnection waiting for response headers. Both roles continued serving. The completed control and the subsequent candidates use [a fresh HTTP connection per request](../probes/pd_spec_replay.py) and zero retries inside each cohort.

Three mixed-prefix waves reported 39,296 real host-hit tokens. The networked scored retrieval produced 8/8 correct after host restoration and 8/8 following a cold flush. All eight scored requests hit the host cache and totaled 15,232 host tokens. In a dedicated transfer pass, 36/36 target byte vectors and 24/24 packed draft byte vectors equalled the owning host rows. The separate prefill-only warmup-sender control scored 8/8 on cold letter retrieval and registered zero host hits.

[Individual requests, pool configuration, byte comparisons and process observations](../samples/four-host-spec-control-f2.json).
