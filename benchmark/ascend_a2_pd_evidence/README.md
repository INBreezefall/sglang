# Ascend 910B4 PD component benchmarks

These measurements use four eight-device Ascend 910B4-1 hosts with 64 GiB per device, CANN 9.1.0, torch 2.10.0+cpu and torch-npu 2.10.0.post6. The reference revisions were upstream `sglang` at `40a64e1b5ba00c73b7f8a123a903a2123b7d467c` and `sgl-kernel-npu` at `bc2b60032f99bf36dd098babc914e9c42bbe7cea`. Each candidate uses the reference tree plus exactly one PR diff. Synthetic token IDs and KV contents are used throughout.

| Candidate | PR head | Device measurement |
| --- | --- | --- |
| SGLang #48096 | `4702f1d847ccbbe3c3b706a099654d86c80fc016` | Eviction and prefix reuse under allocation pressure |
| SGLang #48098 | `9edb89b3c468bafa9a15e309a5daffd1fa00d810` | Actual controller dispatch and device-resident prefix matches |
| sgl-kernel-npu #947 | `0b42dacab33cfcc6259291c8b507e52a76f4b6d8` | Correctness, submission, stream and synchronized wall time for H2D |

The four-host GLM-5.2 W4A8C8 PD launch stopped during argument parsing on the reference revision. Its `--kv-cache-dtype` choices omit `int8`. The DSA main KV resolver has no A2 INT8 branch. There are no model-forward, TTFT, decode throughput, output TPS or model accuracy results in this dataset. An independent BF16 diagnostic with Mooncake 0.3.13.post1 also exposed a thread-local ACL context issue. Neither diagnostic is used as a timing baseline.

## Reproducing

Use a device-enabled ARM64 environment. Check out the exact reference SHAs into `/bench/base/sglang` and `/bench/base/kernel`. For each candidate, copy its upstream tree and apply the PR diff at the recorded head. The patch base for both SGLang diffs is `11d687870f813782bfa6122360d8f57430d07ece`. The kernel patch base is `46f9367a10569d3ade70952c7311161ea09f598f`. Generate each patch with `git diff --binary PATCH_BASE PR_HEAD` and apply it to the corresponding newer reference tree. The examples below use `/bench/d-48096`, `/bench/p-48098`, and `/bench/kernel-947`. The NPU exercises use Torch's Ascend backend. The SGLang runs use its Python tree core, transformers 5.19.0, xgrammar 0.2.7 and tokenizers 0.23.3. For prefill dispatch the harness replaces the service socket send to exercise the selected rank's cache directly.

### Decode eviction

Run these commands from the directory containing the scripts.

```sh
PYTHONPATH=/bench/base/sglang/python SGLANG_PYTHON_RADIX_EVICT_TAIL_CHUNK_SIZE=0 python3 d-tail.py main --paged-pressure
PYTHONPATH=/bench/d-48096/python SGLANG_PYTHON_RADIX_EVICT_TAIL_CHUNK_SIZE=4096 python3 d-tail.py opt --paged-pressure
```

Each arm fills a 65,536-slot `NPUPagedTokenToKVPoolAllocator` with a 32,768-token evictable leaf and 32,768 other occupied slots. A new allocation fails before eviction. After eviction, the new allocation overwrites its slots. The probe checks the retained next-turn slots and their contents on the NPU. Page size is 128. The 1,280-, 8,192- and 32,768-token shortfalls run three times per arm on each of two hosts. The output includes individual walk timings, including the cold first call. The fast path's stated benefit is retained KV. [Device measurements](decode-eviction.json)

### Prefill placement

```sh
PYTHONPATH=/bench/base/sglang/python python3 p-replay.py main --sessions 8
PYTHONPATH=/bench/p-48098/python python3 p-replay.py opt --sessions 8
# Run both arms again with each of --seed 11 and --seed 29.
PYTHONPATH=/bench/base/sglang/python python3 p-replay.py main --sessions 24 --seed 11
PYTHONPATH=/bench/p-48098/python python3 p-replay.py opt --sessions 24 --seed 11
```

Each DP rank owns a radix cache on a different physical NPU. `total_tokens` on main and `prefill_session_tokens` on the candidate receive the same unpinned tokenized requests, unchanged session IDs and output allowance. The eight-rank load snapshot is set to idle at each wave boundary. During the wave the controller applies its normal optimistic load increments. Prefix matches and allocation misses come from the selected rank's actual cache. The C24 fixture uses the native NPU paged allocator, 32,768 slots per rank, eviction under pressure, and both fixed and permuted arrival order. Each session grows 512 tokens per turn. Missing slots measure device prefix misses. They do not directly measure DMA bytes, recomputed tokens or service latency. [Per-request selections and cache counts](prefill-routing.json)

### H2D transfer

Place `kernel-build.sh` and `h2d-paired.py` in `/bench` and put the diff for #947 at `/bench/pr-947.patch`. The build script compiles only the relevant host operator and its dispatcher registration from each revision. Both use the same compiler flags and CANN/Torch libraries. The paired build changes the operator namespace so both revisions can run against identical tensor buffers in one process. The copy implementation itself is unchanged. The original namespace is used for the published PR stream test.

```sh
cd /bench
bash kernel-build.sh
PYTHONPATH=/bench/op-pr python3 -m pytest -q kernel-947/tests/python/sgl_kernel_npu/test_transfer_kv_dim_exchange.py::TestTransferKV::test_coalesced_h2d_preserves_page_order_and_scale
REGISTRATION_STYLE=paired bash kernel-build.sh
python3 -u h2d-paired.py
python3 -u h2d-paired.py --model-layout
```

The harness pins CPU execution to the NUMA node of device 0; adjust its PCI address for other hardware. Physical placement of the driver's pinned VMA was not observable. Within each shape, all three arms share the same pinned source, device destination, indices and stream. The arms alternate in rotated order. The first pass overwrites and verifies all transferred bytes and both untouched sentinel pages in each arm. Seven further samples are timed. Two allocations in reverse shape order on each of two hosts give four blocks per case. `submit_ms` times the host call, `stream_ms` spans events on the NPU stream and can include host submission gaps, and `wall_ms` ends after event synchronization. A five-dimensional zero-element V tensor gives both native implementations the same valid input for timing. `off` and `on` select flags 2 and 6 on the candidate.

The model-layout test submits the 78-layer, 656-byte INT8 packed MLA row, 21 physical INT8 indexer layers of width 128, and one FP16 scale per indexer row. These are target-only synthetic tensors with the checkpoint's layout. Each request uses pages of 128 tokens. For the gaps case the source pages are visited in odd-then-even order, while destinations increase. The mean of the four block medians gives equal weight to both hosts and both allocations. [All transfer samples](h2d-samples.json) · [Per-block medians and routing aggregates](summary.json)
