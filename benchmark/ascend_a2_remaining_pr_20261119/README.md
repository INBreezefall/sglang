# Ascend 910B4 comparisons of 20 isolated SGLang PRs

All 20 candidates were rebased onto upstream main `1ed3e1d04b25ab007b4c80481b7b6e3a36836eb7` before measurement on November 19, 2026. Each page gives the exact two commits, test scope, sample counts and measured outcomes. Infrastructure and diagnostic changes report their own primary metrics. Model throughput appears only for matched full-service deployments.

| PR | Measured feature | Candidate |
| --- | --- | --- |
| [#48096](results/48096.md) | Paged radix eviction | `4529862bc1ba` |
| [#48520](results/48520.md) | Input-only prefix retention | `778d5a2d93c9` |
| [#48537](results/48537.md) | Mooncake allocation registration | `877b20c1303a` |
| [#48538](results/48538.md) | MLA draft layout selection | `05717bb9fda5` |
| [#48540](results/48540.md) | Prefill MoE tiling | `f3abcca21bab` |
| [#48543](results/48543.md) | Exact-size ACL host allocation | `2877db5d3bf6` |
| [#48574](results/48574.md) | DeepEP Normal and LowLatency dtypes | `9ba34b92034e` |
| [#48575](results/48575.md) | Pinned allocator prewarming | `d225360ab183` |
| [#48576](results/48576.md) | Per-request prefetch timing | `a14908162bc2` |
| [#48577](results/48577.md) | HiCache backup metrics | `5239c4443cfb` |
| [#48597](results/48597.md) | PD rebootstrap failure handling | `d09125946350` |
| [#48599](results/48599.md) | Resident L2 hit origins | `7541d7410e7c` |
| [#48630](results/48630.md) | Prefill admission and forward tracing | `f0f024dcfd70` |
| [#48631](results/48631.md) | Packed NZ checkpoint round trip | `50cf45054aae` |
| [#48632](results/48632.md) | Mooncake worker binding failures | `77ba9f6f799f` |
| [#48843](results/48843.md) | Local shared-memory writer ports | `3a760a253bf5` |
| [#49029](results/49029.md) | Pageable KV backup fallback | `7d66973c2b1e` |
| [#49077](results/49077.md) | Ascend Direct endpoint naming | `bb9c53a98e82` |
| [#49181](results/49181.md) | Docker bytecode exclusions | `d394549ff289` |
| [#49183](results/49183.md) | Independent draft host budget | `1b789051388c` |

## Hardware and reproducibility

Four Linux ARM64 hosts each provided eight Ascend 910B4-1 accelerators with 64 GiB per device. Control and candidate components ran on the same device or physical rank group. Each arm started in a fresh process with separate compiler caches. The Docker comparison used four independent local builds. Every service used the same 96 GLM-5.2 W4A8C8 shards. All four hosts verified the full shard SHA-256 values, tokenizer, configuration, weight index and staged SGLang Python blobs. [Hardware and versions](runtime/environment.json) · [Checkpoint hashes](runtime/checkpoint.json) · [Source hashes](runtime/source-provenance.json)

Select the stated Git revision before each run. The native benchmarks require the CANN 9.1 environment, torch 2.10, torch-npu 2.10.0.post6 and the ARM64 NPU kernel package. The Mooncake probes use the Ascend Direct wheel recorded in the environment manifest. Set TEST_HOST_ADDR to the local network interface address for those probes. The DeepEP probe requires the installed vendor operator and eight visible NPUs. The packed-weight probe requires the matching checkpoint under /model. [SGL kernel release archive](https://github.com/sgl-project/sgl-kernel-npu/releases/tag/2026.11.0.post8)

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export ARM_SHA=$(git -C "$CHECKOUT" rev-parse HEAD)
export PYTHONPATH="$CHECKOUT/python:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=1 HF_HUB_OFFLINE=1
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export PYTORCH_NPU_ALLOC_CONF=pinned_reserve_segment_size_mb:64
export ASCEND_RT_VISIBLE_DEVICES=0
# Use a fresh directory and select the probe for the corresponding PR below.
OUT=$(mktemp -d)
export TRITON_CACHE_DIR="$OUT/triton" ASCEND_CACHE_PATH="$OUT/cann" SGLANG_CACHE_DIR="$OUT/sglang"
mkdir -p "$TRITON_CACHE_DIR" "$ASCEND_CACHE_PATH" "$SGLANG_CACHE_DIR"
python "$PROBE" "$PR" "$CANDIDATE_FLAG" "$OUT/result.json"
```

Use flag 0 for main and 1 for the PR. The listed rank count is the number of ranks in the probe. The four Gloo ranks for #48843 run on the host CPU. For #49181, supply BENCH_ROOT with jobs/ and src/{main,49181}/, place the matching Git ignore files and a SHA file in each checkout, and set BENCH_OWNER to the contents of the scratch owner file. Run its script directly on a host with Docker. It removes the tagged temporary images after inspection.

| PR | Executable source | Ranks | Raw results |
| --- | --- | ---: | --- |
| 48096 | [probe](probes/probe-97d0841a817b.py) | 1 | [samples](samples/48096.json) |
| 48520 | [probe](probes/probe-97d0841a817b.py) | 1 | [samples](samples/48520.json) |
| 48537 | [probe](probes/probe-b1a426a75127.py) | 2 | [samples](samples/48537.json) |
| 48538 | [probe](probes/probe-28c37fb9ffcd.py) | 1 | [samples](samples/48538.json) |
| 48540 | [probe](probes/probe-2ebd0678168f.py) | 1 | [samples](samples/48540.json) |
| 48543 | [probe](probes/probe-2ebd0678168f.py) | 1 | [samples](samples/48543.json) |
| 48574 | [probe](probes/probe-8b5652eb6194.py) | 8 | [samples](samples/48574.json) |
| 48575 | [probe](probes/probe-2ebd0678168f.py) | 1 | [samples](samples/48575.json) |
| 48576 | [probe](probes/probe-7b3180a698ac.py) | 1 | [samples](samples/48576.json) |
| 48577 | [probe](probes/probe-7b3180a698ac.py) | 1 | [samples](samples/48577.json) |
| 48597 | [probe](probes/probe-560f32dde003.py) | 1 | [samples](samples/48597.json) |
| 48599 | [probe](probes/probe-28c37fb9ffcd.py) | 1 | [samples](samples/48599.json) |
| 48630 | [probe](probes/probe-d9458cb74b0f.py) | 1 | [samples](samples/48630.json) |
| 48631 | [probe](probes/probe-07a339d3efd1.py) | 1 | [samples](samples/48631.json) |
| 48632 | [probe](probes/probe-560f32dde003.py) | 1 | [samples](samples/48632.json) |
| 48843 | [probe](probes/probe-a10c0cdd6e74.py) | 4 | [samples](samples/48843.json) |
| 49029 | [probe](probes/probe-2ebd0678168f.py) | 1 | [samples](samples/49029.json) |
| 49077 | [probe](probes/probe-b1a426a75127.py) | 2 | [samples](samples/49077.json) |
| 49181 | [probe](probes/probe-130299af05e0.py) | 1 | [samples](samples/49181.json) |
| 49183 | [probe](probes/probe-28c37fb9ffcd.py) | 1 | [samples](samples/49183.json) |

Scripts have been formatted for review. Their Python abstract syntax trees match the exact measured sources. The raw JSON SHA-256 values can be reproduced by encoding each process object with Python json.dumps using indent=2 and adding a trailing newline. [Script checksums](probes/checksums.json) · [Serving method and matched cohorts](serving/README.md)
