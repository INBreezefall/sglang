# Ascend 910B4 per-PR measurements

These hardware measurements compare ten SGLang changes on the November 19, 2026 upstream baseline `f7e7ccbd9f1b9fc7f9373de19ea5ea91a81d359b`. The indexer change in #48562 is stacked on the rebased #48560 and uses that head as its control. Each results page links to the exact samples used in its table.

| PR | Measured path | Candidate |
| --- | --- | --- |
| [#48098](results/48098.md) | Prefill session placement | `405ac4fc3d75` |
| [#48539](results/48539.md) | HiCache LPM reload cost | `374aec581e92` |
| [#48541](results/48541.md) | Decode session placement | `00c371a40f73` |
| [#48542](results/48542.md) | Early prefill transfer retirement | `d6b00bad4714` |
| [#48555](results/48555.md) | Compact PD and HiCache indexers | `94d9156ae932` |
| [#48560](results/48560.md) | Packed INT8 main KV | `89ddc8785e27` |
| [#48562](results/48562.md) | Graph-safe INT8 DSA indexer | `b476e575d222` |
| [#48633](results/48633.md) | Paged KV allocator Triton specialization | `2a88b4028692` |
| [#49103](results/49103.md) | Bounded decode KV backfill | `5d1571b7884b` |
| [#49117](results/49117.md) | Eager sparse attention KV lengths | `24c97ca43300` |

## Hardware and software

The test machines each have eight Ascend 910B4-1 NPUs with 64 GiB per device. The runtime uses Linux arm64, Python 3.12, CANN 9.1.0, torch 2.10.0 and torch-npu 2.10.0.post6. Every comparison runs both arms on the same host and device. The routing comparison uses eight devices on one host, one physical radix cache and allocator process per rank. The results pages identify the number of processes, repetitions and unit of timing. The two 21-layer indexer runs form a separate same-device pair.

Pool backup and restore use the ARM64 torch 2.10 / CANN 9.1 wheel from the upstream kernel release archive. The wheel manifest includes the `ascend910b` compute unit and the native KV exchange operator. The compiler and sparse-attention probes use the image's A2 runtime. Where the image lacked the newer Python operator package, the Python layer came from sgl-kernel-npu `62a5187d10f7`. The graph-safe indexer uses both real `torch.ops.custom` vendor entry points. [Kernel release](https://github.com/sgl-project/sgl-kernel-npu/releases/tag/2026.11.0.post8)

All four hosts independently verified every Python source blob under `python/sglang` in the baseline and ten staged candidate trees against Git. The manifest lists the number of Python files, the commit SHA and the SHA-256 of the sorted path and Git-blob-ID pairs. [Source manifest](source-provenance.json)

## Reproduction

Use one checkout per commit in the source manifest. The examples below use `/bench/src/main-latest` for the control and `/bench/src/PR_NUMBER` for a candidate. Mount a GLM-5.2 W4A8C8 model configuration at `/model/config.json` for the pool tests. Make the named native libraries and the model files available to both arms before running.

From the kernel-enabled runtime, select a checkout and a fresh output directory for each process. The same directory holds that process's cold compiler caches and samples.

```bash
export ARM=/bench/src/main-latest
export ARM_SHA=$(git -C "$ARM" rev-parse HEAD)
export PYTHONPATH="$ARM/python:${PYTHONPATH:-}"
export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 TASK_QUEUE_ENABLE=2
export RUN=$(mktemp -d)
export TRITON_CACHE_DIR="$RUN/triton" ASCEND_CACHE_PATH="$RUN/cann"
mkdir -p "$TRITON_CACHE_DIR" "$ASCEND_CACHE_PATH"
python scripts/allocator_bench.py "$RUN/result.json"
```

For each row below, invoke the script with a fresh output path followed by the arguments shown. Set `TASK_QUEUE_ENABLE=1` for graph capture. For route replays, expose NPU devices 0 through 7. The decode queue fixture is imported from the candidate test module at `/bench/src/49103` in both arms. The framework and allocator always import from the arm selected in `PYTHONPATH`.

| Script | Control arguments | Candidate arguments |
| --- | --- | --- |
| `route_bench.py` | `prefill off` or `decode off` | `prefill on` or `decode on` |
| `lpm_bench.py` | `0` | `64` |
| `backfill_bench.py`, `retire_bench.py` | `0` | `1` |
| `allocator_bench.py`, `sparse_bench.py` | No additional arguments | No additional arguments |
| `pool_bench.py` | `dense` | `compact` or `packed` |
| `numeric_bench.py` | `bf16` | `int8` |
| `index_bench.py` | `bf16` on #48560 | `int8` on #48562 |
| `index_stack_bench.py` | `bf16 21` on #48560 | `int8 21` on #48562 |

The scripts were collected from the completed hardware jobs and formatted for this archive. The Python abstract syntax tree of each published script was verified against its measured copy. [Script checksums](scripts/checksums.json)
