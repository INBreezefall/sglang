# Isolated Ascend hardware validation

The 2026-11-20 follow-up uses GLM-5.2 W4A8C8 and Ascend 910B4-1. Each result identifies the upstream baseline, the individual candidate and the measurement boundary. The raw measurements use synthetic identifiers or public evaluation IDs.

| PR | Validation |
| --- | --- |
| [#48538](https://github.com/sgl-project/sglang/pull/48538) | [MLA draft layout selection](results/48538.md) |
| [#48539](https://github.com/sgl-project/sglang/pull/48539) | [Host-aware LPM scheduling](results/48539.md) |
| [#48540](https://github.com/sgl-project/sglang/pull/48540) | [NPU expert activation tiling](results/48540.md) |
| [#48541](https://github.com/sgl-project/sglang/pull/48541) | [Decode session placement](results/48541.md) |
| [#48543](https://github.com/sgl-project/sglang/pull/48543) | [Pinned host allocation and last-view release](results/48543.md) |
| [#48555](https://github.com/sgl-project/sglang/pull/48555) | [Compact DSA indexer buffers](results/48555.md) |
| [#48560](https://github.com/sgl-project/sglang/pull/48560) | [Packed INT8 main KV rows](results/48560.md) |
| [#48562](https://github.com/sgl-project/sglang/pull/48562) | [INT8 DSA indexer graph replay](results/48562.md) |
| [#48631](https://github.com/sgl-project/sglang/pull/48631) | [Packed-NZ checkpoint restore](results/48631.md) |
| [#48633](https://github.com/sgl-project/sglang/pull/48633) | [Public NPU allocator cold compilation](results/48633.md) |
| [#49183](https://github.com/sgl-project/sglang/pull/49183) | [Separate draft host budget](results/49183.md) |

[Physical P/D transport checks and runtime compatibility](runtime/transport-checks.md).

[Four-host shared model control](runtime/four-host-control.md).

[Upstream test status and the per-PR CI jobs](runtime/upstream-ci-gate.md).

[Four-host NEXTN host-cache control](runtime/nextn-control.md).
