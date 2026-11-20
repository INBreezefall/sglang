# Upstream test status for the ten retested pull requests

Checked 2026-11-20 at 19:00 Singapore time. GitHub reported all ten PRs as Open with draft mode disabled. None carried the upstream `run-ci` label. On each measured tip the only failed check was `pr-test-finish`. Its job log says the label is required to run the gated test suites. The benchmark evidence and the local registered tests are recorded on each individual PR.

| PR | Checked head | Workflow job |
| --- | --- | --- |
| [#48538](https://github.com/sgl-project/sglang/pull/48538) | `e2e1c3acc1d2` | [Upstream CI gate](https://github.com/sgl-project/sglang/actions/runs/44721507092/job/136238411241) |
| [#48539](https://github.com/sgl-project/sglang/pull/48539) | `4e344c895359` | [Upstream CI gate](https://github.com/sgl-project/sglang/actions/runs/44700697569/job/136168343183) |
| [#48540](https://github.com/sgl-project/sglang/pull/48540) | `f3c1a5bec090` | [Upstream CI gate](https://github.com/sgl-project/sglang/actions/runs/44700712794/job/136168385640) |
| [#48541](https://github.com/sgl-project/sglang/pull/48541) | `3518681e48b2` | [Upstream CI gate](https://github.com/sgl-project/sglang/actions/runs/44721583704/job/136238693120) |
| [#48543](https://github.com/sgl-project/sglang/pull/48543) | `166bf929ff0e` | [Upstream CI gate](https://github.com/sgl-project/sglang/actions/runs/44698983461/job/136162758627) |
| [#48555](https://github.com/sgl-project/sglang/pull/48555) | `e49b95c6e357` | [Upstream CI gate](https://github.com/sgl-project/sglang/actions/runs/44700739627/job/136168446469) |
| [#48560](https://github.com/sgl-project/sglang/pull/48560) | `288e6f168afb` | [Upstream CI gate](https://github.com/sgl-project/sglang/actions/runs/44721611174/job/136238796575) |
| [#48562](https://github.com/sgl-project/sglang/pull/48562) | `d7dad05d1ed4` | [Upstream CI gate](https://github.com/sgl-project/sglang/actions/runs/44721631119/job/136238868729) |
| [#48631](https://github.com/sgl-project/sglang/pull/48631) | `39f97b506c1c` | [Upstream CI gate](https://github.com/sgl-project/sglang/actions/runs/44700781434/job/136168649642) |
| [#49183](https://github.com/sgl-project/sglang/pull/49183) | `b19942a2f5a4` | [Upstream CI gate](https://github.com/sgl-project/sglang/actions/runs/44721547558/job/136238574726) |
