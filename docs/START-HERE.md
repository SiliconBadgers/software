# Software starting material

September 22, 2026. Initial investigations for team discussion; no personal assignments or deadlines.

## Shared starting points

- [Editable architecture diagram](https://github.com/SiliconBadgers/architecture/blob/main/docs/accelerator-diagram.md) and [candidate boundaries](https://github.com/SiliconBadgers/architecture/blob/main/contracts/accelerator-boundaries.md).
- [Workload cases and source shapes](https://github.com/SiliconBadgers/architecture/blob/main/docs/workload-cases.md).
- [Measured llama.cpp report](https://github.com/SiliconBadgers/software/blob/main/experiments/llama-cpp/2026-09-22/REPORT.md) and [reproduction procedure](https://github.com/SiliconBadgers/software/blob/main/experiments/llama-cpp/2026-09-22/README.md).
- [Parallel team investigations](https://github.com/SiliconBadgers/planning/blob/main/docs/team-start.md).

The diagram and engine split are proposals. Start from available shapes and
reference cases now; use explicit parameters or stubs where decisions remain
open. Software's broader profiling study is not a prerequisite. Preserve the
source revision, assumptions, commands and limits of each result. Members and
leads can choose a different investigation that resolves a relevant uncertainty.


## Evidence already available

The dated [llama.cpp package](../experiments/llama-cpp/2026-09-22/README.md) contains
the C++ harness, instrumentation patch, pinned inputs, baseline results, compressed
traces, saved logits and reproduction/analysis scripts. Reuse this work. The
[next investigations](profiling-next-steps.md) explain the outstanding questions.

## First useful extension

Choose one representative workload beyond the repeated passage and retain
matched tokens/settings across comparisons. Measure prefill and decode separately.
Extend operation traces with a selected dependency chain, intermediate layouts
and state/buffer lifetimes needed by Compute, Memory or Control. Publish that
small result as soon as it is useful, without waiting for the whole study.

Compare candidate offload boundaries including packing, dispatch, transfer and
host fallback. Map projections and the output head as well as attention,
recurrence, normalization and copies. Study numerical formats with Verification;
the mixed Q4_K_M baseline does not validate a custom uniform INT4 representation.

## Reproduce and check

Follow the dated package's exact commands and Python dependency requirements.
Reruns go in new directories; never replace recorded results. Preserve
source/model hashes and investigate the 8K timing anomaly separately. Include
saved-output checks and state/tolerance rationale with new instrumentation.

A useful submission has commands, inputs, latency/throughput, relevant shapes and
layouts, instrumentation overhead, numerical checks and a bounded recommendation.
CPU operation timing, logical byte counts and accelerator cost predictions must
remain distinct. All other teams can use today's published shapes immediately.
