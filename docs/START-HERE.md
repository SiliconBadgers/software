# Software profiling and runtime evidence: current work

Use llama.cpp profiling and numerical evidence to guide compute-unit boundaries, shared resources and host/device work. Software also owns operation mapping, backend/runtime and host integration.

## Assignment

For maintained source, baselines and reproduction boundaries, see
[scientific code organization](DESIGN-RULES.md).

- [Extend llama.cpp profiling and recommend compute boundaries](https://github.com/SiliconBadgers/software/issues/3)

1. Reproduce and read the recorded 2026-09-22 baseline, report and CPU profiler methodology before extending it. Keep recorded results unchanged.
2. Extend representative workloads and continuations with prefill and decode separated. Capture shapes, formats/layouts, fusion/dependency chains, buffer/state lifetimes and operation timing.
3. Compare candidate offload and shared-compute boundaries, including packing, transfer, synchronization, dispatch and host fallback costs. Retain numerical-output and profiler-overhead checks.
4. Publish reproducible procedures, code, annotated traces and an evidence-backed recommendation. Share intermediate results with both independent Compute teams, Memory and Control as they become useful.

## Starting evidence

- [Central diagram](https://github.com/SiliconBadgers/architecture/blob/main/docs/accelerator-diagram.md)
- [Recorded Software profiling package](https://github.com/SiliconBadgers/software/tree/main/experiments/llama-cpp/2026-09-22)

## Artifact locations

| Location | What belongs here |
|---|---|
| [experiments/llama-cpp/](../experiments/llama-cpp/README.md) | Recorded baseline in 2026-09-22/. Put each new run in a separate dated or named directory with a manifest, commands, source/model hashes, host/backend, inputs, results and limitations. |
| [research/compute-mapping/](../research/compute-mapping/README.md) | Operation/dependency maps, offload comparisons and recommendations. Link each claim to a run or source trace; separate measured time/traffic from estimates. |

## What runs today

A CPU-reference profiling workflow with optional Metal/CUDA execution, captured
CDFGs and the existing analytical engine are available. See the [current report](../REPORT.md),
[resolved baseline](../engine/designs/README.md), [study reproduction](../engine/sweeps/README.md)
and [component checks](../tests/README.md). CPU time shares are not accelerator area
or speedup; mixed Q4_K_M is not a custom uniform INT4 validation.

These structures organize the work; they do not close the research objective.
Use explicit evidence, keep unresolved decisions provisional and publish useful
intermediate results through the repository's existing branch/PR review process.

Follow [CONTRIBUTING.md](../CONTRIBUTING.md) before editing or committing.
