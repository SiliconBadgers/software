# Software profiling and runtime evidence

Use llama.cpp profiling and numerical evidence to guide compute-unit boundaries, shared resources and host/device work. Software also owns operation mapping, backend/runtime and host integration.

Open the [software workspace](https://siliconbadgers.com/software/) for
model comparisons, captured graphs, execution estimates and source links.
[Workspace development](web/README.md) describes components and branch previews.

## Start here

1. Read [the current assignment and artifact locations](docs/START-HERE.md).
2. Read the [current report](REPORT.md), [scientific code organization](docs/DESIGN-RULES.md)
   and [contribution guide](CONTRIBUTING.md). Work on a branch and open an
   issue-linked PR for review under the repository's existing requirements.

## Current issues

- [Extend llama.cpp profiling and recommend compute boundaries](https://github.com/SiliconBadgers/software/issues/3)

## Repository structure

| Location | Purpose |
|---|---|
| [experiments/llama-cpp/](experiments/llama-cpp/README.md) | Recorded baseline in 2026-09-22/. Put each new run in a separate dated or named directory with a manifest, commands, source/model hashes, host/backend, inputs, results and limitations. |
| [research/compute-mapping/](research/compute-mapping/README.md) | Operation/dependency maps, offload comparisons and recommendations. Link each claim to a run or source trace; separate measured time/traffic from estimates. |

## Current material and scope

The profiling runner supports CPU reference execution, Metal and optional CUDA
iteration. [Shared CDFG source](core/cdfg/), [mapping and evaluation](engine/README.md),
[resolved baselines](engine/designs/README.md) and [reproducible studies](engine/sweeps/README.md)
organize the existing analysis. [Explicit tests](tests/README.md) cover these paths.
CPU time shares are not accelerator area or speedup; mixed Q4_K_M does not validate
a custom uniform INT4 format. The small MAC reference remains an example.

[Shared diagram](https://github.com/SiliconBadgers/architecture/blob/main/docs/accelerator-diagram.md) · [Software evidence](https://github.com/SiliconBadgers/software/tree/main/experiments/llama-cpp/2026-09-22)

[CHARTER.md](CHARTER.md) and [OBJECTIVES.md](OBJECTIVES.md) describe the
longer-term purpose. Current issues and the starting guide specify the work
assigned now. [SETUP.md](SETUP.md) describes existing example commands and scope.

Device firmware and llama.cpp integration probes: [integration/](integration/README.md).
