# Software profiling and runtime evidence

Use llama.cpp profiling and numerical evidence to guide compute-unit boundaries, shared resources and host/device work. Software also owns operation mapping, backend/runtime and host integration.

## Start here

1. Read [the current assignment and artifact locations](docs/START-HERE.md).
2. Work on a branch and open a PR for `@abhinavnandwani` using
   [CONTRIBUTING.md](CONTRIBUTING.md). Main requires a code-owner approval;
   admins can bypass.

## Current issues

- [Extend llama.cpp profiling and recommend compute boundaries](https://github.com/SiliconBadgers/software/issues/3)

## Repository structure

| Location | Purpose |
|---|---|
| [experiments/llama-cpp/](experiments/llama-cpp/README.md) | Recorded baseline in 2026-09-22/. Put each new run in a separate dated or named directory with a manifest, commands, source/model hashes, host/backend, inputs, results and limitations. |
| [research/compute-mapping/](research/compute-mapping/README.md) | Operation/dependency maps, offload comparisons and recommendations. Link each claim to a run or source trace; separate measured time/traffic from estimates. |

## Current material and scope

A reproducible CPU/Metal profiling package and the small MAC reference example exist. CPU time shares are not accelerator area or speedup. Mixed Q4_K_M results do not validate the slide's custom INT4/group-64 proposal.

[Shared diagram](https://github.com/SiliconBadgers/architecture/blob/main/docs/accelerator-diagram.md) · [Software evidence](https://github.com/SiliconBadgers/software/tree/main/experiments/llama-cpp/2026-09-22)

[CHARTER.md](CHARTER.md) and [OBJECTIVES.md](OBJECTIVES.md) describe the
longer-term purpose. Current issues and the starting guide specify the work
assigned now. [SETUP.md](SETUP.md) describes existing example commands and scope.
