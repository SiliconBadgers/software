# Software

Software connects llama.cpp workloads to the accelerator: profiling execution, understanding numerical behavior, mapping operations to candidate hardware, and developing the backend, runtime and host integration. Workload evidence helps determine the compute boundaries; the proposed block diagram is a starting hypothesis.

## Start here

- [Profiling findings](experiments/llama-cpp/2026-09-22/REPORT.md) and [reproduction guide](experiments/llama-cpp/2026-09-22/README.md): the completed Qwen3.5-2B llama.cpp baseline.
- [Next investigations](docs/profiling-next-steps.md): extend that evidence while the hardware teams work in parallel.
- [Charter](CHARTER.md) and [objectives](OBJECTIVES.md): responsibilities and shared decisions.
- [Setup](SETUP.md) and [Git AI](docs/git-ai.md): experiment entry points, the existing MAC example and contribution provenance.

## Current evidence

The September 22 experiment includes pinned llama.cpp source/model provenance, CPU and Metal baselines, CPU operation traces and saved-output checks. At a 2K prompt, matrix products account for about 80% of timed CPU prefill work and 82% of decode; the vocabulary output head alone accounts for about 25% of decode. These measurements support investigating shared matrix arithmetic across several model functions. They do not select a final compute split or predict accelerator performance.

The baseline uses a mixed-format Q4_K_M GGUF on one Mac and a repeated passage. It is not a validation of the proposed custom INT4 format or model quality. The report retains the anomalous 8K tracing diagnostics separately from the accepted timing conclusions.

Software's next work extends this baseline with representative workloads, numerical comparisons, useful operation/state traces and candidate kernel/backend experiments. Other teams can use the current shapes and results immediately; they do not need to wait for a complete Software study.

## Repository structure

| Location | Purpose |
|---|---|
| [docs/](docs/README.md) | Software design, interfaces, decisions, contribution setup and next investigations. |
| [research/](research/README.md) | Workload, quantization, execution-mapping and runtime studies. |
| [experiments/](experiments/README.md) | Reproducible profiling and kernel experiments with results and limitations. |
| [tests/](tests/README.md) | Numerical, mapping and runtime checks; currently includes the MAC reference tests. |
| [fixtures/](fixtures/README.md) | Compact reference inputs, outputs and state cases with provenance. |

The team may extend this structure as useful. Existing example entry points stay
in their current locations, described in [SETUP.md](SETUP.md).

## Contributing

Choose a bounded question, experiment, implementation or explanation that advances the [objectives](OBJECTIVES.md). Share useful intermediate findings and discuss changes to numerical behavior or shared interfaces with the affected teams. These documents do not assign people or prescribe a ticket backlog.

The small MAC reference, four unit tests and deterministic vector generator remain available for learning and numerical-reference work; [SETUP.md](SETUP.md) states their scope. They do not define the accelerator's numerical format. Software works with a conventional CPU/ISA and owns software execution mapping directly; there is no separate compiler-team handoff in this scope.
