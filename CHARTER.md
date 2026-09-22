# Software: team charter

## Purpose

Make the accelerator useful through measured workloads, trustworthy numerical behavior and an executable software path. Software owns llama.cpp profiling, model/workload references, operation mapping, and accelerator backend/runtime/host integration.

Use evidence to help choose hardware boundaries. The current functional diagram does not establish that every box needs a dedicated engine. Software and the hardware teams compare candidates using workload behavior, data movement, numerical requirements and implementation cost.

## Responsibilities

### Profiling and workloads

Maintain reproducible llama.cpp workloads and prefill/decode measurements. Capture operation shapes, formats, fusion boundaries, state and data behavior. Separate backend-specific timing from arithmetic counts, logical tensor sizes and measured traffic; explain what each result can support.

### Numerical reasoning

Study quantization, accumulation, rounding, state precision and model-quality effects. Distinguish mathematical intent, chosen finite-precision semantics and observed behavior. A mixed-format GGUF benchmark does not validate a different hardware encoding.

### Reference behavior

Develop understandable reference computations, representative inputs and intermediate/output/state comparisons with Verification. State the scope of each reference and whether a comparison expects exact agreement, numerical tolerance or task-quality equivalence.

### Operation mapping and integration

Investigate how llama.cpp/ggml operations map to candidate accelerator kernels, including tiling, layouts, fusion and host fallback. Develop backend/runtime software for buffer handling, submission, completion and errors as shared interfaces become stable. Include transfers, packing and dispatch when evaluating an offload benefit. Firmware software can be part of this path; CPU, bus and reset hardware integration remains a shared boundary with Control and the system designers.

### Reproducible knowledge

Preserve methods, source/model provenance, raw evidence and numerical explanations so results can be revisited. Share useful traces and findings before a complete study is finished. Follow the [Git AI setup](docs/git-ai.md) when contributing with AI assistance.

## Boundaries and shared decisions

Software owns software execution mapping directly; it does not depend on a separate compiler team. Custom CPU or ISA development is outside this charter. The working approach uses ordinary CPU instructions to interact with the accelerator through agreed software/device interfaces.

Compute owns candidate arithmetic organizations, Memory owns storage/transfer design, Control owns device sequencing, and Physical Design contributes implementation constraints. Software informs these choices and implements its side of the agreed contracts. Verification retains independent reasoning and checks. CPU profiles alone do not determine accelerator area, bandwidth, throughput or the final division of compute units.

All teams can start from the existing shapes, references and proposed diagram. Use provisional parameters for unresolved details, exchange incremental evidence and implement stable pieces in parallel. Discuss a shared contract change before depending on it across repositories.

## Member autonomy

Members can choose profiling, workload studies, quantization experiments, references, kernel/runtime implementation, numerical comparisons or teaching material. Methods are selected to answer the question. Changes that redefine expected numerical behavior or workloads used for shared claims are discussed with affected teams. The existing MAC example does not set the team's numerical or architectural scope.

## Collaboration

| Partners | Shared concerns |
|---|---|
| Compute 1 and Compute 2 | Compare resource sharing, specialization and kernel boundaries against actual shapes, formats and execution behavior. |
| Memory Control and Top-Level Control | Agree layouts, lifetimes, transfers, submission/completion and dependency handling; estimate integration overhead. |
| Verification | Share references and fixtures, investigate discrepancies, and agree numerical/state acceptance criteria. |
| Architecture, Physical Design and system integration | Reconcile workload goals, software requirements, physical constraints and complete-system behavior. |

## Possible directions

The current priority is extending the completed [llama.cpp baseline](experiments/llama-cpp/2026-09-22/REPORT.md). [Next investigations](docs/profiling-next-steps.md) describe useful directions, not personal assignments or a fixed architecture. Small, well-scoped studies can be valuable without large-model training or new infrastructure.

## What progress means

Progress means the project makes better-supported compute and interface choices, numerical references become more trustworthy, and the software/device path becomes more useful and reproducible. A failed offload experiment or a negative numerical result can improve the design by ruling out an expensive mistake.

Leads help members interpret this purpose, find collaborators, access resources
and share what they learn. Members choose their questions and contributions.
Research, design reasoning, experiments, implementation, documentation and
teaching can all advance the charter; success is not measured by the number of
code changes or completed tickets.

The team can revise this charter as its understanding evolves. Changes to a
shared boundary or commitment are discussed with the teams affected by them.
The [objectives](OBJECTIVES.md) describe durable outcomes, and the
[repository structure](README.md#repository-structure) provides places to develop
work without specifying a mandatory project or sequence.
