# ML models, workloads and numerical methods: team charter

## Purpose

Connect the accelerator project to meaningful machine-learning workloads and trustworthy numerical understanding. The team studies algorithms, representations and workload behavior, and develops references that help other teams reason about both correctness and usefulness.

A correct hardware result is meaningful only relative to a well-understood computation. This team keeps the project grounded in algorithmic intent, finite-precision behavior and the characteristics of the workloads it chooses to support. Modeling can range from mathematical explanation to executable references and empirical workload studies.

## Responsibilities

### Workload understanding

Investigate relevant algorithms, model structures, operator mixes, dimensions and data behavior. Explain which characteristics matter to architectural and implementation choices.

### Numerical reasoning

Study representations, quantization, accumulation, rounding and error behavior as appropriate to the selected workloads. Distinguish mathematical intent, chosen finite-precision semantics and observed model quality.

### Reference behavior

Develop and maintain understandable reference computations and representative inputs that other teams can use. State what a reference establishes, its assumptions and the conditions it does not cover.

### Reproducible knowledge

Preserve methods, provenance, experiment context and numerical explanations so results can be understood and revisited. Make model and workload knowledge accessible to hardware and software contributors.

## Boundaries and shared decisions

ml-models owns numerical and workload references. Architecture uses that knowledge to establish shared system semantics; compute implements arithmetic; ml-compiler maps algorithms to supported execution; verification uses references alongside independent reasoning. A model implementation is evidence with assumptions, not an automatic definition of correct hardware behavior. Timing-oriented architectural modeling belongs with architecture unless a specific study is jointly owned.

## Member autonomy

Members can choose algorithm or workload studies, quantization experiments, reference implementations, numerical comparisons, dataset characterization or teaching material. Methods and frameworks are selected by the team as needed. Changes that redefine expected numerical behavior or the workload used for shared claims are discussed with the affected teams. The existing MAC reference does not set the scope of the team’s research.

## Collaboration

| Partners | Shared concerns |
|---|---|
| architecture and ml-compiler | Share workload structure, representation choices and numerical expectations that inform capability and mapping decisions. |
| rtl-compute and rtl-memory | Explain operation semantics and data characteristics that matter for datapath and storage design. |
| verification and accelerator | Provide references and representative cases, interpret discrepancies and clarify the limits of system-level correctness or quality claims. |

## Possible directions

Members might study a model family, compare numerical representations, investigate sensitivity to quantization, explain a reference operator, capture a representative workload or improve reproducibility. Small studies with clear assumptions can be useful without requiring large-model training or infrastructure.

## What progress means

Progress means algorithmic intent and approximation choices become clearer, references become more trustworthy, and the project can explain why its selected computations matter. A negative numerical result or a clarified reference assumption can materially improve the design.

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
