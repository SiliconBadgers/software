# Software: high-level objectives

These objectives express the outcomes this team exists to advance. Members
choose which questions to pursue, the approach, the scale and the contribution
format. The order is not a priority ranking, and the examples of evidence are
illustrative. They are not a checklist, required deliverables or assignments.

Read the [charter](CHARTER.md) for scope and shared decision boundaries.

## Measured workload behavior

Help the project understand what llama.cpp executes and which candidate offloads or shared compute resources can help the intended workloads.

Evidence might include reproducible prefill/decode profiles, operation/state traces, shape and format inventories, or comparisons across representative contexts and inputs. The [initial baseline](experiments/llama-cpp/2026-09-22/REPORT.md) is complete; extend its coverage and interpretation.

## Understood numerical behavior

Reveal how representation and approximation choices affect correctness or workload quality.

Evidence might include analytical reasoning, quantization studies, comparisons or documented limits.

## Trustworthy reference computations

Give other teams useful numerical anchors with explicit scope and assumptions.

Evidence might include well-explained reference operators, reviewed examples, deterministic fixtures or discrepancy investigations.

## Useful execution mapping and integration

Connect workload operations to accelerator kernels and an executable backend/runtime/host path, with meaningful host fallback and error handling.

Evidence might include candidate kernel comparisons, a mapping or layout study, backend prototypes, or integration measurements accounting for transfers, packing and dispatch. Software owns this mapping; it does not require a separate compiler team or custom CPU/ISA.

## Reproducible shared learning

Make workload, numerical and integration findings accessible and possible to revisit. Give other teams incremental evidence they can use while their own investigations proceed.

Evidence might include documented methods, compact experiments, interpretation guides or educational material.

## Using these objectives

Use the objectives to explain why a chosen line of work matters and to reflect
on what has been learned. A member can advance several objectives through one
study, or explore one objective deeply. Leads support the conversation and help
connect related interests across the team. Members can propose new directions
or changes to these objectives when they can explain how the charter would be
better served.
