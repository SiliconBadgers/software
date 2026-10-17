# Scientific code organization

Software uses captured workloads and analytical estimates to study compute,
memory and host integration. The [Architecture diagram](https://github.com/SiliconBadgers/architecture/blob/main/docs/accelerator-diagram.md)
is the shared design reference. The organization below describes current
calculation ownership and can evolve with the work.

## Stages and ownership

```text
pinned execution -> capture -> CDFG -> mapping -> simulation -> evaluation/sweeps
                                      ^                           |
                               resolved configuration             v
                                                           reports and web
```

| Owner | Responsibility |
|---|---|
| `core/runner/`, `core/harness/` | Builds, instrumented execution, graph/trace capture and provenance. |
| `core/cdfg/` | Tensors, operations, operand slots, aliases, storage classification, conservative dependencies and shared dense-work counting. Hardware-independent; standard-library only. |
| `engine/sbengine/mapping.py` | Capability, implementation, tile, lowering and fusion selection using CDFG identities. Memory-dependent selection remains explicit. |
| `costs.py`, `memory.py`, `schedule.py` | Arithmetic/service equations, residency/traffic accounting and resource scheduling. |
| `model.py`, `sweep.py`, CLI | Compose evaluation and run enumeration, sensitivity, Monte Carlo and Pareto analyses. |
| `web/`, reports | Present canonical calculations and results; update native imports and browser packaging together when shared sources move. |

Lower stages do not import evaluation or presentation. Compatibility imports
retain public paths without maintaining a second implementation. Share calculations
where conventions match; CPU-fit features, measured-operation groups, structural
fusion and rendering have distinct semantics.

## Configurations and studies

The [pinned experiment](../core/experiments/qwen35-2b-q4km.json) identifies
the source and mixed-Q4_K_M model. Reproduction names an explicit run and workload.
[Resolved baselines](../engine/designs/README.md) record complete configurations,
captures, assumptions, parent revisions and review provenance. Partial profiles
retain their existing override behavior; complete profiles reject missing fields.

[Study specifications](../engine/sweeps/README.md) declare axes, fixed assumptions,
objectives and coupled prerequisites. Primary compute/local-memory studies hold
external-memory and platform settings fixed. A reviewed baseline revision records
its parent, selected configuration, supporting study and rationale; earlier
revisions remain available.

## Evidence and checks

Keep equations readable, with explicit units and meaningful intermediates.
Explain approximations and limitations beside the calculation. Distinguish
measured execution, graph-derived structure and assumed target behavior: logical
bytes are not measured traffic, and a unitless cost proxy is not physical area
or power. Report prefill and decode separately.

Commit source, specifications and compact summaries; keep raw traces, candidate
rows, weights, checkpoints and rendered intermediates in ignored `build/` or
`work/`. Preserve dated evidence and historical models with their own assumptions.
The recorded inventory protects the original files while allowing new research
under the normal artifact rules. Study checks retain the original regression
cases and accept additional valid specifications.

Use [compatibility checks](refactor-characterization.md) for source extractions
and the [component tests](../tests/README.md) for affected behavior. Explain
result-changing fixes separately, retain comparison tolerances and report missing
evidence explicitly. CPU-reference instrumentation, optional backend provenance
and numerical comparisons retain their existing contracts.

Technical details live in the engine's [equations](../engine/docs/EQUATIONS.md),
[assumptions](../engine/docs/ASSUMPTIONS.md) and [validation](../engine/docs/VALIDATION.md).
Contribution and review guidance is in [CONTRIBUTING.md](../CONTRIBUTING.md).
