# Scientific code organization

Software uses execution and graph evidence to evaluate compute boundaries, shared
resources and host integration. The [Architecture diagram](https://github.com/SiliconBadgers/architecture/blob/main/docs/accelerator-diagram.md)
is the shared design reference, not a list of already justified hardware engines.

## Stages and dependency direction

```text
pinned execution -> capture -> CDFG -> mapping -> simulation -> evaluation/sweeps
                                      ^                           |
                               resolved configuration             v
                                                           reports and web
```

| Owner | Responsibility |
|---|---|
| `core/runner/`, `core/harness/` | Existing executable builds, instrumented execution, graph/trace export and provenance. Executable-to-CDFG is capture, not decompilation. |
| `core/cdfg/` | Captured tensors/operations, operand slots, aliases, storage kinds and conservative data/state dependencies. No hardware resources, bandwidth, latency, engine or NumPy dependency. |
| `engine/sbengine/mapping.py` | Existing implementation, capability, tile and lowering choices. Reference CDFG IDs, not a copied graph; keep memory-dependent selection explicit. |
| `costs.py`, `memory.py`, `schedule.py` | Existing arithmetic/service equations, residency/traffic accounting and resource scheduling. No new allocator, alias analysis or simulation fidelity. |
| `model.py`, `sweep.py`, CLI | Compose the stages and reuse existing enumeration, sensitivity, Monte Carlo and Pareto methods. No automatic baseline promotion or search service. |
| `web/`, maintained reports | Consume canonical source and results. Shared-source moves update native imports and browser packaging together. No duplicate equations in presentation. |

Lower stages do not import evaluation, presentation or dated research. Compatibility
imports preserve old public paths without a second implementation. Share dense work
counting only where conventions match. CPU-fit features, measured-operation grouping,
structural fusion and rendering have distinct semantics and remain distinct.

## Scientific style

Use readable calculations, explicit units and meaningful intermediates. Introduce
helpers for a scientific responsibility or real duplication, not line-count goals.
Comments explain equations, dependencies, approximations and limitations; preserve
attribution. Avoid progress narration, speculative frameworks and hardware-command
proposals in maintained scientific source.

Use capture, CDFG, mapping, lowering, schedule, baseline, decision, sweep and cost
proxy consistently. Separate measured execution, graph-derived structure and
assumed target behavior. Logical bytes are not measured memory traffic; a unitless
cost proxy is not physical area, power or implementation cost. A resource knee is
not a globally optimal design. Keep prefill and decode separate.

## Baselines and decisions

Use `core/experiments/qwen35-2b-q4km.json` for exact model/source identities.
Q4_K_M is mixed captured quantization, not uniform INT4. Reproduction names an
explicit run and workload, not `latest`. Complete resolved profiles remain
compatible with the existing loader; accompanying records cite captures, assumptions,
parent revision, evidence and approval status. Defaults are provisional unless an
approval record says otherwise.

Compute, local-memory and joint studies declare varied paths, fixed assumptions and
coupled prerequisites. External memory/platform assumptions stay fixed in primary
studies. Use existing fields and methods, not another parameter or decision system.
Freezing is manual review: add a new revision with parent, selected configuration,
supporting study and rationale; preserve earlier revisions and rerun affected studies.

## Evidence and validation

Preserve dated inputs, reports and reproduction entry points. Historical JS models
retain their different assumptions; they are preserved consumers, not new optimization
targets. Keep CPU-reference isolation, optional backends/diagnostics, fallback
provenance, checkpoint bit identity and explicit missing-evidence status. CUDA
accelerates iteration, not accelerator architecture evidence.

Commit source, specifications and compact summaries. Keep weights, raw traces,
candidate rows, timelines, checkpoints and rendered intermediates under ignored
`work/` or `build/`. Preserve existing historical bulk until its consumers and
retrieval/regeneration paths are verified. Do not rewrite old evidence to resolve
configuration mismatches.

Compare graph interpretation, mapping, counts, traffic, footprints, feasibility,
schedules and public outputs against [characterization](refactor-characterization.md).
Unexplained differences stop an extraction. Result-changing correctness fixes are
separate explained changes; do not relax tolerances. Run affected native/browser
checks after source moves and all suites before handoff. Report skips and unavailable
inference, hardware and licensed-tool checks explicitly.

Keep commits small and independently usable. Use issue-linked PRs and request
the maintainer-designated reviewers. Follow the repository's existing approval
and merge requirements; these design rules do not change them.

Ponytail is optional advisory review, not required installation, hooks or source
policy. Scientific authorities remain [equations](../engine/docs/EQUATIONS.md),
[assumptions](../engine/docs/ASSUMPTIONS.md) and [validation](../engine/docs/VALIDATION.md).
