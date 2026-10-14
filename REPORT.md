# Current Software analysis and refactor

## Question and method

How can the existing profiling/model work support reproducible compute and
local-memory decisions without changing its scientific meaning?

Starting from reviewed main `0e06023`, the refactor extracts the existing captured
representation into `core/cdfg/`, implementation/tile/lowering/fusion selection
into `sbengine/mapping.py`, and makes the evaluator compose mapping, work/traffic,
residency and scheduling explicitly. Old graph imports and costing entry points
remain compatible. No new simulator, optimizer, search service, physical model,
hardware command encoding or architecture choice is introduced.

The [design guide](docs/DESIGN-RULES.md) defines ownership and evidence policy.
[Characterization](docs/refactor-characterization.md) records starting behavior
and known defects. [Complete baseline/decisions](engine/designs/README.md) and
[thin study reproduction](engine/sweeps/README.md) organize existing methods.
No local CDFG/prototype source was included; unidentified work elsewhere is not
discarded or presumed migrated.

## Inputs and assumptions

- Exact experiment identity: [pinned Qwen configuration](core/experiments/qwen35-2b-q4km.json),
  `Qwen3.5-2B-Q4_K_M.gguf`, llama.cpp `f46bc30`; mixed captured quantization.
- Explicit run: [2026-09-27_1839_windows-2](results/2026-09-27_1839_windows-2/EVIDENCE.md),
  all 22 workload pairs / 44 phases. Primary study: pp512-fa-on, batch-1 capture.
- `characterized-v1` records the fully resolved previous balanced configuration
  and capture/source hashes. It is provisional, not an accepted Architecture design.
  Hardware counts/rates, external memory, residency/fusion/recurrence choices,
  scheduling and unitless cost coefficients remain assumptions.
- Authoritative explanations remain [equations](engine/docs/EQUATIONS.md),
  [assumptions](engine/docs/ASSUMPTIONS.md) and [validation](engine/docs/VALIDATION.md).
  Those component records describe their dated experiments, not newly measured results.

## Results and interpretation

All 76 characterized evaluations preserve exact graph interpretation, raw costs,
feasibility, traffic, footprints, bounds, operations/reservations and public
outputs. The separate original/new mapping audit checked 254,784 node costs
across profiles, unavailable capabilities, fallback, recurrence, memory-fit and
state-residency classes. Existing dense-work conventions now have one owner;
measured CPU-fit features and structurally different grouping/rendering remain
distinct. Historical input files and gitlink identity are preserved.

[Compact study summaries](docs/sweeps/README.md) record existing-grid reproductions:

| Study | Model-feasible / sampled | Interpretation |
|---|---:|---|
| Matrix capacity | 3 / 4 | Eight arrays exceed the assumed usable DSP budget; feasibility for the others is not physical fit. |
| L1 service | 4 / 4 | Banking is an assumption/cost-proxy axis; this is not a recommended SRAM organization. |
| Matrix + L1 | 12 / 16 | Joint candidates expose interactions under fixed external/platform assumptions; no selection is promoted. |

Frontiers use explicitly defined cost-proxy/decode objectives. They are mathematical
frontiers of sampled estimates, not calibrated architecture optima. Prefill remains
reported separately. Candidate validity counts and record digests accompany every
summary; complete rows remain ignored under `build/sweeps/`.

The dated [profiling findings](experiments/llama-cpp/2026-09-22/REPORT.md),
[operation/dependency study](research/compute-mapping/2026-10-02-op-dependency-map/README.md)
and [recurrence sensitivity](research/compute-mapping/2026-10-03-recurrence-multiplier-sensitivity/REPORT.md)
remain available unchanged. Historical JS equations/schedulers are preserved,
not silently replaced with the Python model. Optional CUDA remains an iteration
tool; CPU-reference instrumentation, fallback labels and numerical comparisons
are unchanged.

## Verification and limits

[Complete local acceptance](docs/refactor-verification.md) records fresh-checkout
results, exact comparisons, visual checks, preserved inputs and unavailable gates.

Local verification uses existing Python 3.13.15, NumPy 2.5.3, Matplotlib 3.11.2,
Node 24.19.0, Playwright 1.62.1 and native Chrome. [Explicit test commands](tests/README.md)
cover core, engine, repository/MAC, publication and all five JS suites. Native/worker
parity, controls, graphs, exports, study transfer and desktop/mobile workflows
are checked after shared-source moves. Detailed logs and comparisons are contained
in ignored `build/refactor-final/`; no generated replay package enters Git.

Known numeric-domain, composite-capability/reservation, invalid-design reporting,
heterogeneous-CSV and empty-parity defects are listed in the characterization
audit and preserved for separate correctness changes. Three measured-evidence
tests skip because raw traces/summaries are absent. Earlier host-dependent fit
failures are not resolved by those skips. Analytical accounting/shape checks do
not qualify hardware timing, physical cost, energy, numerical quality or a real
chunked implementation. No new inference, SoC, RTL or licensed-tool qualification
is claimed; missing CPU validation remains NOT CHECKED.

Baseline freezing remains manual review, with parent, selected study, rationale
and approval evidence. All refactor commits remain local; no remote writes or
review requests are made without approval. The Software research objective is
not completed merely by source organization.

AI assistance: OpenAI Codex assisted this refactor; the exact model identifier
was unavailable in runtime metadata. Existing contributor and AI attribution is
preserved in the component sources and documentation.
