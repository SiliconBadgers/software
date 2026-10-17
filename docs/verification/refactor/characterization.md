# Pre-refactor characterization

Status: first local preparation step, not the completed refactor or a new simulator.
Starting source: reviewed main `0e06023d74c06cc12bb9e05d9e65102347ecc3f0`
(merged Software PRs #18 and #20). The proposed layout and sequence remain revisable.
This step changes test selection and adds regression evidence; production equations,
CLI behavior, publication, recorded results and browser code are unchanged.

## Reference and evidence contract

- Experiment: `core/experiments/qwen35-2b-q4km.json`; model
  `Qwen3.5-2B-Q4_K_M.gguf`, mixed captured Q4_K_M, not uniform INT4.
  The compact fixture preserves the complete pinned experiment, source/model hashes
  and repository run-manifest hash. No weights are added to Git.
- Explicit capture run: `results/2026-09-27_1839_windows-2`, all 22 workload pairs
  (44 phases). Six fixed-length pairs: pp128/pp512/pp8192, flash attention on/off;
  sixteen natural-prompt pairs: all eight prompts, on/off. No implicit `latest`.
- All three registered profiles are resolved in full in the fixture. Existing
  `accel-balanced` and `accel-efficient` files inherit defaults; these names alone
  are not immutable baseline configurations. `legacy-equations` deliberately
  models uniform INT8 as an alternative scenario, not a new capture identity.
- Ten separate switch cases on pp512-fa-on cover fusion none/groups, direct weight
  streaming, serial scheduling, no within-op overlap, serial/matrix-4x recurrence,
  uniform-INT4, batch 2 and serial softmax. Batch 2 and uniform precision are model
  scenarios, not measured workloads. Memory/platform defaults remain assumptions,
  not accepted Architecture decisions or optimization targets.
- Graph records cover tensor identities, operand slots, aliases, storage roots,
  scheduler positions, metadata/zero-node filtering, dependencies and dense MACs.
  Evaluations cover feasibility, resources, work, traffic, service times, bounds,
  selected pools/notes, operations and reservations. Raw per-node costs before
  fusion are also retained, using the existing evaluator and cost functions.
- Exact SHA-256 digests detect changes; they do not establish physical correctness.
  Existing analytic and independent accounting tests remain necessary. No rounding
  or relaxed tolerance is used to make a comparison pass.
- The historical inventory covers 446 tracked files under `experiments/`,
  `research/` and `results/`: exact repository blob identities/modes plus Git's
  working-tree comparison. OS checkout line endings are not new scientific
  content; existing CRLF CSV blobs must not be normalized indiscriminately.
  Untouched local-byte SHA-256 hashes are also retained in the ignored comparisons.
  Historical gitlink identities are recorded separately; submodule contents are
  not folded into this inventory.

The compact Git fixture is
`engine/tests/fixtures/characterization-0e06023.json`. Full comparison data and
starting-suite logs stay in ignored `build/refactor-characterization/0e06023/`.
The generator refuses to overwrite an existing output directory and does not
update the tracked fixture. Any intentional fixture change needs an explained,
reviewed diff and separately retained prior evidence.

From the repository root, using an already configured Python environment:

```text
python -B -m unittest discover -s engine/tests -v
python -B engine/tests/characterization.py --out build/refactor-characterization/0e06023/new-comparison --compare engine/tests/fixtures/characterization-0e06023.json
```

Missing reference inputs fail the new characterization tests, not skip them.
Older measured-trace tests can still skip absent local-only evidence. Test helpers
now select the explicit committed run; the user-facing `latest_run` API and CLI
defaults have not been changed. Extra local captures cannot silently replace the
test baseline.

## Compatibility inventory before moving sources

| Surface | Current contract / consumers |
|---|---|
| Profiling entry points | `windows/run.ps1` and `unix/run.sh` bootstrap dependencies, then invoke `core/runner/cli.py`; retain both platform entry points. |
| Core commands | `profile`, `doctor`, `analyze`; top-level `--platform-dir`. `doctor` shares profile options except its own dry-run action. |
| Core options | `--experiment`, `--work-dir`, `--results-root`, `--lengths`, `--threads`, `--repetitions`, `--trace-repetitions`, `--decode-steps`, `--long-continuation`, `--jobs`, `--metal`, `--skip-metal`, `--cuda`, `--cuda-controls`, `--cuda-no-host-control`, `--no-trace`, `--no-graphs`, `--include-diagnostic`, `--no-prompt-graphs`, `--no-prompt-timings`, `--prompts-only`, `--prompt-set`, `--smoke`, `--model`, `--dry-run`. |
| Backend roles | CPU reference by default; `--cuda on` adds optional execution, `--cuda only` is the fixed-length fast path without traces/capture. CPU-in-CUDA controls are opt-in diagnostics. Existing Metal auto/on/off behavior remains. CPU fallback provenance excludes fallback runs from CUDA speedups. These are execution tools, not proposed accelerator behavior. |
| Engine entry point | From `engine/`, `python -m sbengine`; commands `list`, `eval`, `sweep`, `ablation`, `switches`, `groups`, `sensitivity`, `montecarlo`, `parity`, `validate-cpu`, `serve`. |
| Engine options | Common evaluation options `--run`, `--graph`, `--profile`, repeated `--set`; `eval --json/--nodes`; `sweep --axis/--out/--name`; analysis commands `--json`; sensitivity/Monte Carlo `--delta`; Monte Carlo `--designs/--samples/--seed`; validation `--report`; server `--host/--port/--open`. Keep current defaults, units and exit behavior visible. |
| Graph imports | `sbengine.graph.Tensor`, `Node`, `Graph`, `Workload`, operation constants, `layer_of`, `capture_path`, `has_capture`, `load_workload`, `latest_run`; retain compatibility adapters during extraction. Graph node IDs and original scheduler positions are different identities. |
| Other maintained owners | `config` merging/validation/hash; `costs.Ctx`, `Cost`, `node_cost`; `memory` fusion/residency; `schedule.schedule`; `model.evaluate`; `sweep` summaries/grid/Pareto/ablation/sensitivity/Monte Carlo; `groups` structural grouping; `trace` and `cpu_validate` measured evidence. |
| Graph callers | Engine costs/groups/memory/model, CLI/web/CPU validation and tests; `web/build.py` and `web/components/python/bridge.py`. Core graph reporting currently has its own raw-capture representation. |
| Core analysis | `core/analysis/graph.py` dense MAC helper is consumed by fusion and offload reports. Measured operation grouping, structural fusion, offload boundaries and engine groupings are not interchangeable. CPU-fit attention/GDN features are not universally actual MAC counts. |
| Native engine server | `sbengine/web.py`: existing meta/eval/sweep/ablation/sensitivity routes, invalid-config handling and nonfinite-JSON sanitization; `engine/tests/test_web.py` retains API checks. |
| Browser Python | Registry copies engine `.py` sources to `assets/python/`; worker installs them under virtual `/engine`, imports bridge, which imports `sbengine.graph`. A graph move must change packaged paths, manifests and imports in the same step; native imports alone are insufficient. |
| Browser/export contracts | JS and Python controls, graph/dependency maps, remembered config/engine, sweep/comparison tables, JSON/CSV and timeline exports, local server and mobile layout. Existing browser suites exercise these against native evaluation. |
| Publication | `web/registry.json`, source attribution/manifest hashes, build metadata, existing main/preview isolation, workflows and `web/publish.py`. Preserve, do not mix auth/deployment changes into this refactor. |
| Historical consumers | Frozen Zeb explorer/equations, JS scheduler/sensitivity/joint/recurrence studies, recorded Mac/Windows evidence, dated dependency maps and reproduction scripts. Their different memory scheduling and modeling assumptions are intentional historical contracts, not duplicate code to casually replace. |
| Other integration | Root MAC reference/tests/vector generation, sibling contracts and existing host/offload evidence remain. No new compiler/ISA/hardware validation is implied. |

`profile --dry-run` applies inside the runner; platform wrappers can still create
an environment/install dependencies first. `doctor` fetches/configures source;
`analyze` writes derived reports/manifests. Root `make doctor` is the small MAC
example, not the profiler preflight. Avoid using any of these as a read-only audit
shortcut without accounting for their writes.

## Known gaps, separate from source reorganization

These were pressure-tested on the starting revision. They are not silently fixed
or treated as physically valid just because current outputs are preserved.

| Class | Observed problem and follow-up boundary |
|---|---|
| Numeric domains | Enabled DMA with zero bytes/cycle validates then divides by zero; infinite matrix rates and negative cost coefficients validate. Audit related finite/range/dependent constraints across fields, not just one input. |
| Capability completeness | Removing matrix requant, vector quant and scalar support can still report a valid matrix workload: missing requantization is silently skipped. Test full composite legality in a correctness change. |
| Composite reservations | Requantization tails and chunked recurrence charge secondary compute without auxiliary reservations; serial softmax reserves only part of its vector work. Serial matrix recurrence has a separate fix already in main. Reconcile all composite paths, not only that instance. |
| Invalid-design analysis | Sensitivity and Monte Carlo can report latency statistics for model-infeasible designs; ablation drops feasibility context. Pareto already filters invalid designs. Do not interpret unmapped zero-duration work as an improvement. |
| Heterogeneous CSV rows | A config-invalid first sweep point fixes the CSV columns to assignment/valid/errors, discarding later feasible-point metrics. Review the record-schema class and retain feasibility evidence. |
| Missing-evidence status | `parity` can exit successfully with `all match` when there were no summaries to compare. CPU validation correctly says NOT CHECKED. Missing checks need consistent explicit denominators/status. |
| Host-dependent tests | Prior newer-host checks exposed a `>1 s` measured-prefill assertion and a 16.8% held-out prefill fit error against a 10% threshold. This run has no traces, so neither failure is resolved or reproduced by its three skips. Do not relax thresholds during refactoring. |
| Baseline provenance | Named profiles inherit defaults and CLI discovery uses newest manifests. Full resolved profiles and explicit workload identity are captured here; reviewed Architecture baselines/decision studies are still future work. |
| Documentation drift | Older Abhinav review guidance and draft-CUDA wording lag later team direction/merged work. Reconcile owners and guidance explicitly; do not infer remote policy changes or send review requests from this local step. |
| Scientific scope | CPU shape/accounting fits are not hardware timing, measured memory traffic, quality or implementation-cost validation. Unitless cost and resource-sensitivity curves do not establish a global optimum. |

The next low-risk step is a concise design-rule document and agreement on extraction
boundaries. Behavior-changing correctness work should be isolated from moves and
adapters, with changed outputs and evidence explained. Do not promote an assumed
hardware configuration or expand CUDA characterization into the Software objective.

## Local preservation and verification

No CDFG/prototype source was selected for inclusion. None was identified in the
current workspace/branch/worktree inventory; that is not proof none exists
elsewhere. No older development branch or prototype was deleted or folded in.
PR #18's preserved benchmark archive and separated publication-auth/browser patch
remain under ignored `work/`, outside the refactor; all 104 local files match their
before/after SHA-256 inventory. No new worktree, global tool,
model download, remote branch, PR, comment or document write is needed here.

Starting suites on reviewed main: core 88; engine 82 with three evidence-dependent
skips; publication 7; MAC 4; JavaScript 46 across all five model/study suites.
After this change: engine 86 with the same three skips, including the new exact
76-evaluation contract and compressed/plain import checks. Repeat full comparison
matches. Historical hashes match. Static build packages 188 assets. Both browser
suites pass, including Python/native parity, workload switching, sweeps, exports,
study navigation and mobile layout. The final clean-checkout compatibility gate
belongs after the staged refactor and is not claimed here. None of these checks
reruns inference or qualifies RTL.

Local tool versions: Python 3.13.15, NumPy 2.5.3, Matplotlib 3.11.2,
Node 24.19.0 and Playwright 1.62.1 with existing native Chrome. No dependency
installation or browser download was needed. The loopback preview server is
stopped after testing. The current portable comparison is `verified-compact.json`
in the ignored audit folder; earlier full snapshots retain the same graph/result
digests and original local-byte inventories.

AI assistance: OpenAI Codex assisted this local characterization. An exact model
identifier was not available in the task's tool/runtime metadata; none is inferred.
