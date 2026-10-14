# Complete local refactor acceptance

Verified 2026-10-14 on local branch `refactor/scientific-characterization`, source
`2e53a124a30a84aa969a1cabe2332b73abae9309`, against reviewed main
`0e06023d74c06cc12bb9e05d9e65102347ecc3f0`. This record is documentation added after
testing that source; no numerical change is hidden in it. At this verification
checkpoint on 2026-10-14, nothing had been published.

## Gates

| Scope | Result |
|---|---|
| Starting contract | 44 captured phases, all three original profiles and ten switch cases: 76 complete evaluations match exactly. Graphs, raw costs, feasibility, traffic, footprints, bounds, operations/reservations and result schemas retain their digests. |
| Graph extraction | Plain/compressed parity across all 44 phases; old graph exports reference the canonical classes. CDFG contains no assumed resources or engine/NumPy dependency. |
| Mapping extraction | 254,784 exact original/new per-node cost comparisons across profiles, capability/fallback/recurrence/memory-fit classes and state-residency endpoints. Costing consumes explicit mapping without selecting again. |
| Baseline | Existing loader reproduces the characterized configuration/hash from a complete resolved profile. Changed defaults do not alter it; missing fields fail. Approval remains provisional. |
| Studies | Existing grid/evaluation/Pareto methods reproduced, with unchanged explicit candidates/objectives. Joint study reproduces the committed record digest and frontier in the clean checkout. External/platform assumptions remain fixed; no promotion. |
| Boundaries | Standard-library import/reference/artifact checks pass, including positive/negative cases. Current historical corpus is explicitly preserved. No new hooks/lint stack/blanket JSON ban. |
| Core Python | 88 tests pass, including harness-equivalence and optional backend/provenance cases. |
| Engine Python | 100 tests run: 97 pass, three missing-evidence skips. No tolerance was relaxed. |
| Repository/MAC | 9 tests pass: four numerical MAC references and five boundary-policy cases. |
| Publication Python | 7 tests pass; deployment/authentication source remains unchanged. |
| Historical JavaScript | All five suites pass: model 26, sensitivity 4, scheduler 7, joint 6, recurrence 3. Total 46. |
| Browser | Both suites pass from the main workspace and isolated clean checkout: native/worker parity, graph/control workflows, comparisons, exports, study transfer and desktop/mobile behavior. |
| Visual inspection | Native Chrome captures of desktop/mobile Python model and Sources pages inspected: no overlapping labels/cards or document-width overflow found on these pages. This is not exhaustive usability testing. |
| Evidence preservation | All 446 historical local-byte hashes match the starting inventory; repository inventory and historical llama.cpp gitlink remain unchanged. Cached Qwen model SHA-256 still matches the pinned experiment. No prototype/archive content was folded in or removed. |

The clean checkout was a contained local shared clone under ignored
`build/refactor-final/clean-checkout`, using the existing interpreter/dependencies
and browser. It started with no untracked or modified files and no model/source
downloads or initialized submodules. Complete Python, JS and browser checks passed;
the task-local preview servers are stopped separately after testing. Automatic
removal of the temporary checkout was blocked by the execution policy, so that
copy remains contained and ignored; it contains no unique source changes and can
be removed manually or recreated from the tested commit. Detailed logs, screenshots
and full comparisons remain ignored and contained.

Full characterization is under
`build/refactor-characterization/0e06023/final/`; final/clean suite logs and visual
captures are under `build/refactor-final/`. These generated packages are not part
of Git. Native tool versions are recorded in the [current report](../REPORT.md).

## Unchanged scope and remaining limits

The final diff does not change dated `experiments/`, `research/` or `results/`,
measurement harnesses, scheduler equations, native sweep algorithms, optional
backend execution, integration probes or publication/authentication behavior.
Required graph/cost/fusion compatibility paths remain; no redundant maintained
representation or alternate evaluator was introduced. Candidate-pricing formulas
needed for resource-dependent lowering selection were moved intact, not recalibrated.

The three skips are MAC-summary parity, measured host fallback and held-out CPU
fit; absent traces/summaries cannot certify them. The [known defects](refactor-characterization.md)
remain separate result-changing work: numeric domains, composite legality and
reservations, invalid-design statistics, heterogeneous CSV rows and empty parity.
Prior host-dependent failures remain unresolved. Missing CPU validation remains
NOT CHECKED, not a numerical pass.

No new inference, quantization-quality, real lowered-kernel, accelerator/SoC,
FPGA/ASIC, physical bandwidth/cost or licensed-tool qualification was performed.
This completes the requested source/configuration/reproduction refactor, not the
broader Software research objective or an optimal hardware design. Any new baseline
selection still needs manual review and evidence.

AI assistance: OpenAI Codex. Local task metadata verified `gpt-6.1-sol` for the
latest task on 2026-10-14; exact model identifiers for earlier stages were not
verified. This does not attribute every earlier stage to that model.
