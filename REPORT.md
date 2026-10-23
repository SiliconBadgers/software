# Software analysis

Software combines captured workload graphs with configurable resource models to
study compute capacity, memory service and operation mapping. The
[stage guide](docs/DESIGN-RULES.md) describes calculation ownership;
[engine documentation](engine/README.md) covers the analytical model.

## Baseline and reproduction

- Model: `Qwen3.5-2B-Q4_K_M.gguf`, with exact model/source identities in the
  [pinned experiment](core/experiments/qwen35-2b-q4km.json).
  Q4_K_M is mixed captured quantization, not uniform INT4.
- Capture: [2026-09-27_1839_windows-2](results/2026-09-27_1839_windows-2/EVIDENCE.md),
  containing 22 workload pairs / 44 phases.
- Configuration: [`characterized-v1`](engine/designs/README.md), a complete,
  provisional reproduction of the balanced configuration. Resource rates,
  external memory, residency and cost coefficients remain model assumptions.

Use the [baseline evaluation](engine/designs/README.md),
[study runner](engine/sweeps/README.md) and [component tests](tests/README.md)
for reproduction. [Compatibility checks](docs/refactor-characterization.md)
compare graph interpretation, mapping, costs and outputs against the recorded
reference.

## Resource studies

The [compact summaries](docs/sweeps/README.md) report pp512-fa-on under the
same baseline, with external-memory and platform assumptions held fixed.

| Study | Model-feasible / sampled | Interpretation |
|---|---:|---|
| Matrix capacity | 3 / 4 | Eight arrays exceed the assumed usable DSP budget. |
| L1 service | 4 / 4 | Bank count varies modeled service and the cost proxy. |
| Matrix + L1 | 12 / 16 | Joint candidates expose interactions between those axes. |

The frontiers minimize a unitless cost proxy and decode latency; prefill is
reported separately. Model feasibility does not establish physical fit, and
sampled frontiers are not globally optimal hardware designs. No study candidate
has been adopted as a hardware baseline.

## Evidence and limitations

The [profiling findings](experiments/llama-cpp/2026-09-22/REPORT.md),
[operation/dependency study](research/compute-mapping/2026-10-02-op-dependency-map/README.md)
and [recurrence sensitivity](research/compute-mapping/2026-10-03-recurrence-multiplier-sensitivity/REPORT.md)
retain their original inputs, methods and results. CPU measurements, historical
JS scheduling and Python estimates have different assumptions and should not be
treated as interchangeable evidence.

Known correctness gaps remain in numeric-domain validation, composite capability
checks and resource reservations, infeasible-design statistics, heterogeneous
CSV exports and empty parity comparisons. The
[supporting characterization](docs/verification/refactor/characterization.md#known-gaps-separate-from-source-reorganization)
records the concrete cases.

MAC-summary parity, measured host fallback and held-out CPU fit require local
evidence that is not included in the checkout. Their skips do not resolve earlier
host-dependent fit failures; missing CPU validation is NOT CHECKED.
The [model validation record](engine/docs/VALIDATION.md) and
[assumptions](engine/docs/ASSUMPTIONS.md) distinguish accounting checks from
uncalibrated timing, cost and lowering assumptions. Physical performance,
FPGA/ASIC fit and numerical quality still need independent validation.

Detailed [refactor verification records](docs/verification/refactor/README.md)
are retained as supporting evidence.
