# Qwen3.5-2B dependency/resource-aware sensitivity

September 30, 2026. This is the controlled follow-up to the
[serial one-dimensional sensitivity study](../2026-09-28-resource-sensitivity/REPORT.md).
It leaves Zeb Taylor's operation-cost equations unchanged and schedules the
same modeled operations over their captured dependencies and finite resources.

> [!IMPORTANT]
> These are uncalibrated analytical list-schedule results, not hardware
> measurements, a cycle-accurate simulator, or an optimal accelerator. The
> scheduler is useful for comparing sensitivities under one explicit contention
> abstraction. It does not establish physical L1 saturation, SRAM banking,
> timing closure, area, energy or numerical quality.

## Result

At the frozen default point, legal cross-operation overlap reduces modeled
prefill from 4,664.30 ms to 4,477.99 ms (1.042x) and decode from 46.78 ms to
44.53 ms (1.050x). Most serial-model knees remain unchanged. The changes are
small and phase-specific: the decode vector and DMA knees move to the lowest
positive sample, and the prefill dedicated-recurrence knee moves from eight to
four units. The prefill matrix knee remains unobserved within the feasible
region; decode remains consistent with shared-L1 service demand being the main
modeled constraint.

This does not make the default configuration optimal. It shows that the
captured DAG exposes only modest additional overlap under this scheduler and
that the original resource-sensitivity conclusions are mostly stable to this
first dependency/resource model.

## Frozen objective and workload

The experiment retains the previous study's batch-1, 512-token prefill,
2,048-token decode context, W8/A16 precision and all default hardware settings.
One resource changes at a time. The reported knee remains the first positive
sample after which every available exact doubling improves scheduled latency by
less than 2%. Fixed-work throughput is the inverse of scheduled latency and is
not an independent objective.

The objective here is therefore **phase-specific saturation characterization**,
not global architecture optimization. A physical optimization still needs an
explicit calibrated cost or separate area/energy objectives plus latency,
throughput, capacity, quality and implementation constraints.

## Scheduler semantics

[`scheduler.js`](../../../experiments/llama-cpp/2026-09-24-qwen35-2b/explorer/scheduler.js)
implements deterministic captured-order list scheduling:

1. An operation launches only after all active source producers complete.
2. Metadata-only and zero-sized nodes are bypassed while their nearest active
   producers remain dependencies.
3. Read-after-write, write-after-read and write-after-write hazards are added
   for in-place `CPY` and `SET_ROWS` aliases. This prevents, for example, an
   attention operation from reading a KV cache before its captured update.
4. Each modeled operation uses its selected compute family, shared-L1 service
   and HBM service concurrently. A resource serves only one operation at a time
   for its modeled service duration. Dispatch is serialized.
5. The engine's existing assumption that all configured units of one compute
   family cooperate on one operation is preserved; this scheduler does not
   partition a family among concurrent operations.
6. Serial mode reproduces the original engine exactly. Dependency schedules
   are deterministic, include every active operation, do not overcommit a
   resource, respect dependencies, and cannot beat their DAG critical-path or
   aggregate resource-demand bounds.

The exclusivity model is deliberately coarse. It does not model fair-share or
priority arbitration, ports/banks, backpressure, buffering, pipeline fill,
prefetch, double buffering, tile-level overlap, command queues or physical
interconnect. A different legal scheduling policy may produce different
latencies and knees.

## Default-point bounds

| Phase | Serial | Scheduled | Speedup | DAG critical path | Aggregate-resource bound | Dominant aggregate demand |
|---|---:|---:|---:|---:|---:|---|
| Prefill | 4,664.30 ms | 4,477.99 ms | 1.042x | 3,748.54 ms | 3,149.20 ms | Matrix, 3,149.20 ms |
| Decode | 46.78 ms | 44.53 ms | 1.050x | 35.37 ms | 42.90 ms | Shared L1, 42.90 ms |

The decode schedule lies close to the aggregate shared-L1 demand bound. The
precise supported statement is that shared-L1 service dominates this analytical
schedule; it is not evidence that a physical SRAM implementation with 32 banks
would sustain or saturate at the modeled rate.

## Knee comparison

| Resource | Serial prefill | Scheduled prefill | Serial decode | Scheduled decode |
|---|---:|---:|---:|---:|
| Matrix units | Not established before 32 is DSP-infeasible | Same | At or below 1 | Same |
| Vector units | 4 | 4 | 2 | At or below 1 |
| Dedicated recurrence units | 8 | 4 | At or below 1 | Same |
| DMA engines | At or below 1 | Same | 2 | At or below 1 |
| Shared-L1 banks | 32 | 32 | 256 | 256 |
| Peak HBM bandwidth | At or below 25 GB/s | Same | 100 GB/s | 100 GB/s |
| Shared-L1 capacity | 0.125 MiB | Same | 0.125 MiB | Same |

The moved knees are scheduler/model effects, not hardware prescriptions. At the
default point, decode alone still does not justify additional matrix compute or
HBM bandwidth in this model; prefill continues to benefit from matrix compute.
The 256-bank decode knee remains an extrapolated response from a model with no
bank/interconnect cost or physical feasibility calculation.

## What this adds—and what remains

The result distinguishes pure captured-order serialization from a legal
operation-level schedule and shows that the one-dimensional conclusions are not
being driven by a large amount of ignored DAG parallelism under this policy.
It is still insufficient for the architecture decision:

- the operation service equations and efficiencies remain uncalibrated;
- one-at-a-time sweeps omit interactions and fixed-budget substitutions;
- only one frozen workload point is swept;
- the resource families cannot be partitioned dynamically;
- memory contention is service serialization, not a physical arbitration model;
- cost, area, energy, quality and timing constraints are incomplete.

The next bounded experiment should be a small joint sweep such as matrix count
versus L1 bandwidth under an explicit DSP/SRAM or calibrated cost budget. That
can test whether fewer matrix units plus more L1 service beats the converse for
prefill/decode targets. Results should be reported as feasible tradeoffs until
the competing objectives or a calibrated constrained objective are defined.

## Reproduction

From the repository root with Node.js and the repository `.venv`:

```powershell
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\test_model.cjs
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\test_sensitivity.cjs
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\test_scheduler.cjs
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\sweep_dependency_sensitivity.cjs `
  --spec research\compute-mapping\2026-09-30-dependency-resource-sensitivity\sweep-spec.json `
  --json research\compute-mapping\2026-09-30-dependency-resource-sensitivity\sweeps.json `
  --csv research\compute-mapping\2026-09-30-dependency-resource-sensitivity\sweeps.csv
.\.venv\Scripts\python.exe `
  experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\plot_sensitivity.py `
  --input research\compute-mapping\2026-09-30-dependency-resource-sensitivity\sweeps.json `
  --output-dir research\compute-mapping\2026-09-30-dependency-resource-sensitivity\figures
```

[`sweeps.json`](sweeps.json) records SHA-256 hashes of the engine, captured
graph, scheduler and sweep specification. [`sweeps.csv`](sweeps.csv) contains
the serial/scheduled comparison and bounds. [`figures/`](figures/) contains one
plot per resource and a combined PDF.
