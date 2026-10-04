# Qwen3.5-2B F32 recurrence multiplier sensitivity

October 3, 2026. This study audits the gated-delta recurrence arithmetic class
before testing Zeb's proposed 4x and 16x shared-matrix penalties. It extends the
[dependency/resource-aware scheduler](../2026-09-30-dependency-resource-sensitivity/REPORT.md)
and the [bounded joint-resource study](../2026-10-01-joint-resource-tradeoffs/REPORT.md).

> [!IMPORTANT]
> Neither 4x nor 16x is established hardware performance. They are sampled
> costs, expressed as W8 x A16-equivalent PE cycles per 32-bit recurrence MAC.
> The results characterize the analytical model and identify what must be
> calibrated; they do not select an FPGA or ASIC architecture.

## Result

Zeb's correction is material, but the claim that shared GeMM remains ahead at
4x is conditional rather than general.

- All 72 gated-delta instances in the four authoritative captures belong to
  one arithmetic/dtype class: F32 Q, K, V, gate, beta, state and output, with
  `S=128`, `H=16`, and 18 recurrence layers per capture.
- Each head/token has two state-vector products and one outer-product update,
  or `3*S*S` 32-bit MACs, plus state decay, scalar/vector work and one
  exponential. Surrounding W8/A16 or W4/A8 weight projection choices do not
  change this captured F32 recurrence class.
- At a 4x matrix penalty, shared matrix recurrence beats generic vector
  recurrence in 92/120 prefill and 96/120 decode scenario cells. It wins every
  cell once at least four matrix arrays are available.
- At 16x, those counts fall to 52/120 prefill and 61/120 decode. It wins every
  cell only at eight or more matrix arrays.
- Against four dedicated recurrence units, 4x shared matrix wins only 24/120
  prefill and 12/120 decode cells; 16x wins 0/120 prefill and 1/120 decode.
  Eight or sixteen unconstrained dedicated units are always faster in this
  grid, but consume additional modeled DSP/cost and are not equal-resource
  comparisons.

The defensible conclusion is therefore: a 4x shared-matrix implementation can
remove the need for dedicated recurrence hardware in some sufficiently
matrix-rich configurations, while a 16x implementation makes dedicated units
substantially more competitive. Neither mapping dominates across workload,
precision, bandwidth and resource classes.

## Source and class audit

The source is the pinned llama.cpp commit
`308883b335798865e73f8fee0d9a5fbfa13c8480`. Its F32 CPU implementation performs,
for every head and token:

1. state decay by `exp(g)`;
2. `state * k` to form the prediction;
3. an outer-product state update from `k` and delta;
4. `state * q` to form the output.

The automated audit reads every `GATED_DELTA_NET` instance from:

| Capture | Phase | Instances | Dtype class |
|---|---|---:|---|
| pp128 | prefill | 18 | all inputs/output F32 |
| pp128 | decode | 18 | all inputs/output F32 |
| pp512 | prefill | 18 | all inputs/output F32 |
| pp512 | decode | 18 | all inputs/output F32 |

All 72 also have a `128 x 128 x 16` recurrent state. This proves one captured
arithmetic class, not a universal fact about every model, kernel lowering or
future quantized recurrence implementation.

## Sweep coverage

The sweep contains 1,440 feasible configurations:

- workloads: captured pp128, captured pp512, target batch 1 and target batch 8;
- surrounding precisions: W8/A16 and W4/A8;
- memory profiles: 32 banks/100 GB/s, 128 banks/300 GB/s, and 256 banks/460 GB/s;
- matrix counts: 1, 2, 4, 8 and 16;
- shared-matrix penalties: 1x, 2x, 4x, 8x, 16x and 32x;
- alternatives: generic vector recurrence and 1, 2, 4, 8 or 16 dedicated
  recurrence units.

The dependency-aware scheduler is used throughout. It preserves captured
dependencies, in-place alias hazards and per-resource contention. For the
shared-matrix lowering, it conservatively reserves the matrix pool for the
complete fused gated-delta operation and also reserves vector or RISC-V for the
non-matrix tail. This deliberately double-reserves that interval rather than
claiming unproven matrix/vector pipelining.

### Shared matrix versus vector by matrix capacity

Each cell below is the number of wins across 24 combinations of workload,
surrounding precision and memory profile.

| Penalty / phase | M=1 | M=2 | M=4 | M=8 | M=16 |
|---|---:|---:|---:|---:|---:|
| 4x prefill | 4/24 | 16/24 | 24/24 | 24/24 | 24/24 |
| 4x decode | 8/24 | 16/24 | 24/24 | 24/24 | 24/24 |
| 16x prefill | 0/24 | 0/24 | 4/24 | 24/24 | 24/24 |
| 16x decode | 0/24 | 3/24 | 10/24 | 24/24 | 24/24 |

This is the central class-level qualification. “GeMM is ahead at 4x” is true
throughout the sampled classes at `M>=4` only when the alternative is the
generic vector path. It is not true against every dedicated-unit count.

## Reference slice

The table below uses target batch 1, prompt 512, context 2,048, W8/A16,
256 modeled L1 banks and 460 GB/s peak HBM. Matrix mappings add no dedicated
recurrence DSPs; dedicated mappings do.

| Matrix units | Recurrence mapping | DSP | Cost proxy | Prefill ms | Decode ms |
|---:|---|---:|---:|---:|---:|
| 4 | vector | 1,088 | 3,912 | 3,728.72 | 10.123 |
| 4 | matrix, 4x | 1,088 | 3,912 | 3,484.91 | 9.643 |
| 4 | matrix, 16x | 1,088 | 3,912 | 3,862.39 | 10.380 |
| 4 | dedicated, 4 units | 1,600 | 9,064 | 3,247.52 | 9.280 |
| 8 | vector | 2,112 | 4,936 | 2,241.09 | 7.975 |
| 8 | matrix, 4x | 2,112 | 4,936 | 1,849.24 | 7.203 |
| 8 | matrix, 16x | 2,112 | 4,936 | 2,037.99 | 7.572 |
| 8 | dedicated, 4 units | 2,624 | 10,088 | 1,759.89 | 7.285 |
| 16 | vector | 4,160 | 6,984 | 1,497.56 | 7.907 |
| 16 | matrix, 4x | 4,160 | 6,984 | 982.03 | 7.074 |
| 16 | matrix, 16x | 4,160 | 6,984 | 1,076.40 | 7.258 |
| 16 | dedicated, 4 units | 4,672 | 12,136 | 1,016.37 | 7.217 |

This slice explains Zeb's observation: at 16 matrix units, 4x shared matrix is
slightly faster than four dedicated units in both phases without their added
modeled resources. At 4 matrix units the same claim reverses. At 16x, four
dedicated units are faster throughout this reference slice.

## What changed in the model

The explorer now exposes an opt-in shared-matrix recurrence path and an editable
32-bit penalty. Dedicated recurrence retains first priority; disabling the new
path preserves prior configurations. The matrix path accounts for all three
`S*S` MAC classes and separately charges the remaining decay/vector/special
tail. Regression tests cover MAC completeness, penalty monotonicity, dedicated
priority, tail accounting and input validation.

Earlier reports remain reproducible because the new path defaults off. Their
matrix/L1/HBM studies are still valid under their fixed four-dedicated-unit
assumption, but their recurrence-unit conclusions must not be generalized to a
shared-matrix F32 lowering. This report supersedes that recurrence-specific
interpretation.

## Remaining blockers

The analysis cannot determine whether 4x, 16x or another value is physically
correct. The next evidence should be a small recurrence-kernel calibration,
not a larger unconstrained analytical sweep:

1. specify the intended FPGA and ASIC 32-bit arithmetic semantics (integer,
   fixed point or floating point, including accumulation width);
2. measure or synthesize the two GEMV-like products and outer-product update on
   the candidate matrix datapath;
3. record achieved cycles/MAC, utilization, fmax, DSP/area and L1 traffic;
4. feed those calibrated points back into the existing penalty sweep;
5. then run bounded joint studies under explicit DSP/BRAM/area/power budgets.

The generated frontier minimizes scheduled prefill latency, scheduled decode
latency, modeled DSP and the existing cost proxy. It is mathematically defined
for comparison, but the cost proxy is uncalibrated and therefore is not a
physical architecture frontier.

## Reproduction

From the repository root:

```powershell
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\test_model.cjs
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\test_scheduler.cjs
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\test_recurrence_penalty.cjs
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\sweep_recurrence_penalty.cjs `
  --spec research\compute-mapping\2026-10-03-recurrence-multiplier-sensitivity\sweep-spec.json `
  --json research\compute-mapping\2026-10-03-recurrence-multiplier-sensitivity\sweeps.json `
  --csv research\compute-mapping\2026-10-03-recurrence-multiplier-sensitivity\sweeps.csv
.\.venv\Scripts\python.exe `
  experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\plot_recurrence_penalty.py `
  --input research\compute-mapping\2026-10-03-recurrence-multiplier-sensitivity\sweeps.json `
  --output research\compute-mapping\2026-10-03-recurrence-multiplier-sensitivity\recurrence-sensitivity.png
```

[`sweeps.json`](sweeps.json) contains the source hashes, capture-class audit,
all configurations, fixed-count comparisons and frontiers.
[`sweeps.csv`](sweeps.csv) is the flat review table, and
[`recurrence-sensitivity.png`](recurrence-sensitivity.png) visualizes the key
penalty/capacity interaction.
