# Qwen3.5-2B bounded joint-resource tradeoffs

October 1, 2026. This study follows the
[serial sensitivity](../2026-09-28-resource-sensitivity/REPORT.md) and
[dependency/resource-aware sensitivity](../2026-09-30-dependency-resource-sensitivity/REPORT.md)
work by varying the three core resources together instead of combining isolated
one-dimensional knees.

> [!IMPORTANT]
> The latency champion below is optimal only within this finite analytical grid
> for the two modeled latency objectives. It is not a physical or cost optimum.
> In particular, modeled L1 banks have no routing, arbitration, timing, area or
> power cost, and the operation service equations remain uncalibrated.

## Result

All 125 sampled configurations are feasible under the explorer's DSP check and
the study's 460 GB/s HBM ceiling. The single configuration minimizing both
scheduled prefill and scheduled decode latency in this grid is:

| Setting | Value |
|---|---:|
| Matrix units | 16 |
| Vector units (fixed) | 4 |
| Dedicated recurrence units (fixed) | 4 |
| DMA engines (fixed) | 1 |
| Modeled shared-L1 banks | 256 |
| Shared-L1 capacity (fixed) | 8 MiB |
| Peak HBM bandwidth | 460 GB/s |
| Modeled DSP demand | 4,672 of 7,219 usable |
| Scheduled prefill | 1,016.37 ms; 503.76 tokens/s |
| Scheduled decode | 7.22 ms/token; 138.55 tokens/s |

This is the **sampled latency champion**, not a recommended implementation.
Once prefill latency, decode latency, DSP demand, L1 bank count and HBM bandwidth
are kept as five separate objectives, 60 of the 125 configurations are
non-dominated. Selecting one requires physical bank feasibility and explicit
product targets or calibrated area/power/cost weights.

## Frozen scope and objective

The workload remains batch 1, prompt 512, decode context 2,048 and W8/A16.
Vector count 4, dedicated recurrence count 4, one DMA engine and 8 MiB L1 are
held fixed at the dependency-aware phase-envelope settings. The grid is:

- matrix units: 1, 2, 4, 8, 16;
- modeled L1 banks: 16, 32, 64, 128, 256;
- peak HBM: 100, 200, 300, 400, 460 GB/s.

The formal reported frontier minimizes five explicit quantities: scheduled
prefill latency, scheduled decode latency, modeled DSP demand, modeled L1 bank
count and peak HBM bandwidth. That makes the Pareto calculation mathematically
defined, but it does not turn the quantities into physical area, energy or
price.

## Interaction results

The joint sweep confirms that isolated knees cannot simply be combined. At 16
matrix units and 256 modeled L1 banks, raising HBM from 100 to 460 GB/s changes
decode from 29.09 ms to 7.22 ms. The earlier 100 GB/s one-dimensional HBM knee
only applied while the default shared-L1 service remained dominant.

### L1-bank ceiling at 16 matrix units and 460 GB/s

| Modeled banks | Prefill ms | Decode ms | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|
| 16 | 1,168.70 | 84.34 | 438.09 | 11.86 |
| 32 | 1,028.84 | 42.88 | 497.65 | 23.32 |
| 64 | 1,021.19 | 21.32 | 501.38 | 46.90 |
| 128 | 1,017.45 | 11.04 | 503.22 | 90.58 |
| 256 | 1,016.37 | 7.22 | 503.76 | 138.55 |

Prefill is nearly insensitive beyond 32 modeled banks, while decode continues
to consume every sampled increase. This divergence is the main architecture
tradeoff, but the model cannot determine whether 128 or 256 physical banks are
implementable.

### Matrix count at 256 modeled banks and 460 GB/s

| Matrix units | Modeled DSP | Prefill ms | Decode ms |
|---:|---:|---:|---:|
| 1 | 832 | 12,659.35 | 35.18 |
| 2 | 1,088 | 6,359.70 | 17.88 |
| 4 | 1,600 | 3,247.52 | 9.28 |
| 8 | 2,624 | 1,759.89 | 7.29 |
| 16 | 4,672 | 1,016.37 | 7.22 |

Decode is effectively at its matrix knee by eight units in this high-bandwidth
slice, while prefill still improves materially from eight to sixteen. Decode
alone therefore does not justify sixteen matrix units; prefill is what drives
that choice.

### HBM at 16 matrix units and 256 modeled banks

| HBM GB/s | Prefill ms | Decode ms |
|---:|---:|---:|
| 100 | 1,022.23 | 29.09 |
| 200 | 1,018.48 | 14.86 |
| 300 | 1,017.23 | 10.26 |
| 400 | 1,016.61 | 7.96 |
| 460 | 1,016.37 | 7.22 |

At this L1 setting, decode remains HBM-sensitive throughout the available F2
range. Prefill is matrix-dominated and gains little beyond 100 GB/s.

## Decision tiers without invented cost weights

For orientation only, the table below asks for both phase throughputs to reach
a fraction of the sampled latency champion and then removes configurations
dominated in DSP, bank count and HBM bandwidth. Each shown tier has one
resource-minimal sampled survivor.

| Required fraction of champion throughput in both phases | Matrix | Banks | HBM GB/s | DSP | Prefill tokens/s | Decode tokens/s |
|---:|---:|---:|---:|---:|---:|---:|
| 50% | 8 | 128 | 300 | 2,624 | 290.7 | 90.6 |
| 75% | 16 | 256 | 400 | 4,672 | 503.6 | 125.7 |
| 90% | 16 | 256 | 400 | 4,672 | 503.6 | 125.7 |
| 95% | 16 | 256 | 460 | 4,672 | 503.8 | 138.6 |

These thresholds are sensitivity landmarks, not product requirements. The 50%
tier is the more defensible provisional architecture-analysis point because it
uses fewer matrix units, banks and HBM while retaining substantial modeled
throughput. It is still conditional on the same uncalibrated model.

## Decision

There is no support for calling any point a physical optimum. The evidence does
support the following bounded conclusions:

1. `16 / 256 / 460` is the latency champion inside this exact grid.
2. `8 / 128 / 300` is the unique sampled resource-minimal point meeting at
   least half of the champion throughput in both phases.
3. Prefill drives matrix provisioning; decode drives modeled L1 bandwidth and,
   once L1 is increased, HBM bandwidth.
4. A final selection is blocked on realistic L1-bank limits/costs and explicit
   prefill/decode targets—not on another unconstrained analytical sweep.

## Reproduction

From the repository root:

```powershell
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\test_model.cjs
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\test_scheduler.cjs
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\test_joint_resources.cjs
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\sweep_joint_resources.cjs `
  --spec research\compute-mapping\2026-10-01-joint-resource-tradeoffs\sweep-spec.json `
  --json research\compute-mapping\2026-10-01-joint-resource-tradeoffs\sweeps.json `
  --csv research\compute-mapping\2026-10-01-joint-resource-tradeoffs\sweeps.csv
.\.venv\Scripts\python.exe `
  experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\plot_joint_resources.py `
  --input research\compute-mapping\2026-10-01-joint-resource-tradeoffs\sweeps.json `
  --output research\compute-mapping\2026-10-01-joint-resource-tradeoffs\joint-tradeoffs.png
```

[`sweeps.json`](sweeps.json) contains every configuration, the explicit
objective definition and SHA-256 provenance. [`sweeps.csv`](sweeps.csv) is the
flat review table, and [`joint-tradeoffs.png`](joint-tradeoffs.png) shows the
core interactions.
