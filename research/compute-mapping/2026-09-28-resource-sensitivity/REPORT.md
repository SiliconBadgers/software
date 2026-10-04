# Qwen3.5-2B explorer resource sensitivity

September 28, 2026. This study independently characterizes the one-dimensional
resource response of Zeb Taylor's checked-in analytical explorer before adding a
dependency-aware scheduler.

> [!IMPORTANT]
> These are uncalibrated analytical-model results, not hardware measurements,
> an FPGA fit, or a physically optimal accelerator. The useful evidence is the
> model's sensitivity and bottleneck transitions under frozen assumptions.

## Result in one paragraph

The serial explorer does not have one resource knee. At its default batch-1,
512-token prefill and 2,048-token decode-context point, decode is predominantly
limited by the modeled shared-L1 service. One matrix unit is already within 2%
of every larger feasible matrix-count sample for decode, while prefill continues
to improve through 16 units and reaches the DSP feasibility ceiling before a
plateau is established. HBM decode latency stops improving at 100 GB/s because
shared L1 becomes dominant. Shared-L1 bank count is the opposite: prefill
plateaus at 32 banks, but modeled decode does not plateau until 256 banks. The
256-bank result is an extrapolated model knee, not a feasible banking proposal;
the explorer has no bank/interconnect cost or timing model.

## Independent audit of the existing tool

The audited implementation is
[`engine.js`](../../../experiments/llama-cpp/2026-09-24-qwen35-2b/explorer/engine.js)
with the captured graph in
[`graph-data.js`](../../../experiments/llama-cpp/2026-09-24-qwen35-2b/explorer/graph-data.js).
The graph records 1,816 tensors, 1,441 scheduled nodes and source-dependency
edges. Under the frozen workload, 845 prefill and 809 decode operations are
nonzero after views and zero-sized operations are removed.

The engine calculates compute, shared-L1 and HBM service for each operation,
uses their maximum when within-operation overlap is enabled, adds launch cost,
and then sums operations in captured scheduler order. It does **not** overlap
dependency-independent graph operations. Its service sums are demands, not
additive latency contributions.

The browser's generic design grid sweeps matrix count/shape, vector count/lanes,
recurrence count, clock, batch, precision and two capability variants. HBM
bandwidth, L1 banks/capacity and DMA controls are editable in the UI but are not
dimensions of that grid. The new command-line facility fills this reproducible
one-dimensional-sweep gap without changing the engine equations.

The model's external structural assumptions agree with the pinned sources:
Qwen3.5-2B has 24 text layers, hidden width 2,048 and intermediate width 6,144
in the [official pinned configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json).
AWS describes one F2 FPGA as having 9,024 DSP slices, 16 GiB HBM and up to 460
GiB/s HBM bandwidth in its [official F2 announcement](https://aws.amazon.com/blogs/aws/now-available-second-generation-fpga-powered-amazon-ec2-instances-f2/).
The explorer deliberately interprets its `hbmGBs` input as decimal GB/s, not
GiB/s. The compute-versus-bandwidth framing is consistent with the original
[Roofline model](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf),
but this explorer is a different, per-operation time model and is not an
empirical Roofline measurement.

## Frozen experiment

All dimensions except the named sweep parameter remain at `engine.js` defaults:

| Setting | Frozen value |
|---|---:|
| Batch | 1 independent sequence |
| Prefill prompt | 512 tokens |
| Decode past context | 2,048 tokens |
| Precision | W8/A16 with INT32 accumulation |
| Clock | 300 MHz |
| Matrix | 4 units, 16 x 16 PEs, 75% efficiency |
| Vector | 2 units x 32 lanes, 70% efficiency |
| Dedicated recurrence | 0 units; vector implementation |
| DMA | 1 engine x 64 bytes/cycle |
| Shared L1 | 8 MiB, 32 banks x 16 bytes/cycle, 65% efficiency |
| HBM | 16 GiB, 460 GB/s peak, 70% efficiency |
| Transfer overlap | Enabled within each operation |

[`sweep-spec.json`](sweep-spec.json) is the authoritative sweep definition.
Every sweep uses powers of two so adjacent pairs are exact resource doublings.
The reported knee is the first positive sample for which that doubling and all
later sampled doublings reduce modeled latency by less than 2%. A knee at the
lowest sampled value is reported as a bound, not an exact crossing. Throughput
is also exported, but with fixed work it is the inverse of latency and therefore
does not provide an independent knee.

## Sensitivity results

| Resource | Prefill knee | Decode knee | Modeled interpretation |
|---|---:|---:|---|
| Matrix units | Not established before 32 becomes DSP-infeasible | At or below 1 | Prefill remains matrix-sensitive; decode is shared-L1-bound. |
| Vector units | 4 | 2 | More vector capacity is hidden behind matrix/L1 service. |
| Dedicated recurrence units | 8 | At or below 1 | The 0-to-1 step changes implementation; later decode scaling is hidden by L1. |
| DMA engines | At or below 1 | 2 | More DMA has little prefill value and quickly becomes L1-limited in decode. |
| Shared-L1 banks | 32 | 256 | Decode moves from shared-L1- to matrix-dominant at 256 banks. This omits banking cost and physical feasibility. |
| Peak HBM bandwidth | At or below 25 GB/s | 100 GB/s | Decode changes from HBM- to shared-L1-dominant between 50 and 100 GB/s. |
| Shared-L1 capacity | 0.125 MiB | 0.125 MiB | A spill/tile-model knee, not a claim that 128 KiB is sufficient physical SRAM. |

Selected quantitative observations:

- Matrix units 1 to 2 to 4 to 8 to 16 reduce prefill latency by 44.6%, 40.3%,
  33.7% and 25.5% per doubling. The corresponding decode reductions are 1.1%,
  0%, 0% and 0%.
- Vector units plateau at four for prefill: 4 to 8 reduces prefill latency by
  1.33%; later doublings contribute 0.57% and 0.09%. Decode reaches the 2%
  criterion at two units.
- Adding the first dedicated recurrence unit reduces prefill latency by 13.5%
  and decode latency by 2.6%. This is a mapping change from vector recurrence,
  not merely an extra identical unit. Decode improves only another 1.13% from
  one to two units and then stops.
- HBM decode latency falls from 118.94 ms at 25 GB/s to 63.33 ms at 50 GB/s and
  46.78 ms at 100 GB/s; every higher sample is unchanged. The dominant label
  changes from HBM to shared L1 at 100 GB/s.
- Shared-L1 decode latency falls from 46.78 ms at 32 banks to 26.63 ms at 64,
  16.69 ms at 128 and 15.16 ms at 256; it is unchanged at 512 and 1,024. The
  dominant label changes to Matrix at 256 banks.
- Below 0.125 MiB, the L1 capacity proxy has a visible cliff: prefill latency is
  23.14 s at 0.015625 MiB, 12.34 s at 0.03125 MiB, 6.94 s at 0.0625 MiB and
  4.68 s at 0.125 MiB. Above that point, large reductions in the spill proxy do
  not improve latency because another modeled service dominates. This does not
  validate the resulting state-residency or physical storage design.

The raw machine-readable results are in [`sweeps.json`](sweeps.json) and
[`sweeps.csv`](sweeps.csv). [`figures/`](figures/) contains one plot per resource
and a combined PDF.

## What the knees do and do not support

They support statements about the checked-in model under one frozen workload:

- The default 460 GB/s HBM setting is already beyond the modeled bandwidth knee
  for both phases.
- Default matrix count four is well beyond the modeled decode knee but below a
  prefill knee; matrix provisioning is therefore a workload/latency tradeoff.
- Default shared-L1 bank count 32 is at the modeled prefill knee but remains the
  major decode constraint.
- Bottleneck transitions explain the plateaus more usefully than the knee value
  alone.

They do not support a physical choice of 256 SRAM banks, 128 KiB SRAM, a final
engine count, an end-to-end token rate, or an FPGA timing/resource claim. The
model lacks calibrated compute primitives, SRAM arbitration/routing cost,
measured sustained HBM efficiency, host/device overhead, energy, and numerical
quality validation.

## Dependency-aware next step

The graph already contains dependency edges; the next change should consume
them rather than recapture them. Keep the current operation-cost functions as a
first controlled comparison, but replace serial summation with a deterministic
ready-queue scheduler that enforces:

1. An operation starts only after all materialized source dependencies complete.
2. Matrix, vector, recurrence and DMA unit occupancy never exceeds the selected
   configuration.
3. Shared-L1 and HBM demand contend across concurrent operations instead of each
   operation receiving full bandwidth independently.
4. Recurrent token/state hazards remain serialized even when other graph work is
   ready.
5. Serial mode reproduces the current engine exactly.
6. Unlimited-resource timing does not beat the DAG critical path, and finite
   resource timing does not beat aggregate resource-demand lower bounds.

After those invariants pass, rerun this exact sweep specification and compare how
the knees move. That comparison can distinguish bandwidth saturation from lack
of ready DAG parallelism or serialized recurrence. Until then, the current
`lowerBound` is only an optimistic resource-demand bound: defaults are 4,664.30
ms serial versus 3,149.20 ms for prefill, and 46.78 ms versus 42.90 ms for
decode. It is not a dependency-aware schedule.

## Reproduction

From the repository root with Node.js and the repository `.venv`:

```powershell
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\test_model.cjs
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\test_sensitivity.cjs
node experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\sweep_sensitivity.cjs `
  --spec research\compute-mapping\2026-09-28-resource-sensitivity\sweep-spec.json `
  --json research\compute-mapping\2026-09-28-resource-sensitivity\sweeps.json `
  --csv research\compute-mapping\2026-09-28-resource-sensitivity\sweeps.csv
.\.venv\Scripts\python.exe `
  experiments\llama-cpp\2026-09-24-qwen35-2b\scripts\plot_sensitivity.py `
  --input research\compute-mapping\2026-09-28-resource-sensitivity\sweeps.json `
  --output-dir research\compute-mapping\2026-09-28-resource-sensitivity\figures
```

The JSON records SHA-256 hashes of the engine, graph and sweep specification so
the results cannot silently drift away from their inputs.
