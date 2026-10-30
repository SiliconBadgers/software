# HBM latency and the command-format progress counter (2026-10-25)

Question from Zeb: the [command format](https://github.com/SiliconBadgers/rtl/blob/codex/command-format-spec/docs/command-format.md)
gives each queue one wrapping counter and publishes milestones in command
order. Can one slow HBM transfer hold the Memory counter back after later
prefetches have already landed, and how much does that cost compared with a
per-command scoreboard?

`sim.py` is a small event-ordered model of the Memory queue feeding Matrix and
Vector over the F2 HBM IP's 32 AXI3 ports. It compares four variants on the
same latency samples:

| Variant | Reorder buffer drain | Progress visibility |
| --- | --- | --- |
| `spec` | In issue order | One counter per queue (the spec) |
| `ordered_scoreboard` | In issue order | Per-command bits |
| `arrival_counter` | On arrival | One counter per queue |
| `ideal` | On arrival | Per-command bits |

Zeb confirmed on 2026-10-26 that the reorder buffer is shared by all 32 ports
and writes to SRAM in order, so `spec` is the variant that matches the design.
No buffer size is fixed yet; smaller is preferred, so the target result is the
smallest buffer that holds a chosen utilization.

## Results

Model outputs, now calibrated with latency measured on AMD's F2 HBM simulation
model (next section). They are not hardware measurements.

1. **The counter itself costs nothing here.** Across all 640 sweep points,
   including the measured latency, `spec` and `ordered_scoreboard` finish at
   the same cycle. With out-of-order drain the counter costs at most 1.3 %. The
   reason is structural: the Memory engine issues in order, so with in-order
   retirement milestones already happen in command order and in-order
   publication delays nothing. A single in-order consumer also cannot gain from
   early visibility of a later transfer, because it has to finish the earlier
   one first. The sanity checks test both cases.
2. **What does cost is retiring the reorder buffer in order.** A slow burst
   holds its slot and every slot behind it. HBM read utilization, decode
   layer, batch 1, HBM-bound Matrix, 1 MiB tiles, measured latency:

   | Shared reorder buffer | 16 KiB | 32 KiB | 64 KiB | 128 KiB | 256 KiB |
   | --- | --- | --- | --- | --- | --- |
   | In-order drain (`spec`) | 0.18 | 0.31 | 0.53 | 0.88 | 0.96 |
   | Drain on arrival (`ideal`) | 0.22 | 0.43 | 0.83 | 0.96 | 0.96 |

   With a fixed latency the two rows are identical, which is the sanity check
   for this effect. The gap only comes from latency variation.
3. **Smallest buffer.** Smallest shared reorder buffer with in-order drain
   that reaches 90 % / 95 % of the unlimited-buffer ideal, measured latency and
   port efficiency, 512 B slots:

   | Matrix consumption | 64 KiB tiles | 256 KiB tiles | 1 MiB tiles |
   | --- | --- | --- | --- |
   | 4x HBM peak (HBM-bound) | 80 / 96 KiB | 128 / 128 KiB | 128 / 128 KiB |
   | 0.25x HBM peak (array-bound) | 32 / 48 KiB | 24 / 32 KiB | 24 / 32 KiB |

   Decode is array-bound in the [memory-traffic study](../../compute-mapping/2026-10-14-memory-traffic/README.md),
   so 32 to 48 KiB covers it. 128 KiB is needed only if HBM becomes the
   bottleneck. The 64 KiB-tile row has a lower ideal because two 64 KiB tile
   buffers already limit bytes in flight.
4. **Bytes in flight set the ceiling.** One port needs about 4 to 8 bursts
   outstanding (2 to 4 KiB) to reach its measured peak, so all 32 ports need
   roughly 64 to 128 KiB in flight. Double-buffered 16 KiB tiles cap
   utilization near 0.27 whatever the reorder buffer size.
5. **Tails get amplified.** A transfer completes when its slowest burst does.
   The measured tail is short (125 of 128 one-at-a-time sequential reads
   finished within 68 cycles), but the `prior_x16` stress case shows
   utilization falling to 0.25 or less if the tail were 16 times longer.

Draft compiler rules:

* Keep 64 to 128 KiB of HBM reads in flight. Prefetch distance (SRAM buffers x
  tile size) and reorder buffer size both have to cover it.
* Prefer fewer, larger transfers over many small ones when the consumer waits
  on the completion milestone.
* A transfer that waits on another queue blocks everything behind it in the
  Memory queue. Put stores whose producer finishes late after the next
  layer's prefetches. Batch 32 runs hit this: utilization stops near 0.30
  because the state store waits on the Vector command.

## Calibration from the AMD HBM model (2026-10-28)

Measured with the latency probe in rtl (`tb/memory/f2_hbm/f2_hbm_latency_tb.sv`,
branch `yoon/hbm-latency-probe` at `b642ad2`), AWS F2 two-stack AMD HBM model
from Vivado 2024.2, VCS Y-2026.03_Full64 on CAE, seed 1. The positive case
logged 19,944 bursts; both expected-failure cases failed with their specific
diagnostics. All values are AXI cycles at 450 MHz, from address acceptance to
the first data beat.

| Measurement | Result | Public U280 reference ([Shuhai](https://arxiv.org/abs/2005.04324)) |
| --- | --- | --- |
| Idle read, own pseudo-channel | 54 (p50), 58 (max) | 55 page hit, switch on |
| Idle read through the switch | 66 to 72 (p50) | up to 77 |
| Row conflict | 68 versus 54; stride bits 15 and up | 69 page miss |
| One port, sequential, depth 1 | 54 (71 %), 61 (23 %), tail to 138 | |
| One port, peak read bandwidth | 28 to 31 B/cycle (12.5 to 13.8 GB/s) | 13.27 GB/s |
| 32 ports streaming | 907.9 B/cycle, 408.6 GB/s, 0.887 of peak | 425 GB/s |
| Outstanding reads accepted per port | about 44 | |
| Read completion order within a port | in order in every phase | |

`sim.py` uses the depth-1 histogram as its latency distribution, the 32-port
result as a per-port efficiency, and 512 B SRAM rows (Abhi's B update,
2026-10-28). As a check, it replays the probe's one-port depth phase:

| Outstanding bursts | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Measured B/cycle | 6.89 | 13.58 | 24.77 | 30.61 | 28.42 | 27.90 | 27.85 |
| Model B/cycle | 6.86 | 13.61 | 25.86 | 27.72 | 27.72 | 27.72 | 27.72 |

Every depth is within 10 %; the run fails if any is outside 15 %. The model
does not reproduce the small peak at depth 8.

## Model

* Times in 450 MHz AXI cycles. Bursts are 16 beats x 32 B, striped
  round-robin over 32 ports. Peak is 1024 B/cycle (460.8 GB/s); the calibrated
  runs scale each port's data bus by the measured 0.887.
* One reorder buffer shared by all 32 ports (Zeb, 2026-10-25). Read bursts
  hold a 512 B slot until retirement. Stores do not use slots and share each
  port's data bus with reads.
* Latency `measured`: the depth-1 histogram above. Values above the 68-cycle
  row conflict count as a stall that delays only its own burst.
* Latency `prior` (kept for comparison): 55 cycles on a page hit
  (p = 0.75), 69 on a miss, plus a stall of up to 72 cycles with p = 0.04,
  from Shuhai Table VI. `prior_x16` stretches the stall 16 times.
* Workload `decode_mix`: per DeltaNet layer, load state (Vector), load the
  layer's INT4 weights as tiles (Matrix, two SRAM buffers), then store state
  after the Vector command. Sizes come from the memory-traffic study, scaled
  by 0.25 for sweep speed with the weight-to-state ratio kept. Matrix consumes
  at 0.25x or 4x the HBM peak; Vector at 0.25x.
* Not modeled: controller reordering for page hits, bank-level refresh
  scheduling, read/write turnaround (the probe's mixed phase shows it raises
  read latency), the per-port limit of about 44 outstanding reads (larger than
  any buffer swept here per port), the AWS shell, and any dependency between
  Matrix and Vector inside a layer.

## Reproduce

```sh
python research/compiler-scheduling/2026-10-25-hbm-counter-hazard/sim.py
python -m unittest tests.test_hbm_counter_study -v
```

Standard library only. All checks, including the probe replay, run before
anything is written; output goes to the ignored `build/hbm-counter/2026-10-25/`
(`results.json` with the sweep, probe replay and smallest-buffer table, and
`sweep.csv`). An existing `results.json` is replaced only with `--force`.
`--quick` runs a small sweep. The full run takes about two minutes on two
cores. The measurement itself is reproduced from rtl with
`make test-f2-hbm-latency` (see `tb/memory/f2_hbm/README.md` on that branch).
