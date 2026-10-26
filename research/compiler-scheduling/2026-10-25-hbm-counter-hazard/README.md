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

## Results so far

All numbers are model outputs under the assumptions below. Latencies come from
public U280 measurements, not yet from the F2 AMD model.

1. **The counter itself costs nothing here.** Across all 480 sweep points,
   `spec` and `ordered_scoreboard` finish at the same cycle. With out-of-order
   drain the counter costs at most 1.3 %. The reason is structural: the Memory
   engine issues in order, so with in-order retirement milestones already
   happen in command order and in-order publication delays nothing. A single
   in-order consumer also cannot gain from early visibility of a later
   transfer, because it has to finish the earlier one first. The sanity
   checks test both cases.
2. **What does cost is retiring the reorder buffer in order.** A slow burst
   holds its slot and every slot behind it. HBM read utilization, decode
   layer, batch 1, HBM-bound Matrix, 1 MiB tiles, prior latency:

   | Shared reorder buffer | 16 KiB | 32 KiB | 64 KiB | 128 KiB | 256 KiB |
   | --- | --- | --- | --- | --- | --- |
   | In-order drain (`spec`) | 0.17 | 0.31 | 0.55 | 0.89 | 0.96 |
   | Drain on arrival (`ideal`) | 0.21 | 0.41 | 0.78 | 0.96 | 0.96 |

   With a fixed latency the two rows are identical, which is the sanity check
   for this effect. The gap only comes from latency variation.
3. **Bytes in flight set the ceiling.** Keeping 1024 B/cycle flowing through
   roughly 80 cycles of latency needs about 80 KiB in flight. Double-buffered
   16 KiB tiles cap utilization near 0.36 even at fixed latency, whatever the
   reorder buffer size.
4. **Tails get amplified.** A transfer completes when its slowest burst does,
   and a 64 KiB tile has 128 bursts. With the stall tail stretched 16x
   (`prior_x16`, a stress case), utilization drops to 0.25 or less for
   in-order drain even at 256 KiB. The real tail is the main unknown.

Draft compiler rules, to firm up once the latency is calibrated:

* Keep at least bandwidth x tail latency in flight: about 80 KiB at the prior
  median, more for the p99. Prefetch distance (SRAM buffers x tile size) and
  reorder buffer size both have to cover it.
* Prefer fewer, larger transfers over many small ones when the consumer waits
  on the completion milestone.
* A transfer that waits on another queue blocks everything behind it in the
  Memory queue. Put stores whose producer finishes late after the next
  layer's prefetches. Batch 32 runs hit this: utilization stops near 0.30
  because the state store waits on the Vector command.

## Model

* Times in 450 MHz AXI cycles. Bursts are 16 beats x 32 B, striped
  round-robin over 32 ports. Peak is 1024 B/cycle (460.8 GB/s).
* One reorder buffer shared by all 32 ports (Zeb, 2026-10-25). Read bursts
  hold a 512 B slot until retirement. Stores do not use slots and share each
  port's data bus with reads.
* Latency `prior`: 55 cycles on a page hit (p = 0.75, the IP's
  `PAGEHIT_PERCENT`), 69 on a miss, plus a stall of up to 72 cycles (tRFCsb
  160 ns) with p = 0.04. Hit/miss values are the switch-on numbers from
  [Shuhai (Wang et al., FCCM 2020)](https://arxiv.org/abs/2005.04324),
  Table VI. A stall delays only its own burst.
* Workload `decode_mix`: per DeltaNet layer, load state (Vector), load the
  layer's INT4 weights as tiles (Matrix, two SRAM buffers), then store state
  after the Vector command. Sizes come from the
  [memory-traffic study](../../compute-mapping/2026-10-14-memory-traffic/README.md),
  scaled by 0.25 for sweep speed with the weight-to-state ratio kept.
  Matrix consumes at 0.25x or 4x the HBM peak; Vector at 0.25x.
* Not modeled: controller reordering for page hits, bank-level refresh
  scheduling, read/write turnaround, AXI ID limits, the AWS shell, and any
  dependency between Matrix and Vector inside a layer.

## Next: calibrate against the AMD model

The F2 HBM controller and DRAM model ship as encrypted simulation sources, so
the timing has to be measured rather than read. The plan is a latency probe
testbench next to `rtl/tb/memory/f2_hbm/` that records AR-to-first-beat and
AR-to-RLAST per burst under controlled patterns (page hit vs miss, local vs
cross-switch port, 1 to N outstanding, mixed read/write) and dumps a CSV. The
fitted distribution replaces `prior_latency`, and the sweep reruns unchanged.

## Reproduce

```sh
python research/compiler-scheduling/2026-10-25-hbm-counter-hazard/sim.py
python -m unittest tests.test_hbm_counter_study -v
```

Standard library only. All checks run before anything is written; output goes
to the ignored `build/hbm-counter/2026-10-25/` (`results.json`, `sweep.csv`),
and an existing `results.json` is replaced only with `--force`. `--quick`
runs a small sweep. The full sweep takes about a minute on two cores.
