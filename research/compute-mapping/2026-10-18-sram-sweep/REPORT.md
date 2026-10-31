# Qwen3.5-2B SRAM (L1) sweep

October 18, 2026. Varies the local-memory (SRAM, `l1.*`) choices of the Python
engine while holding compute (`matrix.*`, vector, recurrence) at the provisional
[`characterized-v1`](../../../engine/designs/baselines/characterized-v1/profile.json)
baseline: 4 matrix units of 32 × 32, 300 MHz, 460 GB/s HBM at 70%. The tiling
study is separate
([`2026-10-17-tiling-sweep`](../2026-10-17-tiling-sweep/REPORT.md)); a small
robustness check below re-ranks SRAM settings at 2 and 8 matrix units but does
not choose a compute configuration.

> [!IMPORTANT]
> These are uncalibrated analytical estimates on one captured workload
> (`pp512-fa-on`, batch 1, mixed Q4_K_M). Cost is a unitless proxy, and
> feasibility flags do not establish physical fit, banking/routing, area,
> power or timing closure. The L1 partition is the engine's static split
> (tiles, resident activations, recurrent state), not a proposed SRAM
> organization.

## Result

1. **L1 bandwidth, not capacity, sets decode latency, and the baseline L1 is
   slower than HBM.** The baseline's effective L1 bandwidth (64 banks × 16 B ×
   300 MHz × 0.65) is 199.7 GB/s, below HBM's 322 GB/s. With weights routed
   through L1, every decode token streams about 2.7 GB through it, so decode
   halves with each doubling of bandwidth (27.38 → 14.38 → 7.93 ms) until about
   800 GB/s, where HBM takes over at 5.32 ms. Prefill is L1-bound below the
   baseline bandwidth (816.7 ms at half) and matrix-bound above it (at most
   0.5% faster).
2. **Capacity buys traffic, not latency.** Going from 4 to 64 MiB cuts prefill
   HBM traffic from 11.4 GB to 1.4 GB (weights 3.82 → 1.27 GB, spilled
   activations 7.48 → 0.10 GB) and decode KV/state traffic from 35.0 to
   11.5 MB per token. Latency moves by at most 1%, because prefill is
   matrix-bound and HBM has headroom (prefill HBM service 35.4 → 4.3 ms
   against 663 ms total). 2 MiB is infeasible because no fused-attention tile
   fits.
3. **The static partition barely moves latency but can double activation
   traffic.** Across the tile/state/activation splits, prefill stays within
   0.2% and decode is unchanged. Leaving only 1 MiB for resident activations
   raises activation HBM traffic by 81% (2.62 → 4.74 GB). The recurrent state
   is 19.3 MiB per sequence, so 1–3 MiB reserves hold only 5–16% of it.
4. **The direct weight path is the largest single decode lever.** Bypassing L1
   for weights drops decode from 14.38 to 5.35 ms (HBM-bound) for a 10% higher
   cost proxy, whose price is itself a placeholder. Anchored group fusion
   cuts prefill activation HBM traffic by 31% (2.62 → 1.80 GB); disabling
   fusion raises it by 53%.

## Capacity (`sram-capacity-v1`, 6 points, 5 feasible)

Grid: `l1.mib` 2, 4, 8, 16, 32, 64, with the baseline split (50% tiles, 25%
state, 25% activations) and 64 banks × 16 B.

| `l1.mib` | Cost | Prefill ms | Decode ms | Tile budget / unit | State resident | Prefill weight HBM GB | Prefill activation HBM GB | Prefill L1 GB | Searched FFN tile | Attention query tile |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 2 (infeasible) | 9,464 | 658.5 | 14.32 | 256 KiB | 3% | 7.22 | 12.97 | 80.2 | 128×64, nr = 8 | none fits |
| 4 | 9,720 | 662.9 | 14.38 | 512 KiB | 5% | 3.82 | 7.48 | 71.5 | 256×128, nr = 4 | 32 tokens |
| 8 (baseline) | 10,232 | 663.7 | 14.38 | 1 MiB | 10% | 3.82 | 2.62 | 66.6 | 512×128, nr = 4 | 96 tokens |
| 16 | 11,256 | 664.8 | 14.38 | 2 MiB | 21% | 2.12 | 2.32 | 64.6 | 512×256, nr = 2 | 224 tokens |
| 32 | 13,304 | 665.6 | 14.38 | 4 MiB | 42% | 2.11 | 1.10 | 63.3 | 1024×256, nr = 2 | 480 tokens |
| 64 | 17,400 | 669.1 | 14.38 | 8 MiB | 83% | 1.27 | 0.10 | 61.5 | 1024×512, nr = 1 | 992 tokens |

`nr` is how many times each weight matrix is streamed from HBM during prefill.
The cost/prefill/decode frontier contains only 4 MiB, because latency is flat
and cost rises by 128 per MiB.

**Why prefill rises slightly with capacity.** The six fused-attention
operations account for the +5.4 ms between 8 and 64 MiB (6 × 0.92 = 5.5 ms;
matrix-pool busy time rises by the same 5.5 ms). A larger tile budget lets
the engine pick a larger query tile (96 → 992 tokens). One large tile cannot
skip the masked upper triangle at tile granularity, so the average number of
keys per query rises from 299 to 512 and each attention operation takes
0.92 ms longer. This is a property of the attention tile choice, not a cost of
SRAM, and it is probably the small non-monotonic prefill effect the tiling
study's robustness check noted.

## Bandwidth (`sram-bandwidth-v1`, 15 points, all feasible)

Grid: `l1.banks` 16, 32, 64, 128, 256 × `l1.bytes_per_bank_cycle` 8, 16, 32.
The engine uses only the product (bytes per cycle), so equal products give
identical latency and traffic.

| Bytes / cycle | Effective GB/s | Configurations (banks × bytes) | Prefill ms | Decode ms | Prefill bound | Decode bound |
|---:|---:|---|---:|---:|---|---|
| 128 | 25.0 | 16 × 8 | 2,706.0 | 109.51 | L1 | L1 |
| 256 | 49.9 | 16 × 16, 32 × 8 | 1,419.7 | 54.76 | L1 | L1 |
| 512 | 99.8 | 16 × 32, 32 × 16, 64 × 8 | 816.7 | 27.38 | L1 | L1 |
| 1,024 (baseline) | 199.7 | 32 × 32, **64 × 16**, 128 × 8 | 663.7 | 14.38 | matrix | L1 |
| 2,048 | 399.4 | 64 × 32, 128 × 16, 256 × 8 | 661.4 | 7.93 | matrix | L1 |
| 4,096 | 798.7 | 128 × 32, 256 × 16 | 660.5 | 5.32 | matrix | HBM |
| 8,192 | 1,597.4 | 256 × 32 | 660.5 | 5.32 | matrix | HBM |

Capacity and partition are fixed here, so traffic is identical at every point
(prefill L1 66.6 GB, decode L1 2.73 GB per token). The frontier contains only
32-byte banks because the cost proxy charges per bank (8 each) but not for
bank width. That is a gap in the cost model, not evidence that wide banks are
free.

## Partition (`sram-partition-v1`, 12 points, 11 feasible)

Grid: `l1.tile_fraction` 0.25, 0.375, 0.5, 0.625 × `l1.state_reserve` 0.125,
0.25, 0.375 of the 8 MiB L1; activations get the remainder. 0.625 + 0.375
leaves no activation space and is rejected by configuration validation.
Decode is 14.38 ms at every feasible point.

| Tile / state | Tile budget / unit | Activations | State (resident) | Prefill ms | Prefill activation HBM GB | Prefill L1 GB | Searched FFN tile | Attention query tile |
|---|---:|---:|---:|---:|---:|---:|---|---|
| 0.25 / 0.125 | 512 KiB | 5 MiB | 1 MiB (5%) | 662.7 | 2.82 | 66.8 | 256×128 | 32 tokens |
| 0.25 / 0.25 | 512 KiB | 4 MiB | 2 MiB (10%) | 662.9 | 2.92 | 66.9 | 256×128 | 32 tokens |
| 0.25 / 0.375 | 512 KiB | 3 MiB | 3 MiB (16%) | 662.9 | 3.14 | 67.1 | 256×128 | 32 tokens |
| 0.375 / 0.125 | 768 KiB | 4 MiB | 1 MiB (5%) | 663.1 | 2.92 | 66.9 | 256×128 | 64 tokens |
| 0.375 / 0.25 | 768 KiB | 3 MiB | 2 MiB (10%) | 663.1 | 3.14 | 67.1 | 256×128 | 64 tokens |
| 0.375 / 0.375 | 768 KiB | 2 MiB | 3 MiB (16%) | 663.1 | 3.22 | 67.2 | 256×128 | 64 tokens |
| 0.5 / 0.125 | 1 MiB | 3 MiB | 1 MiB (5%) | 663.7 | 2.54 | 66.5 | 512×128 | 96 tokens |
| 0.5 / 0.25 (baseline) | 1 MiB | 2 MiB | 2 MiB (10%) | 663.7 | 2.62 | 66.6 | 512×128 | 96 tokens |
| 0.5 / 0.375 | 1 MiB | 1 MiB | 3 MiB (16%) | 663.7 | 4.74 | 68.7 | 512×128 | 96 tokens |
| 0.625 / 0.125 | 1.25 MiB | 2 MiB | 1 MiB (5%) | 664.0 | 2.62 | 66.6 | 512×128 | 128 tokens |
| 0.625 / 0.25 | 1.25 MiB | 1 MiB | 2 MiB (10%) | 664.0 | 4.74 | 68.7 | 512×128 | 128 tokens |

A smaller tile share does not pay for itself with the extra activation space:
below a 1 MiB per-unit tile budget the search picks a 256 × 128 tile, which
re-reads activations more, so 0.25 and 0.375 move more activation traffic
(2.82–3.22 GB) than 0.5 / 0.125 (2.54 GB). Raising the state reserve from 1 to
3 MiB lowers decode KV/state HBM traffic from 35.0 to 31.9 MB per token, with
no latency effect while decode is L1-bound.

## Organization (`sram-organization-v1`, 6 points, all feasible)

Grid: `memory.weight_path` via_l1, direct × `memory.fusion` chains, groups,
none. `direct` is an unpriced-organization alternative in the decision
registry, not an approved design; its cost uses the placeholder
`direct_weight_buffer_per_pe` (0.25 per PE, +1,024 here).

| Weight path | Fusion | Cost | Prefill ms | Decode ms | Decode bound | Prefill activation HBM GB | Prefill L1 GB |
|---|---|---:|---:|---:|---|---:|---:|
| via_l1 | chains (baseline) | 10,232 | 663.70 | 14.38 | L1 | 2.62 | 66.6 |
| via_l1 | groups | 10,232 | 663.71 | 14.22 | L1 | 1.80 | 65.0 |
| via_l1 | none | 10,232 | 665.19 | 14.39 | L1 | 4.00 | 69.5 |
| direct | chains | 11,256 | 660.65 | 5.35 | HBM | 2.62 | 48.7 |
| direct | groups | 11,256 | 660.82 | 5.35 | HBM | 1.80 | 47.2 |
| direct | none | 11,256 | 662.14 | 5.35 | HBM | 4.00 | 51.7 |

With `direct`, decode L1 service falls from 13.69 to 0.99 ms and the HBM weight
stream (1.27 GB per token at 322 GB/s) becomes the limit.

## Robustness: unit count and workload

[`robustness.csv`](robustness.csv) re-evaluates `l1.mib` 4, 8, 16, 32 and
`l1.banks` 32, 64, 128, 256 at 2, 4 and 8 matrix units on `pp512-fa-on`,
`pp128-fa-on` and `pp512-fa-off`. The ranking does not change for feasible
designs: capacity stays within 1% in prefill and leaves decode unchanged, and
decode follows bank count the same way on every workload (about 27.3, 14.4,
7.9 and 5.3 ms for 32–256 banks).

Eight matrix units exceed the usable DSP budget and are infeasible, but they
show the coupling: prefill becomes L1-bound, so bank count then also drives
prefill (467 → 389 ms from 64 to 256 banks on `pp512-fa-on`), and capacity
helps a little (514 → 459 ms from 4 to 32 MiB). If compute grows past the
baseline, SRAM bandwidth becomes a prefill decision too.

## Hand-offs

- **Memory:** the deciding number is effective L1 bandwidth
  (`banks × bytes_per_bank_cycle × clock × efficiency`). The baseline's
  199.7 GB/s sits below HBM's 322 GB/s, which is why decode is L1-bound.
  Achievable aggregate SRAM bandwidth and arbitration efficiency on the target
  should be confirmed before relying on that conclusion. The cost proxy should
  price bank width as well as bank count.
- **Tiling:** the two studies are coupled through the per-unit tile budget
  (`l1.mib × tile_fraction / matrix.count`). The fused-attention query-tile
  choice ignores lost causal skipping, so larger budgets select slower
  attention tiles. A combined check of the best few array shapes against the
  best few SRAM settings should follow both studies.
- **Architecture:** `memory.weight_path = direct` is the largest decode lever
  in this model, but it is an organization decision the engine does not size
  or price beyond a placeholder.

## Reproduction

From the repository root:

```text
python -B engine/sweeps/run.py engine/sweeps/sram/capacity.json
python -B engine/sweeps/run.py engine/sweeps/sram/bandwidth.json
python -B engine/sweeps/run.py engine/sweeps/sram/partition.json
python -B engine/sweeps/run.py engine/sweeps/sram/organization.json
python -B research/compute-mapping/2026-10-18-sram-sweep/analyze.py
```

The first four commands write full rows to ignored `build/sweeps/`.
[`analyze.py`](analyze.py) reuses the same study runner, adds the L1
partition, traffic split, residency and selected tiles per point, and writes
[`points.csv`](points.csv), [`robustness.csv`](robustness.csv) and
[`summary.json`](summary.json) (record hashes, frontiers, spec hashes and
source revision).
