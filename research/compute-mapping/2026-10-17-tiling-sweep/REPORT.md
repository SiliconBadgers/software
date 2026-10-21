# Qwen3.5-2B matrix-array and tile sweep

October 17, 2026. Varies the compute-engine tiling choices of the Python engine
while holding SRAM (`l1.*`) at the provisional
[`characterized-v1`](../../../engine/designs/baselines/characterized-v1/profile.json)
baseline: 8 MiB L1, 64 banks × 16 B/cycle, 25% state reserve, 50% tile share.
The SRAM study is separate; a small robustness check below re-ranks shapes at
4 and 16 MiB but does not choose an SRAM configuration.

> [!IMPORTANT]
> These are uncalibrated analytical estimates on one captured workload
> (`pp512-fa-on`, batch 1, mixed Q4_K_M). Cost is a unitless proxy, and
> feasibility flags do not establish physical fit or timing closure. Equal-PE
> comparisons assume the engine's array equations (no wiring, accumulator or
> drain penalty for tall or wide arrays).

## Result

1. **Total PE count sets matmul compute time; prefill turns L1-bound only when
   the matmuls' shared-L1 service catches up with it.** At 4,096 PEs, matmul
   compute is about 510 ms for every shape. The three 16-row shapes need
   501–629 ms of matmul L1 service and are L1-bound and 11–22% slower
   (4×16×64: 735.7 ms; 8×16×32: 757.0 ms; 16×16×16: 807.3 ms, infeasible). All
   six feasible shapes with 32 or 64 rows need 161–387 ms, stay matrix-bound and
   are within 0.4% of each other (663.5–665.7 ms). This includes the wide
   2×32×64; aspect ratio alone does not separate the two groups.
2. **Rows per array, then unit count, set the L1 service.** The engine's
   output-stationary feed charges about `M·N·K·(w/cols + a/rows)` bytes, and
   16-bit activations outweigh 4–6-bit weights, so the row count dominates.
   At equal rows, more units add L1 service through smaller per-unit tiles and
   more activation re-reads (32 rows: 2×32×64 273 ms, 4×32×32 315 ms,
   8×32×16 387 ms). The threshold is relative: at 2,048 PEs compute doubles to
   about 1,018 ms and every 16-row shape stays matrix-bound. Decode is L1-bound
   in all but the smallest designs and falls from about 15.0 ms (16 rows) to
   14.38 ms (32) to 14.10 ms (64).
3. **Splitting the same PEs into more units splits the tile budget.** The tile
   share is divided per unit (`l1.mib × tile_fraction / count`), so more units
   choose smaller tiles and re-stream weights more (prefill weight HBM traffic
   2.11 GB at one unit, 3.82 GB at four, 7.22 GB at sixteen). Latency hides this
   while prefill is matrix-bound, but 16×16×16 cannot fit a fused-attention tile
   and is infeasible.
4. **L1 tile parameters move traffic, not latency, at the baseline SRAM.**
   `matrix.tile_k`, `attention.kv_tile` and searched vs fixed tiles change
   prefill by at most 0.3%, except fixed `n = 16` (+3.4%). Decode is unchanged
   across all feasible tile points.

## Array shape (`tiling-array-shape-v1`, 45 points, 26 feasible)

Grid: `matrix.count` 1, 2, 4, 8, 16 × `matrix.rows` 16, 32, 64 ×
`matrix.cols` 16, 32, 64. Of the 19 infeasible points, 18 exceed the usable
DSP budget (7,219; some also fail the attention tile fit) and 16×16×16 fails
only the fused-attention tile fit.

### Equal-PE comparison at 4,096 PEs

| Arrays (count × rows × cols) | Prefill ms | Matmul compute ms | Matmul L1 ms | L1-bound matmul ms | Prefill bound | Decode ms | Prefill weight HBM GB | Searched FFN tile |
|---|---:|---:|---:|---:|---|---:|---:|---|
| 1 × 64 × 64 | 665.7 | 511.3 | 161.2 | 4.3 | matrix | 14.10 | 2.11 | 1024×256, nr = 2 |
| 2 × 64 × 32 | 665.1 | 511.2 | 196.8 | 4.3 | matrix | 14.10 | 2.12 | 512×256, nr = 2 |
| 4 × 64 × 16 | 663.6 | 511.2 | 273.5 | 4.3 | matrix | 14.10 | 3.82 | 512×128, nr = 4 |
| 2 × 32 × 64 | 664.7 | 509.9 | 272.7 | 4.3 | matrix | 14.38 | 2.12 | 512×256, nr = 2 |
| 4 × 32 × 32 (baseline) | 663.7 | 509.9 | 315.3 | 4.3 | matrix | 14.38 | 3.82 | 512×128, nr = 4 |
| 8 × 32 × 16 | 663.5 | 509.9 | 386.5 | 48.2 | matrix | 14.38 | 3.82 | 256×128, nr = 4 |
| 4 × 16 × 64 | 735.7 | 509.2 | 501.2 | 331.6 | L1 | 14.95 | 3.82 | 512×128, nr = 4 |
| 8 × 16 × 32 | 757.0 | 509.2 | 538.3 | 353.8 | L1 | 14.95 | 3.82 | 256×128, nr = 4 |
| 16 × 16 × 16 (infeasible) | 807.3 | 509.2 | 629.4 | 408.4 | L1 | 14.89 | 7.22 | 128×64, nr = 8 |

Matmul columns sum each prefill matmul's compute and shared-L1 service terms;
"L1-bound matmul ms" is the scheduled time of matmuls whose L1 term exceeds
their compute term. `nr` is how many times each weight matrix is streamed from
HBM during prefill. Rows group the shapes better than aspect ratio does: the
wide 2×32×64 behaves like the other 32-row shapes, not like 4×16×64. Within a
row count, more units raise L1 service, and 8×32×16 already has 48 ms of
L1-bound matmuls. These columns are in [`points.csv`](points.csv).

At 1,024 and 2,048 PEs every shape is matrix-bound: matmul compute is
2,037–2,045 ms and 1,018–1,022 ms, while the largest matmul L1 service is
603 ms and 606 ms (16-row shapes).

The cost/prefill/decode frontier contains only `cols = 16` shapes because cost
depends on PE count alone and more rows reduce decode. That is a property of
the model's feed equation and cost proxy, not evidence that 64×16 arrays are
physically preferable.

## L1 matmul tile (`tiling-matmul-tile-v1`, 30 points, all feasible)

Grid: `matrix.tile_mode` search/fixed × `matrix.legacy_tile_n` 16–256 ×
`matrix.tile_k` 64, 256, 1024. `legacy_tile_n` only applies in fixed mode.
Decode is 14.38 ms at every point.

| Setting | FFN gate tile | Prefill ms | Weight HBM GB | Activation HBM GB | Prefill L1 GB |
|---|---|---:|---:|---:|---:|
| search, `tile_k` 64 or 256 | 512×128, nr = 4 | 663.7 | 3.82 | 2.62 | 66.6 |
| search, `tile_k` 1024 | 256×64, nr = 8 | 663.7 | 7.22 | 3.22 | 70.6 |
| fixed `n` 16 | 128×16, nr = 32 | 686.5 | 27.64 | 4.43 | 92.2 |
| fixed `n` 32 | 128×32, nr = 16 | 665.6 | 14.03 | 4.43 | 78.6 |
| fixed `n` 64 | 128×64, nr = 8 | 663.7 | 7.22 | 4.43 | 71.8 |
| fixed `n` 128 | 128×128, nr = 4 | 663.7 | 3.82 | 4.43 | 68.4 |
| fixed `n` 256, `tile_k` ≤ 256 | 128×256, nr = 2 | 663.7 | 2.12 | 4.43 | 66.7 |

The search minimizes combined weight and activation traffic, so a wide fixed
tile can stream fewer weights (2.12 vs 3.82 GB) while moving more activations.
A larger `tile_k` slab shrinks the searched tile and doubles weight traffic
without changing latency at 460 GB/s HBM.

## Fused-attention KV tile (`tiling-attention-tile-v1`, 5 points, 4 feasible)

| `attention.kv_tile` | Query tile, passes | Prefill ms | Decode ms |
|---:|---|---:|---:|
| 32 | 147 tokens, 4 | 663.99 | 14.38 |
| 64 | 128 tokens, 4 | 663.99 | 14.38 |
| 128 (baseline) | 96 tokens, 6 | 663.70 | 14.38 |
| 256 | 51 tokens, 11 | 663.00 | 14.38 |
| 512 | does not fit a 1 MiB per-unit budget | infeasible | — |

The effect is below 0.15%. The 512 failure comes from double-buffered K/V
alone filling each of the four units' 1 MiB tile share.

## Robustness: L1 size and prompt length

[`robustness.csv`](robustness.csv) re-evaluates the 4,096- and 2,048-PE shapes
at 4, 8 and 16 MiB L1 and at both `pp512-fa-on` and `pp128-fa-on`. The
grouping by rows does not change. At 4,096 PEs, the 32- and 64-row shapes stay
matrix-bound and within 1% of each other, except 8×32×16 at 4 MiB on
`pp512-fa-on`, which turns L1-bound and is also infeasible. 4×16×64 and
8×16×32 stay L1-bound and 10–20% slower in prefill. Every 2,048-PE shape stays
matrix-bound. At 4 MiB, every 8- and 16-unit shape fails the fused-attention
tile fit.

A larger L1 raises prefill by up to 0.7% for some shapes. Eric Wang traced this
to causal-attention accounting, and the `l1_size_control` record in
[`summary.json`](summary.json) reproduces it: from 8 to 16 MiB, the query tile
grows from two passes to one, so tile-granularity causal skipping removes less
work. Attention accounts for 3.50 of the 3.52 ms increase for 1×64×64 and 6.99
of 7.01 ms for 1×32×64 and 1×64×32. With causal skipping off, the increase is
0.02 ms (activation-residency accounting). This is a model approximation, not
a physical SRAM penalty.

## Hand-offs

- **SRAM study:** the per-unit tile budget couples the two studies. Unit count,
  `l1.mib` and `l1.tile_fraction` together decide tile size, weight
  re-streaming and attention-tile feasibility. A combined check of the best few
  array shapes against the best few SRAM settings should follow both studies.
- **Compute:** under the current feed model and baseline L1 bandwidth, 16-row
  arrays become L1-bound at 4,096 PEs, while 32- and 64-row arrays do not,
  whatever their column count. The threshold moves with total PEs, unit count
  and L1 bandwidth, so it is not a fixed rule. Before relying on it, the feed
  equation and the absence of any row/column-dependent array penalty need an
  RTL or microbenchmark check.

## Reproduction

From the repository root:

```text
python -B engine/sweeps/run.py engine/sweeps/tiling/array-shape.json
python -B engine/sweeps/run.py engine/sweeps/tiling/matmul-tile.json
python -B engine/sweeps/run.py engine/sweeps/tiling/attention-tile.json
python -B research/compute-mapping/2026-10-17-tiling-sweep/analyze.py
```

The first three commands write full rows to ignored `build/sweeps/`.
[`analyze.py`](analyze.py) reuses the same study runner, adds the selected
tiles, traffic split and matmul compute/L1 terms per point, and writes
[`points.csv`](points.csv), [`robustness.csv`](robustness.csv) and
[`summary.json`](summary.json). The summary has record hashes, frontiers, spec
hashes, the L1-size control and the source revision.

Every capture the script evaluates, including the robustness and control
workloads that bypass the study runner, is checked against the baseline
record's decoded capture SHA-256 before use. The validated hashes are listed in
`validated_decoded_capture_sha256`. `uncommitted_source_paths` covers source
files only; it is not a capture-integrity check.
[`test_tiling_study.py`](../../../engine/tests/test_tiling_study.py) confirms
that a substituted capture in either phase, or a single edited operation
parameter, makes the robustness path fail, while an equivalent plain-JSON form
is accepted.
