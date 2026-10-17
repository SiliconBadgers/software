# Qwen3.5-2B memory traffic for mapping decisions

October 14, 2026. Part of the mapping/software-optimization track and
[issue #3](https://github.com/SiliconBadgers/software/issues/3). It estimates
SRAM and HBM traffic for matrix operations tiled onto the candidate matrix
unit, and which of those bytes the current architecture decisions remove.

> [!IMPORTANT]
> Every figure here is a **logical byte count** derived from captured tensor
> shapes. None is a measured SRAM or HBM transfer, a cycle count, or an FPGA
> result. Bank conflicts, buffer residency beyond what each case states, DMA
> overhead and overlap are not modeled.

## Inputs

| Input | Revision | Status |
|---|---|---|
| [Op dependency map](../2026-10-02-op-dependency-map/README.md) `out/pp128-decode`, `out/pp512-prefill` | frozen 2026-10-02 study | structural graph data |
| Abhinav Gummadi's [Timeloop baseline](https://github.com/SiliconBadgers/software/blob/abhinav/timeloop-accelergy-qwen35/research/timeloop-accelergy-qwen35/REPORT.md) | `34f09ed` on `abhinav/timeloop-accelergy-qwen35` | model output; constants copied into `compute.py` |
| [Compute team decisions](https://github.com/SiliconBadgers/architecture/blob/codex/compute-team-decisions/decisions/compute-team-decisions.md) | `70ee075` on `codex/compute-team-decisions` | INT8 activations, INT32 accumulation, INT4/INT8 weights |
| [Command format draft](https://github.com/SiliconBadgers/rtl/blob/codex/command-format-spec/docs/command-format.md) | `codex/command-format-spec` | `clear_acc` / `write_output` K-tile flags, requantization at output write |
| SRAM port widths 8 / 32 / 64 B per cycle for 8x8 / 16x16 / 32x32 | Software meeting 2026-10-11 | proposal |

## Reproduce

Python 3.9 or newer, standard library only. From the repository root:

```
python research/compute-mapping/2026-10-14-memory-traffic/compute.py
```

The script prints the four tables below, writes `results.json` beside itself,
and exits non-zero if a check fails. Checks:

- Decode weight-matrix elements summed from `nodes.csv` equal 1,881,276,416.
- For all five Timeloop cases, `k_groups * M * N` equals the C update count in
  the Timeloop baseline, so the partial-sum estimates use the same tiling.

## Results

### 1. Partial sums across K tiles

Every captured matmul has K > 8, and the candidate 8x8 RTL emits one 8-wide
dot product per pass. Partial sums are INT32 (4 bytes). Three cases:

- **keep**: accumulators stay in the array across K tiles and are written once.
  This is what the command format's `clear_acc` / `write_output` flags encode.
- **timeloop**: one write per MainMemory-level K group (2 for `ffn_gate` and
  the head, 6 for `ffn_out`) and one read back for all but the first, matching
  the baseline mappings. In those mappings the GlobalBuffer holds partials
  across the inner 128 K steps, so this column is MainMemory traffic.
- **per8**: partials leave the array after every 8-wide K tile, which is the
  candidate RTL as it stands.

| Case | Weight bytes | keep | timeloop | per8 |
|---|---:|---:|---:|---:|
| ffn_gate/up decode | 12.58 MB | 0.025 MB | 0.074 MB (0.6%) | 12.6 MB (100%) |
| ffn_gate/up pp128 prefill | 12.58 MB | 3.15 MB | 9.44 MB (75%) | 1,607 MB (12,775%) |
| ffn_out decode | 12.58 MB | 0.008 MB | 0.090 MB (0.7%) | 12.6 MB (100%) |
| ffn_out pp128 prefill | 12.58 MB | 1.05 MB | 11.5 MB (92%) | 1,610 MB (12,792%) |
| output head | 508.6 MB | 0.99 MB | 2.98 MB (0.6%) | 508 MB (100%) |

Percentages are relative to the case's weight bytes. In decode, spilling every
8-wide tile doubles total traffic. In prefill, weight reuse makes weights cheap
and partial sums dominate. The Timeloop tiling moves another 75-92% of the
weight bytes as partials, and per-tile spilling moves about 128x. Prefill needs
the output retention in the command format, which the candidate RTL does not
implement yet.

### 2. DeltaNet state per decode step

Each of the 18 DeltaNet layers reads and writes 1,122,304 bytes per sequence
per token (1 MiB F32 recurrent state, 72 KiB conv state), per the dependency
map's `summary.json`. Total: 19.27 MiB each way. Q8.24 decay math keeps the
state at 32 bits, so the size does not shrink under the current decisions.
Weights total 1.88 GB at INT8 and 0.94 GB at INT4, shared across a batch.

| Batch | SRAM-resident | State HBM per step | SRAM needed | vs INT8 weights | vs INT4 weights |
|---:|---:|---:|---:|---:|---:|
| 1 | 0% | 40.4 MB | 0 | 2.1% | 4.3% |
| 1 | 100% | 0 | 19.3 MiB | 0% | 0% |
| 8 | 0% | 323 MB | 0 | 17% | 34% |
| 8 | 100% | 0 | 154 MiB | 0% | 0% |
| 32 | 0% | 1,293 MB | 0 | 69% | 137% |
| 32 | 100% | 0 | 617 MiB | 0% | 0% |

At batch 1, state traffic is a few percent of weight traffic, so keeping state
in HBM costs little. State scales with batch while weights do not; at batch 32
with INT4 weights, state traffic exceeds weight traffic. Full residency at
batch 1 needs 19.3 MiB, about half of the roughly 40 MB of FPGA block RAM and
UltraRAM mentioned at the October 3 Compute meeting, and nearly 10x the 2 MiB
GlobalBuffer in the Timeloop baseline. We need a target batch size before
spending hardware or software effort on state placement. `results.json`
also has the 50% cases.

### 3. Matrix-to-vector handoff

Bytes moved through SRAM between the FFN matmuls and SwiGLU, per layer, and
for the output head:

- **L0**: INT32 matmul output written to SRAM and read by the vector unit.
- **L1**: requantization at matrix output write (command format draft), INT8
  round trip through SRAM.
- **L2**: matmul output handed straight to the vector unit; only the SwiGLU
  result goes through SRAM to `ffn_down`.

| Case | L0 | L1 | L2 | INT8 weights |
|---|---:|---:|---:|---:|
| FFN decode | 0.111 MB (0.3%) | 0.037 MB (0.1%) | 0.012 MB | 37.7 MB |
| FFN pp512 prefill | 56.6 MB (150%) | 18.9 MB (50%) | 6.3 MB (17%) | 37.7 MB |
| Output head | 1.99 MB | 0.50 MB | 0 | 509 MB |

Prefill projects only the last token through the output head, so its head
traffic equals decode. Handoff bytes are negligible in decode. In prefill,
requantization at write removes two thirds of the L0 traffic, and a direct
matrix-to-vector path would remove most of the rest. That path is a
forwarding feature the October 6 Controller meeting listed as a stretch goal.

### 4. Decode weight feed versus SRAM port

With one token, each weight is used once. To keep an A-stationary array busy,
SRAM must deliver one weight per PE per cycle.

| Array | Port | INT8 need | INT8 utilization bound | INT4 need | INT4 utilization bound |
|---|---:|---:|---:|---:|---:|
| 8x8 | 8 B/cycle | 64 B | 12.5% | 32 B | 25% |
| 16x16 | 32 B/cycle | 256 B | 12.5% | 128 B | 25% |
| 32x32 | 64 B/cycle | 1,024 B | 6.2% | 512 B | 12.5% |

HBM at the explorer's 460 GB/s setting supplies 1,840 B/cycle at 250 MHz, so
in this proxy the SRAM-to-array port limits decode and HBM has headroom. The
proposed port widths assume reuse across tokens, which prefill has
(arithmetic intensity equals the token count, for example 128 MAC per weight
byte at pp128) and decode does not (1 MAC per byte at INT8). A larger array
lowers the decode bound unless the port grows with it.

## Limitations

- Logical bytes from CPU-captured F32/BF16 graphs remapped to the decided
  integer widths. No numerical validation of INT8/INT4 execution.
- One layer's FFN represents all 24 layers; attention projections, the DeltaNet
  `attn_qkv` stack and vector kernels other than SwiGLU are not covered.
- Partial-sum cases assume a write per K group and a read back for all but the
  first group, with no residency in a register file or buffer between them.
- State residency is a fraction of the aggregate state, not a placement trace.
  Batch scaling is an analytical extrapolation, as in the explorer.
- Port widths, clocks and the 460 GB/s figure are proposals or explorer
  settings, not committed hardware. Utilization bounds ignore pipeline fill,
  shadow-register swaps and command overhead.
- The Timeloop constants are copied from `34f09ed`. If that branch reruns with
  different mappings, update `TIMELOOP_CASES` in `compute.py`.

## Questions for Compute and Memory

1. What batch sizes should decode target? The answer sets how much DeltaNet
   state residency is worth.
2. Will SRAM-to-matrix port width scale with array size, and is decode
   utilization a design goal?
3. Is a direct matrix-to-vector path planned for prefill, or does all handoff
   go through SRAM?
4. When will the candidate RTL gain the output retention that `clear_acc` /
   `write_output` require?
