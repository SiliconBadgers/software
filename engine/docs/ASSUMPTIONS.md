# Assumption register

Every constant the engine uses is either **derived** from the captured graphs, **measured** (CPU op traces) or an
**assumption**. Nothing here is a Compute-, Memory- or Control-team decision. The last columns say how much the answer moves
and who could replace the placeholder with a measurement.

Format rates matter only when the op is compute-bound, hence 0.00 for `via_l1` decode (L1-bound). Sensitivity is `d ln(time) / d ln(parameter)` for a +/-30 % change on `pp512-fa-on` with the balanced design
(`python -m sbengine sensitivity`); "-1.0" means time falls in proportion as the parameter rises, "0.00" means no effect in
these workloads. Two binding regimes are shown because the answer changes: `via_l1` (weights cross L1, the explorer's
assumption; decode is L1-bound) and `direct` (decode is HBM-bound).

## Derived from the capture (not assumptions)

Tensor shapes, dtypes and byte sizes; operand slots and view chains; scheduled order and dependencies; MAC counts
(`MUL_MAT`, `SSM_CONV`; checked against every capture summary); KV-cache and recurrent/conv state sizes; head counts and
head size (`FLASH_ATTN_EXT`: 8 query heads, 2 KV heads, head size 256); `S = 128`, `H = 16` for the recurrence; which
weights each op reads and in which format. **Caveat:** captured with flash attention on and off, batch 1, context = 4x prompt
(a capture convention), on CPU; a target accelerator may schedule, fuse and buffer differently.

## Measured

| Item | Where used | Limits |
|---|---|---|
| Per-node CPU time (`cpu-op-trace.jsonl.gz`) | `host.use_measured_cpu_time` fallback; `validate-cpu` | One instrumented run per graph; profiler overhead is measured separately by `core/`; fa-on graphs only; prompts split into micro-batches are excluded |

## Hardware parameters (assumptions)

Defaults copy the explorer's saved "balanced" design; those in turn are placeholders (F2-class FPGA framing, "your 460 GB/s"
HBM setting). Sensitivity: `via_l1` decode / prefill, then `direct` decode where it differs.

| Parameter | Default | Elasticity | Who can settle it |
|---|---|---|---|
| `clock_mhz` | 300 | -1.00 / -1.00 (direct decode -0.30) | Compute: synthesis timing closure |
| `matrix.count`, `rows`, `cols` | 4, 32x32 | design axes, not assumptions | Compute |
| `matrix.efficiency` | 0.75 | 0.00 / -0.81 | Compute: RTL GEMM/GEMV utilisation |
| `matrix.rates.w4` | 2 | 0.00 / -0.38 (67 % of prefill MACs are q4_K) | Compute: packing of 4-bit weights |
| `matrix.rates.w5` | 1 | 0.00 / -0.26 (22 % of prefill MACs are q5_K) | Compute: which formats the array supports |
| `matrix.rates.w6` | 1 | 0.00 / -0.14 (11 % are q6_K) | Compute |
| `matrix.rates.w8`, `w16`, `w32` | 1, 0.5, 0.25 | 0.00 / 0.00 (q8_0 is 0.1 % of MACs; f16/f32 are not matmul weights here) | Compute |
| `matrix.rates.aa` (activation x activation) | 0.5 | 0.00 / -0.06 | Compute |
| `matrix.pipelined_tiles` | true | reverting to per-tile fill/drain costs prefill +3.1 % | Compute |
| `matrix.tile_k` | 256 | 0.00 / 0.00 | Compute |
| `matrix.conv_penalty` | 4.0 | unused by default (no `conv` cap) | Compute |
| `vector.count`, `lanes` | 4, 32 | design axes | Compute |
| `vector.efficiency` | 0.7 | -0.01 / -0.17 | Compute |
| `vector.special_cycles` | 8 | 0.00 / +0.07 | Compute: exp/rsqrt unit latency |
| `recurrent.count`, `lanes`, `scratch_kib` | 2, 128, 64 | design axes | Compute |
| `recurrent.efficiency` | 0.7 | -0.05 / 0.00 (direct decode -0.12) | Compute |
| `recurrent.lowering`, `chunk` | `auto`, 64 | **prefill +38 % if serial-only** (see below) | Software + Compute: an actual lowering |
| `recurrent.matrix_penalty` (only with `lowering = matrix`) | 16 | none at 64 L1 banks (the op is L1-bound); +16 % prefill from 1x to 16x at 256 banks | Compute: synthesize the per-token update loop |
| `dma.count`, `bytes_per_cycle` | 2, 64 | -0.04 / -0.01 | Memory |
| `scalar.*` (fallback core) | 1 core, ipc 1, 300 MHz | only if a design lacks units | Control |
| `launch_cycles` | 40 | 0.00 / 0.00 | Control: command issue cost |
| `sync_cycles` (cross-unit hand-off) | 100 | +0.01 / 0.00 | Control: **no measurement behind it** |
| `l1.mib` | 8 | 0.00 / 0.00 | Memory |
| `l1.banks`, `bytes_per_bank_cycle`, `efficiency` | 64, 16, 0.65 | **-0.90 / -0.08 each** (direct decode ~0) | Memory: SRAM banking and arbitration test |
| `l1.state_reserve`, `tile_fraction` | 0.25, 0.50 | 0.00 (static partition) | Memory |
| `hbm.gib` | 16 | feasibility only | Platform |
| `hbm.gbs`, `efficiency` | 460, 0.7 | 0.00 / 0.00 (**direct decode -0.72 each**) | Memory: measured sustained bandwidth |
| `host.link_gbps`, `link_latency_us` | 16, 20 | host fallback only (off by default) | Platform |

## Design and modelling switches (choices, not physical constants)

| Switch | Default | Effect if flipped |
|---|---|---|
| `memory.weight_path` | `via_l1` | `direct`: decode 14.4 -> 5.35 ms (pp512); an **architecture decision** that this model does not size |
| `precision.weights` | `captured` | `uniform_int4` / `uniform_int8`: weights 1.18 GiB captured vs int4 < captured < int8; Q4_K_M does not validate a custom INT4 |
| `precision.act_bits`, `kv_bits`, `state_bits` | 16, 16, 32 | storage widths; the explorer's activation remap (the capture has f32 activations) |
| `memory.fusion` | `chains` | `none` is the pessimistic bracket (activation HBM +50 % at pp512); `groups` is the optimistic one (anchored dependency-map groups, a further -34 %) |
| `memory.residency` | `belady` | `per_op` restores the explorer's proxy (undercounts activation traffic ~3x) |
| `attention.causal_skip` | true | false: fused attention does the dense work |
| `attention.mask_onchip` | true | false: the mask is read from HBM (0.5 GiB at 8192) |
| `attention.kv_padding` | `captured` | `exact` reads only live tokens |
| `attention.overlap_softmax` | true | softmax hides behind the matrix work |
| `attention.kv_tile` | 128 | 0.00 |
| `attention.capacity_tokens` | graph's | rescales the KV footprint |
| `schedule.mode`, `within_op_overlap` | `pools`, true | `serial` is the explorer's sum |

**The recurrence lowering is the single largest assumption in the engine.** With `lowering = serial` the gated-delta-net op costs
315.7 ms at pp512 (8.4x the chunked form at every length) and prefill is 38 % slower. Chunked lowering assumes the matrix units
run the chunk GEMMs at `matrix.efficiency` and that software can lower the operation that way; neither is demonstrated.

`lowering = matrix` runs the captured serial update on the matrix arrays at `recurrent.matrix_penalty` instead. It needs no new
software lowering and covers decode, but its penalty is unmeasured and its result here is set by the assumed five state passes per
token through shared L1 (`EQUATIONS.md` 2.8). `python -m sbengine switches` lists every switch above with its pessimistic setting
and what each one is worth for a given design.

## Resource and cost proxies (unitless, unchanged coefficients except two new ones)

`dsp_budget` 9024 and `shell_reserve` 0.2 (F2 framing, placeholders); `dsp_per_pe` 1, `dsp_per_vector_lane` 0.5,
`dsp_per_recurrent_lane` 1; cost per matrix PE 1, vector lane 8, recurrent lane 10, SRAM 128 per MiB, capability 40, scalar core
200, DMA engine 40. **New placeholders:** `l1_bank` 8 per bank and `direct_weight_buffer_per_pe` 0.25, because the explorer
priced SRAM by capacity alone and so treated L1 bandwidth as free. Cost is not area, power or price. Passing the DSP check
does not establish FPGA fit, and LUT/BRAM/URAM and timing closure are not checked.

## Uncertainty in one number

Perturbing every assumption above by +/-30 % (log-uniform, 60 samples, seed 0; `python -m sbengine montecarlo`) moves the
balanced design's pp512 decode time across **9.4 / 17.4 / 33.8 ms (5th / 50th / 95th percentile)** and prefill p50 to 745 ms
(point estimate 663.7 ms). Read every absolute time in this engine as an estimate with roughly that width until the
placeholders above are replaced by measurements.
