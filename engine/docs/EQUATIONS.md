# Equations: what changed from the explorer, and why

**Explorer** = `experiments/llama-cpp/2026-09-24-qwen35-2b/explorer/engine.js` (Zeb Taylor, origin/main `11d36ed`).
**Engine** = `engine/sbengine/`. Every change below has a switch that restores the explorer's version, so its effect is
measured by `python -m sbengine ablation`, not argued. Numbers are from run `2026-09-27_1839_windows-2`, balanced design
(`profiles/accel-balanced.json`), and are estimates under the assumptions in `ASSUMPTIONS.md`.

Status labels: **derived** (follows from captured shapes/dtypes/order), **assumption** (a design placeholder that the data
do not settle), **measured** (from CPU op traces).

## 0. Notation

`f` clock (Hz). `B` batch (independent sequences). A matmul has output `M x N`, reduction `K`, `G` independent groups
(GGML order: `M = ne0`, `N = ne1`). `w`, `a` = bytes per weight / activation element. `rows x cols` = matrix array,
`U` = matrix units. `eff` = efficiency. Node duration `d = max(compute, L1 service, HBM service) + launch`.
`L1 service = L1 bytes / (banks * bytes_per_bank * f * eff)`, `HBM service = HBM bytes / (GB/s * eff)`.

## 1. Kept from the explorer (no data to justify a change)

| Piece | Equation |
|---|---|
| GEMM cycles | `tiles = ceil(M/rows) ceil(N/cols) G`; `waves = ceil(tiles/U)`; `cycles = waves * K/rate` (+ fill/drain, see 2.4) |
| GEMV lane reuse (`N < cols`, cap `gemv`) | `active = min(U, ceil(M/rows) G)`; `cycles = ceil(M N K G / (active rows cols rate)) + rows + cols` |
| Matrix time | `cycles / (f * eff)` |
| Vector / scalar time | `(ops + special * special_cycles) / (units * lanes * f * eff)`; scalar: `(ops * cycles_per_op + special * cycles_per_special) / (cores * ipc * f_core)` |
| Vector op counts per element | RMSNorm 3 (+1 special per row), softmax 5 (+1), SiLU/sigmoid 3 (+1), softplus 2 (+2), SwiGLU 4 (+1), RoPE 3 (+0.5), mul/add/scale 1 |
| Serial recurrence | `work = (7 S^2 + 3 S) H T B`, `special = H T B`; time `= (work + special * special_cycles) / (min(units, H B) * lanes * f * eff)` |
| Recurrence traffic on a unit that lacks scratch / on vector | state read 5x and written 2x per token through L1 |
| Persistent-state residency | `resident = min(1, L1 * state_reserve / (state_per_seq * B))` of state avoids HBM |
| Mapping priority | dedicated unit, then matrix, then vector if all caps, then scalar; missing everything is infeasible |
| Resource proxies | DSP demand and the unitless cost sum (extended, see 2.13) |
| Feasibility | every op mapped, tile fits L1, DSP <= budget * (1 - shell reserve), HBM footprint <= capacity |

The explorer's mixed uniform-precision modes (`W8/A16`, `W4/A8`) survive as `precision.weights = uniform_int8 | uniform_int4`.

## 2. Changed

Effects quote **pp512, flash attention off** (the explorer's own graph), cumulative `explorer -> engine`, then the
**leave-one-out** cost of reverting just that change from the full engine. `weights HBM` etc. are prefill unless noted.

### 2.1 Weight storage and MAC rate: per tensor, from the capture  (`precision.weights`)  [derived + assumption]
- Explorer: two uniform modes. `bytes = e * bits / 8 + ceil(e / group) * scale` for matrices, 32-bit otherwise; MAC rate
  `W8A16 = 1`, `W4A8 = 2`.
- Engine: `bytes(t) = logical_bytes` of the captured tensor (block formats: q4_K 4.5, q5_K 5.5, q6_K 6.56, q8_0 8.5
  bits/element, checked against every weight in `test_formats_config`); MAC rate looked up by format class
  (`matrix.rates`: `w4 2, w5 1, w6 1, w8 1, w16 0.5, w32 0.25, aa 0.5`). Uniform policies remain.
- Why: the capture stores 187 weight tensors in four formats totalling 1.27 GB. The explorer's BF16 capture stores about
  3.8 GB and its uniform 8-bit mode 1.94 GB. Nothing else moves the estimate as much.
- Effect: prefill -20.5 %, decode -30.2 % (leave-one-out +37.6 % / +44.1 %). The rates for `w5/w6/w8/w16/w32` are
  **assumptions**; the accelerator's number format is undecided, and Q4_K_M does not validate a custom INT4.

### 2.2 Fused attention (`FLASH_ATTN_EXT`)  [derived + assumption]
- Explorer: none; its graph has flash attention off, so QK, softmax and AV are separate ops with materialised scores
  (`8192 x 8192 x 8` f32 = 2 GiB per layer at 8192 tokens).
- Engine: for the captured fused op with `d = Q.ne0`, `Tq = Q.ne1`, `Hq = Q.ne2`, `Tk = K.ne1`, `Hkv = K.ne2`, `gq = Hq/Hkv`:
  ```
  q_rows_tile = floor((cap - 4 kt d kv_b) / (d a + 4 d + 4 kt)),  cap = L1 * tile_fraction / U
  tok_tile    = max(1, q_rows_tile / gq);  nq = ceil(Tq / tok_tile)
  avg_keys    = keys_read                                   (no causal skip)
              = min(keys_read, past + Tq (nq + 1) / (2 nq))  (causal skip, tile granularity)
  MACs        = 2 d Tq Hq B avg_keys;   scores = Hq Tq B avg_keys
  time        = max( T_mat(QK) + T_mat(AV),  T_vec(5 scores ops, scores exps) )     (softmax overlaps the matrices)
  KV from HBM = nq * avg_keys * d * Hkv * 2 * kv_b * B;   L1 extra = scores * (8 + 2 a)
  ```
  Scores never reach HBM. The attention mask is generated on-chip by default (`attention.mask_onchip`), because the
  capture's 8192-token mask alone is 0.5-1 GiB of a tensor the accelerator would not load.
- Why: fused and unfused attention differ by 29x in boundary bytes at pp8192 (write-up in `results/`), and the model needs
  the fused form to price what a real kernel does.
- Effect: fa-on vs fa-off prefill saves 1.5 % (pp128), 2.3 % (pp512), **21.7 % (pp8192: 12.2 s vs 15.6 s)**; decode < 2 %.
- Check: with causal skipping off, the fused MACs equal the QK + AV MACs of the *independently captured* fa-off graph
  exactly, at pp512 and pp8192, both phases (`test_fused_attention_does_the_same_matmul_work...`).
  `kv_tile = 128`, the L1 partition and the score hand-off cost `(8 + 2a)` bytes per score are **assumptions**.

### 2.3 Matmul tiling: searched, not fixed  (`matrix.tile_mode`)  [derived]
- Explorer: tile `m = 4 rows`, `n = tileN = 64`, `k = tileK`, halved until `2(m k w + k n a + 4 m n) <= L1 (1 - reserve) / U`.
  Weights are re-streamed `ceil(N / 64)` times.
- Engine: choose `(m, n)` from power-of-two and array-multiple candidates to minimise `A_total ceil(N/n) + X_total ceil(M/m)`
  subject to the same footprint. `A` is weights (or K/V), `X` the activations.
- Why: with real prefill lengths (up to 8192 tokens) the fixed `n = 64` re-reads every weight `N/64` times (128x at 8192).
- Effect: prefill weights HBM 7.2 -> 3.8 GB (pp512) and 109 -> 55 GB (pp8192). Time barely moves (0.0 % / -0.9 %) because prefill is
  matrix-bound; it matters for HBM footprint and bandwidth design, not latency, at this operating point. The test
  proves optimality over the candidate set and that more L1 never means more traffic.

### 2.4 Array fill/drain once per op  (`matrix.pipelined_tiles`)  [assumption]
- Explorer: `cycles = waves * (K/rate + rows + cols - 2)`. Engine: `waves * K/rate + (rows + cols - 2)`.
- Why: double-buffered accumulators overlap a tile's drain with the next tile's fill. Prefill -2.1 % (leave-one-out +3.1 %).

### 2.5 L1 feed traffic at array level  (`memory.l1_feed`)  [derived]
- Explorer: `L1 reads = (M K w ceil(N/n) + K N a ceil(M/m)) G`, reuse counted at L1-tile level; weight *scale* bytes counted
  in HBM traffic but not in the L1 read.
- Engine (output-stationary array): `tiles * K * (min(rows, M) w + min(cols, N) a)`, `tiles = ceil(M/rows) ceil(N/cols) G`, with
  weight bytes including scales. Every output tile streams both operands past the PEs.
- Why: L1 bandwidth is the binding resource in decode (13.4 of 14.4 ms; elasticity -0.9), so its traffic must be right.
  Effect: decode +3.8 % (almost all scale bytes; a node-by-node comparison found 30 MB of 3.9 GB), prefill ~0.

### 2.6 Activation traffic from liveness and residency  (`memory.residency`)  [derived + assumption]
- Explorer: every op's inputs and outputs are charged to L1; HBM adds a proxy `2 * max(0, working_set - L1 (1 - reserve))`
  per op (none for matmul/recurrence); HBM footprint uses the peak working set of one op.
- Engine: simulate the whole schedule. A result is written once and stays in L1 while it fits (activation share of L1 =
  `1 - state_reserve - tile_fraction`) and is still needed; otherwise it is written to HBM and re-read at each use. Eviction
  removes the tensor needed farthest in the future (Belady: optimal for equal sizes, a heuristic for mixed sizes). Tensors
  larger than the share stream from HBM with their reuse factor. `peak spilled bytes` is a real liveness footprint.
- Why: the proxy undercounts round trips. Prefill activation HBM at pp512: **1.32 GB (proxy) vs 4.10 GB (simulation)**, 3.1x.
- Effect on time is small (compute-bound): +0.2 % prefill. The static L1 partition (25 % state / 50 % tiles / 25 % activations)
  is an **assumption**; the results are insensitive to it in these workloads (elasticity 0.00).

### 2.7 Fusion of single-consumer epilogues  (`memory.fusion`)  [assumption]
- Engine (new): an edge is fused, and its tensor never stored, when the producer's result has exactly one consumer node, that
  consumer is an elementwise epilogue (mul/add/scale/activations/RoPE) and the producer is a compute op. 355 edges at pp512.
- Why: dependency and consumer counts come from the capture. Optimistic (assumes the units can apply the epilogue in place);
  `fusion = none` is the pessimistic bracket. Activation HBM 4.10 -> 2.72 GB; time -0.1 %.

### 2.8 Recurrence: serial or chunked, cheaper wins  (`recurrent.lowering`)  [assumption, high impact]
- Explorer: serial only (`work = (7 S^2 + 3 S) H T B` per layer).
- Engine adds the chunkwise-parallel form, mapped to the matrix units, for `T > 1`: per head per chunk of `C` tokens
  ```
  K K^T: C^2 S   triangular solve: C^3/3   A K: C^2 S   A V: C^2 S   W S: C S^2   K^T(U - W S): C S^2
  Q S: C S^2     Q K^T: C^2 S              (Q K^T)(U - W S): C^2 S        =>  3 C S^2 + 5 C^2 S + C^3/3
  ```
  plus `12 S H T B` vector ops for gating/decay, added serially. `auto` picks the cheaper of serial and chunked.
- Why: recurrence is 16-25 % of the explorer's predicted prefill time, and per-token serial recurrence is a dependency chain. The
  chunk-parallel form is the standard formulation for gated delta networks (Yang et al., "Gated Delta Networks: Improving
  Mamba2 with Delta Rule"; Yang et al., "Parallelizing Linear Transformers with the Delta Rule over Sequence Length").
  The MAC counts are derived here from that structure, **not** taken from or validated against an implementation.
- Effect: the recurrence op costs 315.7 ms serial vs 37.7 ms chunked at pp512 (8.4x at every length), which is 28.8 % of prefill
  (leave-one-out +38.3 %). It has no effect on decode (`T = 1`). **This is the most assumption-dependent number in the engine:**
  it presumes the matrix units can run the chunk GEMMs at `matrix.efficiency` and that the software can lower the op this way.

### 2.9 Whole-graph time: bounds and a dependency-aware schedule  (`schedule.mode`)  [derived + assumption]
- Explorer: sum of op durations in captured order (no overlap between ops).
- Engine reports three numbers: `serial` (the explorer's sum), `list` (in-order issue per unit pool, ops wait for dependencies
  plus a hand-off, ops on different pools overlap, never below total L1 or HBM service time) and `lower_bound` (max of the
  critical path, busiest pool, total L1 and total HBM service). Dependencies come from the capture, plus explicit
  read-after-write and write-after-read ordering on the KV cache and recurrent/conv state, which tensor dataflow omits.
- Effect: decode -9.6 % (leave-one-out +10.6 %), prefill -1.3 %. Pools are exclusive and memory systems are shared floors, so
  this is a bound-respecting estimate, not a simulation.

### 2.10 Cross-unit hand-off and host fallback  (`sync_cycles`, `host.*`)  [assumption; measured for the host]
- Engine (new): a dependency crossing unit pools costs `sync_cycles / f` in the schedule (placeholder 100 cycles).
  With `host.enabled`, an op no accelerator unit can run takes its **measured CPU time** from the op trace plus
  `bytes / link_GB/s + 2 * link_latency`. This is the only place the CPU measurements enter an accelerator estimate.
- Why: issue #3 asks for dispatch, transfer, synchronisation and host-fallback costs, which the explorer omitted. The default
  100 cycles has no measurement behind it (elasticity ~0.01 on decode).

### 2.11 KV, state and persistent traffic from the graph  [derived]
- Explorer: `kv = 6 * 2 * 512 * round(max(prompt, context + 1), 256) * B * kv_bits / 8` and
  `state = 18 (128 * 128 * 16 + 3 * 6144) * state_bits / 8`, both hard-coded to this model, with state HBM traffic charged at the
  recurrence op.
- Engine: KV and state sizes are the `cache_k/v_l*` and `cache_r/s_l*` tensors in the graph. Traffic is charged where it
  happens: `GET_ROWS` from state, `CPY` into state, `SET_ROWS` into the cache, K/V reads by attention.
  `attention.capacity_tokens` rescales the KV footprint (the capture's context is 4x the prompt by convention, the
  explorer pads live context to 256).

### 2.12 Memory-op DMA and double counting  [fix]
- The explorer's DMA time uses `(read + write) / (engines * bytes_per_cycle * f)`; here it is `2 * bytes` for every memory op
  (gathers and scatters previously used one direction). A byte entering L1 from HBM is charged once (ingress, in the HBM
  count) and one leaving once (egress), so the unit-side copy of the same stream is not charged again.
- The unfused requantisation epilogue (`3 E` vector ops) is charged only when the op actually ran on the matrix unit.

### 2.13 Weight path and costed L1 bandwidth  (`memory.weight_path`, `resources.cost.*`)  [assumption]
- Explorer: "no direct inter-unit channels": HBM -> L1 -> array, so every weight byte crosses L1 twice. Decode is then 93 % L1-bound
  (13.4 of 14.4 ms) and HBM bandwidth has no effect.
- Engine: `weight_path = via_l1` (default, explorer-compatible) or `direct` (weights stream HBM -> per-PE staging, skipping L1).
  `direct` takes decode from 14.4 ms to **5.35 ms** at pp512 and makes HBM bandwidth the binding resource (elasticity -0.72).
- The explorer priced SRAM by capacity only, so extra banks (L1 bandwidth) and a direct path were free and a sweep put
  `l1.banks = 256` on the Pareto front at no cost. New placeholder cost terms: `l1_bank = 8` per bank and
  `direct_weight_buffer_per_pe = 0.25`; the capability count also gains one entry per supported weight-format class.
  These are **placeholders**; setting them to 0 restores the explorer's "free" behaviour.

## 3. Ablation, pp512 fa-off (`python -m sbengine ablation --graph pp512-fa-off`)

| Step (cumulative) | Prefill | Decode | Traffic |
|---|---:|---:|---|
| explorer equations | 1241.3 ms | 22.27 ms | weights HBM 11.85 GB, act HBM 1.32 GB, decode L1 4.00 GB |
| + captured weight formats | 986.2 (-20.5 %) | 15.53 (-30.2 %) | weights HBM 7.22 GB, decode L1 2.65 GB |
| + searched tiles | 986.2 (0.0 %) | 15.40 (-0.9 %) | weights HBM 3.82 GB |
| + pipelined fill/drain | 965.2 (-2.1 %) | 15.40 | |
| + array-level L1 feed | 965.4 | 15.98 (+3.8 %) | decode L1 2.74 GB |
| + residency simulation | 967.1 (+0.2 %) | 15.99 | act HBM 4.10 GB |
| + fusion | 965.7 (-0.1 %) | 15.98 | act HBM 2.72 GB |
| + chunked recurrence | **687.8 (-28.8 %)** | 15.98 | |
| + causal skipping / hand-off | 687.8 | 15.98 | (no effect on this graph) |
| + list schedule | **679.0 (-1.3 %)** | **14.45 (-9.6 %)** | |

Leave-one-out from the full engine (prefill / decode): captured formats +37.6 % / +44.1 %, recurrence lowering +38.3 % / 0,
list schedule +1.3 % / +10.6 %, fill/drain +3.1 % / 0, L1 feed -0.0 % / -3.9 %, all others |x| < 0.6 %. The same ordering holds at
pp128 and pp8192 (formats +37.8 % / +26.0 % prefill, recurrence +38.1 % / +26.6 %).

## 4. Fidelity to the explorer when everything is switched back

`profiles/legacy-equations.json` resets every switch. On the explorer's graphs, both saved designs, 128 and 512 tokens
(8 points): **prefill differs by <= 0.001 %, decode by +0.46 % to +0.99 %**. At balanced/512 the decode gap is +0.10 ms of
22.17: +0.15 ms in matmuls from weight scale bytes now counted in the L1 feed (2.5), partly offset by -0.05 ms from slightly
different ingress accounting in `GET_ROWS` and `CONCAT` (2.12); recurrent-state traffic moved from the recurrence op to the
state read/write ops (2.11) without changing any op's time there. Every other op agrees to within 0.002 ms. Asserted in
`test_legacy_switches_reproduce_the_explorer_on_its_own_graphs`.

## 5. Not modelled

Energy and power; NoC/multi-core; bank conflicts and arbitration beyond one efficiency number; weight caching in L1; prefill
micro-batching (a 628-token prompt ran as two micro-batches in the profiled run but is one captured graph); decode-context growth
within a run (one decode step per capture); batch > 1 beyond analytical scaling; the vision encoder, tokenizer, sampling and host
orchestration; numerical accuracy of any quantisation.
