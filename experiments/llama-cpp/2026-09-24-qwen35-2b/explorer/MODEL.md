# QwenProfiling analytical model v1

This is an interactive architecture exploration model, not a hardware benchmark, synthesis tool, numerical quantization validator, or cycle-accurate simulator. Its useful outputs are sensitivity, bottlenecks, workload tradeoffs, and candidate architectures under explicit assumptions. Absolute timings and the Pareto frontier require calibration.

## Provenance and workload

The input is the full, successfully executed GGML text graph of Qwen3.5-2B from llama.cpp commit `308883b335798865e73f8fee0d9a5fbfa13c8480`. The original captures and model checksum are preserved one directory above. There are 24 layers, including 18 gated-delta layers and 6 full-attention layers. This version does not include vision, MTP, tokenization, sampling, host transfer, or multi-FPGA communication.

`scripts/export_explorer_data.py` derives tensor shape expressions from four captures: prefill lengths 128 and 512, and one decode step after each. The exporter verifies tensor names, operation names, and source edges agree. Shape expressions encode constants, affine token-length terms, phase-specific shapes, attention length rounded to 256, and context capacity. Unrecognized expressions fail export rather than being guessed. The capture's four matrix MAC totals are regression checks.

Batch means independent sequences processed together. Projection weights are shared across those sequences. Activations, KV storage, and recurrent states scale with batch. Weighted GEMM flattens tokens and sequences into its N dimension. Attention keeps independent head/sequence groups. This batching is an analytical extrapolation, not a separately captured batch graph.

Prefill starts at past length zero, processes the chosen prompt length, and projects only the last position into logits. Decode processes one new token at the chosen past length. These workload lengths may be set independently. `tokens/s per sequence = 1 / decode_step_seconds`; aggregate decode rate is batch times that rate. Prefill throughput is prompt length times batch divided by prefill time. Prefill latency excludes CPU tokenization, transfer and sampling, so it is not end-to-end TTFT.

## Precision

The captured model stores BF16 matrices. The explorer remaps rank-two model weights to W8 or W4 and activation storage to integer A16 or A8, respectively. Matrix accumulation and accumulator-tile capacity use INT32. Attention matrix products require A16xA16 or A8xA8 support separately from weight products. Weight metadata costs `ceil(weight_count / group_size) * scale_bytes`.

One-dimensional model tensors are retained as 32-bit. KV storage defaults to 16-bit; recurrent state to 32-bit. Index tensors retain their integer widths. Gated-delta output includes both sequence activations and final state, which are accounted for separately. Nonlinear/special-function throughput is modeled through configurable vector/scalar costs; it does not imply integer exp/log/normalization is numerically equivalent to the BF16 model. Dequantization/requantization quality, clipping, overflow policy, scale granularity beyond byte overhead, and model accuracy remain unverified.

Changing the active precision changes arithmetic packing rates and traffic, but not the physical unit count or area proxy of a multi-format architecture. Per-format MAC rates are editable assumptions, not guaranteed DSP packing. Fused matrix requantization eliminates the separately charged epilogue; otherwise 3 basic operations per output element and intermediate traffic are charged to vector or scalar execution. More detailed conversion costs can be calibrated in the engine.

## Compute and mapping

The captured scheduler order is preserved. Views/reshapes/permutations/transposes are metadata-only; zero-element operations are skipped. Materialization is charged through CONT/CPY/CONCAT. This version uses all configured units of the selected family cooperatively within one operation and does not overlap independent graph operations. It therefore does not predict a dependency-aware multikernel schedule.

For a matrix result with dimensions M by N, reduction K, and G independent groups:

```
MACs = M * N * K * G
array_tiles = ceil(M / rows) * ceil(N / columns) * G
cycles = ceil(array_tiles / unit_count) * (K / precision_rate + rows + columns - 2)
time = cycles / (frequency * compute_efficiency)
```

With GEMV lane reuse and `N < columns`, reduction parallelism can reuse the narrow token dimension:

```
active_units = min(unit_count, ceil(M / rows) * G)
cycles = ceil(M*N*K*G / (active_units*rows*columns*precision_rate)) + rows + columns
```

This is a simplified reconfigurable GEMV mapping. It is not a claim that an arbitrary systolic array implements that reuse for free. Test the capability both on and off and calibrate costs. Optional matrix depthwise convolution currently has a 4x utilization penalty. General vector software GEMM, when enabled, uses two basic arithmetic operations per MAC.

Vector time is `(basic_ops + special_calls * special_cycles) / (units * lanes * frequency * efficiency)`. Basic counts include 3 operations/element for RMSNorm plus one rsqrt per row; 5 operations and one exp/element for softmax; 3 and one special for sigmoid/SiLU; 2 and two specials for softplus; 4 and one special for SwiGLU; and 3 and one-half special per element for RoPE. These are algebraic cost proxies, not generated instruction counts. Reduction tree latency and small-vector utilization are folded into efficiency and special-function costs.

For gated-delta recurrence with state dimension S=128 and H=16 heads, each sequence token performs approximately `(7*S*S + 3*S)*H` basic arithmetic operations and H exponentials: state decay, state-vector prediction, delta update, outer product, and output projection. This comes from the pinned GGML recurrence implementation. The dedicated unit supports the whole update; its parallel unit count is capped by independent heads times batch. Token recurrence remains serial. Generic vector recurrence requires recurrence-loop, elementwise, reduction, and exp capabilities; otherwise the operation falls back to RISC-V. This version does not implement a separate chunked/prefix-scan GDN lowering to GEMM.

Missing capabilities send the whole operation to RISC-V, rather than charging unsupported work as zero. Scalar time is `(basic_ops * instructions_per_basic_op + special_calls * instructions_per_special) / (cores * IPC * core_frequency)`. Multiple cores assume software parallelism; the default is one. These costs are editable software placeholders. No implementation plus no RISC-V means an infeasible design. Mapping is deterministic priority (dedicated, matrix where applicable, vector if complete, scalar), not a compiler search for the fastest legal route.

Memory operations use configured DMA bytes/cycle or vector/scalar fallback. Their service demand shares L1 with every other unit.

## Shared L1, HBM and capacity

There are no direct inter-unit channels. Operation input reads, output writes, and HBM DMA ingress/egress all consume the same shared SRAM service budget:

```
L1_bandwidth = banks * bytes_per_bank_per_cycle * frequency * bank_efficiency
HBM_bandwidth = peak_GB_per_second * 1e9 * sustained_efficiency
```

F2 defaults are 16 GiB HBM and the user's 460 **GB/s** setting. AWS's announcement uses 460 **GiB/s**, approximately 493.9 GB/s. These units are deliberately distinguished. Both peak and sustained settings are editable.

Matrix tiles begin at M=min(M,4*array_rows), N=min(N,max_token_tile), K=min(K,reduction_tile). Double-buffered storage is `2*(m*k*weight_bytes + k*n*activation_bytes + m*n*4)`; the last term is INT32 accumulation. Available tile storage is the non-state L1 allocation divided by matrix count. Tiles shrink until they fit. A design that cannot fit a one-element tile is infeasible. Accumulators stay in the tile; reduction-tile segmentation does not by itself imply spilling accumulators to HBM.

Weight matrices stream from HBM once per N tile, not once per batch element. L1 weight reads repeat per N tile; activation reads repeat per M tile. Attention KV reads/writes include the captured padding and grouped-head reuse assumptions. All weights are treated as streamed, even if extra L1 could hold some; there is no persistent weight-cache policy in this version.

A configurable share of L1 retains a proportional fraction of the aggregate recurrent/conv state. Nonresident gated-delta state reads/writes go to HBM (prefill writes final state; decode reads and writes it). Vector recurrence uses a multipass L1 traffic proxy: five state reads and two state writes per token, plus initial/final state and inputs. A dedicated recurrence unit avoids those passes when its per-unit scratch fits one head's state. This persistent allocation is a fractional approximation, not an actual cache placement trace.

For other operations, a working set beyond available non-state L1 incurs `2*max(working_set - available_L1, 0)` bytes of HBM spill traffic. This is an explicit conservative capacity proxy, not an LRU simulation or a bound for every fused schedule. It can overestimate tiled vector kernels and underestimate simultaneous live-tensor pressure. Repeated traffic is never counted as resident allocation. Estimated HBM footprint includes unique stored weights, KV, recurrent state, and a peak operator working set; it is not an exact liveness allocation.

## Time and overlap

For each nonzero operation:

```
compute = implementation time, including an unfused epilogue when needed
L1_service = shared_L1_bytes / effective_L1_bandwidth
HBM_service = HBM_bytes / effective_HBM_bandwidth
operation_time = max(compute, L1_service, HBM_service) + launch_cycles / frequency
```

With overlap disabled, the three service terms are added. The graph time is the sum of operation times. Service bars show aggregate demands and must not be added when overlap is enabled. This transfer-overlap comparison is not an overall confidence interval. Per-operation overlap, banking, register forwarding, pipeline fill, residency, and future fusion choices all warrant hardware calibration.

## Resource proxy, feasibility and Pareto

Whole-device F2 DSP ceiling defaults to 9,024. A 20% shell reserve is only a placeholder. Estimated DSP demand is physical matrix PEs, vector lanes, and recurrence lanes times their editable DSP coefficients. This excludes actual LUT implementation, nonlinear logic, DMA/control complexity, and the AWS shell's exact resources. Passing this check does **not** establish FPGA fit or timing closure. SRAM bits are counted in the cost proxy, but no BRAM/URAM fit test is claimed.

Cost sums editable matrix-PE, vector-lane, recurrent-lane and SRAM coefficients; enabled capability/type overhead once per unit family; RISC-V cost; and a fixed 40-unit DMA-engine proxy. It has no physical unit. Disabled units contribute no corresponding capability cost. Dedicated scratch also consumes the SRAM cost proxy. This is a starting point for calibrated area/resource scores, not tapeout area, power or price.

A feasible result has numerical parameters, a mapping for every operation, an admissible matrix tile, sufficient HBM under the footprint approximation, and estimated DSP demand below the configured budget. Four objectives are supported: cost versus decode latency, cost versus aggregate decode throughput, decode latency versus aggregate throughput, and cost versus prefill latency. The frontier excludes infeasible candidates and removes dominated/duplicate objective outcomes. Cross-precision comparisons do not assert equal model quality. Batch sweeps trade per-sequence latency for aggregate throughput.

## Validation and extension points

Run `node ../scripts/test_model.cjs` from this directory. Checks cover all four captured matrix-MAC totals; full mapping coverage; missing-capability fallback; no-fallback infeasibility; weight sharing across batch; traffic versus storage distinction; HBM and DSP capacity; L1/HBM sensitivity; SRAM tiling; recurrence implementation; overlap; precision versus physical cost; Pareto dominance; and invalid parameters.

The explorer runs entirely in the browser and works from `file://` without a server. `graph-data.js` is the compact graph, `engine.js` contains all formulas, and `app.js` handles interaction. Configurations import/export as JSON, sweeps export CSV, and settings persist in browser-local storage. QA mode (`?qa=1`) skips persistence.

Useful next calibration inputs are an RTL GEMM/GEMV microbenchmark across the four supported integer pairs, a vector primitive throughput/latency table, a RISC-V software primitive benchmark, a shared-SRAM arbitration test, measured HBM efficiency, and synthesis resources for each optional capability. More sophisticated scheduling, explicit tiling/dataflow search, nonlinear numeric validation, energy/power, and process-node area models can be added independently.

## Sources

- [AWS F2 announcement: VU47P, 9,024 DSP slices, 16 GiB HBM and bandwidth](https://aws.amazon.com/blogs/aws/now-available-second-generation-fpga-powered-amazon-ec2-instances-f2/)
- [Official Qwen3.5-2B configuration](https://huggingface.co/Qwen/Qwen3.5-2B/blob/15852e8c16360a2fea060d615a32b45270f8a8fc/config.json)
- [Pinned GGML CPU operation implementations, including gated-delta recurrence](https://github.com/ggml-org/llama.cpp/blob/308883b335798865e73f8fee0d9a5fbfa13c8480/ggml/src/ggml-cpu/ops.cpp)
