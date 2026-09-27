# Qwen3.5-2B graph inventory

Workload: pp128. Graphs captured during successful CPU inference.

| Phase | Input tokens | Past tokens | Scheduled nodes | Matrix MACs | Conv MACs |
|---|---:|---:|---:|---:|---:|
| prefill | 128 | 0 | 1,441 | 177,021,648,896 | 56,623,104 |
| decode | 1 | 128 | 1,441 | 1,887,567,872 | 442,368 |

MAC = multiply-accumulate; the work inventory counts two arithmetic operations per MAC. These counts exclude vector/reduction work and fused recurrent operations. No accelerator latency or tokens/s is estimated yet.

## Operation counts

| Operation | Prefill | Decode |
|---|---:|---:|
| GET_ROWS | 74 | 74 |
| RMS_NORM | 115 | 115 |
| MUL | 121 | 121 |
| RESHAPE | 228 | 228 |
| VIEW | 254 | 254 |
| SCALE | 72 | 72 |
| CPY | 72 | 72 |
| MUL_MAT | 199 | 199 |
| TRANSPOSE | 18 | 18 |
| CONCAT | 18 | 18 |
| SSM_CONV | 18 | 18 |
| SILU | 36 | 36 |
| ADD | 66 | 66 |
| SOFTPLUS | 18 | 18 |
| SIGMOID | 24 | 24 |
| GATED_DELTA_NET | 18 | 18 |
| SWIGLU | 24 | 24 |
| ROPE | 12 | 12 |
| SET_ROWS | 12 | 12 |
| PERMUTE | 24 | 24 |
| SOFT_MAX | 6 | 6 |
| CONT | 12 | 12 |

## Interpretation

- MAC counts cover MUL_MAT and SSM_CONV only; they are not total model FLOPs.
- Vector functions, reductions and fused GATED_DELTA_NET need separate hardware cost models.
- Logical tensor bytes and spans are not DRAM traffic; views alias storage and kernels reuse data.
- Dependencies are tensor dataflow. Stateful aliases and backend fusion require additional scheduling constraints.
- CPU-selected GGML graph with flash attention disabled; a target accelerator may lower or fuse it differently.

Raw JSON contains all dependencies, data types, shapes, byte strides, alias offsets, operation parameters, and scheduler order. DOT files contain full dataflow; layer 0 and layer 3 SVGs show representative recurrent and full-attention layers.
