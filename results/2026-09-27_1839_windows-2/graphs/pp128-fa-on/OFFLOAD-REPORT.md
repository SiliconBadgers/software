# Offload/compute-unit boundary comparison

Groups follow ops.py's software-facing taxonomy so they line up with operation-summary.json. Boundary bytes are the logical size of every tensor crossing into or out of a group in the scheduled dataflow -- what a candidate offload of that group would have to pack/transfer/synchronize, not a measured transfer time. Arithmetic intensity (arithmetic ops per boundary byte) is a compute-bound-vs-transfer-bound signal that needs no bandwidth assumption.


## prefill

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 115,964,116,992 | 301,989,888 | 768.00 | 47.7% |
| Attention/DeltaNet projections | 114 | 59,743,666,176 | 248,807,424 | 480.24 | 41.8% |
| DeltaNet recurrence | 18 | 0 | 151,289,856 | n/a | 4.0% |
| Copies/gather/state movement | 182 | 0 | 293,740,688 | n/a | 2.3% |
| Vector/norm/activation/other | 1000 | 0 | 1,010,221,200 | n/a | 2.0% |
| Convolution | 18 | 56,623,104 | 114,573,312 | 0.99 | 1.0% |
| Attention QK/softmax/AV | 6 | 0 | 15,728,640 | n/a | 0.6% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 0.5% |

## decode

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 905,969,664 | 2,359,296 | 768.00 | 36.7% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 22.4% |
| Attention/DeltaNet projections | 114 | 466,747,392 | 1,943,808 | 480.24 | 21.5% |
| Copies/gather/state movement | 182 | 0 | 106,471,568 | n/a | 12.2% |
| Vector/norm/activation/other | 1000 | 0 | 170,046,096 | n/a | 3.4% |
| DeltaNet recurrence | 18 | 0 | 57,362,688 | n/a | 2.8% |
| Attention QK/softmax/AV | 6 | 0 | 3,244,032 | n/a | 0.7% |
| Convolution | 18 | 442,368 | 2,211,840 | 0.40 | 0.3% |

## Limitations

- Boundary bytes come from one captured graph per phase; different prompt lengths keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, so absolute bytes scale with prompt length even though the ranking usually will not.
- Measured CPU time is a CPU interval share on this host, not accelerator latency, area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).
- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more or less resident than this structural count implies.
