# Offload/compute-unit boundary comparison

Groups follow ops.py's software-facing taxonomy so they line up with operation-summary.json. Boundary bytes are the logical size of every tensor crossing into or out of a group in the scheduled dataflow -- what a candidate offload of that group would have to pack/transfer/synchronize, not a measured transfer time. Arithmetic intensity (arithmetic ops per boundary byte) is a compute-bound-vs-transfer-bound signal that needs no bandwidth assumption.


## prefill

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 173,040,205,824 | 450,625,536 | 768.00 | 46.1% |
| Attention/DeltaNet projections | 114 | 89,148,751,872 | 371,267,328 | 480.24 | 41.2% |
| Copies/gather/state movement | 182 | 0 | 386,637,968 | n/a | 4.7% |
| DeltaNet recurrence | 18 | 0 | 197,883,648 | n/a | 4.0% |
| Vector/norm/activation/other | 1000 | 0 | 1,427,000,976 | n/a | 2.2% |
| Convolution | 18 | 84,492,288 | 170,311,680 | 0.99 | 0.9% |
| Attention QK/softmax/AV | 6 | 0 | 21,921,792 | n/a | 0.5% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 0.3% |

## decode

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 905,969,664 | 2,359,296 | 768.00 | 37.7% |
| Attention/DeltaNet projections | 114 | 466,747,392 | 1,943,808 | 480.24 | 21.0% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 20.8% |
| Copies/gather/state movement | 182 | 0 | 106,471,568 | n/a | 12.1% |
| Vector/norm/activation/other | 1000 | 0 | 170,046,096 | n/a | 4.3% |
| DeltaNet recurrence | 18 | 0 | 57,362,688 | n/a | 2.8% |
| Attention QK/softmax/AV | 6 | 0 | 3,244,032 | n/a | 0.9% |
| Convolution | 18 | 442,368 | 2,211,840 | 0.40 | 0.4% |

## Limitations

- Boundary bytes come from one captured graph per phase; different prompt lengths keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, so absolute bytes scale with prompt length even though the ranking usually will not.
- Measured CPU time is a CPU interval share on this host, not accelerator latency, area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).
- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more or less resident than this structural count implies.
