# Offload/compute-unit boundary comparison

Groups follow ops.py's software-facing taxonomy so they line up with operation-summary.json. Boundary bytes are the logical size of every tensor crossing into or out of a group in the scheduled dataflow -- what a candidate offload of that group would have to pack/transfer/synchronize, not a measured transfer time. Arithmetic intensity (arithmetic ops per boundary byte) is a compute-bound-vs-transfer-bound signal that needs no bandwidth assumption.


## prefill

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 278,132,686,848 | 724,303,872 | 768.00 | 46.0% |
| Attention/DeltaNet projections | 114 | 143,291,449,344 | 596,749,056 | 480.24 | 41.2% |
| Copies/gather/state movement | 182 | 0 | 557,686,928 | n/a | 5.0% |
| DeltaNet recurrence | 18 | 0 | 283,675,392 | n/a | 3.9% |
| Vector/norm/activation/other | 1000 | 0 | 2,197,550,736 | n/a | 1.8% |
| Convolution | 18 | 135,806,976 | 272,941,056 | 1.00 | 1.0% |
| Attention QK/softmax/AV | 6 | 0 | 36,470,784 | n/a | 0.9% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 0.2% |

## decode

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 905,969,664 | 2,359,296 | 768.00 | 36.2% |
| Attention/DeltaNet projections | 114 | 466,747,392 | 1,943,808 | 480.24 | 21.7% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 21.5% |
| Copies/gather/state movement | 182 | 0 | 106,471,568 | n/a | 12.0% |
| Vector/norm/activation/other | 1000 | 0 | 173,191,824 | n/a | 3.6% |
| DeltaNet recurrence | 18 | 0 | 57,362,688 | n/a | 2.9% |
| Attention QK/softmax/AV | 6 | 0 | 6,389,760 | n/a | 1.7% |
| Convolution | 18 | 442,368 | 2,211,840 | 0.40 | 0.4% |

## Limitations

- Boundary bytes come from one captured graph per phase; different prompt lengths keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, so absolute bytes scale with prompt length even though the ranking usually will not.
- Measured CPU time is a CPU interval share on this host, not accelerator latency, area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).
- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more or less resident than this structural count implies.
