# Offload/compute-unit boundary comparison

Groups follow ops.py's software-facing taxonomy so they line up with operation-summary.json. Boundary bytes are the logical size of every tensor crossing into or out of a group in the scheduled dataflow -- what a candidate offload of that group would have to pack/transfer/synchronize, not a measured transfer time. Arithmetic intensity (arithmetic ops per boundary byte) is a compute-bound-vs-transfer-bound signal that needs no bandwidth assumption.


## prefill

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 239,175,991,296 | 622,854,144 | 768.00 | 46.4% |
| Attention/DeltaNet projections | 114 | 123,221,311,488 | 513,165,312 | 480.24 | 40.8% |
| Copies/gather/state movement | 182 | 0 | 494,280,848 | n/a | 5.2% |
| DeltaNet recurrence | 18 | 0 | 251,873,280 | n/a | 3.8% |
| Vector/norm/activation/other | 1000 | 0 | 1,913,082,000 | n/a | 1.8% |
| Convolution | 18 | 116,785,152 | 234,897,408 | 0.99 | 0.9% |
| Attention QK/softmax/AV | 6 | 0 | 32,243,712 | n/a | 0.9% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 0.2% |

## decode

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 905,969,664 | 2,359,296 | 768.00 | 35.6% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 22.0% |
| Attention/DeltaNet projections | 114 | 466,747,392 | 1,943,808 | 480.24 | 21.1% |
| Copies/gather/state movement | 182 | 0 | 106,471,568 | n/a | 12.9% |
| Vector/norm/activation/other | 1000 | 0 | 173,191,824 | n/a | 3.5% |
| DeltaNet recurrence | 18 | 0 | 57,362,688 | n/a | 2.8% |
| Attention QK/softmax/AV | 6 | 0 | 6,389,760 | n/a | 1.6% |
| Convolution | 18 | 442,368 | 2,211,840 | 0.40 | 0.5% |

## Limitations

- Boundary bytes come from one captured graph per phase; different prompt lengths keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, so absolute bytes scale with prompt length even though the ranking usually will not.
- Measured CPU time is a CPU interval share on this host, not accelerator latency, area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).
- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more or less resident than this structural count implies.
