# Offload/compute-unit boundary comparison

Groups follow ops.py's software-facing taxonomy so they line up with operation-summary.json. Boundary bytes are the logical size of every tensor crossing into or out of a group in the scheduled dataflow -- what a candidate offload of that group would have to pack/transfer/synchronize, not a measured transfer time. Arithmetic intensity (arithmetic ops per boundary byte) is a compute-bound-vs-transfer-bound signal that needs no bandwidth assumption.


## prefill

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 137,707,388,928 | 358,612,992 | 768.00 | 47.3% |
| Attention/DeltaNet projections | 114 | 70,945,603,584 | 295,458,816 | 480.24 | 42.1% |
| DeltaNet recurrence | 18 | 0 | 169,039,872 | n/a | 3.9% |
| Copies/gather/state movement | 182 | 0 | 329,130,128 | n/a | 2.6% |
| Vector/norm/activation/other | 1000 | 0 | 1,168,994,448 | n/a | 2.0% |
| Convolution | 18 | 67,239,936 | 135,806,976 | 0.99 | 1.0% |
| Attention QK/softmax/AV | 6 | 0 | 18,087,936 | n/a | 0.7% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 0.4% |

## decode

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 905,969,664 | 2,359,296 | 768.00 | 35.7% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 22.4% |
| Attention/DeltaNet projections | 114 | 466,747,392 | 1,943,808 | 480.24 | 20.7% |
| Copies/gather/state movement | 182 | 0 | 106,471,568 | n/a | 13.8% |
| Vector/norm/activation/other | 1000 | 0 | 170,046,096 | n/a | 3.4% |
| DeltaNet recurrence | 18 | 0 | 57,362,688 | n/a | 2.9% |
| Attention QK/softmax/AV | 6 | 0 | 3,244,032 | n/a | 0.8% |
| Convolution | 18 | 442,368 | 2,211,840 | 0.40 | 0.4% |

## Limitations

- Boundary bytes come from one captured graph per phase; different prompt lengths keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, so absolute bytes scale with prompt length even though the ranking usually will not.
- Measured CPU time is a CPU interval share on this host, not accelerator latency, area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).
- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more or less resident than this structural count implies.
