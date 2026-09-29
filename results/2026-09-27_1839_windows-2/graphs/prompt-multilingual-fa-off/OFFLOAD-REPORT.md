# Offload/compute-unit boundary comparison

Groups follow ops.py's software-facing taxonomy so they line up with operation-summary.json. Boundary bytes are the logical size of every tensor crossing into or out of a group in the scheduled dataflow -- what a candidate offload of that group would have to pack/transfer/synchronize, not a measured transfer time. Arithmetic intensity (arithmetic ops per boundary byte) is a compute-bound-vs-transfer-bound signal that needs no bandwidth assumption.

No matching CPU operation trace was given; only structural figures are shown.

## prefill

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte |  |
|---|---:|---:|---:|---:|
| MLP projections | 72 | 137,707,388,928 | 358,612,992 | 768.00 |
| Attention/DeltaNet projections | 114 | 70,945,603,584 | 295,458,816 | 480.24 |
| Other matrix products | 12 | 956,301,312 | 33,030,144 | 57.90 |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 |
| Convolution | 18 | 67,239,936 | 135,806,976 | 0.99 |
| Copies/gather/state movement | 188 | 0 | 348,790,928 | n/a |
| Vector/norm/activation/other | 1018 | 0 | 1,203,597,456 | n/a |
| DeltaNet recurrence | 18 | 0 | 169,039,872 | n/a |

## decode

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte |  |
|---|---:|---:|---:|---:|
| MLP projections | 72 | 905,969,664 | 2,359,296 | 768.00 |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 |
| Attention/DeltaNet projections | 114 | 466,747,392 | 1,943,808 | 480.24 |
| Other matrix products | 12 | 6,291,456 | 3,342,336 | 3.76 |
| Convolution | 18 | 442,368 | 2,211,840 | 0.40 |
| Copies/gather/state movement | 188 | 0 | 111,288,464 | n/a |
| Vector/norm/activation/other | 1018 | 0 | 174,961,296 | n/a |
| DeltaNet recurrence | 18 | 0 | 57,362,688 | n/a |

## Limitations

- Boundary bytes come from one captured graph per phase; different prompt lengths keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, so absolute bytes scale with prompt length even though the ranking usually will not.
- Measured CPU time is a CPU interval share on this host, not accelerator latency, area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).
- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more or less resident than this structural count implies.
