# Offload/compute-unit boundary comparison

Groups follow ops.py's software-facing taxonomy so they line up with operation-summary.json. Boundary bytes are the logical size of every tensor crossing into or out of a group in the scheduled dataflow -- what a candidate offload of that group would have to pack/transfer/synchronize, not a measured transfer time. Arithmetic intensity (arithmetic ops per boundary byte) is a compute-bound-vs-transfer-bound signal that needs no bandwidth assumption.

No matching CPU operation trace was given; only structural figures are shown.

## prefill

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte |  |
|---|---:|---:|---:|---:|
| MLP projections | 72 | 212,902,871,040 | 554,434,560 | 768.00 |
| Attention/DeltaNet projections | 114 | 109,685,637,120 | 456,794,880 | 480.24 |
| Other matrix products | 12 | 1,478,492,160 | 49,348,608 | 59.92 |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 |
| Convolution | 18 | 103,956,480 | 209,240,064 | 0.99 |
| Copies/gather/state movement | 188 | 0 | 480,911,504 | n/a |
| Vector/norm/activation/other | 1018 | 0 | 1,770,579,600 | n/a |
| DeltaNet recurrence | 18 | 0 | 230,425,344 | n/a |

## decode

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte |  |
|---|---:|---:|---:|---:|
| MLP projections | 72 | 905,969,664 | 2,359,296 | 768.00 |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 |
| Attention/DeltaNet projections | 114 | 466,747,392 | 1,943,808 | 480.24 |
| Other matrix products | 12 | 6,291,456 | 3,342,336 | 3.76 |
| Convolution | 18 | 442,368 | 2,211,840 | 0.40 |
| Copies/gather/state movement | 188 | 0 | 112,861,328 | n/a |
| Vector/norm/activation/other | 1018 | 0 | 176,534,160 | n/a |
| DeltaNet recurrence | 18 | 0 | 57,362,688 | n/a |

## Limitations

- Boundary bytes come from one captured graph per phase; different prompt lengths keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, so absolute bytes scale with prompt length even though the ranking usually will not.
- Measured CPU time is a CPU interval share on this host, not accelerator latency, area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).
- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more or less resident than this structural count implies.
