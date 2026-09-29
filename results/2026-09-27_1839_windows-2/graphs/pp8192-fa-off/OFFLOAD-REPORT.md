# Offload/compute-unit boundary comparison

Groups follow ops.py's software-facing taxonomy so they line up with operation-summary.json. Boundary bytes are the logical size of every tensor crossing into or out of a group in the scheduled dataflow -- what a candidate offload of that group would have to pack/transfer/synchronize, not a measured transfer time. Arithmetic intensity (arithmetic ops per boundary byte) is a compute-bound-vs-transfer-bound signal that needs no bandwidth assumption.

No matching CPU operation trace was given; only structural figures are shown.

## prefill

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte |  |
|---|---:|---:|---:|---:|
| MLP projections | 72 | 7,421,703,487,488 | 19,327,352,832 | 768.00 |
| Attention/DeltaNet projections | 114 | 3,823,594,635,264 | 15,923,675,136 | 480.24 |
| Other matrix products | 12 | 1,649,267,441,664 | 26,675,773,440 | 123.65 |
| Convolution | 18 | 3,623,878,656 | 7,249,084,416 | 1.00 |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 |
| Copies/gather/state movement | 188 | 0 | 13,191,225,488 | n/a |
| Vector/norm/activation/other | 1018 | 0 | 81,231,986,832 | n/a |
| DeltaNet recurrence | 18 | 0 | 6,115,295,232 | n/a |

## decode

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte |  |
|---|---:|---:|---:|---:|
| MLP projections | 72 | 905,969,664 | 2,359,296 | 768.00 |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 |
| Attention/DeltaNet projections | 114 | 466,747,392 | 1,943,808 | 480.24 |
| Other matrix products | 12 | 207,618,048 | 107,151,360 | 3.88 |
| Convolution | 18 | 442,368 | 2,211,840 | 0.40 |
| Copies/gather/state movement | 188 | 0 | 307,896,464 | n/a |
| Vector/norm/activation/other | 1018 | 0 | 475,378,320 | n/a |
| DeltaNet recurrence | 18 | 0 | 57,362,688 | n/a |

## Limitations

- Boundary bytes come from one captured graph per phase; different prompt lengths keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, so absolute bytes scale with prompt length even though the ranking usually will not.
- Measured CPU time is a CPU interval share on this host, not accelerator latency, area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).
- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more or less resident than this structural count implies.
