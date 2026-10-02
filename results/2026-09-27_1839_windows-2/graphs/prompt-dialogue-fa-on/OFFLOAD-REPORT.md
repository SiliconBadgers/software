# Offload/compute-unit boundary comparison

Groups follow ops.py's software-facing taxonomy so they line up with operation-summary.json. Boundary bytes are the logical size of every tensor crossing into or out of a group in the scheduled dataflow -- what a candidate offload of that group would have to pack/transfer/synchronize, not a measured transfer time. Arithmetic intensity (arithmetic ops per boundary byte) is a compute-bound-vs-transfer-bound signal that needs no bandwidth assumption.


## prefill

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 212,902,871,040 | 554,434,560 | 768.00 | 47.3% |
| Attention/DeltaNet projections | 114 | 109,685,637,120 | 456,794,880 | 480.24 | 40.8% |
| Copies/gather/state movement | 182 | 0 | 451,518,608 | n/a | 4.6% |
| DeltaNet recurrence | 18 | 0 | 230,425,344 | n/a | 3.8% |
| Vector/norm/activation/other | 1000 | 0 | 1,718,085,264 | n/a | 1.6% |
| Attention QK/softmax/AV | 6 | 0 | 26,247,168 | n/a | 0.9% |
| Convolution | 18 | 103,956,480 | 209,240,064 | 0.99 | 0.8% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 0.3% |

## decode

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 905,969,664 | 2,359,296 | 768.00 | 36.7% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 21.7% |
| Attention/DeltaNet projections | 114 | 466,747,392 | 1,943,808 | 480.24 | 21.5% |
| Copies/gather/state movement | 182 | 0 | 106,471,568 | n/a | 12.5% |
| Vector/norm/activation/other | 1000 | 0 | 170,046,096 | n/a | 3.1% |
| DeltaNet recurrence | 18 | 0 | 57,362,688 | n/a | 2.9% |
| Attention QK/softmax/AV | 6 | 0 | 3,244,032 | n/a | 1.3% |
| Convolution | 18 | 442,368 | 2,211,840 | 0.40 | 0.4% |

## Limitations

- Boundary bytes come from one captured graph per phase; different prompt lengths keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, so absolute bytes scale with prompt length even though the ranking usually will not.
- Measured CPU time is a CPU interval share on this host, not accelerator latency, area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).
- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more or less resident than this structural count implies.
