# Offload/compute-unit boundary comparison

Groups follow ops.py's software-facing taxonomy so they line up with operation-summary.json. Boundary bytes are the logical size of every tensor crossing into or out of a group in the scheduled dataflow -- what a candidate offload of that group would have to pack/transfer/synchronize, not a measured transfer time. Arithmetic intensity (arithmetic ops per boundary byte) is a compute-bound-vs-transfer-bound signal that needs no bandwidth assumption.


## prefill

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 463,856,467,968 | 1,207,959,552 | 768.00 | 46.4% |
| Attention/DeltaNet projections | 114 | 238,974,664,704 | 995,229,696 | 480.24 | 40.6% |
| Copies/gather/state movement | 182 | 0 | 859,971,728 | n/a | 5.3% |
| DeltaNet recurrence | 18 | 0 | 435,290,112 | n/a | 3.8% |
| Vector/norm/activation/other | 1000 | 0 | 3,553,738,896 | n/a | 1.8% |
| Attention QK/softmax/AV | 6 | 0 | 56,623,104 | n/a | 1.1% |
| Convolution | 18 | 226,492,416 | 454,311,936 | 1.00 | 0.9% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 0.2% |

## decode

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 905,969,664 | 2,359,296 | 768.00 | 36.8% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 20.8% |
| Attention/DeltaNet projections | 114 | 466,747,392 | 1,943,808 | 480.24 | 20.6% |
| Copies/gather/state movement | 182 | 0 | 106,471,568 | n/a | 12.0% |
| Vector/norm/activation/other | 1000 | 0 | 176,337,552 | n/a | 4.3% |
| DeltaNet recurrence | 18 | 0 | 57,362,688 | n/a | 2.7% |
| Attention QK/softmax/AV | 6 | 0 | 9,535,488 | n/a | 2.4% |
| Convolution | 18 | 442,368 | 2,211,840 | 0.40 | 0.3% |

## Limitations

- Boundary bytes come from one captured graph per phase; different prompt lengths keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, so absolute bytes scale with prompt length even though the ranking usually will not.
- Measured CPU time is a CPU interval share on this host, not accelerator latency, area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).
- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more or less resident than this structural count implies.
