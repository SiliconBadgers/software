# Offload/compute-unit boundary comparison

Groups follow ops.py's software-facing taxonomy so they line up with operation-summary.json. Boundary bytes are the logical size of every tensor crossing into or out of a group in the scheduled dataflow -- what a candidate offload of that group would have to pack/transfer/synchronize, not a measured transfer time. Arithmetic intensity (arithmetic ops per boundary byte) is a compute-bound-vs-transfer-bound signal that needs no bandwidth assumption.


## prefill

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 64,323,846,144 | 167,510,016 | 768.00 | 48.1% |
| Attention/DeltaNet projections | 114 | 33,139,064,832 | 138,010,368 | 480.24 | 41.3% |
| DeltaNet recurrence | 18 | 0 | 109,133,568 | n/a | 3.9% |
| Copies/gather/state movement | 182 | 0 | 209,690,768 | n/a | 2.3% |
| Vector/norm/activation/other | 1000 | 0 | 633,134,736 | n/a | 1.8% |
| Attention QK/softmax/AV | 6 | 0 | 10,125,312 | n/a | 0.9% |
| Convolution | 18 | 31,408,128 | 64,143,360 | 0.98 | 0.9% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 0.8% |

## decode

| Group | Nodes | Matrix MACs | Boundary bytes (in+out) | Arithmetic ops/byte | Measured CPU time |
|---|---:|---:|---:|---:|:--|
| MLP projections | 72 | 905,969,664 | 2,359,296 | 768.00 | 32.9% |
| Copies/gather/state movement | 182 | 0 | 106,471,568 | n/a | 22.0% |
| Attention/DeltaNet projections | 114 | 466,747,392 | 1,943,808 | 480.24 | 19.4% |
| Vocabulary output head | 1 | 508,559,360 | 8,192 | 124160.00 | 19.3% |
| Vector/norm/activation/other | 1000 | 0 | 170,046,096 | n/a | 3.1% |
| DeltaNet recurrence | 18 | 0 | 57,362,688 | n/a | 2.5% |
| Attention QK/softmax/AV | 6 | 0 | 3,244,032 | n/a | 0.4% |
| Convolution | 18 | 442,368 | 2,211,840 | 0.40 | 0.3% |

## Limitations

- Boundary bytes come from one captured graph per phase; different prompt lengths keep the same op sequence (see `graphs/prompt-comparison.md`) with different shapes, so absolute bytes scale with prompt length even though the ranking usually will not.
- Measured CPU time is a CPU interval share on this host, not accelerator latency, area or bandwidth (see AGENTS.md/docs/profiling-next-steps.md).
- Boundary bytes assume no buffer reuse across the cut; a real backend may keep more or less resident than this structural count implies.
