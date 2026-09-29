# Operation-sharing evidence: chains and category boundaries

Workload: pp128-fa-off.

A **chain** is a maximal run of ops where each step's only computed input is the previous step's output, and that output has no other consumer -- the dataflow condition needed to keep an intermediate out of memory, not a claim about what any specific backend or accelerator fuses. A **category transition** is every scheduled producer-consumer edge, tagged `chainable` (meets that condition) or `shared` (the producer feeds more than one consumer, or the consumer merges several computed inputs, so materializing that edge cannot be avoided by fusion choices alone).

## prefill

783 chains covering 1441 scheduled ops.

### Highest-MAC chains

| Length | Matrix MACs | Categories | Backend-fused edges |
|---:|---:|---|---|
| 3 | 1,610,612,736 | matrix -> metadata | not checked |
| 1 | 1,610,612,736 | matrix | not checked |
| 1 | 1,610,612,736 | matrix | not checked |
| 2 | 1,610,612,736 | vector -> matrix | not checked |
| 3 | 1,610,612,736 | matrix -> metadata | not checked |
| 1 | 1,610,612,736 | matrix | not checked |
| 1 | 1,610,612,736 | matrix | not checked |
| 2 | 1,610,612,736 | vector -> matrix | not checked |
| 3 | 1,610,612,736 | matrix -> metadata | not checked |
| 1 | 1,610,612,736 | matrix | not checked |

### Category-boundary crossings (top by producer bytes)

| From | To | Kind | Edges | Producer bytes |
|---|---|---|---:|---:|
| matrix | vector | shared | 78 | 182,452,224 |
| vector | metadata | shared | 54 | 169,869,312 |
| vector | matrix | shared | 144 | 150,994,944 |
| metadata | memory | shared | 300 | 143,474,832 |
| vector | vector | shared | 136 | 142,606,336 |
| vector | vector | chainable | 164 | 125,075,456 |
| matrix | metadata | chainable | 108 | 104,103,936 |
| metadata | vector | chainable | 156 | 103,858,176 |
| vector | matrix | chainable | 30 | 81,788,928 |
| recurrent | metadata | shared | 36 | 75,497,472 |
| metadata | metadata | chainable | 48 | 69,206,016 |
| memory | metadata | shared | 18 | 57,950,208 |
| memory | convolution | shared | 18 | 57,950,208 |
| convolution | vector | chainable | 18 | 56,623,104 |
| vector | recurrent | shared | 54 | 37,896,192 |

## decode

783 chains covering 1441 scheduled ops.

### Highest-MAC chains

| Length | Matrix MACs | Categories | Backend-fused edges |
|---:|---:|---|---|
| 5 | 508,559,360 | vector -> memory -> matrix | not checked |
| 3 | 12,582,912 | matrix -> metadata | not checked |
| 1 | 12,582,912 | matrix | not checked |
| 1 | 12,582,912 | matrix | not checked |
| 2 | 12,582,912 | vector -> matrix | not checked |
| 3 | 12,582,912 | matrix -> metadata | not checked |
| 1 | 12,582,912 | matrix | not checked |
| 1 | 12,582,912 | matrix | not checked |
| 2 | 12,582,912 | vector -> matrix | not checked |
| 3 | 12,582,912 | matrix -> metadata | not checked |

### Category-boundary crossings (top by producer bytes)

| From | To | Kind | Edges | Producer bytes |
|---|---|---|---:|---:|
| metadata | memory | shared | 300 | 85,733,520 |
| recurrent | metadata | shared | 36 | 38,043,648 |
| metadata | metadata | shared | 36 | 20,201,472 |
| memory | metadata | chainable | 36 | 20,201,472 |
| metadata | recurrent | shared | 54 | 19,022,976 |
| metadata | metadata | chainable | 48 | 3,661,824 |
| metadata | matrix | shared | 18 | 3,194,880 |
| memory | metadata | shared | 18 | 1,769,472 |
| memory | convolution | shared | 18 | 1,769,472 |
| matrix | vector | shared | 78 | 1,425,408 |
| vector | metadata | shared | 54 | 1,327,104 |
| vector | matrix | shared | 144 | 1,179,648 |
| vector | vector | shared | 136 | 1,114,112 |
| vector | vector | chainable | 164 | 977,152 |
| matrix | metadata | chainable | 108 | 813,312 |

## Limitations

- Chains and transitions come from one captured graph per phase; a different prompt length keeps the same shapes-independent structure (see `graphs/prompt-comparison.md`) but was not re-checked here.
- `backend_fusion` reflects ggml's CPU scheduler on this host; a target accelerator's backend may fuse a different set of adjacent ops.
- Bytes are logical tensor sizes (`logical_bytes`), not measured DRAM traffic; views alias storage and kernels may reuse buffers the byte count does not know about.
- "Shared" edges are a lower bound on real materialization: dependency-graph fan-out does not capture scheduler reordering or buffer-lifetime overlap.
