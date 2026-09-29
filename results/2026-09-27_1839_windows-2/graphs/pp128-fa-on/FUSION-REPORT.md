# Operation-sharing evidence: chains and category boundaries

Workload: pp128-fa-on.

A **chain** is a maximal run of ops where each step's only computed input is the previous step's output, and that output has no other consumer -- the dataflow condition needed to keep an intermediate out of memory, not a claim about what any specific backend or accelerator fuses. A **category transition** is every scheduled producer-consumer edge, tagged `chainable` (meets that condition) or `shared` (the producer feeds more than one consumer, or the consumer merges several computed inputs, so materializing that edge cannot be avoided by fusion choices alone).

## prefill

765 chains covering 1411 scheduled ops.

### Highest-MAC chains

| Length | Matrix MACs | Categories | Backend-fused edges |
|---:|---:|---|---|
| 3 | 1,610,612,736 | matrix -> metadata | 0/2 |
| 1 | 1,610,612,736 | matrix | 0/0 |
| 1 | 1,610,612,736 | matrix | 0/0 |
| 2 | 1,610,612,736 | vector -> matrix | 0/1 |
| 3 | 1,610,612,736 | matrix -> metadata | 0/2 |
| 1 | 1,610,612,736 | matrix | 0/0 |
| 1 | 1,610,612,736 | matrix | 0/0 |
| 2 | 1,610,612,736 | vector -> matrix | 0/1 |
| 3 | 1,610,612,736 | matrix -> metadata | 0/2 |
| 1 | 1,610,612,736 | matrix | 0/0 |

### Category-boundary crossings (top by producer bytes)

| From | To | Kind | Edges | Producer bytes |
|---|---|---|---:|---:|
| matrix | vector | shared | 78 | 182,452,224 |
| vector | metadata | shared | 54 | 169,869,312 |
| vector | matrix | shared | 138 | 144,703,488 |
| vector | vector | shared | 136 | 142,606,336 |
| metadata | memory | shared | 288 | 138,756,240 |
| vector | vector | chainable | 164 | 125,075,456 |
| metadata | vector | chainable | 156 | 103,858,176 |
| matrix | metadata | chainable | 102 | 97,812,480 |
| vector | matrix | chainable | 30 | 81,788,928 |
| recurrent | metadata | shared | 36 | 75,497,472 |
| metadata | metadata | chainable | 42 | 67,633,152 |
| memory | metadata | shared | 18 | 57,950,208 |
| memory | convolution | shared | 18 | 57,950,208 |
| convolution | vector | chainable | 18 | 56,623,104 |
| vector | recurrent | shared | 54 | 37,896,192 |

## decode

765 chains covering 1411 scheduled ops.

### Highest-MAC chains

| Length | Matrix MACs | Categories | Backend-fused edges |
|---:|---:|---|---|
| 5 | 508,559,360 | vector -> memory -> matrix | 1/4 |
| 3 | 12,582,912 | matrix -> metadata | 0/2 |
| 1 | 12,582,912 | matrix | 0/0 |
| 1 | 12,582,912 | matrix | 0/0 |
| 2 | 12,582,912 | vector -> matrix | 0/1 |
| 3 | 12,582,912 | matrix -> metadata | 0/2 |
| 1 | 12,582,912 | matrix | 0/0 |
| 1 | 12,582,912 | matrix | 0/0 |
| 2 | 12,582,912 | vector -> matrix | 0/1 |
| 3 | 12,582,912 | matrix -> metadata | 0/2 |

### Category-boundary crossings (top by producer bytes)

| From | To | Kind | Edges | Producer bytes |
|---|---|---|---:|---:|
| metadata | memory | shared | 288 | 82,575,504 |
| recurrent | metadata | shared | 36 | 38,043,648 |
| metadata | metadata | shared | 36 | 20,201,472 |
| memory | metadata | chainable | 36 | 20,201,472 |
| metadata | recurrent | shared | 54 | 19,022,976 |
| metadata | metadata | chainable | 42 | 3,649,536 |
| metadata | vector | shared | 42 | 3,391,488 |
| memory | metadata | shared | 18 | 1,769,472 |
| memory | convolution | shared | 18 | 1,769,472 |
| matrix | vector | shared | 78 | 1,425,408 |
| vector | metadata | shared | 54 | 1,327,104 |
| vector | matrix | shared | 138 | 1,130,496 |
| vector | vector | shared | 136 | 1,114,112 |
| vector | vector | chainable | 164 | 977,152 |
| matrix | metadata | chainable | 102 | 764,160 |

## Limitations

- Chains and transitions come from one captured graph per phase; a different prompt length keeps the same shapes-independent structure (see `graphs/prompt-comparison.md`) but was not re-checked here.
- `backend_fusion` reflects ggml's CPU scheduler on this host; a target accelerator's backend may fuse a different set of adjacent ops.
- Bytes are logical tensor sizes (`logical_bytes`), not measured DRAM traffic; views alias storage and kernels may reuse buffers the byte count does not know about.
- "Shared" edges are a lower bound on real materialization: dependency-graph fan-out does not capture scheduler reordering or buffer-lifetime overlap.
