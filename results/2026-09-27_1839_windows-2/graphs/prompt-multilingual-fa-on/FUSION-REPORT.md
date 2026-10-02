# Operation-sharing evidence: chains and category boundaries

Workload: prompt-multilingual-fa-on.

A **chain** is a maximal run of ops where each step's only computed input is the previous step's output, and that output has no other consumer -- the dataflow condition needed to keep an intermediate out of memory, not a claim about what any specific backend or accelerator fuses. A **category transition** is every scheduled producer-consumer edge, tagged `chainable` (meets that condition) or `shared` (the producer feeds more than one consumer, or the consumer merges several computed inputs, so materializing that edge cannot be avoided by fusion choices alone).

## prefill

765 chains covering 1411 scheduled ops.

### Highest-MAC chains

| Length | Matrix MACs | Categories | Backend-fused edges |
|---:|---:|---|---|
| 3 | 1,912,602,624 | matrix -> metadata | 0/2 |
| 1 | 1,912,602,624 | matrix | 0/0 |
| 1 | 1,912,602,624 | matrix | 0/0 |
| 2 | 1,912,602,624 | vector -> matrix | 0/1 |
| 3 | 1,912,602,624 | matrix -> metadata | 0/2 |
| 1 | 1,912,602,624 | matrix | 0/0 |
| 1 | 1,912,602,624 | matrix | 0/0 |
| 2 | 1,912,602,624 | vector -> matrix | 0/1 |
| 3 | 1,912,602,624 | matrix -> metadata | 0/2 |
| 1 | 1,912,602,624 | matrix | 0/0 |

### Category-boundary crossings (top by producer bytes)

| From | To | Kind | Edges | Producer bytes |
|---|---|---|---:|---:|
| matrix | vector | shared | 78 | 216,662,016 |
| vector | metadata | shared | 54 | 201,719,808 |
| vector | matrix | shared | 138 | 171,835,392 |
| vector | vector | shared | 136 | 169,345,024 |
| metadata | memory | shared | 288 | 149,373,072 |
| vector | vector | chainable | 164 | 148,527,104 |
| metadata | vector | chainable | 156 | 119,543,808 |
| matrix | metadata | chainable | 102 | 116,152,320 |
| vector | matrix | chainable | 30 | 97,124,352 |
| recurrent | metadata | shared | 36 | 82,575,360 |
| metadata | metadata | chainable | 42 | 79,724,544 |
| memory | metadata | shared | 18 | 68,567,040 |
| memory | convolution | shared | 18 | 68,567,040 |
| convolution | vector | chainable | 18 | 67,239,936 |
| vector | recurrent | shared | 54 | 45,001,728 |

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
