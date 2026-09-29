# Operation-sharing evidence: chains and category boundaries

Workload: prompt-hardware-fa-on.

A **chain** is a maximal run of ops where each step's only computed input is the previous step's output, and that output has no other consumer -- the dataflow condition needed to keep an intermediate out of memory, not a claim about what any specific backend or accelerator fuses. A **category transition** is every scheduled producer-consumer edge, tagged `chainable` (meets that condition) or `shared` (the producer feeds more than one consumer, or the consumer merges several computed inputs, so materializing that edge cannot be avoided by fusion choices alone).

## prefill

765 chains covering 1411 scheduled ops.

### Highest-MAC chains

| Length | Matrix MACs | Categories | Backend-fused edges |
|---:|---:|---|---|
| 3 | 2,894,069,760 | matrix -> metadata | 0/2 |
| 1 | 2,894,069,760 | matrix | 0/0 |
| 1 | 2,894,069,760 | matrix | 0/0 |
| 2 | 2,894,069,760 | vector -> matrix | 0/1 |
| 3 | 2,894,069,760 | matrix -> metadata | 0/2 |
| 1 | 2,894,069,760 | matrix | 0/0 |
| 1 | 2,894,069,760 | matrix | 0/0 |
| 2 | 2,894,069,760 | vector -> matrix | 0/1 |
| 3 | 2,894,069,760 | matrix -> metadata | 0/2 |
| 1 | 2,894,069,760 | matrix | 0/0 |

### Category-boundary crossings (top by producer bytes)

| From | To | Kind | Edges | Producer bytes |
|---|---|---|---:|---:|
| matrix | vector | shared | 78 | 327,843,840 |
| vector | metadata | shared | 54 | 305,233,920 |
| vector | matrix | shared | 138 | 260,014,080 |
| vector | vector | shared | 136 | 256,245,760 |
| vector | vector | chainable | 164 | 224,744,960 |
| metadata | memory | shared | 288 | 183,877,776 |
| matrix | metadata | chainable | 102 | 175,756,800 |
| metadata | vector | chainable | 156 | 170,522,112 |
| vector | matrix | chainable | 30 | 146,964,480 |
| metadata | metadata | chainable | 42 | 119,021,568 |
| recurrent | metadata | shared | 36 | 105,578,496 |
| memory | metadata | shared | 18 | 103,071,744 |
| memory | convolution | shared | 18 | 103,071,744 |
| convolution | vector | chainable | 18 | 101,744,640 |
| vector | recurrent | shared | 54 | 68,094,720 |

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
