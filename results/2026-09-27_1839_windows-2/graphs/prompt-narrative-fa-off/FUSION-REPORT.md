# Operation-sharing evidence: chains and category boundaries

Workload: prompt-narrative-fa-off.

A **chain** is a maximal run of ops where each step's only computed input is the previous step's output, and that output has no other consumer -- the dataflow condition needed to keep an intermediate out of memory, not a claim about what any specific backend or accelerator fuses. A **category transition** is every scheduled producer-consumer edge, tagged `chainable` (meets that condition) or `shared` (the producer feeds more than one consumer, or the consumer merges several computed inputs, so materializing that edge cannot be avoided by fusion choices alone).

## prefill

783 chains covering 1441 scheduled ops.

### Highest-MAC chains

| Length | Matrix MACs | Categories | Backend-fused edges |
|---:|---:|---|---|
| 3 | 2,403,336,192 | matrix -> metadata | not checked |
| 1 | 2,403,336,192 | matrix | not checked |
| 1 | 2,403,336,192 | matrix | not checked |
| 2 | 2,403,336,192 | vector -> matrix | not checked |
| 3 | 2,403,336,192 | matrix -> metadata | not checked |
| 1 | 2,403,336,192 | matrix | not checked |
| 1 | 2,403,336,192 | matrix | not checked |
| 2 | 2,403,336,192 | vector -> matrix | not checked |
| 3 | 2,403,336,192 | matrix -> metadata | not checked |
| 1 | 2,403,336,192 | matrix | not checked |

### Category-boundary crossings (top by producer bytes)

| From | To | Kind | Edges | Producer bytes |
|---|---|---|---:|---:|
| matrix | vector | shared | 78 | 272,252,928 |
| vector | metadata | shared | 54 | 253,476,864 |
| vector | matrix | shared | 144 | 225,312,768 |
| vector | vector | shared | 136 | 212,795,392 |
| vector | vector | chainable | 164 | 186,636,032 |
| metadata | memory | shared | 300 | 173,691,024 |
| matrix | metadata | chainable | 108 | 155,342,592 |
| metadata | vector | chainable | 156 | 145,032,960 |
| vector | matrix | chainable | 30 | 122,044,416 |
| metadata | metadata | chainable | 48 | 101,720,064 |
| recurrent | metadata | shared | 36 | 94,076,928 |
| memory | metadata | shared | 18 | 85,819,392 |
| memory | convolution | shared | 18 | 85,819,392 |
| convolution | vector | chainable | 18 | 84,492,288 |
| vector | recurrent | shared | 54 | 56,548,224 |

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
| metadata | memory | shared | 300 | 87,306,384 |
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
