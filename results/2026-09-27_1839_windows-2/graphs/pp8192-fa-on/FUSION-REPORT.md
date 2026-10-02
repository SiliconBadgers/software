# Operation-sharing evidence: chains and category boundaries

Workload: pp8192-fa-on.

A **chain** is a maximal run of ops where each step's only computed input is the previous step's output, and that output has no other consumer -- the dataflow condition needed to keep an intermediate out of memory, not a claim about what any specific backend or accelerator fuses. A **category transition** is every scheduled producer-consumer edge, tagged `chainable` (meets that condition) or `shared` (the producer feeds more than one consumer, or the consumer merges several computed inputs, so materializing that edge cannot be avoided by fusion choices alone).

## prefill

765 chains covering 1411 scheduled ops.

### Highest-MAC chains

| Length | Matrix MACs | Categories | Backend-fused edges |
|---:|---:|---|---|
| 3 | 103,079,215,104 | matrix -> metadata | not checked |
| 1 | 103,079,215,104 | matrix | not checked |
| 1 | 103,079,215,104 | matrix | not checked |
| 2 | 103,079,215,104 | vector -> matrix | not checked |
| 3 | 103,079,215,104 | matrix -> metadata | not checked |
| 1 | 103,079,215,104 | matrix | not checked |
| 1 | 103,079,215,104 | matrix | not checked |
| 2 | 103,079,215,104 | vector -> matrix | not checked |
| 3 | 103,079,215,104 | matrix -> metadata | not checked |
| 1 | 103,079,215,104 | matrix | not checked |

### Category-boundary crossings (top by producer bytes)

| From | To | Kind | Edges | Producer bytes |
|---|---|---|---:|---:|
| matrix | vector | shared | 78 | 11,676,942,336 |
| vector | metadata | shared | 54 | 10,871,635,968 |
| vector | matrix | shared | 138 | 9,261,023,232 |
| vector | vector | shared | 136 | 9,126,805,504 |
| vector | vector | chainable | 164 | 8,004,829,184 |
| matrix | metadata | chainable | 102 | 6,259,998,720 |
| metadata | vector | chainable | 156 | 5,374,230,528 |
| vector | matrix | chainable | 30 | 5,234,491,392 |
| metadata | metadata | chainable | 42 | 4,227,858,432 |
| metadata | memory | shared | 288 | 3,706,011,792 |
| memory | metadata | shared | 18 | 3,625,205,760 |
| memory | convolution | shared | 18 | 3,625,205,760 |
| convolution | vector | chainable | 18 | 3,623,878,656 |
| recurrent | metadata | shared | 36 | 2,453,667,840 |
| vector | recurrent | shared | 54 | 2,425,356,288 |

## decode

765 chains covering 1411 scheduled ops.

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
| metadata | metadata | chainable | 42 | 104,312,832 |
| metadata | vector | shared | 42 | 104,054,784 |
| metadata | memory | shared | 288 | 82,575,504 |
| recurrent | metadata | shared | 36 | 38,043,648 |
| metadata | metadata | shared | 36 | 20,201,472 |
| memory | metadata | chainable | 36 | 20,201,472 |
| metadata | recurrent | shared | 54 | 19,022,976 |
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
