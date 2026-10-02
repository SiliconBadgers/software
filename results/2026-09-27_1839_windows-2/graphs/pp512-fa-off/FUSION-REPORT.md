# Operation-sharing evidence: chains and category boundaries

Workload: pp512-fa-off.

A **chain** is a maximal run of ops where each step's only computed input is the previous step's output, and that output has no other consumer -- the dataflow condition needed to keep an intermediate out of memory, not a claim about what any specific backend or accelerator fuses. A **category transition** is every scheduled producer-consumer edge, tagged `chainable` (meets that condition) or `shared` (the producer feeds more than one consumer, or the consumer merges several computed inputs, so materializing that edge cannot be avoided by fusion choices alone).

## prefill

783 chains covering 1441 scheduled ops.

### Highest-MAC chains

| Length | Matrix MACs | Categories | Backend-fused edges |
|---:|---:|---|---|
| 3 | 6,442,450,944 | matrix -> metadata | not checked |
| 1 | 6,442,450,944 | matrix | not checked |
| 1 | 6,442,450,944 | matrix | not checked |
| 2 | 6,442,450,944 | vector -> matrix | not checked |
| 3 | 6,442,450,944 | matrix -> metadata | not checked |
| 1 | 6,442,450,944 | matrix | not checked |
| 1 | 6,442,450,944 | matrix | not checked |
| 2 | 6,442,450,944 | vector -> matrix | not checked |
| 3 | 6,442,450,944 | matrix -> metadata | not checked |
| 1 | 6,442,450,944 | matrix | not checked |

### Category-boundary crossings (top by producer bytes)

| From | To | Kind | Edges | Producer bytes |
|---|---|---|---:|---:|
| matrix | vector | shared | 78 | 729,808,896 |
| vector | metadata | shared | 54 | 679,477,248 |
| vector | matrix | shared | 144 | 629,145,600 |
| vector | vector | shared | 136 | 570,425,344 |
| vector | vector | chainable | 164 | 500,301,824 |
| matrix | metadata | chainable | 108 | 416,415,744 |
| metadata | vector | chainable | 156 | 354,828,288 |
| metadata | memory | shared | 300 | 327,499,920 |
| vector | matrix | chainable | 30 | 327,155,712 |
| metadata | metadata | chainable | 48 | 270,532,608 |
| memory | metadata | shared | 18 | 227,819,520 |
| memory | convolution | shared | 18 | 227,819,520 |
| convolution | vector | chainable | 18 | 226,492,416 |
| recurrent | metadata | shared | 36 | 188,743,680 |
| vector | recurrent | shared | 54 | 151,584,768 |

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
| metadata | memory | shared | 300 | 95,170,704 |
| recurrent | metadata | shared | 36 | 38,043,648 |
| metadata | metadata | shared | 36 | 20,201,472 |
| memory | metadata | chainable | 36 | 20,201,472 |
| metadata | recurrent | shared | 54 | 19,022,976 |
| metadata | metadata | chainable | 48 | 9,953,280 |
| metadata | matrix | shared | 18 | 9,486,336 |
| memory | metadata | shared | 18 | 1,769,472 |
| memory | convolution | shared | 18 | 1,769,472 |
| matrix | vector | shared | 78 | 1,425,408 |
| vector | metadata | shared | 54 | 1,327,104 |
| vector | matrix | shared | 144 | 1,277,952 |
| vector | vector | shared | 136 | 1,114,112 |
| vector | vector | chainable | 164 | 977,152 |
| matrix | metadata | chainable | 108 | 813,312 |

## Limitations

- Chains and transitions come from one captured graph per phase; a different prompt length keeps the same shapes-independent structure (see `graphs/prompt-comparison.md`) but was not re-checked here.
- `backend_fusion` reflects ggml's CPU scheduler on this host; a target accelerator's backend may fuse a different set of adjacent ops.
- Bytes are logical tensor sizes (`logical_bytes`), not measured DRAM traffic; views alias storage and kernels may reuse buffers the byte count does not know about.
- "Shared" edges are a lower bound on real materialization: dependency-graph fan-out does not capture scheduler reordering or buffer-lifetime overlap.
