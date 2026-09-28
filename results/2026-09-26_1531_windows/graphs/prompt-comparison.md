# Per-prompt graph comparison

Each row is one captured prefill + single-decode-step graph. **Op sequence** is the control flow (which operations run, in order); **structure** also includes data types and shapes. Prompts of equal token count share a structure; prompts of different length usually keep the same op sequence with different shapes. Token contents never change a graph.

## Flash attention on

| Prompt | Tokens | Context | Nodes (prefill/decode) | Matrix MACs prefill | Matrix MACs decode | Op sequence | Structure |
|---|---:|---:|---:|---:|---:|---|---|
| repetitive | 71 | 512 | 1411/1411 | 97,971,470,336 | 1,881,276,416 | seq-1 | shape-1 |
| narrative | 191 | 768 | 1411/1411 | 262,697,517,056 | 1,881,276,416 | seq-1 | shape-2 |

Distinct op sequences: 1. Distinct structures: 2.

## Flash attention off

| Prompt | Tokens | Context | Nodes (prefill/decode) | Matrix MACs prefill | Matrix MACs decode | Op sequence | Structure |
|---|---:|---:|---:|---:|---:|---|---|
| repetitive | 71 | 512 | 1441/1441 | 98,418,163,712 | 1,887,567,872 | seq-1 | shape-1 |
| narrative | 191 | 768 | 1441/1441 | 263,899,185,152 | 1,887,567,872 | seq-1 | shape-2 |

Distinct op sequences: 1. Distinct structures: 2.
