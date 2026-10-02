# Per-prompt graph comparison

Each row is one captured prefill + single-decode-step graph. **Op sequence** is the control flow (which operations run, in order); **structure** also includes data types and shapes. Prompts of equal token count share a structure; prompts of different length usually keep the same op sequence with different shapes. Token contents never change a graph.

## Flash attention on

| Prompt | Tokens | Context | Nodes (prefill/decode) | Matrix MACs prefill | Matrix MACs decode | Op sequence | Structure |
|---|---:|---:|---:|---:|---:|---|---|
| repetitive | 71 | 512 | 1411/1411 | 97,971,470,336 | 1,881,276,416 | seq-1 | shape-1 |
| multilingual | 152 | 768 | 1411/1411 | 209,161,551,872 | 1,881,276,416 | seq-1 | shape-2 |
| narrative | 191 | 768 | 1411/1411 | 262,697,517,056 | 1,881,276,416 | seq-1 | shape-3 |
| hardware | 230 | 1024 | 1411/1411 | 316,233,482,240 | 1,881,276,416 | seq-1 | shape-4 |
| dialogue | 235 | 1024 | 1411/1411 | 323,097,067,520 | 1,881,276,416 | seq-1 | shape-5 |
| math | 264 | 1280 | 1411/1411 | 362,905,862,144 | 1,881,276,416 | seq-1 | shape-6 |
| code | 307 | 1280 | 1411/1411 | 421,932,695,552 | 1,881,276,416 | seq-1 | shape-7 |
| json_records | 628 | 2560 | 1411/1411 | 862,574,870,528 | 1,881,276,416 | seq-1 | shape-8 |

Distinct op sequences: 1. Distinct structures: 8.

## Flash attention off

| Prompt | Tokens | Context | Nodes (prefill/decode) | Matrix MACs prefill | Matrix MACs decode | Op sequence | Structure |
|---|---:|---:|---:|---:|---:|---|---|
| repetitive | 71 | 512 | 1441/1441 | 98,418,163,712 | 1,887,567,872 | seq-1 | shape-1 |
| multilingual | 152 | 768 | 1441/1441 | 210,117,853,184 | 1,887,567,872 | seq-1 | shape-2 |
| narrative | 191 | 768 | 1441/1441 | 263,899,185,152 | 1,887,567,872 | seq-1 | shape-3 |
| hardware | 230 | 1024 | 1441/1441 | 317,680,517,120 | 1,887,567,872 | seq-1 | shape-4 |
| dialogue | 235 | 1024 | 1441/1441 | 324,575,559,680 | 1,887,567,872 | seq-1 | shape-5 |
| math | 264 | 1280 | 1441/1441 | 366,227,750,912 | 1,893,859,328 | seq-1 | shape-6 |
| code | 307 | 1280 | 1441/1441 | 425,795,649,536 | 1,893,859,328 | seq-1 | shape-7 |
| json_records | 628 | 2560 | 1441/1441 | 874,427,973,632 | 1,900,150,784 | seq-1 | shape-8 |

Distinct op sequences: 1. Distinct structures: 8.
