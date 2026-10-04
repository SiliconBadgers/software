# Qwen3.5-2B operation dependency map

October 2, 2026. Part of [issue #3](https://github.com/SiliconBadgers/software/issues/3):
operation chains, dependencies, persistent-state access and fusion/parallel
candidates for the captured llama.cpp graphs.

> [!IMPORTANT]
> This is a **structural analysis of captured dataflow graphs**. It contains no
> timing, no measured memory traffic and no accelerator estimate. Byte counts are
> logical tensor sizes from the capture. Fusion groups are candidates allowed by
> the dependencies, not a claim that a given kernel or hardware unit implements
> them.

## Inputs

The four graphs from the [2026-09-24 Qwen3.5-2B capture](../../../experiments/llama-cpp/2026-09-24-qwen35-2b/README.md)
(read only; nothing under `experiments/` is modified):

| Graph | Input tokens | Past tokens | SHA-256 (first 12) |
|---|---:|---:|---|
| `graphs/pp128/decode.json` | 1 | 128 | `39cb60765c88` |
| `graphs/pp512/decode.json` | 1 | 512 | `b690ad7db798` |
| `graphs/pp128/prefill.json` | 128 | 0 | `5a1a56b6f4e2` |
| `graphs/pp512/prefill.json` | 512 | 0 | `deb6604ff967` |

Full hashes are in each `out/<run>/summary.json`. These are **BF16 CPU graphs
with flash attention off**. The [2026-09-22 timing baseline](../../../experiments/llama-cpp/2026-09-22/REPORT.md)
used Q4_K_M with flash attention on, so its time shares do not map one-to-one
onto these graphs.

## Reproduce

Python 3.9 or newer, standard library only. From the repository root:

```sh
G=experiments/llama-cpp/2026-09-24-qwen35-2b/graphs
python research/compute-mapping/2026-10-02-op-dependency-map/build_map.py \
  $G/pp128/decode.json $G/pp128/prefill.json $G/pp512/decode.json $G/pp512/prefill.json
```

Each graph writes `out/<workload>-<phase>/`. The script prints its validation
checks and exits non-zero if any fail. It refuses to write under `experiments/`.
Checked with Python 3.12.6 and 3.9.0 on Windows 11.

## Outputs

| File | Contents |
|---|---|
| `nodes.csv` | Every operation that does work: layer, layer type, category, shape (ggml order), dtype, bytes, GEMM M/K/N, MACs, weights, state access, unit, fusion group |
| `edges.csv` | Data edges after collapsing views, plus `state_raw`/`state_war`/`state_waw` ordering edges for in-place cache and state writes |
| `chain_<type>.csv` | The operation chain of one representative layer per type (`deltanet`, `full_attention`, `embedding`, `output_head`). `#k` refers to step k in the same file; `W:`, `S:` and `I:` are weight, persistent-state and index/mask inputs |
| `parallel_groups.csv` | Weight matrix multiplies in one layer that read the same activation |
| `fusion_groups.csv` | Fusion candidates with their pattern, categories, internal bytes and handoffs removed |
| `handoffs.csv` | Edges between different categories, split into `fused`, `remaining` and `inter-layer` |
| `summary.json` | Counts, per-layer-type totals, validation results and method notes |

## Method

1. **Collapse metadata.** `VIEW`, `RESHAPE`, `PERMUTE` and `TRANSPOSE` do no
   work; edges pass through them to the real producer. Zero-sized nodes (batch-1
   bookkeeping for multi-sequence state copies) are dropped the same way.
2. **Add hidden ordering.** `CPY`/`SET_ROWS` write into persistent input buffers
   (`cache_k_lN`, `cache_v_lN`, `cache_r_lN`, `cache_s_lN`) through views, so no
   source edge links the write to later reads. The script adds read-after-write,
   write-after-read and write-after-write edges per buffer, at whole-buffer
   granularity.
3. **Categorize.** `weight_matrix` (MUL_MAT with a weight), `act_matrix`
   (QK and AV against the KV cache), `conv`, `recurrent` (GATED_DELTA_NET),
   `memory` (CPY, SET_ROWS, GET_ROWS, CONCAT, CONT) and `vector` (everything
   else).
4. **Parallel groups.** Weight MUL_MATs in one layer that share an input could
   be stacked into one larger multiply.
5. **Fusion groups.** Merge producer→consumer when the producer has exactly one
   consumer in the same layer, allowing at most one anchor unit per group (a
   parallel group, the QK/AV attention core, or one matrix/conv/recurrent op).
   Vector ops merge in schedule order, so an op between two anchors joins the
   earlier anchor as its epilogue.
6. **Check.** Every layer of a type has the same structure; the captured
   schedule respects every data and state edge; every op is in exactly one
   fusion group; every written buffer has ordering edges; MAC totals equal the
   capture's own `*.summary.json`.

All checks pass for all four graphs.

## Results

All 18 DeltaNet layers are structurally identical, as are all 6 full-attention
layers. The tables below are per layer.

| | DeltaNet | Full attention |
|---|---:|---:|
| Operations that do work (decode / prefill) | 35 / 37 | 29 / 29 |
| Weight MUL_MATs | 8 | 7 |
| Parallel groups | 2 | 2 |
| Fusion groups (decode / prefill) | 13 / 15 | 9 |
| Cross-category edges before → after fusion | 25 → 13 | 20 → 7 |
| Persistent state read per token (decode) | 1.07 MiB | 0.5 MiB at 128 past, 1.5 MiB at 512 past |
| Persistent state written per token (decode) | 1.07 MiB | 4 KiB |

**Parallel groups.** Every DeltaNet layer has `attn_qkv` (6144 rows) + `ssm_alpha`
(16) + `ssm_beta` (16) + `attn_gate` (2048) on `attn_norm`, 8,224 rows in total.
Every attention layer has `attn_q` (4096) + `attn_k` (512) + `attn_v` (512), 5,120
rows. Every layer has `ffn_gate` + `ffn_up` on `attn_post_norm`, 12,288 rows.
Stacking them means the 16-row alpha/beta projections need not run alone on a
large matrix array.

**DeltaNet state.** Each token, each DeltaNet layer reads its 1 MiB recurrent
state through `GET_ROWS` and writes it back through `CPY`, plus 72 KiB of
convolution state. Across 18 layers that is about 19 MiB read and 19 MiB written
per token (logical). The `GET_ROWS` fuses into the recurrence as a prologue; the
write-back `CPY` stays separate because the recurrence output also feeds the
output norm. A recurrence unit that updates state in place would remove both
copies. In prefill, llama.cpp first clears each state buffer in place (a
`SCALE` by zero), which adds the write-after-write edges.

**Attention core.** `q_norm → RoPE → QK → softmax → AV → gate` forms one fusion
group (the flash-attention pattern plus Qwen's output gate). K and V each fuse
with their cache write (`SET_ROWS`). Decode reads 256 cache positions at 128
past tokens and 768 at 512, which is more than the live tokens because llama.cpp
pads the KV view.

**Remaining handoffs.** After fusion, most cross-category edges left are a
vector result feeding a weight multiply: 7 per DeltaNet layer and 6 per
attention layer. In DeltaNet, 4 are `attn_norm` feeding the four stacked
projections (they land in different fusion groups because each has a different
epilogue), 1 is `attn_post_norm` feeding gate/up, and 2 are an epilogue result
feeding the next projection (gated output → `ssm_out`, SwiGLU → `ffn_down`).
Attention has the same pattern with three Q/K/V projections, plus one
matrix→vector edge from `attn_q` into the attention core. DeltaNet also keeps 3
vector→recurrence, 1 recurrence→vector, 1 recurrence→memory (state write-back)
and 1 memory→conv edge. If matrix and vector work run on separate units, these
are the synchronization points per layer.

**Prefill vs decode.** The fusion structure is the same; the bytes inside fused
groups grow with token count (DeltaNet: 1.3 MB at decode, 26 MB at 128 tokens,
102 MB at 512 tokens per layer). Fusion mainly removes handoffs and launches in
decode and removes real intermediate bytes in prefill.

## Limitations

- CPU-selected GGML graph, BF16 weights, flash attention off, batch 1. Another
  backend may lower or fuse operations differently.
- Logical bytes are not DRAM or SRAM traffic. Views, caches and reuse change
  physical traffic.
- State ordering is whole-buffer and therefore conservative.
- The fusion rule is one deterministic policy. A different merge order or
  anchor definition gives different groups, and fusibility here says nothing
  about hardware cost.
- Only the text trunk is covered: no vision, multi-token prediction,
  tokenizer or sampling.
- Buffer lifetimes and peak live memory (step 3) and a prefill/decode
  comparison report are not yet produced.
