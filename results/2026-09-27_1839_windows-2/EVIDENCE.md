# Evidence set: 2026-09-27_1839_windows-2

This is the curated, committed part of one local run folder. The raw outputs stay local; they are listed with
SHA-256 in [`omitted-files.sha256`](omitted-files.sha256) (584 files, 160.6 MB) so any copy of them can be
checked. Provenance is in [`run-manifest.json`](run-manifest.json).

To browse it, open [`../index.html`](../index.html): one self-contained page with every graph's operation
sequence, offload and fusion tables, the prompt comparison, the timing summary and this run's write-up, plus
the graph SVGs committed under `graphs/`.

A second, interactive page models an accelerator on these graphs: `cd engine && python -m sbengine serve`
(see [`engine/README.md`](../../engine/README.md)). It reads the gzipped graph captures committed here. Its outputs
are analytical estimates under assumed hardware parameters, not measurements.

## What was run

The default `profile` command on one Windows host. Nothing here was run on Linux, macOS or Metal.

| | |
|---|---|
| Command | `windows\run.ps1 profile` (manifest: `cli.py --platform-dir '...\windows' profile`), no flags |
| Time | 2026-09-27 23:39:25 to 2026-09-28 00:17:33 UTC, manifest status `complete` |
| Host | Windows 11 (10.0.26200), Intel Core i7-13620H, 16 logical CPUs, 15.6 GiB RAM |
| Toolchain | LLVM clang 21.1.8, CMake 4.4.3, Ninja 1.13.2, Python 3.13.14, Graphviz 16.1.0 |
| Runtime | llama.cpp `f46bc30cb6a7f68a67e34a00061e20a4ad1eff43`, CPU backend, 6 threads, one sequence, flash attention on for timing runs, OpenMP not found (manifest `build.effective`) |
| Model | Qwen3.5-2B `Qwen3.5-2B-Q4_K_M.gguf` (mixed Q4_K_M; SHA-256 in the manifest) |

| Workload | Configuration | Result in this folder |
|---|---|---|
| Untraced timing (`cpu-baseline`) | prompt tokens 128, 512, 2048; 3 reps; 32 teacher-forced decode steps | `cpu-baseline.jsonl`, `analysis/summary.*` |
| Untraced timing at 8192 (`cpu-8k`) | 8192 tokens; 3 reps; 32 decode steps. Excluded from timing conclusions by the experiment's policy | `cpu-8k.jsonl`, `analysis/summary.*` |
| Traced CPU operations (`cpu-profile`) | 128, 512, 2048; 2 reps; 32 decode steps. **8192 not traced** (opt-in diagnostic, not run) | `cpu-profile.jsonl`, `analysis/operation-summary.*`, `analysis/profile-output-validation.json` (traced vs untraced logits bit-identical, 6 of 6 checks) |
| Decode cost vs context | from the runs above | `analysis/decode-curve.*` |
| Graph capture, fixed length | 128 and 512 tokens, context 4x, flash attention on and off. **No graph at 2048** | `graphs/pp{128,512}-fa-{on,off}/*.svg`, `../index.html` |
| Graph capture, per prompt | 8 prompts (71 to 628 tokens), flash attention on and off | `graphs/prompt-*/*.svg`, `../index.html` (includes the prompt comparison) |
| Per-prompt timing and trace | 8 prompts; untraced 3 reps, traced 2 reps; 32 decode steps | `prompts/<id>/cpu-*.jsonl`, `prompt-profiles.json` |

Not run: Linux, macOS, Metal, other thread counts, batch > 1, `--long-continuation`, the traced 8192 diagnostic.

## Added to this folder after the run finished

The run-folder rule is "never edited afterwards". These additions break it and are listed here instead
(times are file modification times, UTC):

| What | When | How |
|---|---|---|
| `graphs/pp8192-fa-on/`, `graphs/pp8192-fa-off/` | 2026-09-28 01:19 (on), 02:09 (off prefill); fa-off `decode.json` rewritten 2026-09-29 00:28 | `sb-capture-graph` at 8192 tokens, context 32768, 6 threads, same model file (from each `prefill.json` metadata). The exact command line was **not recorded**, and the manifest does not list these graphs. `core/experiments/qwen35-2b-q4km.json` was changed afterwards to add 8192 to `graphs.prompt_lengths`, so a new default run captures them itself. |
| Graph DOT/SVG re-render (omitted files) | 2026-09-29 00:24 | Command not recorded; `windows\run.ps1 analyze` is the tool that writes these, and it only reads the captured graph JSON |
| `graphs/*/FUSION-REPORT.md`, `graphs/*/OFFLOAD-REPORT.md` (omitted), `research/` | regenerated 2026-10-03 | `python core/analysis/compute_mapping.py results/2026-09-27_1839_windows-2` |
| `../index.html` | rebuilt 2026-10-03 | `python core/analysis/site.py results/2026-09-27_1839_windows-2 --no-raw-links` |

## What is committed (and what is not)

| Committed | Omitted (local, hashed in `omitted-files.sha256`) |
|---|---|
| `run-manifest.json`, `prompt-profiles.json` | saved logits `*.f32.gz` (needed only for the bit-identity check, already recorded in `analysis/profile-output-validation.json`) |
| measured timing records `cpu-baseline.jsonl`, `cpu-profile.jsonl`, `cpu-8k.jsonl`, `prompts/*/cpu-*.jsonl` | CPU operation traces `cpu-op-trace.jsonl.gz`, `prompts/*/cpu-op-trace.jsonl.gz` |
| `analysis/` summaries, CSVs, `decode-curve.png`, `profiling-summary.png`, `matrix-weight-inventory.json` | `analysis/operator-shapes.json`, `profiling-summary.pdf` (same figure as the PNG) |
| `graphs/`: the SVG renders (92 files, 17.3 MB): per-layer `{prefill,decode}.layer{0,3}.svg` for every graph and whole-graph `{prefill,decode}.svg` for pp8192; and the captured graphs gzipped, `{prefill,decode}.json.gz` (44 files, 2.5 MB, byte-identical to the local `.json` once decompressed), which the engine needs | everything else in `graphs/`: uncompressed graph JSON, `*.ops.csv`, `*.summary.json`, `*.dot`, per-graph `index.html`, `workload.json`, `REPORT.md`, `FUSION-REPORT.md`, `OFFLOAD-REPORT.md`, the fusion/offload JSON, `prompt-comparison.*`. Their content is embedded in `../index.html` |
| `research/compute-mapping.md`, `compute-mapping-index.json` | `*.log`, `*-prompt.txt`, `*-tokens.json` |

`../index.html` was built with `--no-raw-links`, so it links only to files that are committed (the SVGs and
the two `analysis/` PNGs).

## Regenerating the derived files

From the repository root, with the omitted raw files restored into this folder:

```
python core/analysis/compute_mapping.py results/2026-09-27_1839_windows-2   # fusion, offload, research/
windows\run.ps1 analyze results\2026-09-27_1839_windows-2                    # graph reports, SVGs, analysis/
python core/analysis/site.py results/2026-09-27_1839_windows-2 --no-raw-links  # results/index.html
```

`compute_mapping.py` reads only `graphs/*/{prefill,decode}.json`, `graphs/*/*.ops.csv`, `cpu-op-trace.jsonl.gz`,
`prompts/*/cpu-op-trace.jsonl.gz`, `analysis/summary.json`, `prompt-profiles.json` and `run-manifest.json`.
Without the raw files, a fresh `windows\run.ps1 profile` (or `unix/run.sh profile`) produces a new run folder
with the same file set; its numbers will differ by host and run.

## How to read the results

- Timings and operation shares are CPU wall-clock measurements on this host only. They are not comparable to
  other hosts or to the recorded 2026-09-22 Apple M5 Pro baseline, and they are not accelerator area,
  bandwidth or speedup.
- Graph-derived figures (MACs, boundary bytes, leaf-input bytes, largest intermediates) are logical sizes from
  the CPU-scheduled graph, not measured memory traffic.
- `research/compute-mapping.md` labels each statement as measured, structural or candidate. Its candidate
  boundaries are discussion points, not hardware recommendations. It does not establish SRAM traffic, FPGA or
  ASIC stalls, or the need for dedicated recurrence arithmetic.
