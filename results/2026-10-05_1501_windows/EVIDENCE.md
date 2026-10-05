# Evidence set: 2026-10-05_1501_windows

This is the curated, committed part of one local run folder. The raw outputs stay local; they are listed with
SHA-256 in [`omitted-files.sha256`](omitted-files.sha256) (430 files, 147.1 MB) so any copy of them can be
checked. Provenance is in [`run-manifest.json`](run-manifest.json).

It is a second Windows host running the same default `profile` command as
[`2026-09-27_1839_windows-2`](../2026-09-27_1839_windows-2/EVIDENCE.md), with the same pinned source, model,
prompts and settings. It adds no new workload: no long continuation, no batch above 1.

[`../index.html`](../index.html) and the `engine/` defaults still describe the 2026-09-27 run; neither was
rebuilt or repointed for this one.

## What was run

The default `profile` command on one Windows host. Nothing here was run on Linux, macOS or Metal.

| | |
|---|---|
| Command | `windows\run.ps1 profile` (manifest: `cli.py --platform-dir '...\windows' profile`), no flags. `SB_CC` / `SB_CXX` pointed at the clang bundled with Visual Studio 2022 |
| Time | 2026-10-05 20:01:51 to 20:15:09 UTC, manifest status `complete` |
| Host | Windows 11 (10.0.26200), AMD Ryzen 7 8840U w/ Radeon 780M Graphics, 16 logical CPUs, 29.8 GiB RAM |
| Toolchain | LLVM clang 18.1.8 (Visual Studio 2022 component), CMake 4.4.3, Ninja 1.13.2, Python 3.13.4. Graphviz not installed |
| Runtime | llama.cpp `f46bc30cb6a7f68a67e34a00061e20a4ad1eff43`, CPU backend, 6 threads, one sequence, flash attention on for timing runs, OpenMP not found (manifest `build.effective`) |
| Model | Qwen3.5-2B `Qwen3.5-2B-Q4_K_M.gguf` (mixed Q4_K_M; SHA-256 `aaf42c8b...9223`, in the manifest) |

| Workload | Configuration | Result in this folder |
|---|---|---|
| Untraced timing (`cpu-baseline`) | prompt tokens 128, 512, 2048; 3 reps; 32 teacher-forced decode steps | `cpu-baseline.jsonl`, `analysis/summary.*` |
| Untraced timing at 8192 (`cpu-8k`) | 8192 tokens; 3 reps; 32 decode steps. Excluded from timing conclusions by the experiment's policy | `cpu-8k.jsonl`, `analysis/summary.*` |
| Traced CPU operations (`cpu-profile`) | 128, 512, 2048; 2 reps; 32 decode steps. **8192 not traced** (opt-in diagnostic, not run) | `cpu-profile.jsonl`, `analysis/operation-summary.*`, `analysis/profile-output-validation.json` |
| Decode cost vs context | from the runs above | `analysis/decode-curve.*` |
| Graph capture, fixed length | 128, 512 and 8192 tokens, context 4x, flash attention on and off, all recorded in the manifest. **No graph at 2048** | not committed (see below) |
| Graph capture, per prompt | 8 prompts (71 to 628 tokens), flash attention on and off | not committed (see below) |
| Per-prompt timing and trace | 8 prompts; untraced 3 reps, traced 2 reps; 32 decode steps | `prompts/<id>/cpu-*.jsonl`, `prompt-profiles.json` |

Not run: Linux, macOS, Metal, other thread counts, batch > 1, `--long-continuation`, the traced 8192 diagnostic,
`core/analysis/compute_mapping.py` (so there is no `research/` write-up for this run) and `core/analysis/site.py`.

A two-minute `--smoke` run on the same host preceded this one (`2026-10-05_1454_windows`); it is not committed.

## Added to this folder after the run finished

| What | When | How |
|---|---|---|
| `EVIDENCE.md` | 2026-10-05 | Written from `run-manifest.json` and the `analysis/` files |
| `omitted-files.sha256` | 2026-10-05 | SHA-256 of every file in the folder that `.gitignore` excludes, sorted by path |

Nothing else in the folder was added, edited or regenerated.

## What is committed (and what is not)

| Committed (32 files, 0.5 MB, plus this file and the hash list) | Omitted (local, hashed in `omitted-files.sha256`) |
|---|---|
| `run-manifest.json`, `prompt-profiles.json` | saved logits `*.f32.gz` (46 files) |
| measured timing records `cpu-baseline.jsonl`, `cpu-profile.jsonl`, `cpu-8k.jsonl`, `prompts/*/cpu-*.jsonl` | CPU operation traces `cpu-op-trace.jsonl.gz`, `prompts/*/cpu-op-trace.jsonl.gz` (9 files) |
| `analysis/` summaries, CSVs, `decode-curve.png`, `profiling-summary.png`, `matrix-weight-inventory.json` | `analysis/operator-shapes.json`, `profiling-summary.pdf` (same figure as the PNG) |
| | all of `graphs/` (332 files): captured graph JSON, `*.ops.csv`, `*.summary.json`, `*.dot`, per-graph `index.html` and reports, `prompt-comparison.*` |
| | `*.log`, `*-prompt.txt`, `*-tokens.json` |

**Why no graphs are committed.** Graphviz was not installed, so no SVG was rendered. The captured graphs
themselves duplicate the committed 2026-09-27 captures: for all 44 files (22 workloads, prefill and decode),
`scheduled_nodes` and `tensors` are equal to that run's `{prefill,decode}.json.gz`, and the metadata differs
only in the local `model_file` path (checked 2026-10-05 by loading both and comparing). Committing them would
also make `engine/` pick this run as its default, since it reads the newest run that has a `graphs/` folder.

## Checks

- Traced and untraced logits are bit-identical at all 6 fixed-length checkpoints
  (`analysis/profile-output-validation.json`) and for all 8 prompts (`prompt-profiles.json`, `checks`). This
  covers those checkpoints, not every intermediate value.
- `analysis/cpu-metal-logit-check.json` is empty: there is no Metal backend on Windows.
- The fixed-length input tokens and prompt text (`cpu-baseline-tokens.json`, `cpu-baseline-prompt.txt`) are
  byte-identical to the 2026-09-27 run's (same SHA-256 as in that run's `omitted-files.sha256`). The saved logits are **not** byte-identical to that host's. Its logits
  are not committed, so the size of the difference is unknown.
- No task-quality or accuracy evaluation was done.

## Measured on this host

Untraced, median of 3 repetitions (`analysis/summary.csv`):

| Prompt tokens | Prefill, s | Prefill tokens/s | Decode, ms/token | Decode tokens/s |
|---:|---:|---:|---:|---:|
| 128 | 0.798 | 160.4 | 29.55 | 33.8 |
| 512 | 3.344 | 153.1 | 30.11 | 33.2 |
| 2,048 | 13.344 | 153.5 | 30.47 | 32.8 |
| 8,192 (excluded by policy) | 59.581 | 137.5 | 37.04 | 27.0 |

Share of timed CPU operation intervals at 2,048 tokens, mean of the 2 traced repetitions
(`analysis/operation-summary.csv`):

| Work | Prefill | Decode |
|---|---:|---:|
| MLP projections | 37.7% | 35.6% |
| Attention/DeltaNet projections | 34.4% | 21.5% |
| Vocabulary output head | 0.1% | 24.7% |
| Copies/gather/state movement | 16.4% | 7.7% |
| Attention QK/softmax/AV | 2.8% | 5.9% |
| DeltaNet recurrence | 3.9% | 2.1% |
| Vector/norm/activation/other | 3.0% | 2.2% |
| Convolution | 1.8% | 0.4% |

Things to keep in mind when reading these:

- **Traced runs are slower than untraced ones here.** Traced / untraced: prefill x1.06, x1.15, x1.12 and decode
  x1.25, x1.20, x1.23 at 128, 512 and 2,048 tokens. The prefill shares above come from the slower traced run.
- **The copy share is one operation.** At 512-token prefill the `Copies/gather/state movement` group takes
  0.90 s (23.5%), and almost all of it is the `CONCAT` that builds each DeltaNet layer's convolution input:
  0.897 s across the 18 layers, against 0.080 s at 128 tokens. That is 11x the time for 4x the data (12.1 MiB
  per tensor against 3.1 MiB). The cause was not investigated; a cache-capacity effect is a guess.
- **`engine/`'s accounting check does not carry over to this host for prefill.** `python -m sbengine validate-cpu`
  on this run (it needs the local traces): held-out prefill mean error 16.8% (max 21.4%, n = 7), decode 1.3%
  (max 6.2%, n = 8), against 2.2% and 7.1% reported for the 2026-09-27 run in `engine/docs/VALIDATION.md`. The
  prefill error is the `CONCAT` non-linearity above; a linear fit through 128 and 512 tokens over-predicts the
  prompts in between.

## Regenerating the derived files

From the repository root, with the omitted raw files restored into this folder:

```
windows\run.ps1 analyze results\2026-10-05_1501_windows
```

Without the raw files, a fresh `windows\run.ps1 profile` produces a new run folder with the same file set; its
numbers will differ by host and run.

## How to read the results

- Timings and operation shares are CPU wall-clock measurements on this host only. They are not comparable to
  other hosts or to the recorded 2026-09-22 Apple M5 Pro baseline, and they are not accelerator area,
  bandwidth or speedup.
- For the same reason this run is not a benchmark against the 2026-09-27 host. One cross-host observation is
  recorded only as a question to follow up: at 2,048-token prefill the copy group took 2.44-2.46 s here and
  2.42-2.46 s there, while total timed prefill was about 15 s here and 48 s there. Whether that work is
  memory-bound has not been tested.
- Mixed Q4_K_M results do not validate a custom uniform INT4/group-64 format.
