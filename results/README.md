# results

One folder per profiling run, written by `unix/run.sh profile` or `windows/run.ps1 profile`.

```
results/2026-09-25_1430_macos/      <date>_<HHMM>_<os>   os = macos | linux | windows
  run-manifest.json                 host, toolchain, pins, config, command, status
  cpu-baseline.jsonl / .log / *-prompt.txt / *-tokens.json
  cpu-baseline-p128-prefill.f32.gz  saved logits (also -decode)
  cpu-profile*.jsonl, cpu-op-trace.jsonl.gz     instrumented CPU run and operation trace
  metal-baseline*                   only on macOS/arm64
  cuda-cpu-baseline*                zero-offload matched control from the CUDA build (`--cuda on`)
  cuda-cpu-no-host-baseline*        optional zero-offload repacking control (`--cuda-no-host-control`)
  cuda-baseline*                    CUDA execution (`--cuda on` or the fast `--cuda only` path)
  analysis/                         summary.json, operation-summary.json, validation JSONs, profiling-summary.png/.pdf,
                                    decode-curve.json/.csv/.png (decode ms per token vs tokens in the KV cache)
  graphs/pp128-fa-on/               scheduled dataflow graph, flash attention on (matches the profiled run)
  graphs/pp128-fa-off/              same, flash attention off (QK/softmax/AV stay visible)
                                    prefill.json decode.json *.dot *.ops.csv REPORT.md index.html [layer SVGs]
  graphs/prompt-<id>-fa-on|off/     one graph per prompt (8 prompts), natural token count, same file set
  graphs/prompt-comparison.md/.json which prompts share a graph structure or an operation sequence
  prompts/<id>/                     per prompt: CPU plus requested GPU baselines, CPU trace, saved logits
  prompt-profiles.json              one row per prompt: timings, trace breakdown, overhead and logit checks, graph pointers
```

The raw file names match the recorded `experiments/llama-cpp/2026-09-22/results/`, so the same analysis code reads both.

## Rules

- A run folder is never reused or edited afterwards. Two runs in the same minute get `-2`, `-3`.
  Exception: `core/runner/cli.py analyze` and `core/analysis/compute_mapping.py` write derived files into an
  existing run; anything else added later must be listed in that run's `EVIDENCE.md`.
- `run-manifest.json` is the provenance. The folder name is only a label; read the manifest for the
  host, compiler, pinned llama.cpp commit, model hash and exact command.
- `datasets` associates each result series with its role, build and measured placement for each case.
  Use `core/analysis/datasets.py` when reading historical or split-build runs; filenames alone do not
  prove CPU-reference identity. `analysis/backend-speedups.{json,csv}` names every speedup denominator.
- **Runs from different hosts are not comparable** to each other or to the recorded Apple M5 Pro
  baseline. CPU operation shares do not describe accelerator area, bandwidth or speedup.
- Model weights never go here (they live in the gitignored `work/`).
- The 2026-09-22 baseline stays in `experiments/` and is not copied here.

## What is committed

Run folders are written locally in full, but only a **curated evidence set** is committed: the manifest,
the measured timing records (`*.jsonl`), `analysis/` summaries and figures, `prompt-profiles.json`, any
`research/` write-up, and from `graphs/` **only the SVG renders and the gzipped graph captures**
(`{prefill,decode}.json.gz`, needed by `engine/`). Everything else (saved logits, operation
traces, captured graph JSON, DOT, per-op CSVs, per-graph reports, logs, tokens and prompt copies) stays local;
`.gitignore` excludes it.

[`index.html`](index.html) is the committed browser for the run: a self-contained page (built by
`core/analysis/site.py --no-raw-links`) with each graph's operation sequence, offload and fusion tables, the
prompt comparison and the write-up embedded, and the committed SVGs linked by relative path.

The second page is the interactive accelerator model in `engine/` (`cd engine && python -m sbengine serve`,
then http://127.0.0.1:8765). It re-runs the engine on the committed `.json.gz` graphs as you change its
controls. Its numbers are estimates under assumed hardware parameters, not measurements.

Each committed run has an `EVIDENCE.md` that lists what is included, what was omitted (with SHA-256 in
`omitted-files.sha256`, so a copy of the raw outputs can be checked against it), anything added to the folder
after the run finished, and the commands that regenerate the derived files. Development and smoke runs are
not committed.
