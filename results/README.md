# results

One folder per profiling run, written by `unix/run.sh profile` or `windows/run.ps1 profile`.

```
results/2026-09-25_1430_macos/      <date>_<HHMM>_<os>   os = macos | linux | windows
  run-manifest.json                 host, toolchain, pins, config, command, status
  cpu-baseline.jsonl / .log / *-prompt.txt / *-tokens.json
  cpu-baseline-p128-prefill.f32.gz  saved logits (also -decode)
  cpu-profile*.jsonl, cpu-op-trace.jsonl.gz     instrumented CPU run and operation trace
  metal-baseline*                   only on macOS/arm64
  analysis/                         summary.json, operation-summary.json, validation JSONs, profiling-summary.png/.pdf,
                                    decode-curve.json/.csv/.png (decode ms per token vs tokens in the KV cache)
  graphs/pp128-fa-on/               scheduled dataflow graph, flash attention on (matches the profiled run)
  graphs/pp128-fa-off/              same, flash attention off (QK/softmax/AV stay visible)
                                    prefill.json decode.json *.dot *.ops.csv REPORT.md index.html [layer SVGs]
  graphs/prompt-<id>-fa-on|off/     one graph per prompt (8 prompts), natural token count, same file set
  graphs/prompt-comparison.md/.json which prompts share a graph structure or an operation sequence
  prompts/<id>/                     per prompt: cpu-baseline*, cpu-profile*, cpu-op-trace.jsonl.gz, saved logits
  prompt-profiles.json              one row per prompt: timings, trace breakdown, overhead and logit checks, graph pointers
```

The raw file names match the recorded `experiments/llama-cpp/2026-09-22/results/`, so the same analysis code reads both.

## Rules

- A run folder is never reused or edited afterwards. Two runs in the same minute get `-2`, `-3`.
- `run-manifest.json` is the provenance. The folder name is only a label; read the manifest for the
  host, compiler, pinned llama.cpp commit, model hash and exact command.
- **Runs from different hosts are not comparable** to each other or to the recorded Apple M5 Pro
  baseline. CPU operation shares do not describe accelerator area, bandwidth or speedup.
- Model weights never go here (they live in the gitignored `work/`).
- The 2026-09-22 baseline stays in `experiments/` and is not copied here.
