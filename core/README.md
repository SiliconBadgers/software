# core

OS-neutral profiling code. Nothing here assumes a shell, a compiler or a directory layout beyond the repo.
The OS-specific parts live in [`../unix/`](../unix/) (Linux and macOS) and [`../windows/`](../windows/);
runs are written to [`../results/`](../results/README.md).

## One command

```sh
unix/run.sh doctor               # preflight: tools, pinned source, trace patch, CMake configure (no model, no compile)
unix/run.sh profile              # everything below, into results/<date>_<HHMM>_<os>/
unix/run.sh profile --smoke      # tiny run to check a new machine
unix/run.sh profile --dry-run    # print every command, run nothing, write nothing
unix/run.sh analyze results/<run>   # redo the analysis and rebuild graph reports/SVGs of an existing run
```

```powershell
windows\run.ps1 doctor
windows\run.ps1 profile [--smoke] [--dry-run]
windows\run.ps1 profile --cuda only                 # fast exploratory CUDA timing
windows\run.ps1 profile --cuda on --smoke           # matched CPU/CUDA validation smoke
```

`profile` runs: **ensure** (pinned llama.cpp + verified model, both cached in the gitignored `work/`) ->
**baseline** (current CPU reference, Metal on macOS/arm64, and optional separate CUDA execution) ->
**trace** (patched runtime, CPU op traces) ->
**graphs** (scheduled dataflow graph per length and flash-attention setting) -> **analyze**
(summaries, validation, `profiling-summary.png/.pdf`) -> `run-manifest.json`.
Useful flags: `--lengths`, `--threads`, `--repetitions`, `--decode-steps`, `--cuda on|only`, `--skip-metal`, `--no-trace`,
`--no-graphs`, `--include-diagnostic` (8K, timing stays excluded), `--model PATH`, `--work-dir`.

**Optional CUDA has three deliberately separate roles.** The default `--cuda off` uses the current pristine
CPU build for its reference baseline, trace validation and graph capture. `--cuda only` is the fast exploratory
path: it builds/runs CUDA only and automatically skips CPU timing, traces, graph capture and per-prompt stages.
It does not produce a CPU numerical comparison, so candidates selected this way should later be checked
against the current CPU reference. `--cuda on` is the freeze/comparison path: it runs the current CPU-reference
build, a zero-offload CPU control inside the separate CUDA-enabled build (`cuda-cpu-*`), and CUDA. The CPU
reference row validates the CPU-only trace; the CUDA-build CPU row is the matched control for backend speedup.

The traced build stays CPU-only, and graph capture hard-codes CPU placement, so GPU kernel selection is never
presented as accelerator-architecture evidence. When both the current CPU reference and CUDA are present, analysis writes
`cpu-cuda-logit-check.json` with finite-logit, maximum/mean difference, top-token and top-10-overlap checks. The
Windows wrapper imports a Visual Studio developer environment when its historical standalone LLVM path is
absent; CUDA itself remains an explicit, optional system prerequisite.

**Longer continuations.** The continuation is the run of single-token decode steps after the prompt.
`--decode-steps N` sets its length (default 32); `--long-continuation` uses the experiment's long setting (256).
Each step is timed on its own, and `analysis/decode-curve.{json,csv,png}` plot decode cost against the number
of tokens in the KV cache (step *j* of an *n*-token prompt sees *n + j + 1*), with a least-squares slope, first
and last window means, and a `context_span_fraction` saying how much the context actually moved. Over the old 32
steps of a 2,048-token prompt that is under 2%, so any slope there is noise. Fixed-length runs take their tokens
from one 9,500-token passage, so `length + steps` must fit in it; the CLI checks this before anything runs.
Traces record every step (about 0.4 MB of text per step and repetition before compression), so a long
continuation makes traced runs slower and larger. Per-prompt continuations repeat `prompts/continuation.txt`
and have no length limit. `graphs/` still capture one decode step at the prompt's own context.

**Per-prompt graphs.** A full `profile` also loops over the 8 prompts in `prompts/` and captures one scheduled
graph per prompt and flash-attention setting (`graphs/prompt-<id>-fa-<on|off>/`), then writes
`graphs/prompt-comparison.md/.json`. Each prompt is used whole, so its natural token count sets the prefill
length. A graph holds shapes and operation order, never token contents: prompts with the same token count give
the same graph, and prompts of different length keep the same operation sequence with different shapes; the
comparison reports exactly that. `--prompts-only` runs just the per-prompt stages, `--no-prompt-graphs` skips
the graphs, `--prompt-set FILE` uses another prompt list.

**Per-prompt timings and traces.** For each prompt, `sb-profile-prompt` (same measurement loop as `profile.cpp`)
runs a canonical untraced CPU run and a traced CPU run (plus Metal on macOS/arm64, or the CUDA-build CPU control
and CUDA in comparison mode) into `prompts/<id>/`, and
`prompt-profiles.json` gets one row per prompt: prompt hash and token count, prefill and decode timings, the
operation-trace breakdown per phase (share by group and by operation, call counts, matrix MACs), the
profiler-overhead ratio, the traced-vs-untraced logit check, and pointers to that prompt's graphs. The
prefill is the whole prompt file; the continuation is **teacher-forced from one shared text**
(`prompts/continuation.txt`, the same paragraph `profile.cpp` repeats), so every prompt gets identical
continuation tokens and decode differences come from the prompt, not from what follows it. `--no-prompt-timings`
skips this stage.

## Layout

| Path | Contents |
|---|---|
| `experiments/*.json` | Everything an experiment fixes: llama.cpp commit, model revision/hash, workload, timing-exclusion policy, graph settings. A new experiment is a new JSON file. |
| `harness/` | `profile.cpp` (the same program as the 2026-09-22 harness, reformatted; `tests/test_harness_equivalence.py` fails if anything but layout, comments and string splitting changes) and `sb-cpu-profile.patch` (byte-identical to the original), `capture_graph.cpp` (Zeb Taylor's, see its header), `compat_win.h` (`setenv` shim, force-included so `profile.cpp` needs no Windows-specific code), `CMakeLists.txt`. |
| `prompts/` | `prompts.json` and the 8 prompt texts (narrative, code, math, JSON records, dialogue, hardware prose, multilingual, repetitive). Add a prompt by adding a file and a line in `prompts.json`. |
| `runner/` | `cli.py` and the pipeline: `fetch`, `build`, `measure`, `capture`, `hostinfo`, `runpaths`, `platform_profile`. |
| `analysis/` | `summarize`, `ops`, `validate`, `plot`, `graph`; `result_io` writes LF-only files so outputs are byte-stable across OSes. Recorded `experiments/` is write-protected. `fusion`, `offload` and `compute_mapping` derive operation-sharing and boundary evidence from an existing run's graphs and traces (`python core/analysis/compute_mapping.py results/<run>`); they never re-run the model. `site` builds `results/index.html`, a self-contained browser over one run's graphs (`--no-raw-links` for the committed copy). |
| `tests/` | `python core/tests/test_regression_0922.py` reproduces the recorded 09-22 analysis outputs from their raw evidence (needs only Python + numpy). |

## Platform profiles

`unix/platform.json` and `windows/platform.json` say what differs by OS: OS labels (macOS and Linux are
separated here, so run folders say `macos` or `linux`), whether Metal is offered, the executable suffix and
compiler/resource-compiler search lists. Override with `SB_CC`, `SB_CXX`, `SB_RC`.

## Things that will bite you

- **Line endings.** Git for Windows often checks files out as CRLF. `core/`, `unix/` and `windows/` force LF
  (`.gitattributes`); the runner normalizes the patch and forces `core.autocrlf=false` on the llama.cpp checkout.
- **Separate builds and source trees.** The CPU reference and CUDA use the pristine `work/llama.cpp` checkout but
separate `work/build-baseline` and `work/build-cuda` directories. The trace uses the patched worktree
`work/llama.cpp-profile` and `work/build-profile`. Each build is checked for trace code and recorded
separately as `build`, `cuda_build` and `profile_build` in `run-manifest.json`.
- **Different hosts, different numbers.** Runs from different hosts or backends are not comparable unless they
  are the matched cases in one run; the build (e.g. OpenMP found or not) also differs. Read `run-manifest.json`.
- **Graphviz** is optional. Without `dot` the graph DOT/JSON/CSV files are written and the layer SVGs are skipped.
- **Graph vs profiled run.** The `fa-on` graph matches the profiled configuration; the `fa-off` graph keeps
  attention as separate operations. The capture sets the micro-batch size to the full prompt, so each prefill is one graph. The configured graph lengths are 128, 512 and 8192 tokens.
