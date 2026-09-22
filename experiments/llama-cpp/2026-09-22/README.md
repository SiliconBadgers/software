# Qwen3.5-2B profiling in llama.cpp

[REPORT.md](REPORT.md) explains the measured findings, proposed investigations and limitations. This dated package preserves the September 22, 2026 evidence: harness, CPU instrumentation patch, raw phase measurements, compressed operation traces and saved logits, analysis scripts, figures and provenance. Model weights, upstream source, build products and environments are downloaded or created separately.

The recorded host was an Apple M5 Pro (Mac17,9), 24 GiB unified memory, macOS 26.6.2, six CPU threads and one sequence. CPU and Metal baselines used three repetitions at 128, 512, 2048 and 8192 prompt tokens, with 32 cached single-token continuation steps. The repeated technical passage is teacher-forced, so backend inputs match; this is not a sampled-chat or task-quality benchmark. Loading, tokenization, sampling and saved-output checks are outside timers; execution synchronization is inside.

## Inspect the recorded evidence without running the model

From this experiment directory, use Python 3.12 or newer (the original environment used Python 3.13). Create an isolated environment; do not install dependencies globally:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

analysis_dir="$(mktemp -d "${TMPDIR:-/tmp}/siliconbadgers-analysis.XXXXXX")"
.venv/bin/python analyze.py --output-dir "$analysis_dir"
.venv/bin/python analyze_ops.py --output-dir "$analysis_dir"
.venv/bin/python validate_outputs.py --output-dir "$analysis_dir"
.venv/bin/python plot_results.py --results-dir "$analysis_dir" --output-dir "$analysis_dir"
```

Every analysis tool accepts `--results-dir` and `--output-dir`. Readers accept plain or gzip-compressed inputs. Default inputs are the packaged `results/`. `validate_outputs.py` only reads and prints checks unless an output directory is supplied; the other tools default to a fresh temporary output directory. All refuse to write into the packaged `results/`, including through symlinks. Regenerating figures uses Matplotlib and may create its normal user font cache.

`artifact-checksums.json` records both the stored file hash and the decompressed original hash/size. Gzip packaging does not change the original measurement bytes. The saved CPU/Metal logits support eight checkpoint comparisons (prefill and final decode at each length); saved instrumented CPU logits support eight bit-for-bit profiler checks. These checks do not establish full-precision model quality or correctness at every intermediate operation.

## Reproduce the workload in a fresh workspace

Prerequisites: Git, a C/C++ toolchain, Python and the pinned packages in `requirements.txt`. On Apple Silicon macOS, install Xcode Command Line Tools. `reproduce.py` enables Metal there automatically; `--metal off` selects CPU only. CPU builds on other supported llama.cpp hosts are possible, but their timings are not comparable to the recorded Apple M5 Pro baselines. The Linux CPU path is provided but was not benchmarked in this experiment. The runner does not enable CUDA or other accelerators.

The setup downloads the pinned source and approximately 1.28 GB GGUF, verifies the model's exact size and SHA256, then builds a pristine runtime and the C++ harness:

```sh
.venv/bin/python reproduce.py --work-dir work setup
```

To reuse an existing model file, pass `setup --model /path/to/Qwen3.5-2B-Q4_K_M.gguf`; the same checksum verification still runs. Setup requires an empty work directory and never resets an existing checkout.

Pinned inputs:

- llama.cpp commit: `f46bc30cb6a7f68a67e34a00061e20a4ad1eff43`.
- Model repository: `unsloth/Qwen3.5-2B-GGUF`; revision: `f6d5376be1edb4d416d56da11e5397a961aca8ae`.
- File: `Qwen3.5-2B-Q4_K_M.gguf`; SHA256: `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`.
- Release build with native CPU feature detection. This GGUF mixes Q4_K, Q5_K, Q6_K, Q8_0 and F32 tensors; it is not the proposal's custom symmetric INT4/group-64 format.

Collect uninstrumented baselines first, then patch and build a separate instrumented runtime:

```sh
.venv/bin/python reproduce.py --work-dir work baseline --output-dir runs/reproduction-1
.venv/bin/python reproduce.py --work-dir work trace --results-dir runs/reproduction-1

.venv/bin/python analyze.py --results-dir runs/reproduction-1 --output-dir runs/reproduction-1/analysis
.venv/bin/python analyze_ops.py --results-dir runs/reproduction-1 --output-dir runs/reproduction-1/analysis
.venv/bin/python validate_outputs.py --results-dir runs/reproduction-1 --output-dir runs/reproduction-1/analysis
.venv/bin/python plot_results.py --results-dir runs/reproduction-1/analysis --output-dir runs/reproduction-1/analysis
```

`baseline` requires a new output directory and unsets tracing variables. It creates the original result prefixes, prompt/token files, saved logits and a run configuration. `trace` requires those CPU baseline measurements first, matches their thread count and decode length, preserves fusion and writes new trace files. Baseline and instrumented binaries/libraries occupy separate build directories; applying the patch never rebuilds or replaces the baseline libraries. Reusing an existing result prefix is rejected. Run benchmarks and compilation sequentially, with other machine workloads minimized and recorded.

The default trace covers 128/512/2048 only. Add `trace --include-8k` to collect an optional 8K diagnostic, which remains flagged as excluded from timing conclusions. The original 8K trace was much slower than its earlier baseline; a later trace-disabled control was also slow, leaving the cause unresolved. Correct logits did not make those timings trustworthy. The absence of an optional new 8K trace is printed by validation rather than counted as a passed profiler check. For CPU-only measurements, add `validate_outputs.py --skip-metal`; plotting the complete recorded figure requires both backend baselines.

For a small fresh smoke run, use `baseline --lengths 128 --repetitions 1 --output-dir runs/smoke-1`, then `trace --repetitions 1 --results-dir runs/smoke-1` and `validate_outputs.py --lengths 128 --results-dir runs/smoke-1`. Decode defaults to 32 steps in all commands; matching `--decode-steps` values may be supplied. Custom result directories must remain separate from `results/`.

## Direct commands and timing interpretation

The runner prints the exact Git, CMake, compiler and harness commands. A direct harness invocation has this interface:

```text
profile MODEL N_GPU_LAYERS THREADS REPETITIONS DECODE_STEPS LENGTHS_CSV OUTPUT_PREFIX
```

CPU runs use GPU layers `0` and disable operation/KQV offloading; Metal runs use `99` to offload all layers. Both use flash attention, physical microbatches capped at 512 (128 in the shortest case), and a full-prefill/eight-step warmup per context. State is cleared before each repetition and retained through the subsequent 32 decode steps. Actual context allocation is rounded beyond the populated prompt; inspect the logs and manifest.

The original stock CPU sanity check can be repeated after setup into a new file:

```sh
work/build-baseline/bin/llama-bench -m work/models/Qwen3.5-2B-Q4_K_M.gguf -p 512 -n 32 -d 512 -r 3 -t 6 -ngl 0 -nopo 1 -fa on -b 2048 -ub 512 -o json
```

If setup reused a model from another location, substitute that verified path. Stock benchmark tokens and prompt-depth setup differ from the passage harness; this check compares throughput, not logits.

See [CPU-PROFILER-METHODOLOGY.md](CPU-PROFILER-METHODOLOGY.md) for manual patch application and timing semantics. Instrumented outer-call latency includes trace serialization and must not replace baseline throughput. CPU operation shares do not decompose Metal kernels or predict ASIC/FPGA cycles, area, power or DRAM traffic.

The report's 2K operation shares can be regenerated from compressed raw traces. The anomalous 8K trace and trace-disabled control remain preserved as diagnostics, explicitly excluded from timing conclusions. [manifest.json](manifest.json), [model-metadata.json](model-metadata.json), and [artifact-checksums.json](artifact-checksums.json) record provenance.
