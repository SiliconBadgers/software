# Evidence set: 2026-10-05_0954_windows

This is a matched CPU/CUDA timing experiment for the existing Qwen3.5-2B profiling harness. It answers a
narrow software-team question: whether optional CUDA execution is useful for faster experimental iteration on
this host. It is not evidence about the SiliconBadgers accelerator architecture, GPU kernel boundaries, ASIC
performance, model quality, or a replacement for canonical CPU traces and graph capture.

The committed evidence contains the manifest, timing records, numerical comparison, summaries and figures.
Large saved logits and logs remain local and are listed with SHA-256 in
[`omitted-files.sha256`](omitted-files.sha256).

## What was run

| | |
|---|---|
| Command | `windows\run.ps1 profile --cuda on --metal off --no-trace --no-graphs --no-prompt-graphs --no-prompt-timings` |
| Time | 2026-10-05 14:54:04 to 14:58:37 UTC; manifest status `complete` |
| Host | Windows 11 (10.0.26200), Intel Core Ultra 7 270K Plus, 24 logical CPUs, 63.3 GiB RAM |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16,303 MiB reported VRAM, compute capability 12.0, driver 610.62 |
| Toolchain | MSVC 19.51.36248, CUDA 13.4 (V13.4.59), CMake 4.4.3, Ninja 1.13.2, Python 3.13.15 |
| Runtime | llama.cpp `f46bc30cb6a7f68a67e34a00061e20a4ad1eff43`; one CUDA-enabled pristine executable; CPU uses `N_GPU_LAYERS=0`, CUDA uses `99`; 6 CPU threads; one sequence |
| Model | Qwen3.5-2B Q4_K_M, revision and SHA-256 recorded in `run-manifest.json` |
| Workload | 128, 512 and 2048 prompt tokens; 3 repetitions; 32 teacher-forced decode steps. The 8192-token case was also run but remains diagnostic and excluded from primary conclusions by existing policy. |

The local CUDA log reports one RTX 5070 Ti, all 25 model layers offloaded, and a 1,211.05 MiB CUDA model
buffer. The log is omitted from Git but its hash is retained.

## Measured timing

Medians are taken using the existing analysis. Speedup is CUDA throughput divided by the matched CPU
throughput from the same executable and build.

| Prompt tokens | CPU prefill tok/s | CUDA prefill tok/s | Prefill speedup | CPU decode tok/s | CUDA decode tok/s | Decode speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 195.9 | 9,606.9 | 49.03x | 41.0 | 346.1 | 8.44x |
| 512 | 188.9 | 17,340.4 | 91.78x | 41.0 | 350.3 | 8.54x |
| 2,048 | 185.8 | 18,199.9 | 97.94x | 39.7 | 336.9 | 8.48x |
| 8,192 diagnostic | 170.6 | 17,799.9 | 104.32x | 33.5 | 325.3 | 9.70x |

The measured improvement is large enough to justify optional CUDA for repeated uninstrumented inference,
quantization and numerical experiments on NVIDIA hosts. It does not accelerate the analytical scheduler or
resource sweeps, which consume captured graphs on CPU.

## Numerical and CPU-path checks

`analysis/cpu-cuda-logit-check.json` contains eight phase/length comparisons. All logits are finite, all eight
top tokens match, top-10 overlap is 9 or 10, the maximum absolute difference is 1.3816814, and the largest mean
absolute difference is 0.1868512. These checks establish backend agreement at this harness checkpoint only;
they are not task-quality or quantization certification.

A second CPU run through the same CUDA-enabled executable reproduced all six saved checkpoints byte-for-byte.
Its median prefill throughput was within 0.7–1.5% of the recorded run and decode throughput within 0.4–2.0%.

A separately compiled `GGML_CUDA=OFF` control was also tested locally. It retained the same top token in all six
checkpoint comparisons, with 9–10 top-10 overlap, but was not byte-identical and its timing differed materially:
prefill was 26.6–28.7% faster while decode was 6.5–9.1% slower than the CUDA-enabled executable's CPU path.
This is why the speedup table compares placements inside one CUDA-enabled build, and why its CPU row must not be
substituted for the canonical CPU-only/historical baseline. The normal default remains `--cuda off`.

## Boundaries and omissions

- No trace, graph capture, per-prompt timing, Metal run, long continuation, multiple batch/sequence study, or
  model-quality evaluation was performed in this run.
- CPU remains the required trace and graph-capture backend. CUDA is opt-in and limited to uninstrumented timing
  and saved-logit comparison.
- The measurements are for this host, driver, toolkit, pinned model and pinned llama.cpp revision. They do not
  predict another GPU or the proposed accelerator.
- Logs, prompts, token dumps, saved logits and the duplicate PDF rendering stay local. The committed JSONL,
  JSON, CSV and PNG files are sufficient to review the timing claim and numerical summary.
