# Evidence set: 2026-10-05_2035_windows

This is a split-build CPU/CUDA timing experiment for the existing Qwen3.5-2B profiling harness. It answers a
narrow software-team question: whether optional CUDA execution is useful for faster experimental iteration on
this host. It is not evidence about the SiliconBadgers accelerator architecture, GPU kernel boundaries, ASIC
performance, model quality, or a replacement for the current CPU trace and graph-capture workflow.

The committed evidence contains the manifest, timing records, numerical comparison, summaries and figures.
Large saved logits and logs remain local and are listed with SHA-256 in
[`omitted-files.sha256`](omitted-files.sha256).

## What was run

| | |
|---|---|
| Command | `windows\run.ps1 profile --cuda on --metal off --no-trace --no-graphs --no-prompt-graphs --no-prompt-timings` |
| Time | 2026-10-06 01:35:49 to 01:43:59 UTC; manifest status `complete` |
| Host | Windows 11 (10.0.26200), Intel Core Ultra 7 270K Plus, 24 logical CPUs, 63.3 GiB RAM |
| GPU | NVIDIA GeForce RTX 5070 Ti, 16,303 MiB reported VRAM, compute capability 12.0, driver 610.62 |
| Toolchain | MSVC 19.51.36248, CUDA 13.4 (V13.4.59), CMake 4.4.3, Ninja 1.13.2, Python 3.13.15 |
| Runtime | llama.cpp `f46bc30cb6a7f68a67e34a00061e20a4ad1eff43`; separate pristine CPU-only and CUDA-enabled executables; 6 CPU threads; one sequence |
| Model | Qwen3.5-2B Q4_K_M, revision and SHA-256 recorded in `run-manifest.json` |
| Workload | 128, 512 and 2048 prompt tokens; 3 repetitions; 32 teacher-forced decode steps. The 8192-token case was also run but remains diagnostic and excluded from primary conclusions by existing policy. |

The three recorded execution roles are explicit in both filenames and `config.backends`:

- `cpu-*`: zero-offload execution from the pristine CPU-only build; the current CPU reference.
- `cuda-cpu-*`: zero-offload execution from the CUDA-enabled build; the matched timing control.
- `cuda-*`: 99 GPU layers in the same CUDA-enabled build as the matched control.

The manifest records the CPU-only build under `build` with `GGML_CUDA=OFF` and the separate CUDA build under
`cuda_build` with `GGML_CUDA=ON`. The local CUDA log reports one RTX 5070 Ti, all 25 model layers offloaded,
and a 1,211.05 MiB CUDA model buffer. Logs are omitted from Git but their hashes are retained.

## Measured timing

Medians are taken using the existing analysis. Speedup is CUDA throughput divided by the matched zero-offload
CPU throughput from the same CUDA-enabled executable.

### Prompt processing

| Prompt tokens | CPU reference tok/s | CUDA-build CPU tok/s | CUDA tok/s | CUDA / matched CPU |
|---:|---:|---:|---:|---:|
| 128 | 245.8 | 195.2 | 9,119.7 | 46.73x |
| 512 | 240.6 | 187.3 | 16,921.1 | 90.36x |
| 2,048 | 233.5 | 181.6 | 17,948.7 | 98.81x |
| 8,192 diagnostic | 212.2 | 172.0 | 17,616.5 | 102.44x |

### Cached single-token processing

| Prompt tokens | CPU reference tok/s | CUDA-build CPU tok/s | CUDA tok/s | CUDA / matched CPU |
|---:|---:|---:|---:|---:|
| 128 | 37.5 | 41.2 | 351.9 | 8.53x |
| 512 | 37.3 | 40.5 | 356.7 | 8.81x |
| 2,048 | 35.9 | 38.6 | 342.1 | 8.85x |
| 8,192 diagnostic | 31.9 | 33.7 | 333.3 | 9.89x |

The phase difference is expected from the workload structure. Prefill exposes matrix work across many prompt
tokens, giving the GPU enough parallel work to use its throughput and amortize launch overhead. Batch-one
decode processes one dependency-ordered token at a time and is more exposed to weight movement and per-kernel
latency, so its acceleration is materially smaller. This is an interpretation of the measured phase behavior,
not a kernel-level profile or an accelerator-architecture conclusion.

The measured improvement is large enough to justify optional CUDA for rapid candidate evaluation on NVIDIA
hosts, followed by checks against the current CPU reference. CUDA alone should not be used to establish subtle
numerical or model-quality conclusions. It does not accelerate the analytical scheduler or resource sweeps,
which consume captured graphs on CPU.

## Numerical and CPU-path checks

`analysis/cpu-cuda-logit-check.json` compares the current CPU reference with CUDA across eight phase/length
checkpoints. All logits are finite, all eight top tokens match, top-10 overlap is 9 or 10, the maximum absolute
difference is 1.0244226, and the largest mean absolute difference is 0.1758775. These checks establish backend
agreement at this harness checkpoint only; they are not task-quality or quantization certification.

The CPU-only and CUDA-enabled builds differ measurably even with GPU offload disabled. Across the primary
cases, the CUDA-build CPU control has 20.6-22.2% lower prefill throughput and 7.7-9.9% higher decode throughput
than the CPU-only reference. Keeping both series explicit prevents the matched speedup control from being
mistaken for the current CPU reference.

## Boundaries and omissions

- No trace, graph capture, per-prompt timing, Metal run, long continuation, multiple batch/sequence study, or
  model-quality evaluation was performed in this run.
- The current implementation performs trace and graph capture on CPU. CUDA is opt-in and limited to
  uninstrumented timing and saved-logit comparison.
- The measurements are for this host, driver, toolkit, pinned model and pinned llama.cpp revision. They do not
  predict another GPU or the proposed accelerator.
- Logs, prompts, token dumps, saved logits and the duplicate PDF rendering stay local. The 43 omitted files
  total 21.83 MiB and are listed in `omitted-files.sha256`.
- The committed JSONL, JSON, CSV and PNG files are sufficient to review the timing claim, provenance and
  numerical summary.
