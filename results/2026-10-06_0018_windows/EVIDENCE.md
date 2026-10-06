# Evidence set: 2026-10-06_0018_windows

Review follow-up to the preserved [split-build experiment](../2026-10-05_2035_windows/EVIDENCE.md).
This tests optional CUDA iteration speed and CPU build/weight-buffer sensitivity on one host. It does not
measure SiliconBadgers hardware, CUDA kernel utilization, model quality, or quantization accuracy.

## Reproduction and provenance

```powershell
windows\run.ps1 profile --cuda on --cuda-no-host-control --metal off --no-trace --no-graphs --no-prompt-graphs --no-prompt-timings
windows\run.ps1 analyze results/2026-10-06_0018_windows
```

The full run completed on 2026-10-06 from 00:18:40 to 00:30:51 America/Chicago (05:18:40 to 05:30:51 UTC).
The manifest records host/tool versions, source/model revisions and hash, separate build configurations,
dataset roles, requested placement, and actual loaded model buffers for each invocation.

- Windows 11; Intel Core Ultra 7 270K Plus; 24 logical CPUs; 64 GiB installed RAM (63.27 GiB Windows-visible).
- NVIDIA RTX 5070 Ti; 16,303 MiB reported VRAM; driver 610.62; CUDA 13.4; MSVC 19.51.36248;
  CMake 4.4.3; Ninja 1.13.2; Python 3.13.15.
- llama.cpp `f46bc30cb6a7f68a67e34a00061e20a4ad1eff43`; Qwen3.5-2B Q4_K_M; 6 CPU threads;
  one sequence; three repetitions after the existing warmup; 32 shared teacher-forced decode steps.
- Identical token inputs across all four series. Primary lengths: 128, 512, 2048. The 8192-token case remains
  diagnostic and excluded from primary conclusions by the existing experiment policy.
- Series run sequentially, not randomized or interleaved. Thermal/background-load variation is not isolated;
  modest differences between equivalent CPU paths should not be interpreted causally.

## Execution roles and actual placement

| Dataset | Role | Build / request | Loaded model-buffer observations |
|---|---|---|---|
| `cpu` | Current CPU reference | `baseline`, CUDA OFF, ngl=0 | `CPU_REPACK`: 514,326,528 bytes; `CPU_Mapped`: 1,269,873,920 bytes |
| `cuda-cpu` | Ordinary CUDA-build CPU control | `cuda`, CUDA ON, ngl=0, no_host=false | `CPU_Mapped`: 1,269,873,920 bytes; no CPU_REPACK model buffer |
| `cuda-cpu-no-host` | CUDA-build CPU repacking control | Same CUDA build, ngl=0, no_host=true | Same CPU_REPACK and CPU_Mapped model-buffer sizes as the CPU reference |
| `cuda` | Accelerated execution | Same CUDA build, ngl=99 | NVIDIA CUDA model buffers: 1,269,876,224 bytes; actual_accelerator=true; cpu_fallback=false |

Both primary and diagnostic invocations report these placements. Mapped/repacked buffers can hold overlapping
weight data: their sizes should not be summed into a unique-weight footprint. The metadata describes model
allocation, not exact offloaded layer counts or which kernels executed. Device descriptions/backend names are
recorded when the pinned runtime exposes them; mapped-buffer device identity can be unknown.

This supports Akanksha's proposed host-buffer/repacking explanation for the prior CPU-path difference. It is
not a kernel-level attribution study. `no_host` remains an explicit additional control, not a silent baseline
change or a requirement for normal CUDA execution.

## Timing sensitivity

`analysis/summary.{json,csv}` contains all 16 phase-paired timing summaries, samples and min/max ranges.
`analysis/backend-speedups.{json,csv}` gives each denominator's name/role and the diagnostic-exclusion flag.
Speedups are ratios of median prefill seconds and median per-repetition decode means, respectively.

| CUDA speedup versus | 128 prefill / decode | 512 prefill / decode | 2048 prefill / decode |
|---|---:|---:|---:|
| CPU reference | 40.00x / 9.76x | 74.00x / 9.21x | 77.79x / 9.16x |
| Ordinary CUDA-build CPU | 48.99x / 8.54x | 93.95x / 8.29x | 98.45x / 8.53x |
| CUDA-build CPU, no_host | 39.39x / 9.33x | 75.41x / 9.24x | 80.64x / 9.36x |

The no_host control's primary prefill latency is within -1.5% to +3.7% of the CPU reference; decode latency
is within -4.4% to +2.1%. Its timing is much closer to the CPU reference than the ordinary control, but the
remaining small differences are not isolated from sequential-run noise. CUDA remains substantially faster
against every CPU denominator on this host. This supports rapid candidate evaluation followed by checks
against the current CPU reference, not CUDA-only certification of subtle numerical findings.

Prefill's larger gain is consistent with parallel matrix work across prompt tokens and launch-overhead
amortization; batch-one decode is dependency-ordered and more exposed to weight movement and per-kernel
latency. This is an interpretation of phase timings, not CUDA kernel profiling. Analytical graph/scheduler
sweeps still run on CPU and are not accelerated by this feature.

## Numerical checks

Each comparison has eight saved phase/length checkpoints, including the diagnostic case:

| Compared with CPU reference | Bit-identical | Largest absolute difference | Largest mean absolute difference | Matching top tokens | Top-10 overlap |
|---|---:|---:|---:|---:|---:|
| Ordinary CUDA-build CPU | 0/8 | 0.9213376 | 0.1522371 | 8/8 | 9-10 |
| CUDA-build CPU, no_host | 8/8 | 0 | 0 | 8/8 | 10 |
| CUDA | 0/8 | 1.0244226 | 0.1758775 | 8/8 | 9-10 |

All saved logits are finite. This is checkpoint agreement on this harness, not task-quality evaluation.
Tracing was disabled in this full timing run: `profile_bit_identical` is null, not a vacuous passing check.
A separate local trace-enabled smoke verifies the current CPU reference against its instrumented build.

## Evidence packaging and limitations

- Committed: eight measured JSONL files, the run manifest, timing/decode/speedup summaries and CSVs, three
  backend numerical-check summaries, empty unchecked trace/Metal-check lists, two PNG figures, and
  `validation-summary.json` extracting the separate local smoke/negative-check records.
- Local-only: logs, copied prompts/tokens, saved logits and duplicate PDF. The 57 omitted files total
  30,508,361 bytes and are listed in `omitted-files.sha256`. SHA-256 covers the compressed bytes as stored.
- After completion, `analyze` regenerated derived summaries/figures and updated the manifest's analysis
  section using the final role/build reader and diagnostic-exclusion reporting. Raw measurements were not
  changed. This evidence note, the omission hash list and the local development-check summary were then added.
- No full trace, graph capture, Metal/AMD/Vulkan inference, long continuation, multi-sequence study or
  model-quality experiment was performed here. Metal build-flag behavior is unit-tested; Metal inference
  and the shared-buffer classification need runtime confirmation on a Mac.
- Default execution remains CPU. CUDA is optional; CPU traces and graph capture remain the current
  architecture-evidence path. Broader validation policy can evolve with measured evidence.
