# Optional CUDA execution: verification

CUDA is an optional fast execution path for software iteration, not a GPU-performance deliverable or
evidence for sizing the SiliconBadgers accelerator. CPU remains the default and current trace/graph
reference. Numerical findings from fast CUDA runs should be checked against that reference.

## Inputs and host

Windows 11, Intel Core Ultra 7 270K Plus, 64 GB installed RAM (63.27 GiB Windows-visible), NVIDIA RTX
5070 Ti (16,303 MiB reported VRAM), driver 610.62, CUDA 13.4, MSVC 19.51.36248, Python 3.13.15.

The existing `core/experiments/qwen35-2b-q4km.json` pins llama.cpp
`f46bc30cb6a7f68a67e34a00061e20a4ad1eff43`, Qwen3.5-2B Q4_K_M and the shared teacher-forced token inputs.
The model SHA-256 is `aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223`.
CPU reference, CUDA execution and instrumented CPU use separate build directories.

## Commands

From the repository root, using the existing local environment and cached model:

```powershell
windows\run.ps1 profile --cuda on --smoke --results-root work/validation-runs
windows\run.ps1 profile --cuda only --smoke --results-root work/validation-runs
windows\run.ps1 profile --cuda on --cuda-no-host-control --smoke --no-trace --no-graphs --results-root work/validation-runs
.venv\Scripts\python.exe -m unittest discover -s core/tests
```

Each fixed-length smoke uses 128 prompt tokens, one repetition and eight decode steps. A short existing
repetitive prompt also exercises the CPU/CUDA prompt path, CPU trace and CPU graph capture. The full
command, inputs, build flags and actual placements are recorded in each local `run-manifest.json`.

## Checks

- Ordinary comparison executes only CPU reference and CUDA, with no diagnostic control datasets or
  control-specific checkpoint exports. CPU-reference/trace checkpoints are bit-identical; CPU graph
  capture remains available. CUDA saved checkpoints are finite but are not assumed bit-identical to CPU.
- The CUDA-only smoke executes only CUDA and skips CPU timing, tracing, graphs and per-prompt stages.
  Loaded model buffers report actual CUDA placement, separately from the requested GPU layers.
- The explicit diagnostic smoke includes both zero-offload CUDA-build CPU controls and labels each
  distinctly. Only this opt-in path adds their comparative ratios/checkpoint reports. Diagnostics
  investigate reference/host-buffer behavior; they are not an everyday iteration requirement.
- A previous negative smoke with process-local `CUDA_VISIBLE_DEVICES=-1` recorded CPU fallback and
  rejected the run. Reanalysis of its saved evidence explicitly labels CPU fallback, emits no accelerated
  speedup, and preserves failed status. Original captured files remain unchanged.
- Core tests cover default/fast/diagnostic selection, separate CPU/Metal/CUDA build flags, historical
  reference identity, fallback in aggregate/case/raw/gzip evidence, failed-run reanalysis, stale derived
  report replacement, per-prompt isolation and original harness measurement-region preservation.

Local checks: 88 core tests and 7 publication tests passed. Both browser suites passed with installed native
Chrome, and the workspace build includes 187 source assets. The publication test's deliberately denied
localhost request can invoke the existing Windows credential helper: its default run timed out here,
then all 7 tests passed with process-local `GCM_INTERACTIVE=never` (restored afterwards). No repository or
global authentication configuration was changed; the unrelated publication fix remains separate.
New-head CI verification follows publication; these local results do not stand in for that check.

Full benchmark packages remain separately preserved, not part of this PR's proposed merge. Existing
recorded experiments and the dashboard's archived graph payload are preserved. Local verification
artifacts, model, builds and logs stay in ignored workspace folders.

## Limits

These are functionality and saved-checkpoint checks, not representative task-quality or quantization
validation. Recorded placement describes loaded buffers, not kernel utilization or exact offloaded layer
counts. AMD/Vulkan and Metal inference were not tested in this Windows run; Metal build flags are
unit-tested, but the new placement metadata still needs a runtime check on a Mac. CUDA operation tracing
and CUDA graph capture are not added. No quantified speedup or accelerator-performance conclusion is
part of this PR.
