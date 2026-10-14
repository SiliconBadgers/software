# Qwen3.5-2B matrix mapping study

This project maps selected captured Qwen3.5-2B matrix operations onto a
provisional Timeloop representation of the Silicon Badgers 8 by 8 matrix RTL.
Read [REPORT.md](REPORT.md) for the modeled outputs, analytical checks,
and limitations.

## Reproduce with the official container

The completed runs used Windows/WSL2, Docker Desktop, exercise repository
`Accelergy-Project/timeloop-accelergy-exercises` at commit
`2d5510807128e9bd5f1cad0705cdf2ec4612fd4e`, and image
`timeloopaccelergy/timeloop-accelergy-pytorch:latest-amd64` at digest
`sha256:4cf9fa1b52a90eee67c6585872fc47e90e83c2ca8e067244e4d1f4ca84e8dce8`.
The container reports PyTimeloop `0.0.1` and Accelergy `0.4`; its `tl` CLI
does not report a version with `--version`.

From Ubuntu, start the official exercise container as its README instructs.
Copy [docker-compose.localhost.yaml](docker-compose.localhost.yaml) beside the
exercise repository's `workspace/` directory to bind Jupyter to localhost
while keeping its token enabled. In a second Ubuntu shell, change to your
local copy of this project and run the script:

```bash
cd /path/to/software/research/timeloop-accelergy-qwen35
bash run_in_container.sh
```

The script copies the checked-in inputs, runs Timeloop inside the container,
and copies reports back into ignored `results/timeloop-run/` and
`results/qwen-ffn-2026-10-12/`. It also runs the result checker. To run only
the checker later, from PowerShell:

```powershell
python software/research/timeloop-accelergy-qwen35/summarize_timeloop.py
```

The summary command asserts MAC counts, minimum tensor element counts, buffer
capacity, and ideal cycles for the seven profiled cases.

## Evidence and status

| Evidence | Revision | Status |
|---|---|---|
| Architecture compute decisions | `70ee075` on `codex/compute-team-decisions` | Team decisions |
| Matrix array candidate | `4926042` on `adam/systolic-array` | Candidate RTL |
| SIMD ISA | `a8385f5` on `docs/simd-isa-32` | Documented vector ISA; outside this model |
| Qwen profiling import | `f264d0f` on `import/qwen35-2b-profiling-2026-09-24` | Captured F32 graph evidence |

The 2048-square results first copied into `results/timeloop-run/` used an
incorrect one-axis tensor projection. Those reports are preserved in ignored
`results/superseded-invalid-projection-2026-10-12/`. The visible
`results/timeloop-run/` now contains corrected reruns. Do not use the old
energy or traffic figures in the superseded archive.
