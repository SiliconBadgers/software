# Existing studies, organized for reproduction

The compute, local-memory and joint specifications use the existing Python engine's
Cartesian enumeration, evaluation and Pareto methods. The joint axes reproduce
the matrix/L1 command in `engine/README.md`; one-dimensional subsets expose each
axis separately. Fixed settings come from the full provisional baseline, including
external memory, platform, precision and cost coefficients. Joint prerequisites
are navigation references, not a task service or automatic optimizer.

From the repository root:

```text
python -B engine/sweeps/run.py engine/sweeps/compute/capacity.json
python -B engine/sweeps/run.py engine/sweeps/local-memory/service.json
python -B engine/sweeps/run.py engine/sweeps/joint/matrix-l1.json
```

Full rows, CSV/meta and local reproduction records go to ignored `build/sweeps/`.
For a repeat, supply a new `--out build/sweeps/<name>` folder; prior output is not
overwritten. Commit only compact, provenance-bearing summaries in `docs/sweeps/`.
These objectives use a unitless cost proxy, not calibrated implementation cost.
No result is promoted to a baseline.

Existing sensitivity, switch/ablation and seeded Monte Carlo remain in the native
CLI. Use explicit `--run`, `--graph` and full baseline `--profile`; Monte Carlo
design paths should name resolved configurations. The existing assumption methods
have known infeasible-design reporting gaps, listed in the characterization audit.

Historical JS studies under `experiments/llama-cpp/2026-09-24-qwen35-2b/scripts/`
retain their original engines, candidates, objectives and seeds. Use their dated
reproduction instructions and [research packages](../../research/compute-mapping/README.md);
do not substitute the Python equations into those results. Preservation is not
approval to extend unrelated backend or platform-performance studies.
