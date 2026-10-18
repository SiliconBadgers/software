# Reproducing resource studies

The compute, local-memory and joint specifications use the Python engine's
Cartesian enumeration, evaluation and Pareto methods. The joint axes reproduce
the matrix/L1 command in `engine/README.md`; one-dimensional subsets expose each
axis separately. Fixed settings come from the resolved provisional baseline,
including external memory, platform, precision and cost coefficients.
Prerequisites record related decision questions; they do not schedule studies.

From the repository root:

```text
python -B engine/sweeps/run.py engine/sweeps/compute/capacity.json
python -B engine/sweeps/run.py engine/sweeps/local-memory/service.json
python -B engine/sweeps/run.py engine/sweeps/joint/matrix-l1.json
```

Full rows, CSV/meta and reproduction records go to ignored `build/sweeps/`.
For a repeat, supply a new `--out build/sweeps/<name>` folder; prior output is not
overwritten. [Compact summaries](../../docs/sweeps/README.md) retain provenance
and key results. The cost objective is unitless, not calibrated implementation
cost; selecting a baseline remains a separate review decision.

The `sram/` specifications vary L1 capacity, banking, the static tile/state/
activation partition and the weight path/fusion organization with compute held
at the baseline. Their partition and traffic analysis and results are in the
[SRAM study](../../research/compute-mapping/2026-10-18-sram-sweep/REPORT.md).

Sensitivity, switch/ablation and seeded Monte Carlo are available in the native
CLI. Use explicit `--run`, `--graph` and resolved `--profile` inputs; Monte Carlo
design paths should name resolved configurations. Review the
[known reporting limitations](../../REPORT.md#evidence-and-limitations)
before interpreting infeasible-design statistics.

Historical JS studies retain their original engines, candidates, objectives and
seeds. Use the dated instructions in their
[research packages](../../research/compute-mapping/README.md).
