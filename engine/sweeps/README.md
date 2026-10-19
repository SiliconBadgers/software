# Reproducing resource studies

The compute, local-memory and joint specifications use the Python engine's
Cartesian enumeration, evaluation and Pareto methods. The joint axes reproduce
the matrix/L1 command in `engine/README.md`; one-dimensional subsets expose each
axis separately. Fixed settings come from the resolved provisional baseline,
including external memory, platform, precision and cost coefficients.
Prerequisites record related decision questions; they do not schedule studies.

The runner and repository checker share the same specification contract.
Objectives are nonempty `[metric, "min"|"max"]` pairs using numeric summary
metrics (cost/DSP, latency, throughput, bounds, memory service and footprint) or
numeric declared axes. Labels and validity flags are not optimization metrics;
each metric appears once, so duplicate or conflicting directions are rejected.
Axes use nonempty value lists; an empty axes object evaluates the baseline alone.
Categorical and capability-list axes remain supported. References and decisions
must resolve, axes cannot overlap or vary fixed assumptions, and malformed,
duplicate-key or nonfinite JSON declarations fail before evaluation or output.
JSON numbers must stay within the finite binary64 range; oversized integers and
nonzero literals that underflow to zero are rejected. Integers in range retain
their exact integer values; ordinary finite/subnormal floats use binary64 rounding.
Frontier summaries retain the axes, cost and phase latencies and include every
selected objective's value, without duplicating fields.
Configuration/physical feasibility remains the evaluator's responsibility.

From the repository root:

```text
python -B engine/sweeps/run.py engine/sweeps/compute/capacity.json
python -B engine/sweeps/run.py engine/sweeps/local-memory/service.json
python -B engine/sweeps/run.py engine/sweeps/joint/matrix-l1.json
```

The `tiling/` specifications vary array shape, L1 matmul tiles and the
fused-attention KV tile with SRAM held at the baseline. Their selected-tile
analysis and results are in the
[tiling study](../../research/compute-mapping/2026-10-17-tiling-sweep/REPORT.md).

Full rows, CSV/meta and reproduction records go to ignored `build/sweeps/`.
For a repeat, supply a new `--out build/sweeps/<name>` folder; prior output is not
overwritten. [Compact summaries](../../docs/sweeps/README.md) retain provenance
and key results. The cost objective is unitless, not calibrated implementation
cost; selecting a baseline remains a separate review decision.

Sensitivity, switch/ablation and seeded Monte Carlo are available in the native
CLI. Use explicit `--run`, `--graph` and resolved `--profile` inputs; Monte Carlo
design paths should name resolved configurations. Review the
[known reporting limitations](../../REPORT.md#evidence-and-limitations)
before interpreting infeasible-design statistics.

Historical JS studies retain their original engines, candidates, objectives and
seeds. Use the dated instructions in their
[research packages](../../research/compute-mapping/README.md).
