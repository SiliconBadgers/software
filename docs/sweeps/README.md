# Compact reproduction summaries

These summaries reproduce the existing Python grid/evaluation/Pareto methods on
the full provisional `characterized-v1` profile and explicit pp512-fa-on capture.
They are not the historical JS studies or a hardware selection.

| Study | Candidates | Model-feasible | Specification | Summary |
|---|---:|---:|---|---|
| Compute | 4 | 3 | [matrix capacity](../../engine/sweeps/compute/capacity.json) | [record](compute-v1.json) |
| Local memory | 4 | 4 | [L1 service](../../engine/sweeps/local-memory/service.json) | [record](local-memory-v1.json) |
| Joint | 16 | 12 | [matrix/L1 grid](../../engine/sweeps/joint/matrix-l1.json) | [record](joint-v1.json) |

Full rows are ignored under `build/sweeps/`; reproduction commands, source/spec
hashes, baseline identity, axes, objectives, fixed assumptions, feasibility counts
and record digests are in each summary. Invalid designs remain in the candidate
denominator but are excluded by the existing Pareto filter. Eight matrix units
exceed this baseline's assumed usable DSP budget; this does not validate physical
fit for the others. Banking beyond the sampled range is not recommended by these
curves, and the cost objective is only a unitless proxy.

Use [the thin runner](../../engine/sweeps/README.md) to reproduce into a new folder.
Source moves preserve existing costs, reservations and known reporting limitations;
these results do not establish global optimality, calibrated energy/area or a
validated recurrence implementation. No candidate is promoted.
