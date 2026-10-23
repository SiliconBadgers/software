# Resource-study summaries

These compact results use the Python engine and the provisional
`characterized-v1` baseline on the pp512-fa-on capture.

| Study | Candidates | Model-feasible | Specification | Summary |
|---|---:|---:|---|---|
| Compute | 4 | 3 | [matrix capacity](../../engine/sweeps/compute/capacity.json) | [record](compute-v1.json) |
| Local memory | 4 | 4 | [L1 service](../../engine/sweeps/local-memory/service.json) | [record](local-memory-v1.json) |
| Joint | 16 | 12 | [matrix/L1 grid](../../engine/sweeps/joint/matrix-l1.json) | [record](joint-v1.json) |
| Tiling | 80 | 60 | [array shape](../../engine/sweeps/tiling/array-shape.json), [matmul tile](../../engine/sweeps/tiling/matmul-tile.json), [attention tile](../../engine/sweeps/tiling/attention-tile.json) | [record](tiling-v1.json) |

The tiling record covers all three tiling specifications. It is written by the
[tiling analysis](../../research/compute-mapping/2026-10-17-tiling-sweep/REPORT.md),
which also pins its full point and robustness rows by content hash.

The JSON records retain axes, objectives, fixed assumptions, source/configuration
identities, feasibility counts, frontiers and reproduction hashes. They are
machine-readable summaries, not raw candidate output. Use the
[study runner](../../engine/sweeps/README.md) to reproduce full rows into a new
ignored folder under `build/sweeps/`.

Invalid designs remain in the candidate denominator and are excluded from the
Pareto frontier. Eight matrix units exceed the baseline's assumed usable DSP
budget. Other feasibility flags do not validate physical fit; bank count is a
modeled service/cost axis, not a recommended SRAM organization. Cost is unitless,
and these sampled estimates do not establish global optimality or select hardware.
