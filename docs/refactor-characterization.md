# Behavior compatibility checks

The characterization fixture records graph interpretation and analytical outputs
from reviewed source `0e06023`. It covers 44 captured phases and 76 evaluations,
including raw costs, feasibility, traffic, footprints, bounds and schedules.
It also records the original 446-file evidence inventory and historical gitlinks.

From the repository root:

```text
python -B -m unittest discover -s engine/tests -v
python -B engine/tests/characterization.py --out build/refactor-characterization/0e06023/new-comparison --compare engine/tests/fixtures/characterization-0e06023.json
```

Choose a new output directory for each comparison. The generator does not
overwrite previous comparisons or update the committed fixture. Inputs are the
pinned Qwen experiment and explicit `2026-09-27_1839_windows-2` capture run;
missing reference inputs fail rather than selecting a different run.

Exact digest agreement establishes compatibility, not physical model accuracy.
The [supporting characterization](verification/refactor/characterization.md)
retains the full input, consumer and known-defect inventory.
[Verification records](verification/refactor/README.md) document the refactor's
acceptance results; [component tests](../tests/README.md) provide current checks.
