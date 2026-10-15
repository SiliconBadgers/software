# Software tests

From the repository root, with the existing Python 3.12+ profiling environment
(NumPy/Matplotlib from `core/requirements.txt`), run each component explicitly:

```text
python -B -m unittest discover -s core/tests -v
python -B -m unittest discover -s engine/tests -v
python -B -m unittest discover -s tests -v
python -B -m unittest discover -s tests/web -v
python -B scripts/check_boundaries.py
```

`tests/` includes the four standard-library MAC tests and boundary-policy cases.
`core/tests` covers capture/profiling semantics and optional-backend provenance;
`engine/tests` includes exact graph/mapping/evaluation characterization. Three
measured-evidence checks skip without local-only traces/summaries; skips are not
validation passes. Missing CPU-fit evidence is explicitly NOT CHECKED.

Boundary tests include real Git repositories and negative evidence/artifact cases.
The committed original-file inventory is checked against the unchanged
characterization digest, so a shallow checkout needs no historical Git objects.
New research is permitted; only the recorded originals retain the legacy bulk
exemption. Study tests retain the original three cases, allow a fourth valid spec
and reject missing originals or invalid additions.

With Node 22+, run all five preserved JS suites:

```text
node experiments/llama-cpp/2026-09-24-qwen35-2b/scripts/test_model.cjs
node experiments/llama-cpp/2026-09-24-qwen35-2b/scripts/test_sensitivity.cjs
node experiments/llama-cpp/2026-09-24-qwen35-2b/scripts/test_scheduler.cjs
node experiments/llama-cpp/2026-09-24-qwen35-2b/scripts/test_joint_resources.cjs
node experiments/llama-cpp/2026-09-24-qwen35-2b/scripts/test_recurrence_penalty.cjs
```

Build and test the [browser workflows](../web/README.md), including Python/native
parity, whenever shared sources move. Use the existing Playwright installation
where available. Browser Python loads its pinned runtime from a CDN on first use;
native import tests alone do not establish that the worker was packaged correctly.
Inference, numerical quality, hardware/SoC and licensed-tool checks remain separate.
See the [current report](../REPORT.md) and [design rules](../docs/DESIGN-RULES.md).

Checks and comparisons support confidence in numerical references, operation mapping and backend/runtime behavior as those implementations develop.

The current [MAC tests](test_reference.py) cover signed extremes, control priority, wrapping and input bounds. Run them with `make test`; see [setup and example scope](../SETUP.md). They do not test llama.cpp or the complete accelerator.

Profiling-output comparisons live with the [experiment](../experiments/llama-cpp/2026-09-22/README.md). State whether new checks establish exact agreement, numerical tolerance or task-level quality, and use independently justified expected behavior where possible.

This is a suggested home for work supporting the [charter](../CHARTER.md) and
[objectives](../OBJECTIVES.md). The team can reorganize or extend it as its work
develops. A directory’s presence does not assign a task or require an artifact.
