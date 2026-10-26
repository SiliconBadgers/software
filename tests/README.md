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
Timestamp-only changes pass; content edits still fail when size and timestamps
are unchanged. Both Git-normalized checkouts and original CRLF blobs are covered.
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

`make test` also checks the uv project and launcher contracts with only the
standard library. CI exercises Linux with Python 3.11 and macOS/Windows with
Python 3.12. The project has no default packages; opt into the relevant group:

```sh
uv run --locked --group engine python -m unittest discover -s engine/tests -v
uv run --locked --group profiling python -m unittest discover -s core/tests -v
uv run --locked python -m unittest discover -s tests/web -v
```

To include the custom GGUF import tests, add `--group model-inspection` to the
core command. Without that optional group only the GGUF-dependent cases skip;
the existing profiling workflows and standard-library CDFG checks remain available.
See the [custom workload guide](../docs/model-baseline.md) for a real pinned-artifact
import, its output safeguards and the distinction from execution/capture evidence.

See [workspace checks](../web/README.md) for browser tests. Use the committed
lockfile; dependency changes and compatibility-export regeneration are described
in [SETUP.md](../SETUP.md).

Profiling-output comparisons live with the [experiment](../experiments/llama-cpp/2026-09-22/README.md). State whether new checks establish exact agreement, numerical tolerance or task-level quality, and use independently justified expected behavior where possible.

This is a suggested home for work supporting the [charter](../CHARTER.md) and
[objectives](../OBJECTIVES.md). The team can reorganize or extend it as its work
develops. A directory’s presence does not assign a task or require an artifact.
