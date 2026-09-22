# Software tests

Checks and comparisons support confidence in numerical references, operation mapping and backend/runtime behavior as those implementations develop.

The current [MAC tests](test_reference.py) cover signed extremes, control priority, wrapping and input bounds. Run them with `make test`; see [setup and example scope](../SETUP.md). They do not test llama.cpp or the complete accelerator.

Profiling-output comparisons live with the [experiment](../experiments/llama-cpp/2026-09-22/README.md). State whether new checks establish exact agreement, numerical tolerance or task-level quality, and use independently justified expected behavior where possible.

This is a suggested home for work supporting the [charter](../CHARTER.md) and
[objectives](../OBJECTIVES.md). The team can reorganize or extend it as its work
develops. A directory’s presence does not assign a task or require an artifact.
