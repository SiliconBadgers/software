# Software tests

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

See [workspace checks](../web/README.md) for browser tests. Use the committed
lockfile; dependency changes and compatibility-export regeneration are described
in [SETUP.md](../SETUP.md).

Profiling-output comparisons live with the [experiment](../experiments/llama-cpp/2026-09-22/README.md). State whether new checks establish exact agreement, numerical tolerance or task-level quality, and use independently justified expected behavior where possible.

This is a suggested home for work supporting the [charter](../CHARTER.md) and
[objectives](../OBJECTIVES.md). The team can reorganize or extend it as its work
develops. A directory’s presence does not assign a task or require an artifact.
