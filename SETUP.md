# Software setup

No tools are required to read the charter, results or design material. Use the setup for the work you are doing; the MAC example and llama.cpp experiment have separate dependencies.

## Python tooling

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) 0.12.23 or newer.
The repository uses `pyproject.toml` and the committed `uv.lock`; CI pins uv 0.12.23.
`uv` creates `.venv` and can download the requested Python when it is not installed.
No manual virtualenv activation or pip install is needed.

The default project is standard-library-only and supports Python 3.11+.
`.python-version` selects Python 3.12 by default. Dependency groups are opt-in
and require Python 3.12+:

- `engine`: the pinned NumPy dependency for contributor-engine checks
- `analysis`: `engine` plus the pinned Matplotlib dependency for profiler analysis
- `profiling`: `analysis` plus the pinned CMake and Ninja build tools

```sh
uv sync --locked                                      # MAC/reference only, no packages
uv run --locked python -m unittest discover -s tests -v
uv run --locked --group engine python -m unittest discover -s engine/tests -v
uv run --locked --group profiling python -m unittest discover -s core/tests -v
uv sync --locked --group profiling                    # full profiler environment
```

Use `uv run --locked` for normal work so stale or missing lockfiles fail clearly.
When intentionally updating dependencies, edit the groups in `pyproject.toml`, run
`uv lock`, and commit both files. `core/requirements.txt` is a generated compatibility
export; regenerate it with `uv export --locked --group profiling --no-hashes
--output-file core/requirements.txt`. Do not edit the export directly.

## llama.cpp profiling

For the current runner, use `unix/run.sh` on Linux/macOS or `windows\run.ps1`
on Windows, as described in [core/README.md](core/README.md). The wrappers select
the locked `profiling` group and Python 3.12. Set `PYTHON` to another Python 3.12+
executable/version if needed. Native compilers and optional CUDA/Graphviz remain
system prerequisites.

For the preserved historical experiment, start with the [measured findings](experiments/llama-cpp/2026-09-22/REPORT.md), then follow the experiment's [reproduction guide](experiments/llama-cpp/2026-09-22/README.md) for its pinned source, model, build and analysis commands. That guide states which dependencies and large artifacts must be obtained locally.

Preserve the recorded results when repeating a run: use new output names, record the source/model/backend configuration, and compare numerical outputs as well as timing. The baseline's Mac CPU and Metal measurements do not establish performance on another host or a proposed accelerator.

## Shared interfaces

Review shared interfaces and numerical expectations with the affected teams.

## Existing example

A small numerical MAC reference, four unit tests and a deterministic vector generator are present. They are examples of reference work, with the scope described by the MAC contract.

Prerequisites: Make and uv (Python 3.11+; 3.12 selected by default). The current Python code uses the standard library.
The unit tests run within this repository. Vector generation and the combined example use paths to sibling components.

From this component directory:

```sh
make setup
make doctor
make test
```

These commands exercise the MAC reference and tooling tests. Set `PYTHON=3.11` to check the supported minimum, for example `make test PYTHON=3.11`. They do not run the llama.cpp experiment or establish full-accelerator, board or physical-implementation correctness.

For vectors, point `CONTRACT` at the shared MAC contract if the sibling layout differs:

```sh
make vectors CONTRACT=../architecture/contracts/mac-v0.json
```

The combined example lives in the sibling `accelerator` repository. Update that checkout for the Software rename: its runner prefers a sibling named `software` and supports an existing `ml-models` folder as a fallback. The reference behavior and tests here are unchanged.

## Example files

- [reference.py](reference.py)
- [generate_vectors.py](generate_vectors.py)
- [tests/test_reference.py](tests/test_reference.py)

For the shared example, see the [workspace checkout guide](https://github.com/SiliconBadgers/soc/blob/main/docs/GETTING_STARTED.md).
