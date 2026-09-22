# Software setup

No tools are required to read the charter, results or design material. Use the setup for the work you are doing; the MAC example and llama.cpp experiment have separate dependencies.

## llama.cpp profiling

Start with the [measured findings](experiments/llama-cpp/2026-09-22/REPORT.md), then follow the experiment's [reproduction guide](experiments/llama-cpp/2026-09-22/README.md) for its pinned source, model, build and analysis commands. That guide states which dependencies and large artifacts must be obtained locally.

Preserve the recorded results when repeating a run: use new output names, record the source/model/backend configuration, and compare numerical outputs as well as timing. The baseline's Mac CPU and Metal measurements do not establish performance on another host or a proposed accelerator.

## Contribution provenance

Follow [Git AI setup](docs/git-ai.md) for AI-assisted contributions. Shared interfaces and numerical expectations should still be reviewed with the affected teams.

## Existing example

A small numerical MAC reference, four unit tests and a deterministic vector generator are present. They are examples of reference work, with the scope described by the MAC contract.

Prerequisites: Make and Python 3.11+. The current Python code uses the standard library.
The unit tests run within this repository. Vector generation and the combined example use paths to sibling components.

From this component directory:

```sh
make setup
make doctor
make test
```

These commands exercise four MAC reference tests. They do not run the llama.cpp experiment or establish full-accelerator, board or physical-implementation correctness.

For vectors, point `CONTRACT` at the shared MAC contract if the sibling layout differs:

```sh
make vectors CONTRACT=../architecture/contracts/mac-v0.json
```

The combined example lives in the sibling `accelerator` repository. Update that checkout for the Software rename: its runner prefers a sibling named `software` and supports an existing `ml-models` folder as a fallback. The reference behavior and tests here are unchanged.

## Example files

- [reference.py](reference.py)
- [generate_vectors.py](generate_vectors.py)
- [tests/test_reference.py](tests/test_reference.py)

For the shared example, see the [workspace checkout guide](https://github.com/SiliconBadgers/accelerator/blob/main/docs/GETTING_STARTED.md).
