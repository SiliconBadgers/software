# ml-models: optional example setup

No tools are required to read the charter or contribute research and design
material. The commands below apply only to the existing technical example.

## Existing example

A small numerical MAC reference, four unit tests and a deterministic vector generator are present. They are examples of reference work, with the scope described by the MAC contract.

Prerequisites: Make and Python 3.11+. The current Python code uses the standard library.
Keep the component folders as siblings for the provided cross-component paths.

From this component directory:

```sh
make setup
make doctor
make test
```

These commands check the current example files. The combined MAC example can be
run from the sibling `accelerator` folder using the same commands. It exercises
four model tests, 261 reference vectors and 131,600 RTL checks. It establishes
no full-accelerator, board or physical-implementation claim.

Run `make clean` from `accelerator` to remove generated build outputs and Python
caches before sharing a folder snapshot. The runner reads the current sibling
files and does not create commits.

## Example files

- [reference.py](reference.py)
- [generate_vectors.py](generate_vectors.py)
- [tests/test_reference.py](tests/test_reference.py)

For the shared example, see the [workspace checkout guide](https://github.com/SiliconBadgers/accelerator/blob/main/docs/GETTING_STARTED.md).
