# Qwen compiler foundation

Foundation for [issue 28](https://github.com/SiliconBadgers/software/issues/28):
validate executed GGML captures and produce symbolic matmul plans and operation
coverage. The semantic-input adapter for [PR 27](https://github.com/SiliconBadgers/software/pull/27)
is follow-up work.

Follow the [source organization](DESIGN-RULES.md) and
[central Architecture diagram](https://github.com/SiliconBadgers/architecture/blob/main/docs/accelerator-diagram.md).

## Use

From `engine/`, select an active operation by its **output tensor ID**:

```sh
uv run --locked python -m sbengine.codegen \
  ../results/2026-09-27_1839_windows-2/graphs/pp128-fa-on/prefill.json.gz \
  --phase prefill --tensor-id 18
```

This projection has `M=6144`, `N=128`, `K=2048`. Omitting `--tensor-id` inventories
all active operations. Unsupported operations produce diagnostics and exit status
2. `--allow-partial` returns 0 for an inventory; it does not implement fallback.
Malformed source input also exits 2. JSON is written to stdout; save generated
plans under ignored `build/compiler/`.

The synthetic fixture requires no model download or NumPy:

```sh
uv run --locked python -m sbengine.codegen tests/fixtures/compiler-matmul.json --phase prefill
```

## Input and plan contract

The supported subset is contiguous, unbatched `MUL_MAT`. GGML lists dimensions
innermost first: operands `[K,M,1,1]` and `[K,N,1,1]` produce `[M,N,1,1]`, recorded
as logical `C=A*B` with dimensions `M,N,K`. Source formats, block strides, byte
extents and storage aliases are checked. Raw operation parameters are retained
without interpreting them as target arithmetic.
Output/input aliasing, incompatible shapes and unsupported layouts are rejected.

`compile_graph(graph, tensor_id=None)` returns plans and diagnostics for the
selected active operations. `lower_matmul(graph, tensor_id)` returns one plan or
raises `LoweringError`. Metadata and zero-size entries remain in the source but
are excluded from active-operation coverage.

Reports use `scope` (`full_graph` or `selected_operation`), operation tensor IDs,
counts and `coverage_by_op` to account for every selected operation exactly once.
Dependencies are source **tensor IDs**, including conservative storage hazards;
`source.external_dependency_tensor_ids` names dependencies outside a selection.
Storage IDs are source identities, not device addresses. The decoded capture
SHA-256 identifies JSON descriptors, not tensor values or model-weight contents.

Every report is `execution_status: symbolic_only`. These plans do not establish
target support or numerical equivalence, allocate hardware buffers, schedule
queues, encode commands or submit device work. The fixture contains descriptors
only. Issue 28 owns the subsequent target, execution and optimization tasks.

## Organization and checks

| File | Responsibility |
|---|---|
| `core/cdfg/source.py` | Capture/Graph coherence, aliases and conservative source dependencies; analytical `Graph.deps()` is unchanged. |
| `engine/sbengine/codegen/plan.py` | Immutable tensor references, matmul plans, diagnostics and validated JSON. |
| `engine/sbengine/codegen/lower.py` | Source-subset checks and symbolic lowering using the existing storage-format definitions. |
| `engine/sbengine/codegen/__main__.py` | Selection, coverage reporting and exit status. |

Run the standard-library checks from the repository root:

```sh
uv run --locked python -m unittest discover -s engine/tests -p test_codegen.py -v
uv run --locked python -m unittest discover -s core/tests -p test_cdfg_source.py -v
uv run --locked python scripts/check_boundaries.py
```

Additional component checks are listed in [tests/README.md](../tests/README.md).

Buffer assignment, transfers and wait targets for one tiled matmul (issue 28,
item E) are described in [compiler-movement.md](compiler-movement.md).
