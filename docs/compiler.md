# Qwen compiler workflow: local foundation

This is the first working path from the shared `core/cdfg` to a readable
compiler plan. It is related to [Software issue 3](https://github.com/SiliconBadgers/software/issues/3)
and builds on [refactor PR 21](https://github.com/SiliconBadgers/software/pull/21).
The [central Architecture diagram](https://github.com/SiliconBadgers/architecture/blob/main/docs/accelerator-diagram.md)
remains the system reference. Source placement follows [DESIGN-RULES.md](DESIGN-RULES.md).

## Run the base

From `engine/`, using the repository's existing locked environment:

```sh
uv run --locked --group engine python -m sbengine.codegen \
  ../results/2026-09-27_1839_windows-2/graphs/pp128-fa-on/prefill.json.gz \
  --phase prefill --tensor-id 18
```

The selected captured projection has `M=6144`, `N=128`, `K=2048`.
The selector is the **output tensor ID**, not a node-list index. The plan retains
the captured mixed-quantized weight format and source buffer descriptors; it does
not assert that the accelerator can execute those formats.

To inventory the whole active graph, omit `--tensor-id`. Unsupported work is
reported in JSON and causes exit status 2. Explicit `--allow-partial` returns 0
for a coverage inventory; it does not implement fallback or enable execution.
Both modes mark their output `execution_status: symbolic_only`.

```sh
mkdir -p ../build/compiler
uv run --locked --group engine python -m sbengine.codegen \
  ../results/2026-09-27_1839_windows-2/graphs/pp128-fa-on/prefill.json.gz \
  --phase prefill --allow-partial > ../build/compiler/pp128-plan.json
```

The synthetic fixture also runs without NumPy or model downloads:

```sh
uv run --locked python -m sbengine.codegen tests/fixtures/compiler-matmul.json \
  --phase prefill
```

Run focused checks from the repository root:

```sh
uv run --locked python -m unittest discover -s engine/tests -p test_codegen.py -v
```

Use the [component checks](../tests/README.md) for affected shared behavior and
`uv run --locked python scripts/check_boundaries.py` for repository boundaries.
The compiler code itself uses only the standard library.

## What this base establishes

`engine/sbengine/codegen/` has three responsibilities:

- `plan.py`: immutable source-buffer references, matmul plans, diagnostics and
  deterministic JSON serialization. Byte extents, shapes and aliases are checked.
- `lower.py`: use the existing `Graph`, retained capture and dependency pass to
  lower a supported unbatched, contiguous `MUL_MAT`. No second graph loader or
  performance engine is introduced.
- `__main__.py`: a small CLI for a selected operation or explicit active-graph
  coverage. Missing/unsupported work cannot silently receive a successful strict
  compilation status.

GGML stores innermost dimensions first. For captured operands `A[K,M,1,1]`,
`B[K,N,1,1]` and result `C[M,N,1,1]`, the plan records conventional `C=A*B` with
logical dimensions `M,N,K`. It preserves numerical formats and raw operation
parameters without interpreting them as target arithmetic. Strides may describe
packed quantization blocks; byte counts are preserved, not inferred as a scalar
bytes-per-element approximation.

Input views retain storage-owner IDs and cumulative byte offsets. Output/input
storage aliases, broadcast/batch dimensions, non-contiguous layouts, missing
descriptors and incompatible dimensions are rejected by this first subset.
The decoded-content SHA-256 is computed by the existing provenance helper, so
whitespace/compression changes do not change input identity.

`plans` and `diagnostics` partition the selected active CDFG nodes. Metadata and
zero-size nodes remain in the source CDFG/capture and are not counted as active
operations. A plan's `dependencies` are captured dependency **tensor IDs**;
they may reference operations outside a selected subgraph. They are not queue
wait targets, and this base does not establish a legal full-graph schedule.

There is no numerical executor, target profile, physical allocation, tiling,
hardware command serialization or runtime submission yet. Coverage of logical
matmul shapes is not accelerator capability, model-quality validation or speedup.
The small fixture contains metadata only, not pretrained weights or activations.

## How this becomes an end-to-end compiler

```text
shared CDFG -> source-semantic plans -> legal implementation/tile choices
 -> commands with buffer effects -> storage allocation + queue schedule
 -> reference validation -> ABI encoding + runtime submission
```

Start in Python because graph semantics, mapping choices, analytical evaluation
and tests already have canonical Python owners. Revisit MLIR when standard
transformations replace enough custom tiling/fusion/layout code to justify its
integration. The command plan and validation fixtures should survive that choice.

| Phase | Work to add | Evidence required before the next phase |
|---|---|---|
| 1: semantics and contracts | Expose reusable missing semantics in `core/cdfg`; pin model/captures; inventory all ops; document supported formats, layouts, accumulator and firmware contracts. | Every operation/state effect classified; unresolved hardware features explicit. |
| 2: conservative lowering | Add concrete tiles, transfers and agreed Vector operations; a functional reference; separate buffers and full-completion dependencies first. | One projection, then FFN chain, numerically checked under declared arithmetic. |
| 3: storage and coverage | Allocate SRAM; preserve aliases and asynchronous lifetimes; generate queue waits; extend attention, convolution and recurrence; implement any claimed host fallback. | Bounds/hazards checked; supported graph runs; repeated decode preserves all state. |
| 4: optimization | Legal tile search, copy elimination, buffer/operand reuse, supported fusion, then overlap. Reuse existing mapping/cost/study owners. | Each optimization independently preserves outputs/state and reports the actual chosen plan's costs. |
| 5: device integration | Encode a defined ABI; bind addresses/counter state; add submission/completion/error handling and a live graph adapter under `integration/`. | Relevant RTL/SoC/device comparisons with unrun checks explicit. |

Use a versioned target contract, not analytical-engine defaults, to authorize
commands. In the reviewed RTL, the raw INT8 Matrix array does not retain partial
sums across A tiles; the command proposal's K accumulation, scaling and Vector
operation IDs need their corresponding implementations/contracts. Pin and review
those sources before enabling a target. A direct small INT8 array fixture and a
symbolic full-projection plan validate different scopes.

## Four-person work division

Assign people to interfaces rather than four independent compiler prototypes:

| Owner | First parallel task | Later responsibility |
|---|---|---|
| A: graph semantics | Capture/operation inventory, descriptors, layouts, aliases and state fixtures. | Full source coverage, partition boundaries and shape guards. |
| B: operation lowering | Matrix tile/arithmetic contracts; one independently checked implementation. | Supported Vector, attention and recurrence decompositions. |
| C: memory and commands | Target constraints and buffer-effects contract; conservative dependencies. | SRAM allocation, transfers, queue waits, safe reuse/overlap. |
| D: validation and integration | Functional oracle, CLI/plan contract, compact fixtures and invalid-plan checks. | State/reference integration, reports and runtime adapters. |

All four can begin on day one after agreeing on record formats. Actual storage
reuse needs buffer effects and completion semantics; numerical integration needs
implemented lowerings; optimization needs a correct baseline. Integrate a shared
fixture daily and cross-review changes at these boundaries. Timeloop can supply
mapping candidates as a bounded parallel study without becoming a CLI dependency.

For four people over two to three weeks, scope the first release to a pinned
offline Qwen configuration and supported reference execution. Report graph,
reference-execution, accelerator and hardware-verification coverage separately.
An unsupported diagnostic is useful accounting but is not working fallback.
All-accelerator execution and new numerical formats depend on external hardware
and firmware work; coding-agent access does not remove those dependencies.

AI assistance: OpenAI Codex. Exact model ID unavailable in this session.
