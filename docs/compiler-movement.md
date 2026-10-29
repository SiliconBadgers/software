# Data movement and ordering (issue 28, item E)

First piece of [issue 28](https://github.com/SiliconBadgers/software/issues/28)
work item E, built on the [compiler foundation](compiler.md). For one tiled
matmul it assigns SRAM buffers, emits the command format's `TRANSFER` and
`MATRIX` records with queue-progress wait targets, and checks the result with
an independent ordering checker. The output is a symbolic plan: nothing is
encoded, addressed in HBM or executed.

## Use

From `engine/`:

```sh
uv run --locked python -m sbengine.codegen.movement \
  ../results/2026-09-27_1839_windows-2/graphs/pp128-fa-on/prefill.json.gz \
  --phase prefill --tensor-id 18 > ../build/compiler/movement-18.json
```

Options: `--tile M,N,K` (stub tiler, default `8,16,8`), `--sram-bytes`,
`--reuse-after {complete,inputs_released}` and `--commands-out PATH` for every
command as JSON lines. The report holds the buffers, counts, bytes moved, a
SHA-256 digest of the command stream, the first commands, the assumptions used
and the checker result. The command exits 1 if the checker finds a violation.

## How the plan is built

| Operand | Placement | Transfers |
| --- | --- | --- |
| B (activations, K x N) | Resident for the whole matmul | One load before any `MATRIX` |
| C (output, M x N) | Resident; `MATRIX` writes tiles into it | One store after the last writer's 3j+3 |
| A (weights, M x K) | Streamed in panels of `tile_m` rows, two slots | One load per panel; panel p reuses the slot of panel p - 2 |

Each `MATRIX` waits for its A panel's load to complete (Memory 2j+2). Because
milestones publish in order, that also covers the earlier B load. A panel load
that reuses a slot waits for the last `MATRIX` reading the previous occupant to
reach its release milestone. The release milestone is configurable:

| `reuse_after` | Waits for | Status |
| --- | --- | --- |
| `inputs_released` (default) | 3j+1 | All of the command's Matrix SRAM reads have finished (Zeb, 2026-10-29) |
| `complete` | 3j+3 | Conservative option; also waits for accumulation and output writes |

The output store waits for the last writer's 3j+3, which waits until the
output writes are committed to SRAM (Zeb, 2026-10-29). The Matrix unit's raw
done signal can fire earlier (Abhi), so the plan never waits on it.

Tiles follow the `TiledMatmul` / `MatmulTile` shape agreed with D (Anu): the
plan, maximum tile sizes and an iterator in run order with k innermost.
`StubTiledMatmul` stands in until D's tiler lands. The planner requires A
panels (m blocks) to be visited in order, each once.

## Checks

`movement_check.py` reads only the commands, the tiles they claim to implement
and the target facts. It does not reuse the planner's bookkeeping. It checks
whole-row transfers inside SRAM and the declared HBM buffers; that every
`MATRIX` input comes from a completed load that brought the right HBM bytes
to the addresses read; that no later load overwrites an input before the
reader's release milestone; that the store waits for every overlapping write
and maps each tile to its HBM bytes; that no write lands in a load
destination; and that the two queues cannot block each other.

`engine/tests/test_movement.py` runs valid plans for whole and edge tiles under
both release policies, and seven deliberately broken plans: a stale input, a
read before its load, an early reuse, a partial-row transfer, an early store, a
wait cycle and an SRAM overflow. Each must fail with the matching violation.
A plan built for `inputs_released` must fail when checked under `complete`.

```sh
uv run --locked python -m unittest discover -s engine/tests -p test_movement.py -v
```

## Projection result (tensor 18)

`M=6144, N=128, K=2048`, stub tiles 8/16/8, default target:

| Item | Value |
| --- | --- |
| `MATRIX` commands | 1,572,864 |
| `TRANSFER` commands | 770 (1 B load, 768 A panel loads, 1 C store) |
| SRAM used | 1,081,344 B (B 256 KiB, C 768 KiB, two 16 KiB A slots) |
| HBM to SRAM | 12,845,056 B (each weight byte once, B once) |
| SRAM to HBM | 786,432 B |
| Checker | passed, 0 violations |
| Run time | about 23 s for planning plus checking |

Two observations for B and D. With 8/16/8 tiles, each `MATRIX` does 1,024
multiply-accumulates, so this one projection needs about 1.6 million commands.
Wait targets reach about 4.7 million on the Matrix counter; `CounterWidth` must
keep live dependencies within half its range, which holds here because each
wait is at most one A panel ahead.

## Assumptions and limits

* Target formats are INT8 for A and B and INT8 requantized output; the
  captured `q5_K` weights and `f32` activations are provenance only. The host
  is assumed to prepare HBM target buffers in these formats, row-major in the
  logical matmul shape, padded to whole SRAM rows.
* 512 B rows and full-row transfers follow the B update (2026-10-28). The
  8 MiB SRAM budget is not confirmed; `--sram-bytes` changes it.
* Only the row-major layout is implemented. Block-quantized target formats
  would need `tile_k` aligned to the format's block size.
* B and C must fit in SRAM. Streaming them, early first-row consumption,
  Vector work, multiple matmuls and counter wraparound are not implemented.
