# Reference checks for generated plans (issue 28, item C)

Work item C of [issue 28](https://github.com/SiliconBadgers/software/issues/28),
built on the [compiler foundation](compiler.md). It provides small
deterministic inputs with expected outputs, and a checker that runs a tiled
matmul plan on those values and compares the result with an untiled reference.
Nothing here encodes commands or runs on a device. Every expected value comes
from this Python reference, not from hardware.

## What is checked

The [foundation](compiler.md) plans carry shapes and dependencies but no values.
D's tiler and E's movement checker check structure and ordering. This step adds
values: a plan is correct only if executing its tiles reproduces the reference
sums exactly.

| File | Responsibility |
| --- | --- |
| `engine/sbengine/codegen/reference.py` | The SoC experiment's arithmetic, seeded input generation and the fixture builder |
| `engine/sbengine/codegen/reference_check.py` | Runs tiles on INT8 operands, records violations and compares C with the reference |
| `engine/tests/fixtures/compiler-reference.json` | Committed cases and expected outputs (24 KB) |
| `engine/tests/test_codegen_reference.py` | Arithmetic, fixture-stability and tiled-plan tests |

## Declared arithmetic

The reference reproduces the [SoC projection experiment](https://github.com/SiliconBadgers/soc/blob/main/docs/HLS.md)
as written in soc `sim/workload_probe.cpp` (`quantize` and `replace`):

1. One symmetric INT8 scale per weight row and per activation vector:
   `scale = peak / 127` in float32, or 1.0 for an all-zero vector.
2. `q = clamp(lround(value / scale), -127, 127)`. Division is float32, halves
   round away from zero and -128 never appears.
3. INT8 × INT8 products summed exactly. The probe sums 8-wide K tiles in INT32
   and adds them on the host in int64.
4. `output = float(sum) * a_scale * b_scale`, rounded to float32 after each step.

This is the experiment's test conversion. It is not the GGUF's arithmetic, an
accepted model quantization, or the proposed Matrix unit's output
requantization, which is not specified yet.

**Cross-check (2026-11-07):** the probe's `quantize` and sum/scale lines were
compiled with g++ 15.2.0 (`-O2`, x86-64) and fed the same float32 bits. They
matched all 304 fixture outputs and 3,409 random outputs (exact halves,
subnormals, tiny and near-overflow magnitudes) on both the integer sum and the
float32 bit pattern. That harness is a local copy of probe code and is not
committed; it does not run the RTL, and a build with flush-to-zero or
fast-math flags could differ on subnormals.

**Underflow:** a vector whose largest magnitude is below about 8.9e-44 (64
float32 subnormal steps) gives a scale of zero. The probe then divides by zero
and calls `lround` on infinity or NaN, which C leaves undefined. The compiled
copy returned -127 for both a tiny value and a zero. The reference raises
`ValueError` for such vectors instead of reproducing that result.

## Cases

| Case | Shape (M×N×K) | Purpose |
| --- | --- | --- |
| `hand_2x2x3` | 2×2×3 | Can be checked by hand: a half rounded away from zero, scale 1.0, a zero row |
| `extremes_4x3x16` | 4×3×16 | Peaks of either sign, a constant row, an outlier that zeroes the rest, tiny values, a zero vector |
| `fixture_5x3x7` | 5×3×7 | Same shape as `compiler-matmul.json`, so it can be checked with a lowered plan |
| `edge_tiles_11x19x21` | 11×19×21 | Edge tiles in M, N and K for any 8/16/8-style tiling |
| `soc_prefill_8x7x2048` | 8×7×2048 | The SoC experiment's prefill shape (8 channels, 7 tokens) |
| `soc_decode_8x1x2048` | 8×1×2048 | The SoC experiment's decode shape (1 token) |

The first two cases are written inline. The rest are generated from seeds using
only `random.random()`, whose sequence Python keeps stable across versions. The
fixture stores the seeds and SHA-256 digests of the inputs and quantized
operands, plus every scale, integer sum and float32 output. Quantized values are
embedded only for the small cases. All inputs are synthetic, not model data.

Regenerate the fixture after an intended arithmetic change, and explain the
change in its own commit:

```sh
cd engine && python -B -m sbengine.codegen.reference > tests/fixtures/compiler-reference.json
```

A test fails if the committed file differs from the regenerated one.

## Checking a tiled plan

```python
from sbengine.codegen.reference import case_inputs, quantize_operands
from sbengine.codegen.reference_check import check_tiles

a_q, _, b_q, _ = quantize_operands(*case_inputs("edge_tiles_11x19x21"))
run = check_tiles(a_q, b_q, tiled.tiles(), plan=plan)   # e.g. D's TiledMatmul
assert run.passed, run.summary()
```

A tile is any object with `m0, n0, k0, m, n, k, clear_acc, write_output`, as in
D's `MatmulTile`. The checker reads only the tiles and operands. It does not
reuse the tiler's bookkeeping.

| Mode | Models | Rules |
| --- | --- | --- |
| `matrix_accumulate` (default) | Command-format Matrix unit | `clear_acc` starts an output tile; later K tiles must continue the same region; `write_output` writes it once its full K range is covered |
| `host_combine` | SoC experiment | Every tile clears and writes; the host adds INT32 partial sums |

Violations reported: `invalid_tile`, `out_of_bounds`, `exceeds_tile_limit`,
`accumulate_without_clear`, `accumulator_region_mismatch`, `k_overlap`,
`incomplete_k`, `duplicate_write`, `accumulator_discarded`, `missing_output`,
`not_standalone`, `accumulator_overflow` and `value_mismatch`. The default tile
limits are the SoC test format (M, K ≤ 8, N ≤ 16). Pass `TileLimits` for
another target.

The tests run every case under four tilings in D's run order (m, then n, then k
innermost), the probe's own command sequence for both SoC shapes, a plan
lowered from `compiler-matmul.json`, and twelve deliberately broken plans that
must each fail with the matching violation. D's real `TiledMatmul` is used when
`sbengine.codegen.tile` is present. Otherwise that test is skipped, and a skip
is not a pass.

```sh
python -B -m unittest discover -s engine/tests -p test_codegen_reference.py -v
```

## Open items (for B and the Matrix interface)

- **Accumulator width and overflow:** not specified for the proposed unit.
  Sums outside INT32 are reported as `accumulator_overflow`, not wrapped or
  saturated. The SoC shapes stay far inside INT32.
- **Output requantization on `write_output`:** the spec says Matrix
  requantizes, while E's plan assumes 4-byte INT32/FP32 C. C holds raw integer
  sums here until B fixes the rule.
- **Block-scaled mode and scale tables:** not modeled.
- **Operand roles and layout:** values follow the plan's A = weights,
  B = activations. E records the proposed unit's swapped roles as an open gap.
- **E's command stream:** not checked yet. The next step is to read E's
  `TRANSFER`/`MATRIX` records instead of bare tiles, once that format settles.
- **Real model data:** waits for item A. Synthetic inputs are enough to check
  plan correctness, but they say nothing about model quality.
