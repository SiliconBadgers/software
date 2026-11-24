"""Value-level reference and tiled-plan checks (issue 28, item C); standard library only."""
from collections import namedtuple
import json
import math
from pathlib import Path
import sys
import unittest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from sbengine.codegen import lower_matmul
from sbengine.codegen.reference import (
    case_inputs, dequantize, digest, f32, fixture_json, int_matmul, quantize_operands,
    quantize_vector, soc_projection)
from sbengine.codegen.reference_check import TileLimits, check_tiles, execute_tiles
from sbengine.graph import Graph

FIXTURE = ENGINE / "tests/fixtures/compiler-reference.json"
SYMBOLIC = ENGINE / "tests/fixtures/compiler-matmul.json"
Tile = namedtuple("Tile", "m0 n0 k0 m n k clear_acc write_output")


def tiles(M, N, K, tm, tn, tk):
    """Test-only tiler in D's documented run order: m, then n, then k innermost."""
    for m0 in range(0, M, tm):
        for n0 in range(0, N, tn):
            for k0 in range(0, K, tk):
                k = min(tk, K - k0)
                yield Tile(m0, n0, k0, min(tm, M - m0), min(tn, N - n0), k, k0 == 0, k0 + k == K)


def soc_probe_tiles(M, N, K):
    """The probe's command sequence: all rows and columns, K in 8s, each standalone."""
    return [Tile(0, 0, k0, M, N, min(8, K - k0), True, True) for k0 in range(0, K, 8)]


def load_cases():
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return {case["name"]: case for case in fixture["cases"]}


class Arithmetic(unittest.TestCase):
    def test_hand_case_by_hand(self):
        # Row [127, -63.5, 1]: peak 127, scale 1, -63.5 rounds away from zero.
        # Column [1, 2, 3]: scale 3/127; 1 -> 42.33 -> 42, 2 -> 84.67 -> 85, 3 -> 127.
        a_q, a_s, b_q, b_s = quantize_operands([[127.0, -63.5, 1.0], [0.0, 0.0, 0.0]],
                                               [[1.0, 2.0, 3.0], [-127.0, 127.0, 0.0]])
        self.assertEqual(a_q, [(127, -64, 1), (0, 0, 0)])
        self.assertEqual(b_q, [(42, 85, 127), (-127, 127, 0)])
        self.assertEqual(a_s, [1.0, 1.0])
        self.assertEqual(b_s, [f32(3 / 127), 1.0])
        sums = int_matmul(a_q, b_q)
        self.assertEqual(sums, [[127 * 42 - 64 * 85 + 127, -127 * 127 - 64 * 127], [0, 0]])
        self.assertEqual(dequantize(sums, a_s, b_s)[0], [f32(21 * f32(3 / 127)), -24257.0])

    def test_rounding_and_clamp(self):
        q, scale = quantize_vector([127.0, 0.5, -0.5, 1.5, -2.5, -127.0])
        self.assertEqual((q, scale), ((127, 1, -1, 2, -3, -127), 1.0))
        q, scale = quantize_vector([0.0, -0.0, 0.0])
        self.assertEqual((q, scale), ((0, 0, 0), 1.0))
        q, _ = quantize_vector([f32(-3.0e38), f32(3.0e38), 1.0])
        self.assertNotIn(-128, q)
        self.assertEqual(q[:2], (-127, 127))

    def test_float32_steps(self):
        # 2^24 + 1 is not a float32: the probe's float(sum) rounds it to 2^24.
        self.assertEqual(dequantize([[2**24 + 1]], [1.0], [1.0]), [[float(2**24)]])
        self.assertEqual(f32(1e39), math.inf)
        self.assertEqual(f32(-1e39), -math.inf)

    def test_rejects_invalid_inputs(self):
        smallest = 2.0 ** -149  # float32's smallest subnormal
        # A peak under 64 subnormal steps underflows the scale to zero; the probe
        # then divides by zero, which has no defined result to reproduce.
        self.assertEqual(quantize_vector([64 * smallest])[1], smallest)
        for values in ([], [0.1], [math.nan], [math.inf], [True], ["1"], [63 * smallest]):
            with self.assertRaises(ValueError):
                quantize_vector(values)
        with self.assertRaises(ValueError):
            quantize_operands([[1.0, 2.0]], [[1.0]])
        with self.assertRaises(ValueError):
            int_matmul([[128]], [[1]])
        with self.assertRaises(ValueError):
            int_matmul([[1, 2]], [[1, 2]], 1, 2)


class Fixture(unittest.TestCase):
    def test_committed_fixture_regenerates_exactly(self):
        self.assertEqual(FIXTURE.read_text(encoding="utf-8"), fixture_json(),
                         "regenerate with: python -m sbengine.codegen.reference")

    def test_generated_inputs_are_stable(self):
        for name, case in load_cases().items():
            if case["inputs"]["kind"] != "generated":
                continue
            a_rows, b_cols = case_inputs(name)
            self.assertEqual(digest(a_rows), case["inputs"]["a_rows_sha256"], name)
            self.assertEqual(digest(b_cols), case["inputs"]["b_cols_sha256"], name)

    def test_expected_values_follow_from_quantized_operands(self):
        for name, case in load_cases().items():
            a_q, a_s, b_q, b_s = quantize_operands(*case_inputs(name))
            expected = case["expected"]
            self.assertEqual(digest(a_q), expected["a_q_sha256"], name)
            self.assertEqual(digest(b_q), expected["b_q_sha256"], name)
            self.assertEqual(int_matmul(a_q, b_q), expected["sums"], name)
            self.assertEqual(dequantize(expected["sums"], a_s, b_s), expected["output"], name)

    def test_quantized_output_tracks_float_matmul(self):
        # Sanity only: the conversion error is small next to the result's scale.
        for name in ("edge_tiles_11x19x21", "soc_prefill_8x7x2048"):
            a_rows, b_cols = case_inputs(name)
            output = soc_projection(a_rows, b_cols)["output"]
            exact = [[math.fsum(x * y for x, y in zip(r, c)) for c in b_cols] for r in a_rows]
            error = max(abs(o - e) for ro, re in zip(output, exact) for o, e in zip(ro, re))
            magnitude = max(abs(e) for row in exact for e in row)
            self.assertLess(error, 0.05 * magnitude, name)


class TiledPlans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_cases()
        cls.operands = {}
        for name in cls.cases:
            a_q, _, b_q, _ = quantize_operands(*case_inputs(name))
            cls.operands[name] = (a_q, b_q)

    def assert_passes(self, name, tile_list, **options):
        a_q, b_q = self.operands[name]
        run = check_tiles(a_q, b_q, tile_list, expected=self.cases[name]["expected"]["sums"], **options)
        self.assertTrue(run.passed, (name, run.summary()))
        self.assertEqual(run.sums, self.cases[name]["expected"]["sums"])
        return run

    def broken(self, name, tile_list, *kinds, **options):
        a_q, b_q = self.operands[name]
        run = check_tiles(a_q, b_q, tile_list, **options)
        found = {v.kind for v in run.violations}
        self.assertFalse(run.passed)
        for kind in kinds:
            self.assertIn(kind, found, run.summary())
        return run

    def test_every_case_under_several_tilings(self):
        for name, case in self.cases.items():
            M, N, K = (case["shape"][d] for d in "mnk")
            for shape in ((8, 16, 8), (3, 5, 2), (1, 1, 8), (8, 1, 1)):
                with self.subTest(name=name, tile=shape):
                    self.assert_passes(name, list(tiles(M, N, K, *shape)))

    def test_soc_probe_command_sequence(self):
        for name in ("soc_prefill_8x7x2048", "soc_decode_8x1x2048"):
            shape = self.cases[name]["shape"]
            run = self.assert_passes(name, soc_probe_tiles(shape["m"], shape["n"], shape["k"]),
                                     mode="host_combine")
            self.assertEqual(run.tiles, 256)

    def test_checks_a_lowered_plan(self):
        plan = lower_matmul(Graph(SYMBOLIC, "prefill"), 30)
        self.assertEqual((plan.m, plan.n, plan.k), (5, 3, 7))
        self.assert_passes("fixture_5x3x7", list(tiles(5, 3, 7, 2, 2, 3)), plan=plan)
        a_q, b_q = self.operands["edge_tiles_11x19x21"]
        with self.assertRaises(ValueError):
            check_tiles(a_q, b_q, [], plan=plan)

    def test_with_d_tiler_when_available(self):
        try:
            from sbengine.codegen.tile import TiledMatmul
        except ImportError:
            self.skipTest("D's sbengine.codegen.tile is not on this branch (not a pass)")
        plan = lower_matmul(Graph(SYMBOLIC, "prefill"), 30)
        self.assert_passes("fixture_5x3x7", list(TiledMatmul(plan, 2, 2, 3).tiles()), plan=plan)

    def test_broken_plans(self):
        name = "edge_tiles_11x19x21"
        good = list(tiles(11, 19, 21, 8, 16, 8))
        middle_k = next(i for i, t in enumerate(good) if not t.clear_acc and not t.write_output)
        first_write = next(i for i, t in enumerate(good) if t.write_output)

        self.broken(name, good[:middle_k] + good[middle_k + 1:], "incomplete_k")
        self.broken(name, good[:middle_k + 1] + good[middle_k:], "k_overlap")
        self.broken(name, good[:first_write] + good[first_write + 1:],
                    "accumulator_discarded", "missing_output")
        self.broken(name, good[:-1], "accumulator_discarded", "missing_output")
        self.broken(name, [good[0]._replace(clear_acc=False)] + good[1:], "accumulate_without_clear")
        self.broken(name, [t._replace(clear_acc=True) if i == middle_k else t
                           for i, t in enumerate(good)], "accumulator_discarded", "incomplete_k")
        self.broken(name, [good[0]._replace(write_output=True)] + good[1:],
                    "incomplete_k", "duplicate_write")
        self.broken(name, [t._replace(n0=t.n0 + 1) if i == middle_k else t
                           for i, t in enumerate(good)], "accumulator_region_mismatch")
        self.broken(name, good + [good[-1]._replace(m0=10)], "out_of_bounds")
        self.broken(name, list(tiles(11, 19, 21, 11, 19, 21)), "exceeds_tile_limit")
        self.broken(name, good, "not_standalone", mode="host_combine")
        self.broken(name, [], "missing_output")

    def test_value_mismatch_is_reported(self):
        name = "fixture_5x3x7"
        a_q, b_q = self.operands[name]
        wrong = [row[:] for row in self.cases[name]["expected"]["sums"]]
        wrong[4][2] += 1
        run = check_tiles(a_q, b_q, list(tiles(5, 3, 7, 8, 16, 8)), expected=wrong)
        self.assertEqual([v.kind for v in run.violations], ["value_mismatch"])
        self.assertIn("C[4][2]", run.violations[0].detail)

    def test_host_combine_coverage(self):
        name = "soc_decode_8x1x2048"
        sequence = soc_probe_tiles(8, 1, 2048)
        self.broken(name, sequence[:-1], "incomplete_k", mode="host_combine")
        self.broken(name, sequence + sequence[:1], "k_overlap", mode="host_combine")

    def test_accumulator_overflow_is_reported_not_wrapped(self):
        # 127 * 127 * 133,200 = 2,148,382,800 > INT32_MAX; overflow behavior is unspecified.
        K = 133_200
        a_q, b_q = [[127] * K], [[127] * K]
        limits = TileLimits(m=1, n=1, k=K)
        run = execute_tiles(a_q, b_q, [Tile(0, 0, 0, 1, 1, K, True, True)], limits=limits)
        self.assertEqual({v.kind for v in run.violations}, {"accumulator_overflow"})
        run = execute_tiles(a_q, b_q, list(tiles(1, 1, K, 1, 1, K // 2)), limits=limits)
        self.assertEqual({v.kind for v in run.violations}, {"accumulator_overflow"})
        self.assertEqual(run.sums, [[127 * 127 * K]])


if __name__ == "__main__":
    unittest.main()
