"""Matmul tiling contracts; standard-library only, without tensor execution."""
from collections import Counter
from pathlib import Path
import sys
import unittest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from sbengine.codegen import MatmulPlan, TensorRef, lower_matmul
from sbengine.codegen.tile import MatmulTile, TiledMatmul
from sbengine.graph import Graph

FIXTURE = ENGINE / "tests/fixtures/compiler-matmul.json"


def f32_ref(tensor_id, inner, outer):
    """A contiguous f32 GGML tensor [inner, outer, 1, 1]."""
    size = inner * outer * 4
    return TensorRef(tensor_id, f"t{tensor_id}", tensor_id, 0, size, (inner, outer, 1, 1),
                     "f32", (4, inner * 4, size, size), size, size)


def plan(m, n, k):
    return MatmulPlan(3, 0, f32_ref(1, k, m), f32_ref(2, k, n), f32_ref(3, m, n),
                      m, n, k, (0,) * 16, ())


def tiles(m, n, k, limits):
    tiled = TiledMatmul(plan(m, n, k), *limits)
    tiled.validate()
    return list(tiled.tiles())


class MatmulTiling(unittest.TestCase):
    def test_single_tile(self):
        """A matmul within the limits is one tile that both clears and writes."""
        self.assertEqual(tiles(2, 2, 2, (8, 16, 8)),
                         [MatmulTile(3, 0, 0, 0, 2, 2, 2, True, True)])

    def test_two_output_tiles(self):
        """M=4, N=2, K=4 with 2/2/2: the top C tile's two K chunks, then the bottom's."""
        self.assertEqual(tiles(4, 2, 4, (2, 2, 2)), [
            MatmulTile(3, 0, 0, 0, 2, 2, 2, True, False),
            MatmulTile(3, 0, 0, 2, 2, 2, 2, False, True),
            MatmulTile(3, 2, 0, 0, 2, 2, 2, True, False),
            MatmulTile(3, 2, 0, 2, 2, 2, 2, False, True),
        ])

    def test_edge_tiles(self):
        """Edge tiles shrink when the fixture (M=5, N=3, K=7) divides unevenly by 2."""
        tiled = TiledMatmul(lower_matmul(Graph(FIXTURE, "prefill"), 30), 2, 2, 2)
        tiled.validate()
        result = list(tiled.tiles())
        self.assertEqual(len(result), 3 * 2 * 4)
        self.assertEqual({(t.m0, t.m) for t in result}, {(0, 2), (2, 2), (4, 1)})
        self.assertEqual({(t.n0, t.n) for t in result}, {(0, 2), (2, 1)})
        self.assertEqual({(t.k0, t.k) for t in result}, {(0, 2), (2, 2), (4, 2), (6, 1)})
        self.assertTrue(all(t.tensor_id == 30 for t in result))

    def test_coverage(self):
        """Every (m, n, k) product is covered exactly once, with no gaps or overlaps."""
        for m, n, k, limits in ((5, 3, 7, (2, 2, 2)), (17, 33, 20, (8, 16, 8)),
                                (8, 16, 8, (8, 16, 8)), (3, 1, 9, (2, 2, 4))):
            with self.subTest(m=m, n=n, k=k, limits=limits):
                covered = Counter((i, j, p) for t in tiles(m, n, k, limits)
                                  for i in range(t.m0, t.m0 + t.m)
                                  for j in range(t.n0, t.n0 + t.n)
                                  for p in range(t.k0, t.k0 + t.k))
                self.assertEqual(len(covered), m * n * k)
                self.assertEqual(set(covered.values()), {1})

    def test_order_and_flags(self):
        """K is innermost; clear_acc marks the first K chunk and write_output the last."""
        m, n, k = 17, 33, 20
        result = tiles(m, n, k, (8, 16, 8))
        for t in result:
            self.assertEqual(t.clear_acc, t.k0 == 0)
            self.assertEqual(t.write_output, t.k0 + t.k == k)
        # Each C tile appears as one contiguous run: clear first, write last, in m-then-n order.
        runs = []
        for t in result:
            if t.clear_acc:
                runs.append([])
            runs[-1].append(t)
        origins = [(run[0].m0, run[0].n0) for run in runs]
        self.assertEqual(origins, sorted(set(origins)))
        for run in runs:
            self.assertEqual({(t.m0, t.n0) for t in run}, {(run[0].m0, run[0].n0)})
            self.assertEqual([t.k0 for t in run], list(range(0, k, 8)))
            self.assertTrue(run[-1].write_output)

    def test_validate(self):
        """Bad tile sizes, a non-plan input and malformed tiles are rejected."""
        good = plan(4, 4, 4)
        for limits in ((0, 2, 2), (2, -1, 2), (2, 2, 2.0), (True, 2, 2)):
            with self.subTest(limits=limits), self.assertRaises(ValueError):
                TiledMatmul(good, *limits).validate()
        with self.assertRaises(ValueError):
            TiledMatmul("not a plan", 2, 2, 2).validate()
        for bad in (MatmulTile(3, 0, 0, 0, 0, 2, 2, True, True),
                    MatmulTile(3, -1, 0, 0, 2, 2, 2, True, True),
                    MatmulTile(3, 0, 0, 0, 2, 2, 2, 1, True)):
            with self.subTest(tile=bad), self.assertRaises(ValueError):
                bad.validate()


if __name__ == "__main__":
    unittest.main()
