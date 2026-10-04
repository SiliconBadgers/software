"""Residency simulation and schedule bounds, with cases whose answers are worked out by hand."""
import random
import unittest

from helpers import ENGINE  # noqa: F401
from sbengine.costs import Cost
from sbengine.memory import per_op_spill, simulate_residency
from sbengine.schedule import schedule


def node(reads=(), writes=()):
    return Cost(reads=[(r, b, 1) for r, b in reads], writes=list(writes))


class Residency(unittest.TestCase):
    def chain(self):
        # 0: writes A(100)   1: reads A, writes B(100)   2: reads A and B, writes O(50) (a graph output)
        return [node(writes=[("A", 100)]), node(reads=[("A", 100)], writes=[("B", 100)]),
                node(reads=[("A", 100), ("B", 100)], writes=[("O", 50)])]

    def test_everything_fits(self):
        r, w, peak, stats = simulate_residency(self.chain(), capacity=250)
        self.assertEqual((sum(r), sum(w), peak), (0.0, 50.0, 0.0))                 # only the output leaves the chip
        self.assertEqual(stats["misses"], 0)

    def test_tight_capacity_spills_the_new_result(self):
        r, w, peak, _ = simulate_residency(self.chain(), capacity=150)
        # B cannot be placed next to the pinned A: written to HBM (100), re-read by node 2 (100), plus the output (50)
        self.assertEqual((sum(r), sum(w), peak), (100.0, 150.0, 100.0))

    def test_evicts_the_tensor_needed_farthest_in_the_future(self):
        costs = [node(writes=[("A", 100)]), node(writes=[("B", 100)]), node(writes=[("C", 100)]),
                 node(reads=[("B", 100)]), node(reads=[("A", 100)]), node(reads=[("C", 100)])]
        r, w, _, _ = simulate_residency(costs, capacity=200)
        # placing C evicts A (used at node 4) rather than B (used at node 3): A is dirty -> written back at node 2, reloaded at 4
        self.assertEqual((w[2], r[3], r[4]), (100.0, 0.0, 100.0))
        self.assertEqual((sum(r), sum(w)), (100.0, 100.0))

    def test_tensor_larger_than_capacity_streams_with_its_reuse_factor(self):
        costs = [node(writes=[("BIG", 1000)]), Cost(reads=[("BIG", 1000, 3)])]
        r, w, peak, _ = simulate_residency(costs, capacity=100)
        self.assertEqual((w[0], r[1], peak), (1000.0, 3000.0, 1000.0))

    def test_more_capacity_never_means_more_traffic(self):
        rng = random.Random(3)
        costs = []
        for i in range(60):
            reads = [(f"t{j}", 40 + 10 * (j % 5)) for j in rng.sample(range(max(1, i)), min(i, rng.randint(0, 3)))]
            costs.append(node(reads=reads, writes=[(f"t{i}", 40 + 10 * (i % 5))]))
        prev = float("inf")
        for cap in (50, 100, 200, 400, 800, 3000):
            r, w, _, _ = simulate_residency(costs, cap)
            total = sum(r) + sum(w)
            self.assertLessEqual(total, prev + 1e-9)
            prev = total

    def test_legacy_spill_proxy(self):
        c = Cost(pool="Vector", reads=[("A", 300, 1)], writes=[("B", 300)])
        spill, peak = per_op_spill([c], capacity=400)
        self.assertEqual((spill, peak), ([2 * (600 - 400)], 600))
        mm = Cost(pool="Matrix", reads=[("A", 300, 1)], writes=[("B", 300)])
        self.assertEqual(per_op_spill([mm], 1)[0], [0.0])                           # matmul carries no proxy spill


class Schedule(unittest.TestCase):
    def test_optional_intervals_preserve_bounds_and_include_auxiliary_work(self):
        args = ([1.0, 1.0, 0.5], ["Matrix", "Vector", "Matrix"],
                [[], [], [0, 1]], [0.1] * 3, [0.2] * 3, 0.1,
                [{"Vector": 0.5}, {}, {}])
        original = schedule(*args)
        detailed = schedule(*args, detail=True)
        self.assertEqual(original, {key: detailed[key] for key in original})
        self.assertEqual(detailed["operations"][2]["start"], 1.6)
        self.assertEqual(detailed["reservations"]["Vector"][0],
                         {"node": 0, "start": 0.0, "end": 0.5})

    def test_independent_ops_on_different_pools_overlap(self):
        s = schedule([1.0, 1.0], ["Matrix", "Vector"], [[], []], [0.0, 0.0], [0.0, 0.0], 0.0, [{}, {}])
        self.assertEqual((s["serial"], s["list"]), (2.0, 1.0))

    def test_same_pool_serializes_and_dependencies_add_handoff(self):
        s = schedule([1.0, 1.0], ["Matrix", "Matrix"], [[], []], [0, 0], [0, 0], 0.0, [{}, {}])
        self.assertEqual(s["list"], 2.0)
        chain = schedule([1.0, 1.0, 1.0], ["Matrix", "Vector", "Matrix"], [[], [0], [1]], [0] * 3, [0] * 3, 0.1, [{}] * 3)
        self.assertAlmostEqual(chain["list"], 3.2)                                   # two cross-pool hops
        self.assertAlmostEqual(chain["critical_path"], 3.2)

    def test_secondary_pool_blocks_that_pool(self):
        # op 0 runs on Matrix and also keeps Vector busy for 1.0; the independent Vector op 1 must wait for it
        s = schedule([1.0, 1.0], ["Matrix", "Vector"], [[], []], [0, 0], [0, 0], 0.0, [{"Vector": 1.0}, {}])
        self.assertEqual(s["list"], 2.0)
        self.assertLessEqual(s["lower_bound"], s["list"])

    def test_memory_service_is_a_floor(self):
        s = schedule([1.0, 1.0], ["Matrix", "Vector"], [[], []], [0.0, 0.0], [1.5, 1.5], 0.0, [{}, {}])
        self.assertEqual(s["list"], 3.0)                                             # 3.0 s of HBM service is a hard floor
        s2 = schedule([1.0, 1.0], ["Matrix", "Vector"], [[], []], [0.0, 0.0], [0.7, 0.7], 0.0, [{}, {}])
        self.assertAlmostEqual(s2["list"], 1.4)

    def test_bounds_are_ordered_on_random_dags(self):
        rng = random.Random(11)
        for _ in range(50):
            n = rng.randint(2, 30)
            durations = [rng.uniform(0.1, 2) for _ in range(n)]
            pools = [rng.choice(["Matrix", "Vector", "DMA"]) for _ in range(n)]
            deps = [sorted(rng.sample(range(i), min(i, rng.randint(0, 3)))) for i in range(n)]
            l1 = [d * rng.uniform(0, 1) for d in durations]
            hbm = [d * rng.uniform(0, 1) for d in durations]
            s = schedule(durations, pools, deps, l1, hbm, 0.05, [{}] * n)
            self.assertLessEqual(s["lower_bound"], s["list"] + 1e-9)
            self.assertLessEqual(s["list"], s["serial"] + 0.05 * n + 1e-9)          # hand-offs are not in the serial sum
            self.assertGreaterEqual(s["list"], s["critical_path"] - 1e-9)


if __name__ == "__main__":
    unittest.main()
