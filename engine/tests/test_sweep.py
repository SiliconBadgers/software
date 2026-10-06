"""Sweeps, Pareto fronts, ablation, sensitivity and Monte-Carlo."""
import json
import tempfile
import unittest
from pathlib import Path

from helpers import real_run
from sbengine import config, sweep
from sbengine.graph import load_workload
from sbengine.model import evaluate

RUN = real_run()


class Cost(unittest.TestCase):
    def test_bandwidth_and_direct_weight_path_are_not_free(self):
        from sbengine.resources import resources
        base = config.make_config()
        more_banks = config.make_config({"l1": {"banks": 128}})
        direct = config.make_config({"memory": {"weight_path": "direct"}})
        self.assertGreater(resources(more_banks)["cost"], resources(base)["cost"])
        self.assertGreater(resources(direct)["cost"], resources(base)["cost"])
        free = config.make_config({"resources": {"cost": {"l1_bank": 0.0, "direct_weight_buffer_per_pe": 0.0}}, "l1": {"banks": 128}})
        self.assertEqual(resources(free)["cost"], resources(config.make_config({"resources": {"cost": {"l1_bank": 0.0}}}))["cost"])


class Pareto(unittest.TestCase):
    def test_dominance_and_direction(self):
        recs = [{"valid": True, "cost": 1, "decode_s": 5}, {"valid": True, "cost": 2, "decode_s": 3},
                {"valid": True, "cost": 2, "decode_s": 4}, {"valid": True, "cost": 3, "decode_s": 3},
                {"valid": False, "cost": 0, "decode_s": 0}]
        front = sweep.pareto(recs)
        self.assertEqual([(r["cost"], r["decode_s"]) for r in front], [(1, 5), (2, 3)])
        dup = recs[:2] + [{"valid": True, "cost": 2, "decode_s": 3, "tag": "tie"}]
        self.assertEqual(len(sweep.pareto(dup)), 2)                                  # identical objective values reported once
        best = sweep.pareto(recs, (("cost", "min"), ("decode_s", "max")))
        self.assertIn((1, 5), [(r["cost"], r["decode_s"]) for r in best])


@unittest.skipIf(RUN is None, "no captured run under results/")
class Analyses(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.w = load_workload(RUN / "graphs" / "pp128-fa-on")
        cls.cfg = config.load_profile("accel-balanced")[0]

    def test_sweep_files_carry_their_assumptions(self):
        axes = {"matrix.count": [1, 2], "recurrent.count": [0, 2]}
        records = sweep.run_sweep(self.w, self.cfg, axes)
        self.assertEqual(len(records), 4)
        self.assertTrue(all(r["valid"] for r in records))
        by = {(r["matrix.count"], r["recurrent.count"]): r for r in records}
        self.assertLessEqual(by[(2, 2)]["prefill_s"], by[(1, 2)]["prefill_s"])      # more matrix units, not slower
        self.assertLess(by[(1, 0)]["cost"], by[(1, 2)]["cost"])
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = sweep.write_sweep(tmp, "t", records, self.cfg, axes, self.w, "test")
            meta = json.loads(Path(tmp, "t.meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["base_config_hash"], config.config_hash(self.cfg))
            self.assertEqual(meta["base_config"]["l1"]["banks"], self.cfg["l1"]["banks"])
            self.assertEqual(meta["workload"], "pp128-fa-on")
            self.assertEqual(len(csv_path.read_text(encoding="utf-8").splitlines()), 5)

    def test_invalid_grid_points_are_kept_and_flagged(self):
        records = sweep.run_sweep(self.w, self.cfg, {"matrix.efficiency": [0.5, 2.0]})
        self.assertEqual([r["valid"] for r in records], [True, False])
        self.assertIn("efficiency", records[1]["errors"])
        with self.assertRaises(ValueError):
            sweep.run_sweep(self.w, self.cfg, {"matrix.count": list(range(50)), "vector.count": list(range(50))}, limit=100)

    def test_ablation_ends_at_the_new_engine(self):
        legacy, new = config.load_profile("legacy-equations")[0], self.cfg
        rows, loo = sweep.ablation(self.w, legacy, new)
        self.assertEqual(len(rows), len(sweep.ABLATION_STEPS) + 1)
        final = evaluate(self.w, new)
        self.assertAlmostEqual(rows[-1]["prefill_s"], final["phases"]["prefill"]["seconds"], places=9)
        self.assertAlmostEqual(rows[-1]["decode_s"], final["phases"]["decode"]["seconds"], places=9)
        self.assertEqual(len(loo), len(sweep.ABLATION_STEPS))

    def test_switch_table_brackets_the_configured_result(self):
        base = config.load_profile("accel-balanced")[0]
        t = sweep.switch_table(self.w, base)
        given, cautious = t["given"], t["conservative"]
        self.assertNotEqual(given["config_hash"], cautious["config_hash"])
        self.assertGreater(cautious["prefill_s"], given["prefill_s"])
        self.assertGreater(cautious["decode_s"], given["decode_s"])
        rows = {r["path"]: r for r in t["rows"]}
        self.assertEqual(set(rows), {s["path"] for s in config.SWITCHES})
        for r in rows.values():                                                # only optimistic switches get a delta
            self.assertEqual("prefill_delta_pct" in r, r["optimistic"], r["path"])
        self.assertGreater(rows["recurrent.lowering"]["prefill_delta_pct"], 20)
        self.assertAlmostEqual(rows["recurrent.lowering"]["decode_delta_pct"], 0)   # chunking never applies to one token
        # a design that is already pessimistic everywhere has nothing to state
        again = sweep.switch_table(self.w, config.conservative(base))
        self.assertEqual(again["given"]["config_hash"], again["conservative"]["config_hash"])
        self.assertEqual(again["conservative"]["prefill_delta_pct"], 0.0)
        self.assertFalse(any(r["optimistic"] for r in again["rows"]))

    def test_switch_table_does_not_compare_infeasible_designs(self):
        """An infeasible design times its unmapped ops at zero, so its seconds must never become a percentage."""
        base = config.load_profile("accel-balanced")[0]
        # valid as configured: the serial update runs on the matrix arrays. Moving recurrent.lowering back to
        # "serial" leaves no unit that can run it (no recurrence unit, no scalar core, vector lacks the capability)
        caps = [c for c in base["vector"]["caps"] if c != "recurrent"]
        cfg = config.make_config({"recurrent": {"lowering": "matrix", "count": 0}, "scalar": {"count": 0},
                                  "vector": {"caps": caps}}, base=base)
        t = sweep.switch_table(self.w, cfg)
        self.assertTrue(t["given"]["valid"])
        rows = {r["path"]: r for r in t["rows"]}
        lowering = rows["recurrent.lowering"]
        self.assertFalse(lowering["alone"]["valid"])
        self.assertTrue(any("GATED_DELTA_NET" in e for e in lowering["alone"]["errors"]))
        self.assertNotIn("prefill_delta_pct", lowering)
        self.assertNotIn("decode_delta_pct", lowering)
        self.assertFalse(t["conservative"]["valid"])                           # it contains the same reversion
        self.assertTrue(t["conservative"]["errors"])
        self.assertNotIn("prefill_delta_pct", t["conservative"])
        feasible = rows["matrix.pipelined_tiles"]                              # other switches are still compared
        self.assertTrue(feasible["alone"]["valid"])
        self.assertIn("prefill_delta_pct", feasible)
        # when the configured design itself is infeasible nothing is compared, even against feasible alternatives
        broken = config.make_config({"matrix": {"count": 0}, "scalar": {"count": 0}}, base=base)
        t = sweep.switch_table(self.w, broken)
        self.assertFalse(t["given"]["valid"])
        self.assertTrue(t["given"]["errors"])
        self.assertNotIn("prefill_delta_pct", t["conservative"])
        self.assertFalse(any("prefill_delta_pct" in r for r in t["rows"]))

    def test_sensitivity_direction_and_binding_flags(self):
        base, rows = sweep.sensitivity(self.w, self.cfg, ["hbm.gbs", "clock_mhz", "l1.banks"], delta=0.3)
        by = {r["param"]: r for r in rows}
        self.assertLessEqual(by["clock_mhz"]["decode_s_elasticity"], 0)               # faster clock never slower
        self.assertLessEqual(by["l1.banks"]["decode_s_elasticity"], 0)
        self.assertLessEqual(by["hbm.gbs"]["decode_s_elasticity"], 1e-9)
        self.assertEqual(base["decode_binding"], "l1")                               # the explorer's default L1-bound decode

    def test_montecarlo_is_seeded_and_bounded(self):
        designs = {n: config.load_profile(n)[0] for n in ("accel-balanced", "accel-efficient")}
        a = sweep.montecarlo(self.w, designs, samples=8, seed=5)
        b = sweep.montecarlo(self.w, designs, samples=8, seed=5)
        self.assertEqual(a, b)
        for r in a.values():
            self.assertTrue(0 <= r["frontier_frequency"] <= 1)
            self.assertLessEqual(r["decode_p05_s"], r["decode_p95_s"])


if __name__ == "__main__":
    unittest.main()
