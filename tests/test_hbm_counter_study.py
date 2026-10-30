"""The HBM counter study passes its sanity checks, compares schemes fairly, and never overwrites silently."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research/compiler-scheduling/2026-10-25-hbm-counter-hazard/sim.py"


def load_study():
    spec = importlib.util.spec_from_file_location("hbm_counter_study", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HbmCounterModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = load_study()

    def test_sanity_checks_pass(self):
        self.assertEqual(self.sim.run_checks(), [])

    def test_fixed_latency_counter_matches_scoreboard(self):
        wl = self.sim.decode_mix(2, 64 * 1024, 1024, 256, weight_bytes=2 * self.sim.MIB)
        r = self.sim.run_variants(wl, self.sim.fixed_latency(62), rob_slots=64)
        self.assertEqual(r["counter_only_loss"], 0.0)
        self.assertEqual(r["loss_vs_ideal"]["arrival_counter"], 0.0)

    def test_counter_only_delays_publication(self):
        wl = self.sim.decode_mix(2, 64 * 1024, 4096, 256, weight_bytes=2 * self.sim.MIB,
                                 stream=True)
        for drain in ("in_order", "arrival"):
            c = self.sim.simulate(wl, "counter", self.sim.prior_latency(16), drain=drain)
            s = self.sim.simulate(wl, "scoreboard", self.sim.prior_latency(16), drain=drain)
            self.assertGreaterEqual(c["makespan_cycles"], s["makespan_cycles"])
            self.assertEqual(s["publish_delay_cycles"], 0.0)

    def test_probe_replay_matches_measurement(self):
        for row in self.sim.validate_against_probe():
            self.assertGreater(row["ratio"], 0.85, row)
            self.assertLess(row["ratio"], 1.15, row)

    def test_measured_histogram_has_all_probe_samples(self):
        self.assertEqual(sum(self.sim.MEASURED_FIRST_BEAT.values()), 128)

    def test_bigger_buffer_never_hurts(self):
        for row in self.sim.min_buffer_table(quick=True):
            fractions = [c["fraction_of_ideal"] for c in row["curve"]]
            self.assertEqual(fractions, sorted(fractions), row)
            self.assertLessEqual(fractions[-1], 1.0 + 1e-9, row)

    def test_wait_on_future_command_is_rejected(self):
        bad = [self.sim.Transfer("load", 512, "matrix", False, 1.0, [("matrix", 3)])]
        with self.assertRaises(ValueError):
            self.sim.simulate(bad, "counter", self.sim.fixed_latency(10))


class HbmCounterOutput(unittest.TestCase):
    def setUp(self):
        self.sim = load_study()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name) / "out"
        self.target = self.out / "results.json"

    def run_main(self, *argv):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return self.sim.main(["--out", str(self.out), "--quick", *argv])

    def test_writes_fresh_output(self):
        self.assertEqual(self.run_main(), 0)
        rows = json.loads(self.target.read_text(encoding="utf-8"))["rows"]
        self.assertTrue(rows)
        self.assertTrue((self.out / "sweep.csv").exists())

    def test_existing_output_needs_force(self):
        self.out.mkdir(parents=True)
        self.target.write_text('{"previous": true}', encoding="utf-8")
        before = self.target.read_bytes()
        self.assertEqual(self.run_main(), 1)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(self.run_main("--force"), 0)
        self.assertNotEqual(self.target.read_bytes(), before)

    def test_failed_check_writes_nothing(self):
        self.sim.run_checks = lambda: ["forced failure"]
        self.assertEqual(self.run_main(), 1)
        self.assertFalse(self.target.exists())


if __name__ == "__main__":
    unittest.main()
