"""The memory-traffic study validates every check before it writes, and never overwrites silently."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research/compute-mapping/2026-10-14-memory-traffic/compute.py"


def load_study():
    spec = importlib.util.spec_from_file_location("memory_traffic_study", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MemoryTrafficOutput(unittest.TestCase):
    def setUp(self):
        self.study = load_study()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.out = Path(temporary.name) / "out"
        self.target = self.out / "results.json"

    def run_main(self, *argv, **overrides):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return self.study.main(["--out", str(self.out), *argv], **overrides)

    def seed_existing(self):
        self.out.mkdir(parents=True)
        self.target.write_text('{"previous": true}', encoding="utf-8")
        return self.target.read_bytes()

    def test_passing_checks_write_fresh_output(self):
        self.assertEqual(self.run_main(), 0)
        result = json.loads(self.target.read_text(encoding="utf-8"))
        self.assertEqual(result["weights_elements"], self.study.EXPECTED_WEIGHTS)
        self.assertEqual(len(result["partial_sums"]), len(self.study.TIMELOOP_CASES))
        self.assertEqual([p.name for p in self.out.iterdir()], ["results.json"])

    def test_weight_mismatch_fails_without_writing(self):
        self.assertEqual(self.run_main(expected_weights=1), 1)
        self.assertFalse(self.out.exists())

    def test_weight_mismatch_keeps_existing_output(self):
        before = self.seed_existing()
        self.assertEqual(self.run_main("--force", expected_weights=1), 1)
        self.assertEqual(self.target.read_bytes(), before)

    def test_timeloop_mismatch_keeps_existing_output(self):
        before = self.seed_existing()
        name, M, K, N, kg, c_updates, a_reads = self.study.TIMELOOP_CASES[0]
        cases = [(name, M, K, N, kg, c_updates + 1, a_reads)] + self.study.TIMELOOP_CASES[1:]
        self.assertEqual(self.run_main("--force", cases=cases), 1)
        self.assertEqual(self.target.read_bytes(), before)

    def test_existing_output_needs_force(self):
        before = self.seed_existing()
        self.assertEqual(self.run_main(), 2)
        self.assertEqual(self.target.read_bytes(), before)
        self.assertEqual(self.run_main("--force"), 0)
        self.assertIn("partial_sums", json.loads(self.target.read_text(encoding="utf-8")))
        self.assertEqual([p.name for p in self.out.iterdir()], ["results.json"])

    def test_default_output_is_ignored(self):
        default = Path(self.study.DEFAULT_OUT).resolve()
        self.assertEqual(default.relative_to(ROOT).parts[0], "build")


if __name__ == "__main__":
    unittest.main()
