"""Old entry points remain usable beside the new internal stage organization."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from helpers import ENGINE, real_run
from cdfg import Graph
from sbengine import costs, mapping
from sbengine.graph import Workload, capture_path, has_capture, latest_run, load_workload
from sbengine.memory import fused_roots

ROOT = ENGINE.parent


class Compatibility(unittest.TestCase):
    def test_old_imports_share_maintained_implementations(self):
        workload = load_workload(real_run() / "graphs/pp512-fa-on")
        self.assertIsInstance(workload, Workload)
        self.assertIsInstance(workload.prefill, Graph)
        self.assertTrue(has_capture(workload.directory))
        self.assertTrue(callable(latest_run))
        self.assertEqual(capture_path(workload.directory, "prefill").suffix, ".gz")
        self.assertIs(costs._tile_plan, mapping._tile_plan)
        self.assertIs(fused_roots, mapping.fused_roots)

    def test_old_and_resolved_profile_commands_match(self):
        with tempfile.TemporaryDirectory() as temp:
            results = []
            for index, profile in enumerate(("accel-balanced", "designs/baselines/characterized-v1/profile.json")):
                out = Path(temp) / f"eval{index}.json"
                result = subprocess.run([sys.executable, "-B", "-m", "sbengine", "eval", "--run", str(real_run()),
                                         "--graph", "pp512-fa-on", "--profile", profile, "--json", str(out)],
                                        cwd=ENGINE, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                results.append(json.loads(out.read_text()))
            self.assertEqual(results[0], results[1])

    def test_core_cli_options_are_available_without_running_capture(self):
        for command in ("profile", "doctor", "analyze"):
            result = subprocess.run([sys.executable, "-B", "core/runner/cli.py", command, "--help"],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            if command != "analyze":
                self.assertIn("--cuda", result.stdout)
                self.assertIn("--dry-run", result.stdout)


if __name__ == "__main__":
    unittest.main()
