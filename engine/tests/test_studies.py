"""Study wrappers reuse existing methods, fixed assumptions and baseline identity."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from helpers import ENGINE
from sbengine import sweep
from sweeps.run import study


class Studies(unittest.TestCase):
    def test_all_specs_reproduce_the_existing_methods(self):
        specs = sorted((ENGINE / "sweeps").glob("*/*.json"))
        self.assertEqual(len(specs), 3)
        for path in specs:
            with self.subTest(spec=path):
                spec, cfg, workload, records, summary = study(path)
                self.assertEqual(records, sweep.run_sweep(workload, cfg, spec["axes"]))
                self.assertEqual(summary["points"], len(records))
                self.assertEqual(summary["valid_points"] + summary["invalid_points"], len(records))
                self.assertEqual(summary["fixed_assumptions"]["hbm"], cfg["hbm"])

    def test_platform_axis_is_not_a_primary_study(self):
        spec = json.loads((ENGINE / "sweeps/compute/capacity.json").read_text())
        spec["axes"] = {"hbm.gbs": [230, 460]}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "wrong.json"
            path.write_text(json.dumps(spec))
            with self.assertRaisesRegex(ValueError, "fixed platform"):
                study(path)


if __name__ == "__main__":
    unittest.main()
