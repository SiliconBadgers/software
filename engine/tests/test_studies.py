"""Study wrappers reuse existing methods, fixed assumptions and baseline identity."""
import copy
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

from helpers import ENGINE
sys.path.insert(0, str(ENGINE.parent / "scripts"))
from check_boundaries import REGRESSION_STUDIES, study_paths
from sbengine import sweep
from sweeps.run import study


class Studies(unittest.TestCase):
    def check_studies(self, paths):
        for path in paths:
            with self.subTest(spec=path):
                spec, cfg, workload, records, summary = study(path)
                self.assertEqual(records, sweep.run_sweep(workload, cfg, spec["axes"]))
                self.assertEqual(summary["points"], len(records))
                self.assertEqual(summary["valid_points"] + summary["invalid_points"], len(records))
                self.assertEqual(summary["fixed_assumptions"]["hbm"], cfg["hbm"])

    def catalog(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        for name in REGRESSION_STUDIES:
            path = root / "engine/sweeps" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ENGINE / "sweeps" / name, path)
        return root

    def test_all_specs_reproduce_the_existing_methods(self):
        self.check_studies(study_paths(ENGINE.parent))

    def test_original_studies_keep_recorded_results(self):
        for name in REGRESSION_STUDIES:
            with self.subTest(study=name):
                *_, summary = study(ENGINE / "sweeps" / name)
                expected = json.loads((ENGINE.parent / "docs/sweeps" / f"{Path(name).parts[0]}-v1.json").read_text())
                for key in ("records_sha256", "frontier", "points", "valid_points", "fixed_assumptions"):
                    self.assertEqual(summary[key], expected[key])

    def test_fourth_valid_spec_is_discovered_and_evaluated(self):
        root = self.catalog()
        spec = json.loads((ENGINE / "sweeps/compute/capacity.json").read_text())
        spec.update(id="additional-compute-case", axes={"matrix.count": [1]})
        (root / "engine/sweeps/compute/additional.json").write_text(json.dumps(spec))
        paths = study_paths(root)
        self.assertEqual(len(paths), 4)
        self.check_studies(paths)

    def test_invalid_additional_spec_fails(self):
        root = self.catalog()
        spec = json.loads((ENGINE / "sweeps/compute/capacity.json").read_text())
        spec["axes"] = {"matrix.unknown": [1]}
        path = root / "engine/sweeps/compute/invalid.json"
        path.write_text(json.dumps(spec))
        self.assertIn(path, study_paths(root))
        with self.assertRaises(KeyError):
            study(path)
        path.write_text("{")
        with self.assertRaises(json.JSONDecodeError):
            study(path)

    def test_missing_original_or_referenced_input_fails(self):
        root = self.catalog()
        path = root / "engine/sweeps/compute/capacity.json"
        spec = json.loads(path.read_text())
        spec["baseline"] = "engine/designs/missing.json"
        path.write_text(json.dumps(spec))
        with self.assertRaises(FileNotFoundError):
            study(path)
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing regression studies: compute/capacity.json"):
            study_paths(root)

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
