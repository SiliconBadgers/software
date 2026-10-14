"""Positive and negative enforcement cases, without plugins or a new lint stack."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("boundaries", ROOT / "scripts/check_boundaries.py")
boundaries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundaries)


class Boundaries(unittest.TestCase):
    def test_repository_passes(self):
        self.assertEqual(boundaries.check(ROOT), [])

    def test_dependency_direction_is_enforced(self):
        for path, source in (("core/cdfg/graph.py", "import numpy"),
                             ("core/cdfg/graph.py", "from sbengine import model"),
                             ("engine/sbengine/mapping.py", "from .costs import Ctx"),
                             ("engine/sbengine/memory.py", "from . import model")):
            with self.subTest(path=path):
                self.assertTrue(boundaries.import_errors(path, source))
        self.assertEqual(boundaries.import_errors("core/cdfg/graph.py", "import json\nfrom .work import dense_macs"), [])
        self.assertEqual(boundaries.import_errors("engine/sbengine/mapping.py", "from . import groups"), [])

    def test_bulk_policy_allows_specs_and_preserved_evidence(self):
        self.assertIsNone(boundaries.tracked_path_error("engine/sweeps/compute/capacity.json"))
        self.assertIsNone(boundaries.tracked_path_error("results/recorded/trace.jsonl.gz", preserved=True))
        for path in ("build/replay.json", "work/model.gguf", "docs/full/records.json", "new/trace.jsonl.gz"):
            self.assertIsNotNone(boundaries.tracked_path_error(path))

    def test_broken_or_escaping_reference_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = root / "record.json"
            good.write_text("{}")
            self.assertEqual(boundaries.reference(root, "record.json"), good)
            for path in ("missing.json", "../elsewhere.json"):
                with self.assertRaisesRegex(ValueError, "reference"):
                    boundaries.reference(root, path)

    def test_invalid_baseline_and_virtual_asset_are_reported(self):
        original = Path.read_text

        def broken(path, *args, **kwargs):
            text = original(path, *args, **kwargs)
            if path.name == "record.json" and "characterized-v1" in str(path):
                record = json.loads(text)
                record["profile"] = "engine/designs/absent.json"
                return json.dumps(record)
            if path == ROOT / "web/registry.json":
                registry = json.loads(text)
                registry["components"][1]["assets"][0]["virtual"] = "../../escaped.py"
                return json.dumps(registry)
            return text

        with mock.patch.object(Path, "read_text", broken):
            errors = boundaries.reference_errors(ROOT)
        self.assertTrue(any("absent.json" in error for error in errors))
        self.assertTrue(any("virtual path" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
