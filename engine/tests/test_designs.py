"""Complete baseline configuration is explicit and does not inherit new defaults."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from helpers import ENGINE
from sbengine import config

BASELINE = ENGINE / "designs/baselines/characterized-v1"


class Baseline(unittest.TestCase):
    def test_existing_loader_reproduces_characterized_configuration(self):
        got, _ = config.load_profile(BASELINE / "profile.json")
        fixture = json.loads((ENGINE / "tests/fixtures/characterization-0e06023.json").read_text())
        record = json.loads((BASELINE / "record.json").read_text())
        self.assertEqual(got, fixture["profiles"]["accel-balanced"])
        self.assertEqual(config.config_hash(got), record["config_hash"])
        self.assertEqual(record["approval"]["status"], "provisional")
        self.assertEqual(record["captures"], fixture["graphs"])

    def test_resolved_values_do_not_inherit_changed_defaults(self):
        with mock.patch.dict(config.DEFAULTS["matrix"], {"count": 9}):
            got, _ = config.load_profile(BASELINE / "profile.json")
        self.assertEqual(got["matrix"]["count"], 4)

    def test_missing_field_fails_instead_of_inheriting(self):
        doc = json.loads((BASELINE / "profile.json").read_text())
        del doc["config"]["l1"]["banks"]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "incomplete.json"
            path.write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, "l1.banks"):
                config.load_profile(path)


if __name__ == "__main__":
    unittest.main()
