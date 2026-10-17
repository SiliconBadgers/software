"""Complete baseline configuration is explicit and does not inherit new defaults."""
import json
import gzip
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from helpers import ENGINE
from sbengine import config
from sbengine.provenance import decoded_capture_sha256

BASELINE = ENGINE / "designs/baselines/characterized-v1"


class Baseline(unittest.TestCase):
    def test_decoded_identity_canonicalization(self):
        expected = "efbd0040190fb0871831e606c581f8a66db79d8e2bb836745a70051306956070"
        for value in ({"b": [2, 3], "a": 1}, {"a": 1, "b": [2, 3]}):
            self.assertEqual(decoded_capture_sha256(value), expected)
        for value in ({"a": 1, "b": [3, 2]}, {"a": True, "b": [2, 3]}, {"a": "1", "b": [2, 3]}):
            self.assertNotEqual(decoded_capture_sha256(value), expected)
        self.assertEqual(decoded_capture_sha256(json.loads('"\\u03b1"')), decoded_capture_sha256("α"))
        for invalid in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaises(ValueError):
                decoded_capture_sha256({"ignored_field": invalid})

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

    def test_full_decoded_identities_match_the_byte_pinned_evidence(self):
        record = json.loads((BASELINE / "record.json").read_text())
        self.assertEqual(set(record["decoded_capture_sha256"]), set(record["captures"]))
        self.assertEqual(len(record["decoded_capture_sha256"]), 44)
        for key, identity in record["captures"].items():
            with self.subTest(capture=key):
                compressed = (ENGINE.parent / record["run"] / "graphs" / f"{key}.json.gz").read_bytes()
                self.assertEqual(hashlib.sha256(compressed).hexdigest(), identity[0])
                parsed = json.loads(gzip.decompress(compressed))
                # Independent of the production identity helper and modeled structure serializer.
                encoded = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                                     allow_nan=False).encode("utf-8")
                self.assertEqual(hashlib.sha256(encoded).hexdigest(), record["decoded_capture_sha256"][key])

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
