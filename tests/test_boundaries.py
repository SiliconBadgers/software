"""Positive and negative enforcement cases, without plugins or a new lint stack."""
import importlib.util
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
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


class GitEvidenceBoundaries(unittest.TestCase):
    """Exercise staged, unstaged and committed changes in disposable real Git repos."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / "repo"
        self.root.mkdir()
        self.git("init", "-q")
        self.git("config", "core.autocrlf", "false")
        self.write("research/original.md", b"Author: original contributor\nRecorded finding.\n")
        self.write("results/original/summary.csv", b"field,value\r\nx,1\r\n")
        self.write("results/original/trace.jsonl.gz", b"original captured bytes\0")
        self.captures = {}
        self.decoded_captures = {}
        for phase in ("prefill", "decode"):
            metadata = {"prompt_tokens": 512, "phase": phase}
            raw = {"metadata": metadata, "tensors": [], "scheduled_nodes": []}
            compressed = gzip.compress(json.dumps(raw).encode(), mtime=0)
            self.write(f"results/original/graphs/sample/{phase}.json.gz", compressed)
            structure = {"phase": phase, "metadata": metadata, "order": [], "tensors": [],
                         "nodes": [], "storage": [], "deps": [], "totals": {},
                         "metadata_nodes": 0, "zero_nodes": 0}
            digest = hashlib.sha256(json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            self.captures[f"sample/{phase}"] = [hashlib.sha256(compressed).hexdigest(), digest, 0, 0]
            encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
            self.decoded_captures[f"sample/{phase}"] = hashlib.sha256(encoded).hexdigest()
        self.git("add", ".")
        self.git("commit", "-qm", "original evidence")
        self.original_commit = self.git("rev-parse", "HEAD").strip()
        (self.root / "experiments/tool").mkdir(parents=True)
        self.git("update-index", "--add", "--cacheinfo", f"160000,{self.original_commit},experiments/tool")
        tracked = boundaries.tracked_files(self.root)
        self.history = {path: identity for path, identity in tracked.items() if identity["mode"] != "160000"}
        fixtures = "engine/tests/fixtures"
        self.write(f"{fixtures}/historical-0e06023.json", json.dumps(self.history))
        digest = hashlib.sha256(json.dumps(self.history, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        self.write(f"{fixtures}/characterization-0e06023.json", json.dumps({
            "historical_files": {"count": len(self.history), "sha256": digest},
            "historical_gitlinks": {"experiments/tool": self.original_commit}}))
        self.write("web/registry.json", json.dumps({"components": []}))
        self.write("engine/designs/baselines/test/record.json", json.dumps({
            "profile": "engine/designs/profile.json", "run": "results/original",
            "decisions": "engine/designs/decisions.json", "captures": self.captures,
            "decoded_capture_sha256": self.decoded_captures,
            "experiment": "engine/designs/experiment.json", "model": {}, "llama_cpp": {},
            "parent": None, "evidence": ["research/original.md"]}))
        self.write("engine/designs/experiment.json", json.dumps({"model": {}, "llama_cpp": {}}))
        self.write("engine/designs/profile.json", json.dumps({"resolved": True, "config": {"matrix": {"count": 1}}}))
        self.write("engine/designs/decisions.json", json.dumps({"branches": [{"id": "compute"}]}))
        self.spec = {"baseline": "engine/designs/baselines/test/record.json", "workload": "sample",
                     "axes": {"matrix.count": [1]}, "decision": "compute", "prerequisites": [],
                     "reference": "research/original.md"}
        for name in boundaries.REGRESSION_STUDIES:
            self.write(f"engine/sweeps/{name}", json.dumps(self.spec))
        self.git("add", ".")
        self.git("update-index", "--add", "--cacheinfo", f"160000,{self.original_commit},experiments/tool")
        self.git("commit", "-qm", "record preservation inventory")

    def git(self, *args, input=None):
        result = subprocess.run(["git", "-c", "user.name=Boundary tests", "-c",
                                 "user.email=boundary-tests@example.invalid", "-c", "commit.gpgSign=false",
                                 *args], cwd=self.root, input=input, capture_output=True, text=True, check=True)
        return result.stdout

    def write(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")

    def test_original_subset_allows_new_attributable_research(self):
        self.assertEqual(boundaries.check(self.root), [])
        for folder in ("experiments", "research", "results"):
            self.write(f"{folder}/new-study/README.md",
                       "Author: new contributor\nSource: recorded capture\nMethod and limitations.\n")
        self.git("add", ".")
        self.assertEqual(boundaries.check(self.root), [])
        self.git("commit", "-qm", "new attributable research")
        self.assertEqual(boundaries.check(self.root), [])
        history, _ = boundaries.historical_files(self.root)
        self.assertEqual(history, self.history)

    def test_new_bulk_is_rejected_inside_historical_directories(self):
        paths = [f"{folder}/new-study/{name}" for folder in ("experiments", "research", "results")
                 for name in ("trace.jsonl", "trace.jsonl.gz", "model.gguf", "model.safetensors",
                              "checkpoint.f32", "checkpoint.f32.gz", "records.json", "timeline.json",
                              "pool-schedule.json")]
        for path in paths:
            self.write(path, "new bulk")
        self.git("add", ".")
        errors = boundaries.check(self.root)
        for path in paths:
            with self.subTest(path=path):
                self.assertIn(f"new bulk output tracked: {path}", errors)

    def test_modified_original_fails_unstaged_staged_and_committed(self):
        self.write("research/original.md", "changed finding\n")
        self.assertTrue(any("historical working tree changed" in error for error in boundaries.check(self.root)))
        self.git("add", "research/original.md")
        self.assertTrue(any("historical identity changed" in error for error in boundaries.check(self.root)))
        self.git("commit", "-qm", "changed evidence")
        self.assertTrue(any("historical identity changed" in error for error in boundaries.check(self.root)))

    def test_timestamp_only_change_preserves_original_evidence(self):
        path = self.root / "research/original.md"
        original = path.read_bytes()
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10_000_000_000))
        self.assertIn("research/original.md", self.git("diff-files", "--name-only"))
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(boundaries.check(self.root), [])
        self.assertEqual(boundaries.historical_files(self.root)[0], self.history)

    def test_same_size_original_edit_with_restored_timestamp_fails(self):
        path = self.root / "research/original.md"
        original = path.read_bytes()
        stat = path.stat()
        changed = original.replace(b"Recorded", b"Altered!")
        self.assertNotEqual(changed, original)
        self.assertEqual(len(changed), len(original))
        self.write("research/original.md", changed)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertTrue(any("historical working tree changed" in error for error in boundaries.check(self.root)))

    def test_line_ending_only_checkout_changes_follow_git_rules(self):
        self.git("config", "core.autocrlf", "true")
        path = self.root / "research/original.md"
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        self.assertEqual(boundaries.check(self.root), [])
        self.assertEqual((self.root / "results/original/summary.csv").read_bytes(), b"field,value\r\nx,1\r\n")

    def test_deleted_original_fails_unstaged_staged_and_committed(self):
        (self.root / "research/original.md").unlink()
        self.assertTrue(any("historical working tree changed" in error for error in boundaries.check(self.root)))
        self.git("add", "-u")
        self.assertTrue(any("historical identity changed or missing" in error for error in boundaries.check(self.root)))
        self.git("commit", "-qm", "deleted evidence")
        self.assertTrue(any("historical identity changed or missing" in error for error in boundaries.check(self.root)))

    def test_replacing_original_file_mode_or_type_fails(self):
        target = self.git("hash-object", "-w", "--stdin", input="different target").strip()
        for mode, blob in (("100755", self.history["research/original.md"]["git_blob"]),
                           ("120000", target), ("160000", self.original_commit)):
            with self.subTest(mode=mode):
                self.git("update-index", "--cacheinfo", f"{mode},{blob},research/original.md")
                self.assertTrue(any("historical identity changed" in error for error in boundaries.check(self.root)))

    def test_renaming_original_fails(self):
        (self.root / "research/original.md").rename(self.root / "research/renamed.md")
        self.git("add", ".")
        self.assertTrue(any("historical identity changed or missing" in error for error in boundaries.check(self.root)))

    def test_inventory_cannot_redefine_original_digest(self):
        for changed in ({}, {**self.history, "research/new.md": self.history["research/original.md"]}):
            with self.subTest(entries=len(changed)):
                self.write("engine/tests/fixtures/historical-0e06023.json", json.dumps(changed))
                self.assertTrue(any("original characterization digest" in error for error in boundaries.check(self.root)))

    def test_original_gitlink_is_protected(self):
        replacement = self.git("rev-parse", "HEAD").strip()
        self.git("update-index", "--cacheinfo", f"160000,{replacement},experiments/tool")
        self.assertTrue(any("historical identity changed or missing: experiments/tool" in error
                            for error in boundaries.check(self.root)))

    def test_shallow_checkout_needs_no_original_commit(self):
        checkout = self.directory / "shallow"
        self.git("clone", "-q", "--depth", "1", self.root.as_uri(), str(checkout))
        shallow = subprocess.check_output(["git", "rev-parse", "--is-shallow-repository"], cwd=checkout, text=True)
        self.assertEqual(shallow.strip(), "true")
        absent = subprocess.run(["git", "cat-file", "-e", self.original_commit], cwd=checkout, capture_output=True)
        self.assertNotEqual(absent.returncode, 0)
        self.assertEqual(boundaries.check(checkout), [])

    def test_catalog_additions_are_checked_and_missing_originals_fail(self):
        additional = "engine/sweeps/compute/fourth.json"
        self.write(additional, json.dumps(self.spec))
        self.git("add", ".")
        self.assertEqual(len(boundaries.study_paths(self.root)), 4)
        self.assertEqual(boundaries.check(self.root), [])
        self.write(additional, json.dumps({**self.spec, "baseline": "missing.json"}))
        self.assertTrue(any("missing.json" in error for error in boundaries.check(self.root)))
        self.write(additional, json.dumps(self.spec))
        (self.root / "engine/sweeps/compute/capacity.json").unlink()
        self.assertTrue(any("missing regression studies: compute/capacity.json" in error
                            for error in boundaries.check(self.root)))

    def test_matching_plain_overrides_pass(self):
        for phase in ("prefill", "decode"):
            captured = self.root / f"results/original/graphs/sample/{phase}.json.gz"
            raw = json.loads(gzip.decompress(captured.read_bytes()))
            self.write(f"results/original/graphs/sample/{phase}.json", json.dumps(raw, indent=1))
        self.assertEqual(boundaries.check(self.root), [])

    def test_mismatched_plain_overrides_fail_baseline_and_study_checks(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                name = f"results/original/graphs/sample/{phase}.json"
                captured = self.root / f"{name}.gz"
                raw = json.loads(gzip.decompress(captured.read_bytes()))
                raw["metadata"]["prompt_tokens"] = 128
                self.write(name, json.dumps(raw))
                errors = boundaries.reference_errors(self.root)
                for prefix in ("engine/designs/baselines/test/record.json", "engine/sweeps/compute/capacity.json"):
                    self.assertTrue(any(error.replace("\\", "/").startswith(prefix)
                                        and f"capture identity differs: sample/{phase}" in error for error in errors), errors)
                (self.root / name).unlink()

    def test_missing_capture_and_missing_identity_fail(self):
        baseline_path = self.root / self.spec["baseline"]
        record = json.loads(baseline_path.read_text())
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                captured = self.root / f"results/original/graphs/sample/{phase}.json.gz"
                original = captured.read_bytes()
                captured.unlink()
                self.assertTrue(any(f"{phase}" in error for error in boundaries.reference_errors(self.root)))
                captured.write_bytes(original)
                missing = {**record, "captures": {key: value for key, value in record["captures"].items()
                                                 if key != f"sample/{phase}"}}
                baseline_path.write_text(json.dumps(missing))
                self.assertTrue(any(f"capture identity missing: sample/{phase}" in error
                                    for error in boundaries.reference_errors(self.root)))
                baseline_path.write_text(json.dumps(record))

    def test_schema_only_override_is_rejected(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                name = f"results/original/graphs/sample/{phase}.json"
                raw = json.loads(gzip.decompress((self.root / f"{name}.gz").read_bytes()))
                raw["schema_version"] = 999
                self.write(name, json.dumps(raw))
                errors = boundaries.reference_errors(self.root)
                self.assertTrue(any(f"capture identity differs: sample/{phase}" in error for error in errors), errors)
                (self.root / name).unlink()

    def test_decoded_hash_cannot_rebind_a_changed_override_to_pinned_evidence(self):
        name = "results/original/graphs/sample/prefill.json"
        raw = json.loads(gzip.decompress((self.root / f"{name}.gz").read_bytes()))
        raw["schema_version"] = 999
        self.write(name, json.dumps(raw))
        path = self.root / self.spec["baseline"]
        record = json.loads(path.read_text())
        encoded = json.dumps(raw, sort_keys=True, separators=(",", ":")).encode()
        record["decoded_capture_sha256"]["sample/prefill"] = hashlib.sha256(encoded).hexdigest()
        self.write(self.spec["baseline"], json.dumps(record))
        errors = boundaries.reference_errors(self.root)
        self.assertTrue(any(error.replace("\\", "/").startswith(self.spec["baseline"])
                            and "capture identity differs: sample/prefill" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
