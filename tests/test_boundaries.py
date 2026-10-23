"""Positive and negative enforcement cases, without plugins or a new lint stack."""
import importlib.util
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("boundaries", ROOT / "scripts/check_boundaries.py")
boundaries = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundaries)
from sbengine import config


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

    def test_bulk_families_are_case_and_wrapper_independent(self):
        names = ("model.gguf", "model.safetensors", "trace.f32", "trace.jsonl",
                 "records.json", "timeline.json", "pool-schedule.json")
        wrappers = ("", ".gz", ".gzip", ".bz2", ".xz", ".lzma", ".zst", ".zstd", ".lz4", ".br",
                    ".zip", ".7z", ".rar", ".tar", ".tgz", ".tbz2", ".txz", ".tar.gz", ".gz.xz")
        for folder in ("research/new", "experiments/new", "results/new"):
            for name in names:
                for wrapper in wrappers:
                    for variant in ((name + wrapper), (name + wrapper).upper(), (name + wrapper).title()):
                        with self.subTest(folder=folder, name=variant):
                            self.assertIsNotNone(boundaries.tracked_path_error(f"{folder}/{variant}"))
        for path in ("Build/file.txt", "WORK/file.txt", ".VENV/file.txt", "build./file.txt", "Work /file.txt",
                     "Build\\file.txt", "new/model.GGUF. ", "new/model.gguf:payload", "new/MODEL.GGUF .GZ"):
            with self.subTest(path=path):
                self.assertIsNotNone(boundaries.tracked_path_error(path))

    def test_bulk_normalization_preserves_legitimate_research_and_historical_exemption(self):
        for path in ("research/new/record.json", "research/new/study.json.gz", "research/new/model.gguf.md",
                     "research/new/MODEL.GGUF.MD", "research/new/model.gguf.md.7z",
                     "research/new/records.json.md.gz", "research/new/build-notes.md",
                     "building/source.py", "workload/spec.json", "research/new/build/README.md"):
            with self.subTest(path=path):
                self.assertIsNone(boundaries.tracked_path_error(path))
        self.assertIsNone(boundaries.tracked_path_error("results/original/TRACE.F32.GZ", preserved=True))
        self.assertIsNotNone(boundaries.tracked_path_error("Build/file.txt", preserved=True))

    def test_broken_or_escaping_reference_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            good = root / "record.json"
            good.write_text("{}")
            self.assertTrue(boundaries.reference(root, "record.json").samefile(good))
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
            "id": "test-baseline", "source_commit": self.original_commit,
            "profile": "engine/designs/profile.json", "run": "results/original",
            "decisions": "engine/designs/decisions.json", "captures": self.captures,
            "decoded_capture_sha256": self.decoded_captures,
            "experiment": "engine/designs/experiment.json", "model": {}, "llama_cpp": {},
            "parent": None, "evidence": ["research/original.md"],
            "config_hash": config.config_hash(config.make_config()), "fixed_assumptions": ["hbm"]}))
        self.write("engine/designs/experiment.json", json.dumps({"model": {}, "llama_cpp": {}}))
        self.write("engine/designs/profile.json", json.dumps({"resolved": True, "config": config.make_config()}))
        self.write("engine/designs/decisions.json", json.dumps({"branches": [{"id": "compute"}]}))
        self.spec = {"id": "fixture-study", "baseline": "engine/designs/baselines/test/record.json", "workload": "sample",
                     "axes": {"matrix.count": [1]}, "decision": "compute", "prerequisites": [],
                     "reference": "research/original.md", "objectives": [["decode_s", "min"]]}
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

    def test_real_git_rejects_case_and_compression_variants_but_allows_research(self):
        prohibited = [f"{folder}/new-study/{name}" for folder in ("experiments", "research", "results")
                      for name in ("MODEL.GGUF", "CHECKPOINT.SAFETENSORS.ZST", "TRACE.F32.GZ", "RUN.JSONL.XZ",
                                   "RECORDS.JSON.GZ", "TIMELINE.JSON.TAR.GZ", "POOL-SCHEDULE.JSON.ZIP",
                                   "model.gguf.7z", "model.safetensors.rar")]
        prohibited += ["Build/probe.txt", "Work/probe.txt", ".VENV/probe.txt"]
        allowed = ["research/new-study/record.json", "research/new-study/model.gguf.md", "results/new-study/study.json.gz"]
        for path in prohibited + allowed:
            self.write(path, "fixture, not actual bulk data")
        self.git("add", ".")
        errors = boundaries.check(self.root)
        self.assertEqual(len(errors), len(prohibited))
        for path in prohibited:
            self.assertTrue(any(error.endswith(": " + path) for error in errors), path)
        for path in allowed:
            self.assertFalse(any(error.endswith(": " + path) for error in errors), path)

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
        (checkout / "research/new").mkdir()
        (checkout / "research/new/MODEL.GGUF.GZ").write_text("fixture")
        (checkout / "research/new/record.json").write_text("{}")
        subprocess.run(["git", "add", "."], cwd=checkout, check=True, capture_output=True)
        self.assertEqual(boundaries.check(checkout), ["new bulk output tracked: research/new/MODEL.GGUF.GZ"])

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

    def test_checker_and_runner_reject_the_same_invalid_study_semantics(self):
        from sweeps import run as runner

        patches = [{"objectives": []}, {"objectives": [["decode_s", "maximize"]]},
                   {"objectives": [["decode_s", None]]}, {"objectives": [["decode_s"]]},
                   {"objectives": [["no_metric", "min"]]}, {"objectives": [["valid", "min"]]},
                   {"objectives": [["decode_s", "min"], ["decode_s", "max"]]},
                   {"decision": "no_decision"}, {"prerequisites": "compute"},
                   {"reference": "missing.md"}, {"axes": {"hbm.gbs": [100]}},
                   {"axes": {"matrix.count": []}}, {"axes": {"matrix.count": "12"}}]
        path = self.root / "engine/sweeps/compute/fourth.json"
        for patch in patches:
            with self.subTest(patch=patch):
                self.write(str(path.relative_to(self.root)), json.dumps({**self.spec, **patch}))
                errors = boundaries.reference_errors(self.root)
                self.assertTrue(any("fourth.json:" in error for error in errors), errors)
                with mock.patch.object(runner, "ROOT", self.root), \
                        mock.patch.object(runner, "load_workload", side_effect=AssertionError("loaded before validation")), \
                        mock.patch.object(runner.sweep, "run_sweep") as evaluate:
                    with self.assertRaises(ValueError):
                        runner.study(path)
                    evaluate.assert_not_called()

    def test_numeric_limits_use_normal_errors_in_checker_and_runner(self):
        from sweeps import run as runner

        path = self.root / "engine/sweeps/compute/fourth.json"
        literals = ["1" + "0" * 310, "-1" + "0" * 310, str(int(sys.float_info.max) + 1),
                    "1" * 5000, "1e999", "-1e999", "1e-10000", "-1e-10000", "NaN", "Infinity"]
        for literal in literals:
            for location in ("axis", "metadata"):
                with self.subTest(literal=literal[:32], location=location):
                    spec = dict(self.spec)
                    if location == "axis":
                        spec["axes"] = {"matrix.count": ["LITERAL"]}
                    else:
                        spec["metadata"] = {"nested": ["LITERAL"]}
                    self.write(str(path.relative_to(self.root)), json.dumps(spec).replace('"LITERAL"', literal))
                    errors = boundaries.reference_errors(self.root)
                    found = [error for error in errors if "fourth.json:" in error]
                    self.assertEqual(len(found), 1, errors)
                    with mock.patch.object(runner, "ROOT", self.root), \
                            mock.patch.object(runner, "load_workload", side_effect=AssertionError("loaded before validation")), \
                            mock.patch.object(runner.sweep, "run_sweep") as evaluate:
                        with self.assertRaises(ValueError) as caught:
                            runner.study(path)
                        self.assertTrue(found[0].endswith(str(caught.exception)), found)
                        evaluate.assert_not_called()

    def test_representable_numeric_metadata_is_accepted_by_both_entry_points(self):
        from sweeps import run as runner
        from sweeps.specs import load_spec

        path = self.root / "engine/sweeps/compute/fourth.json"
        for literal in (str(int(sys.float_info.max)), str(2**53 + 1), repr(sys.float_info.max),
                        repr(sys.float_info.min), "5e-324", "-5e-324", "0e-10000", "-0.0"):
            with self.subTest(literal=literal[:32]):
                raw = json.dumps({**self.spec, "metadata": ["LITERAL"]}).replace('"LITERAL"', literal)
                self.write(str(path.relative_to(self.root)), raw)
                self.assertEqual(boundaries.reference_errors(self.root), [])
                spec, _, _ = load_spec(self.root, path)
                self.assertEqual(spec["metadata"], json.loads(raw)["metadata"])
                with mock.patch.object(runner, "ROOT", self.root), \
                        mock.patch.object(runner.sweep, "run_sweep", return_value=[]) as enumerate_points:
                    runner.study(path)
                    enumerate_points.assert_called_once()

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
