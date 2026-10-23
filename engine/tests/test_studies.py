"""Study wrappers reuse existing methods, fixed assumptions and baseline identity."""
import copy
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from helpers import ENGINE
sys.path.insert(0, str(ENGINE.parent / "scripts"))
from check_boundaries import REGRESSION_STUDIES, study_paths
from sbengine import sweep
from sbengine.graph import Graph
from sbengine.provenance import validate_captures
from characterization import graph_record
from sweeps import run as study_runner
from sweeps.run import study
from sweeps.specs import NUMERIC_METRICS


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
        with self.assertRaisesRegex(ValueError, "missing repository reference"):
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


class CaptureIdentities(unittest.TestCase):
    """Use real pinned captures, but place overrides only in disposable copies."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.spec = json.loads((ENGINE / "sweeps/compute/capacity.json").read_text())
        self.spec["axes"] = {"matrix.count": [1]}
        self.path = self.root / "study.json"
        self.path.write_text(json.dumps(self.spec))
        self.baseline_path = self.root / self.spec["baseline"]
        self.baseline = json.loads((ENGINE.parent / self.spec["baseline"]).read_text())
        for relative in (self.spec["baseline"], self.baseline["profile"], self.baseline["decisions"],
                         self.spec["reference"]):
            destination = self.root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ENGINE.parent / relative, destination)
        self.directory = self.root / self.baseline["run"] / "graphs" / self.spec["workload"]
        self.directory.mkdir(parents=True)
        self.source = ENGINE.parent / self.baseline["run"] / "graphs"
        for phase in ("prefill", "decode"):
            shutil.copyfile(self.source / self.spec["workload"] / f"{phase}.json.gz",
                            self.directory / f"{phase}.json.gz")
        self.root_patch = mock.patch.object(study_runner, "ROOT", self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def assert_rejected(self, phase):
        with mock.patch.object(study_runner.sweep, "run_sweep", return_value=[]) as evaluate, \
                mock.patch.object(study_runner, "load_trace") as trace:
            with self.assertRaisesRegex(ValueError, f"capture identity.*{self.spec['workload']}/{phase}"):
                study(self.path)
            evaluate.assert_not_called()
            trace.assert_not_called()

    def test_fields_omitted_from_modeled_structure_are_rejected(self):
        for phase in ("prefill", "decode"):
            compressed = self.directory / f"{phase}.json.gz"
            original = json.loads(gzip.decompress(compressed.read_bytes()))
            structure = graph_record(Graph(compressed, phase))
            for field in ("schema_version", "op_params_i32", "strides_bytes", "flags", "view_offset_bytes",
                          "extra_field", "removed_field"):
                with self.subTest(phase=phase, field=field):
                    raw = copy.deepcopy(original)
                    rms = next(t for t in raw["tensors"] if t["op_desc"] == "RMS_NORM")
                    view = next(t for t in raw["tensors"] if t["view_source"] >= 0)
                    if field == "schema_version":
                        raw[field] = 999
                    elif field in ("op_params_i32", "strides_bytes"):
                        rms[field][0] += 1
                    elif field == "flags":
                        rms[field] ^= 1
                    elif field == "view_offset_bytes":
                        view[field] += 4
                    elif field == "extra_field":
                        raw[field] = {"future_capture_information": [1, 2, 3]}
                    else:
                        del rms["op_params_i32"]
                    plain = self.directory / f"{phase}.json"
                    plain.write_text(json.dumps(raw))
                    self.assertEqual(graph_record(Graph(plain, phase)), structure)
                    self.assert_rejected(phase)
                    plain.unlink()

    def test_mismatched_plain_override_fails_before_evaluation(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                plain = self.directory / f"{phase}.json"
                plain.write_bytes(gzip.decompress((self.source / "pp128-fa-on" / f"{phase}.json.gz").read_bytes()))
                self.assert_rejected(phase)
                plain.unlink()

    def test_reviewed_128_prefill_512_decode_pair_is_rejected(self):
        plain = self.directory / "prefill.json"
        plain.write_bytes(gzip.decompress((self.source / "pp128-fa-on/prefill.json.gz").read_bytes()))
        with self.assertRaisesRegex(ValueError, "capture identity differs: pp512-fa-on/prefill"):
            study(self.path)  # Real evaluation path, without a mocked sweep.

    def test_matching_plain_gzip_and_recompressed_captures_keep_results(self):
        *_, expected = study(self.path)
        for phases in (("prefill",), ("decode",), ("prefill", "decode")):
            with self.subTest(plain=phases):
                for phase in ("prefill", "decode"):
                    raw = gzip.decompress((self.source / self.spec["workload"] / f"{phase}.json.gz").read_bytes())
                    reformatted = json.dumps(json.loads(raw), indent=1, sort_keys=True, ensure_ascii=False).encode()
                    (self.directory / f"{phase}.json.gz").write_bytes(gzip.compress(reformatted, mtime=123))
                    plain = self.directory / f"{phase}.json"
                    if phase in phases:
                        plain.write_bytes(reformatted)
                    else:
                        plain.unlink(missing_ok=True)
                _, _, workload, _, summary = study(self.path)
                self.assertEqual(summary["records_sha256"], expected["records_sha256"])
                for phase in ("prefill", "decode"):
                    self.assertEqual(workload.graph(phase).path.suffix, ".json" if phase in phases else ".gz")
        for phase in ("prefill", "decode"):
            (self.directory / f"{phase}.json.gz").unlink()
        *_, plain_only = study(self.path)
        self.assertEqual(plain_only["records_sha256"], expected["records_sha256"])

    def test_changed_tensor_with_matching_metadata_is_rejected(self):
        phase = "decode"
        raw = json.loads(gzip.decompress((self.directory / f"{phase}.json.gz").read_bytes()))
        raw["tensors"][-1]["logical_bytes"] += 16
        (self.directory / f"{phase}.json").write_text(json.dumps(raw))
        self.assert_rejected(phase)

    def test_mismatched_compressed_capture_is_rejected(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                path = self.directory / f"{phase}.json.gz"
                original = path.read_bytes()
                shutil.copyfile(self.source / "pp128-fa-on" / path.name, path)
                self.assert_rejected(phase)
                path.write_bytes(original)

    def test_missing_recorded_identity_is_rejected(self):
        for field in ("captures", "decoded_capture_sha256"):
            for phase in ("prefill", "decode"):
                with self.subTest(field=field, phase=phase):
                    record = copy.deepcopy(self.baseline)
                    del record[field][f"{self.spec['workload']}/{phase}"]
                    self.baseline_path.write_text(json.dumps(record))
                    self.assert_rejected(phase)
            for missing in (None, {}):
                with self.subTest(field=field, missing=missing):
                    record = {**self.baseline, field: missing}
                    self.baseline_path.write_text(json.dumps(record))
                    self.assert_rejected("prefill")

    def test_missing_phase_fails_before_evaluation(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                path = self.directory / f"{phase}.json.gz"
                original = path.read_bytes()
                path.unlink()
                with mock.patch.object(study_runner.sweep, "run_sweep", return_value=[]) as evaluate:
                    with self.assertRaisesRegex(ValueError, f"missing captured phase: pp512-fa-on/{phase}"):
                        study(self.path)
                    evaluate.assert_not_called()
                path.write_bytes(original)

    def test_identity_validation_uses_loaded_graphs_not_reread_files(self):
        from sbengine.graph import load_workload

        workload = load_workload(self.directory)
        # Replacing a file after load does not change the objects evaluated.
        shutil.copyfile(self.source / "pp128-fa-on/prefill.json.gz", self.directory / "prefill.json.gz")
        validate_captures(workload, self.baseline)
        workload.decode.capture["metadata"]["prompt_tokens"] = 128
        with self.assertRaisesRegex(ValueError, "capture identity differs: pp512-fa-on/decode"):
            validate_captures(workload, self.baseline)

    def test_rejection_precedes_enabled_host_trace_and_output_creation(self):
        from sbengine import config

        profile_path = self.root / self.baseline["profile"]
        profile = json.loads(profile_path.read_text())
        profile["config"]["host"]["enabled"] = True
        profile_path.write_text(json.dumps(profile))
        record = {**self.baseline, "config_hash": config.config_hash(profile["config"])}
        self.baseline_path.write_text(json.dumps(record))
        raw = json.loads(gzip.decompress((self.directory / "decode.json.gz").read_bytes()))
        rms = next(t for t in raw["tensors"] if t["op_desc"] == "RMS_NORM")
        rms["op_params_i32"][0] += 1
        (self.directory / "decode.json").write_text(json.dumps(raw))
        out = self.root / "build/sweeps/rejected"
        with mock.patch.object(sys, "argv", ["run.py", str(self.path), "--out", str(out)]), \
                mock.patch.object(study_runner, "load_trace") as trace, \
                mock.patch.object(study_runner.sweep, "run_sweep") as evaluate, \
                mock.patch.object(study_runner.sweep, "write_sweep") as write:
            with self.assertRaisesRegex(ValueError, "capture identity differs: pp512-fa-on/decode"):
                study_runner.main()
            trace.assert_not_called()
            evaluate.assert_not_called()
            write.assert_not_called()
        self.assertFalse(out.exists())


class StudyContract(unittest.TestCase):
    """Exercise the public runner and CLI, not just the shared validator."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "study.json"
        self.spec = json.loads((ENGINE / "sweeps/compute/capacity.json").read_text())
        self.spec.update(axes={"matrix.count": [1, 2]}, objectives=[["prefill_s", "min"]])

    def reject_before_use(self, spec=None, raw=None):
        self.path.write_text(json.dumps(spec) if raw is None else raw)
        out = self.directory / "must-not-exist"
        with mock.patch.object(sys, "argv", ["run.py", str(self.path), "--out", str(out)]), \
                mock.patch.object(study_runner, "load_workload", side_effect=AssertionError("loaded before validation")), \
                mock.patch.object(study_runner, "load_trace") as trace, \
                mock.patch.object(study_runner.sweep, "run_sweep") as evaluate, \
                mock.patch.object(study_runner.sweep, "write_sweep") as write:
            with self.assertRaises((ValueError, KeyError)):
                study_runner.main()
            trace.assert_not_called()
            evaluate.assert_not_called()
            write.assert_not_called()
        self.assertFalse(out.exists())

    def test_malformed_objectives_fail_before_capture_trace_evaluation_or_output(self):
        invalid = [None, [], {}, "prefill_s", True, 1, ["prefill_s"], [["prefill_s"]],
                   [["prefill_s", "min", "extra"]], [{"prefill_s": "min"}],
                   [["missing", "min"]], [[[], "min"]], [[1, "min"]],
                   [["valid", "min"]], [["prefill_binding", "min"]], [["config_hash", "min"]],
                   [["prefill_s", "min"], ["prefill_s", "min"]],
                   [["prefill_s", "min"], ["prefill_s", "max"]]]
        invalid += [[["prefill_s", direction]] for direction in ("maximize", "minimize", "MIN", "sideways", "", None,
                                                               True, 1, [], {})]
        for objectives in invalid:
            with self.subTest(objectives=objectives):
                self.reject_before_use({**self.spec, "objectives": objectives})
        self.reject_before_use({key: value for key, value in self.spec.items() if key != "objectives"})

    def test_invalid_axes_metadata_and_references_fail_before_use(self):
        patches = [
            {"axes": []}, {"axes": None}, {"axes": {"matrix.count": []}},
            {"axes": {"matrix.count": "12"}}, {"axes": {"matrix.count": {"value": 1}}},
            {"axes": {"matrix.count": 1}}, {"axes": {"matrix.unknown": [1]}},
            {"axes": {"matrix": [{}], "matrix.count": [1]}}, {"axes": {"hbm.gbs": [230, 460]}},
            {"decision": "missing"}, {"decision": []}, {"prerequisites": "local-memory"},
            {"prerequisites": ["missing"]}, {"prerequisites": [[]]}, {"prerequisites": None},
            {"reference": "missing.md"}, {"reference": "../outside.md"}, {"reference": []},
            {"baseline": "../outside.json"}, {"baseline": str(ENGINE / "designs/baselines/characterized-v1/record.json")},
            {"id": "../outside"}, {"id": ""}, {"id": []}, {"workload": "../pp512-fa-on"},
        ]
        for patch in patches:
            with self.subTest(patch=patch):
                self.reject_before_use({**self.spec, **patch})
        for raw in ("[]", "null", "true", '{"id":"one","id":"two"}'):
            self.reject_before_use(raw=raw)
        for constant in ("NaN", "Infinity", "-Infinity", "1e999"):
            raw = json.dumps(self.spec).replace('"matrix.count": [1, 2]', f'"matrix.count": [{constant}]')
            self.reject_before_use(raw=raw)
        raw = json.dumps(self.spec).replace('"matrix.count": [1, 2]', '"matrix.count": [1], "matrix.count": [2]')
        self.reject_before_use(raw=raw)

    def test_valid_min_max_and_numeric_axis_objectives_keep_their_direction(self):
        for metric in ("prefill_s", "matrix.count"):
            for direction in ("min", "max"):
                with self.subTest(metric=metric, direction=direction):
                    self.path.write_text(json.dumps({**self.spec, "objectives": [[metric, direction]]}))
                    *_, records, summary = study(self.path)
                    selected = (min if direction == "min" else max)(records, key=lambda row: row[metric])
                    self.assertEqual(summary["frontier"][0]["matrix.count"], selected["matrix.count"])
                    numeric = {key for key, value in records[0].items() if type(value) in (int, float)} - {"matrix.count"}
                    self.assertEqual(NUMERIC_METRICS, numeric)

    def test_baseline_only_and_categorical_axes_are_legitimate_studies(self):
        for axes in ({}, {"matrix.tile_mode": ["search", "fixed"]},
                     {"matrix.caps": [["gemm", "requant"], ["gemm"]]}):
            with self.subTest(axes=axes):
                self.path.write_text(json.dumps({**self.spec, "id": "additional-v2.1", "axes": axes}))
                *_, summary = study(self.path)
                self.assertGreater(summary["points"], 0)
        self.reject_before_use({**self.spec, "axes": {"matrix.tile_mode": ["search", "fixed"]},
                                "objectives": [["matrix.tile_mode", "min"]]})

    def test_extreme_numeric_literals_reject_before_use_in_any_location(self):
        literals = ["1" + "0" * 310, "-1" + "0" * 310, str(int(sys.float_info.max) + 1),
                    str(-int(sys.float_info.max) - 1), "1" * 5000, "1e309", "-1e309", "NaN", "Infinity",
                    "-Infinity", "1e-10000", "-1e-10000", "2.4e-324", "-2.4e-324"]
        for literal in literals:
            for location in ("axis", "metadata", "nested-axis"):
                with self.subTest(literal=literal[:32], location=location):
                    spec = copy.deepcopy(self.spec)
                    if location == "axis":
                        spec["axes"]["matrix.count"] = ["LITERAL"]
                    elif location == "nested-axis":
                        spec["axes"] = {"matrix.caps": [["LITERAL"]]}
                    else:
                        spec["metadata"] = {"nested": ["LITERAL"]}
                    self.reject_before_use(raw=json.dumps(spec).replace('"LITERAL"', literal))

    def test_numeric_representation_controls_preserve_values_and_types(self):
        from sweeps.specs import load_spec

        for literal, expected in ((str(int(sys.float_info.max)), int(sys.float_info.max)),
                                  (str(-int(sys.float_info.max)), -int(sys.float_info.max)),
                                  (str(2**53 + 1), 2**53 + 1), (repr(sys.float_info.max), sys.float_info.max),
                                  (repr(sys.float_info.min), sys.float_info.min), ("5e-324", 5e-324),
                                  ("-5e-324", -5e-324), ("0e-10000", 0.0), ("-0.0", -0.0),
                                  ('"1e309"', "1e309"), ("true", True)):
            with self.subTest(literal=literal[:32]):
                self.path.write_text(json.dumps({**self.spec, "metadata": {"value": "LITERAL"}})
                                     .replace('"LITERAL"', literal))
                spec, _, _ = load_spec(ENGINE.parent, self.path)
                value = spec["metadata"]["value"]
                self.assertIs(type(value), type(expected))
                self.assertEqual(value, expected)
                self.assertEqual(json.dumps(value), json.dumps(expected))
        for values in (["1", "2"], [True, False], [None]):
            self.reject_before_use({**self.spec, "axes": {"matrix.count": values},
                                    "objectives": [["matrix.count", "min"]]})

    def test_every_supported_objective_is_present_and_correct_in_frontier(self):
        expected_digest = None
        objectives = [[[metric, direction]] for metric in sorted(NUMERIC_METRICS | {"matrix.count"})
                      for direction in ("min", "max")]
        objectives.append([["decode_tps_total", "max"], ["hbm_required_gib", "min"]])
        for choices in objectives:
            with self.subTest(objectives=choices):
                self.path.write_text(json.dumps({**self.spec, "objectives": choices}))
                spec, _, _, records, summary = study(self.path)
                selected = sweep.pareto(records, [tuple(item) for item in choices])
                keys = [*spec["axes"], "cost", "prefill_s", "decode_s"]
                keys += [metric for metric, _ in choices if metric not in keys]
                self.assertEqual(summary["frontier"], [{key: row[key] for key in keys} for row in selected])
                self.assertEqual(json.loads(json.dumps(summary))["frontier"], summary["frontier"])
                for row in summary["frontier"]:
                    self.assertEqual(list(row), keys)
                expected_digest = expected_digest or summary["records_sha256"]
                self.assertEqual(summary["records_sha256"], expected_digest)

    def test_fixed_descendant_cannot_be_varied_through_parent(self):
        from sweeps import specs

        original = specs._read

        def fixed_child(path):
            value = original(path)
            if path.name == "record.json":
                value["fixed_assumptions"] = ["hbm.gbs"]
            return value

        with mock.patch.object(specs, "_read", side_effect=fixed_child):
            self.reject_before_use({**self.spec, "axes": {"hbm": [{"gbs": 230}]}})

    def test_nonfinite_or_nonnumeric_feasible_results_do_not_reach_output(self):
        self.path.write_text(json.dumps(self.spec))
        out = self.directory / "must-not-exist"
        for value in (None, True, "fast", float("nan"), float("inf"), float("-inf"),
                      10**310, -(10**310), int(sys.float_info.max) + 1):
            with self.subTest(value=value), \
                    mock.patch.object(sys, "argv", ["run.py", str(self.path), "--out", str(out)]), \
                    mock.patch.object(study_runner.sweep, "run_sweep", return_value=[{"valid": True, "prefill_s": value}]), \
                    mock.patch.object(study_runner.sweep, "write_sweep") as write:
                with self.assertRaisesRegex(ValueError, "result for objective"):
                    study_runner.main()
                write.assert_not_called()
                self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
