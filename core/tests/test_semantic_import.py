"""Small synthetic format controls; real export verification is an explicit local run."""
import copy
from contextlib import redirect_stderr
import hashlib
from importlib.util import find_spec
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
if find_spec("gguf") is not None:
    from gguf import GGUFWriter
else:
    GGUFWriter = None

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE))
sys.path.insert(0, str(CORE / "runner"))
from cdfg.graph import Graph
from cdfg.semantic import SemanticGraph
import import_workload as importer
import cli
from policy import load_experiment, DEFAULT_EXPERIMENT


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fixture():
    # Not a substitute model/capture: tiny storage and two semantic nodes.
    return {"schema": "qwenquant.graph.v1", "kind": "semantic_operator_dag", "architecture": "test-compact",
            "model": "test/model", "revision": "test-revision", "symbols": {"B": 1, "T": "input length", "P": "past"},
            "inputs": {"input_ids": {}}, "state_inputs": {}, "state_outputs": {}, "outputs": ["logits"],
            "config": {"hidden_size": 2}, "tensor_aliases": {"head.weight": "embedding.weight"},
            "tensor_manifest": {"embedding.weight": {"shape": [2, 2], "source_dtype": "F16", "ggml_type": 1, "bytes": 8}},
            "contracts": {"lookup": {"semantics": "synthetic test only"}},
            "nodes": [{"id": "embedding", "op": "embedding", "inputs": ["input_ids"], "outputs": ["embedding"],
                       "weights": ["embedding.weight"], "attrs": {"contract": "lookup"}},
                      {"id": "head", "op": "fp16_linear", "inputs": ["embedding"], "outputs": ["logits"],
                       "weights": ["head.weight"]}],
            "source_sha256": {"test.py": digest(b"# never executed\n")}, "source_weights_sha256": "test-weight-hash"}


class SemanticContract(unittest.TestCase):
    def test_inventory_and_declared_dependencies_without_execution(self):
        graph = SemanticGraph(fixture())
        self.assertEqual(graph.dependencies, {"embedding": [], "head": ["embedding"]})
        self.assertEqual(graph.inventory()["stored_tensor_bytes"], 8)
        self.assertFalse(graph.inventory()["execution_validated"])
        self.assertFalse(graph.inventory()["capture_compatible"])

    def test_source_symbols_and_contracts_are_preserved_without_phase_inference(self):
        raw = fixture()
        before = copy.deepcopy(raw)
        graph = SemanticGraph(raw)
        self.assertEqual(graph.raw, before)

    def test_structure_and_format_rejections(self):
        for change in (
            lambda g: g.update(schema="ggml-capture"),
            lambda g: g.update(kind="executed_capture"),
            lambda g: g["symbols"].update(B=2),
            lambda g: g.update(symbols=None),
            lambda g: g.update(source_sha256=[]),
            lambda g: g.pop("config"),
            lambda g: g["nodes"][1].update(id="embedding"),
            lambda g: g["nodes"][0].update(inputs=["logits"]),
            lambda g: g["nodes"][1].update(outputs=["embedding"]),
            lambda g: g["nodes"][0].update(weights=["absent"]),
            lambda g: g["nodes"][0]["attrs"].update(contract="absent"),
            lambda g: g["nodes"][0]["attrs"].update(contract=[]),
            lambda g: g["tensor_aliases"].update({"head.weight": "loop", "loop": "head.weight"}),
            lambda g: g["tensor_manifest"]["embedding.weight"].update(bytes=4),
            lambda g: g["tensor_manifest"]["embedding.weight"].update(source_dtype="Q4_K"),
            lambda g: g["tensor_manifest"]["embedding.weight"].update(ggml_type=True),
            lambda g: g.update(outputs=["absent"]),
            lambda g: g.update(state_outputs={"absent": {}}),
        ):
            graph = fixture()
            change(graph)
            with self.subTest(graph=graph), self.assertRaises(ValueError):
                SemanticGraph(graph)

    def test_semantic_graph_is_rejected_by_capture_loader(self):
        with tempfile.TemporaryDirectory() as temp:
            for phase in ("prefill", "decode"):
                path = Path(temp) / (phase + ".json")
                path.write_text(json.dumps(fixture()))
                with self.assertRaisesRegex(ValueError, "not an executed GGML capture"):
                    Graph(path, phase)

    def test_compact_group_geometry_and_component_types(self):
        graph = fixture()
        graph["contracts"]["compact_int4_linear"] = {"group": 16}
        node = graph["nodes"][1]
        node.update(op="compact_int4_linear", attrs={"group": 16, "weight_shape": [2, 32]}, weights=[])
        for suffix, shape, dtype in (("packed", [2, 16], "U8"), ("scale_codes", [2, 2], "U8"),
                                     ("scale_bases", [2], "F16"), ("zeros", [2, 1], "U8")):
            name = "head.int4_" + suffix
            node["weights"].append(name)
            graph["tensor_manifest"][name] = {"shape": shape, "source_dtype": dtype,
                "ggml_type": 1 if dtype == "F16" else 24, "bytes": int(np.prod(shape)) * (2 if dtype == "F16" else 1)}
        SemanticGraph(graph)
        for change in (lambda g: g["nodes"][1]["attrs"].update(group=32),
                       lambda g: g["nodes"][1]["attrs"].update(weight_shape=[2, 31]),
                       lambda g: g["nodes"][1]["weights"].pop()):
            bad = copy.deepcopy(graph)
            change(bad)
            with self.assertRaises(ValueError):
                SemanticGraph(bad)


@unittest.skipIf(GGUFWriter is None, "requires the model-inspection dependency group")
class GgufImport(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = fixture()

    def artifact(self, role="graph", *, graph=None, arch="test-compact", source_text="# never executed\n",
                 descriptor=True, contains=None, omit=None, raw_graph=None, metadata=None):
        graph = self.raw if graph is None else graph
        text = raw_graph if raw_graph is not None else json.dumps(graph, separators=(",", ":"))
        path = self.root / f"{len(list(self.root.glob('*.gguf')))}.gguf"
        writer = GGUFWriter(path, arch)
        meta = {"qwenquant.graph_json": text, "qwenquant.config_json": json.dumps(graph["config"]),
                "qwenquant.recipe_json": json.dumps({"model": graph["model"], "revision": graph["revision"],
                                                      "export": {"sha256": graph["source_weights_sha256"]}}),
                "qwenquant.runtime_sources_json": json.dumps({"test.py": {"text": source_text,
                                                               "sha256": graph["source_sha256"]["test.py"]}}),
                "qwenquant.source_weights_sha256": graph["source_weights_sha256"]}
        meta.update(metadata or {})
        writer.add_uint32("qwenquant.schema_version", 1)
        writer.add_bool("qwenquant.contains_weights", role == "model" if contains is None else contains)
        for key, value in meta.items():
            if key != omit:
                if type(value) is int:
                    writer.add_uint32(key, value)
                elif isinstance(value, list):
                    writer.add_array(key, value)
                else:
                    writer.add_string(key, value)
        if role == "model" and descriptor:
            for name, info in graph["tensor_manifest"].items():
                dtype = np.float16 if info["source_dtype"] == "F16" else np.int8
                data = np.zeros(info["shape"], dtype=dtype)
                if dtype == np.int8:
                    data.flat[:4] = [0, 127, -128, -1]
                writer.add_tensor(name, data)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        identity = {"file": path.name, "size_bytes": path.stat().st_size, "sha256": digest(path.read_bytes())}
        experiment = {"name": "synthetic", "runtime": {"status": "inspection-only"},
                      "source": {"repo": "test/export", "revision": "export-revision"},
                      "upstream_model": {"repo": graph["model"], "revision": graph["revision"]},
                      "architecture": "test-compact", "embedded_graph_sha256": digest(text.encode()),
                      "artifacts": {role: identity}}
        return path, experiment

    def test_both_payload_roles_import_without_executing_embedded_source(self):
        for role in ("graph", "model"):
            # Valid source text need not even be executable Python.
            self.raw["source_sha256"]["test.py"] = digest(b"NOT EXECUTABLE SOURCE")
            path, experiment = self.artifact(role, source_text="NOT EXECUTABLE SOURCE")
            graph, result = importer.inspect(path, experiment, role)
            self.assertEqual(json.loads(graph), self.raw)
            self.assertEqual(result["tensor_payload_present"], role == "model")
            self.assertEqual(result["evidence_kind"], "semantic_model_import")
            self.assertFalse(result["numerical_execution_validated"])

    def test_size_and_hash_identity_checked_before_reader(self):
        path, experiment = self.artifact()
        for key, value in (("size_bytes", 1), ("sha256", "0" * 64)):
            wrong = copy.deepcopy(experiment)
            wrong["artifacts"]["graph"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                importer.inspect(path, wrong, "graph")

    def test_ordinary_gguf_cannot_be_relabelled_as_the_custom_workload(self):
        path, experiment = self.artifact(arch="qwen35")
        with self.assertRaisesRegex(ValueError, "architecture differs"):
            importer.inspect(path, experiment, "graph")

    def test_missing_metadata_and_runtime_source_corruption_fail(self):
        for options, pattern in (({"omit": "qwenquant.graph_json"}, "missing GGUF field"),
                                 ({"source_text": "changed source"}, "source identity differs"),
                                 ({"contains": True}, "payload role differs")):
            path, experiment = self.artifact(**options)
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, pattern):
                importer.inspect(path, experiment, "graph")

    def test_tensor_directory_must_match_the_declared_payload(self):
        path, experiment = self.artifact("model", descriptor=False)
        with self.assertRaisesRegex(ValueError, "tensor directory differs"):
            importer.inspect(path, experiment, "model")

    def test_unsupported_tensor_encoding_fails_before_output(self):
        from gguf import GGUFReader
        import struct
        path, experiment = self.artifact("model")
        reader = GGUFReader(path)
        tensor_type_offset = reader.tensors[0].field.parts[4].ctypes.data - reader.data.ctypes.data
        del reader
        payload = path.read_bytes()
        # I16 occupies the same bytes as F16 but is not the declared encoding.
        payload = payload[:tensor_type_offset] + struct.pack("<I", 25) + payload[tensor_type_offset + 4:]
        path.write_bytes(payload)
        experiment["artifacts"]["model"]["sha256"] = digest(payload)
        out = self.root / "build/workloads/unsupported-storage"
        with self.assertRaisesRegex(ValueError, "tensor directory differs"):
            importer.import_workload(path, experiment, "model", out, self.root)
        self.assertFalse(out.exists())

    def test_unsigned_packs_remain_unsigned_descriptors_in_i8_containers(self):
        self.raw["tensor_manifest"]["test.int4_packed"] = {
            "shape": [4], "source_dtype": "U8", "ggml_type": 24, "bytes": 4}
        path, experiment = self.artifact("model")
        text, result = importer.inspect(path, experiment, "model")
        self.assertEqual(json.loads(text)["tensor_manifest"]["test.int4_packed"]["source_dtype"], "U8")
        self.assertEqual(result["inventory"]["stored_tensor_bytes"], 12)

    def test_reanalysis_cannot_relabel_old_capture_results_as_custom(self):
        manifest = self.root / "run-manifest.json"
        manifest.write_text(json.dumps({"experiment": {"name": "qwen35-2b-zeb-compact"}}))
        with mock.patch.object(cli.pipeline, "analyze_run") as analyze:
            with self.assertRaisesRegex(SystemExit, "inspection-only"):
                cli.analyze_command(type("Args", (), {"run_dir": self.root})())
            analyze.assert_not_called()
        self.assertFalse((self.root / "analysis").exists())

    def test_source_model_and_embedded_graph_identity_fail_independently(self):
        path, experiment = self.artifact()
        for location, value in (("revision", "other-model"), ("repo", "other/model")):
            wrong = copy.deepcopy(experiment)
            wrong["upstream_model"][location] = value
            with self.assertRaisesRegex(ValueError, "source identity differs"):
                importer.inspect(path, wrong, "graph")
        experiment["embedded_graph_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "graph identity differs"):
            importer.inspect(path, experiment, "graph")

    def test_duplicate_and_nonfinite_json_fail(self):
        for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}', '{"x":1e999}', '{"x":1e-999}',
                     '{"x":' + "9" * 5000 + '}'):
            with self.subTest(text=text[:40], length=len(text)), self.assertRaises(ValueError):
                importer.read_json(text)
        # Normalize the parser's own resource limit, without adding a depth policy.
        with mock.patch.object(importer.json, "loads", side_effect=RecursionError), self.assertRaises(ValueError):
            importer.read_json("{}")

    def test_reader_version_is_not_misreported(self):
        path, experiment = self.artifact()
        with mock.patch.object(importer, "version", return_value="other"), self.assertRaisesRegex(ValueError, "locked gguf"):
            importer.inspect(path, experiment, "graph")

    def test_import_is_contained_and_never_overwrites_evidence(self):
        path, experiment = self.artifact()
        out = self.root / "build/workloads/first"
        importer.import_workload(path, experiment, "graph", out, self.root)
        before = (out / "import.json").read_bytes()
        for destination in (out, self.root / "build/workloads", self.root / "escape"):
            with self.subTest(destination=destination), self.assertRaisesRegex(ValueError, "new output directory"):
                importer.import_workload(path, experiment, "graph", destination, self.root)
        self.assertEqual((out / "import.json").read_bytes(), before)
        self.assertFalse((out / "prefill.json").exists())
        self.assertFalse((out / "decode.json").exists())

    def test_invalid_input_writes_no_output(self):
        path, experiment = self.artifact(arch="other")
        out = self.root / "build/workloads/invalid"
        with self.assertRaises(ValueError):
            importer.import_workload(path, experiment, "graph", out, self.root)
        self.assertFalse(out.exists())

    def test_wrong_metadata_types_fail_before_output(self):
        for key in ("qwenquant.graph_json", "qwenquant.config_json", "qwenquant.recipe_json",
                    "qwenquant.runtime_sources_json", "qwenquant.source_weights_sha256"):
            for value in (7, ["not a scalar string"]):
                path, experiment = self.artifact(metadata={key: value})
                out = self.root / "build/workloads" / (key + str(type(value)))
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    importer.import_workload(path, experiment, "graph", out, self.root)
                self.assertFalse(out.exists())

    def test_duplicate_binary_keys_fail_predictably(self):
        path, experiment = self.artifact(metadata={"test.duplicate_a": "a", "test.duplicate_b": "b"})
        payload = path.read_bytes().replace(b"test.duplicate_b", b"test.duplicate_a")
        path.write_bytes(payload)
        experiment["artifacts"]["graph"]["sha256"] = digest(payload)
        with self.assertRaises(ValueError):
            importer.inspect(path, experiment, "graph")

    def test_cli_validation_error_is_actionable_and_produces_no_evidence(self):
        path, experiment = self.artifact(metadata={"qwenquant.graph_json": 7})
        out = cli.REPO / "build/workloads" / ("invalid-cli-" + self.root.name)
        self.assertFalse(out.exists())
        stderr = io.StringIO()
        with mock.patch.object(cli, "load_experiment", return_value=experiment), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                cli.main(["import-workload", str(path), "--artifact", "graph", "--out", str(out)])
        self.assertEqual(error.exception.code, 2)
        self.assertIn("qwenquant.graph_json must be a scalar str", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())
        self.assertFalse(out.exists())

    def test_malformed_binary_and_unsupported_versions_fail(self):
        import struct
        for replace in (lambda b: b"", lambda b: b[:12], lambda b: b"BAD!" + b[4:],
                        lambda b: b[:4] + struct.pack("<I", 2) + b[8:],
                        lambda b: b[:4] + struct.pack("<I", 99) + b[8:],
                        lambda b: b.replace(b"test/model", b"\xffest/model")):
            path, experiment = self.artifact()
            payload = replace(path.read_bytes())
            path.write_bytes(payload)
            experiment["artifacts"]["graph"].update(size_bytes=len(payload), sha256=digest(payload))
            with self.subTest(size=len(payload)), self.assertRaises(ValueError):
                importer.inspect(path, experiment, "graph")

    def test_import_preserves_exact_utf8_graph_bytes(self):
        text = json.dumps(self.raw, indent=2) + "\n"
        path, experiment = self.artifact(raw_graph=text)
        out = self.root / "build/workloads/multiline"
        importer.import_workload(path, experiment, "graph", out, self.root)
        self.assertEqual((out / "semantic-graph.json").read_bytes(), text.encode("utf-8"))

    def test_redirected_output_root_is_rejected(self):
        path, experiment = self.artifact()
        outside = self.root / "not-generated-evidence"
        outside.mkdir()
        (self.root / "build").mkdir()
        try:
            (self.root / "build/workloads").symlink_to(outside, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"directory symlinks unavailable: {error}")
        with self.assertRaises(ValueError):
            importer.import_workload(path, experiment, "graph", self.root / "build/workloads/new", self.root)
        self.assertEqual(list(outside.iterdir()), [])

    def test_profile_and_doctor_fail_before_build_or_download_for_custom_workload(self):
        for command in ("profile", "doctor"):
            args = cli.build_parser().parse_args([command, "--experiment", "qwen35-2b-zeb-compact"])
            with self.assertRaisesRegex(SystemExit, "inspection-only"):
                cli.make_context(args)
        self.assertEqual(DEFAULT_EXPERIMENT, "qwen35-2b-q4km")
        self.assertEqual(load_experiment()["model"]["sha256"],
                         "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223")


if __name__ == "__main__":
    unittest.main()
