"""Symbolic compiler contracts; standard-library only, without tensor execution."""
import copy
from dataclasses import FrozenInstanceError
import gzip
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ENGINE = Path(__file__).resolve().parents[1]
ROOT = ENGINE.parent
sys.path.insert(0, str(ENGINE))

from sbengine.codegen import LoweringError, compile_graph, lower_matmul
from sbengine.graph import Graph
from sbengine.provenance import decoded_capture_sha256

FIXTURE = ENGINE / "tests/fixtures/compiler-matmul.json"
REAL_CAPTURE = ROOT / "results/2026-09-27_1839_windows-2/graphs/pp128-fa-on/prefill.json.gz"


class SymbolicMatmul(unittest.TestCase):
    def setUp(self):
        self.capture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.serial = 0

    def graph(self, capture=None, *, phase="prefill", compressed=False):
        raw = self.capture if capture is None else capture
        self.serial += 1
        path = Path(self.directory.name) / f"capture-{self.serial}.json"
        data = json.dumps(raw, indent=2).encode("utf-8")
        if compressed:
            path = path.with_suffix(".json.gz")
            data = gzip.compress(data)
        path.write_bytes(data)
        return Graph(path, phase)

    def rejected(self, capture, tensor_id=30):
        with self.assertRaises(LoweringError) as caught:
            lower_matmul(self.graph(capture), tensor_id)
        self.assertTrue(caught.exception.reason)
        self.assertIsInstance(caught.exception, ValueError)

    def cli(self, graph, *extra):
        return subprocess.run(
            [sys.executable, "-B", "-m", "sbengine.codegen", str(graph.path),
             "--phase", graph.phase, *extra],
            cwd=ENGINE, capture_output=True, text=True, timeout=20,
        )

    def with_unsupported(self):
        raw = copy.deepcopy(self.capture)
        unsupported = copy.deepcopy(raw["tensors"][2])
        unsupported.update(id=40, name="fixture.relu", op="UNARY", op_desc="RELU",
                           sources=[{"slot": 0, "tensor": 30}])
        raw["tensors"].append(unsupported)
        raw["scheduled_nodes"].append(40)
        return raw

    def with_views(self):
        raw = copy.deepcopy(self.capture)
        weight = raw["tensors"][0]
        owner = copy.deepcopy(weight)
        owner.update(id=50, name="fixture.owner", shape=[7, 10, 1, 1],
                     strides_bytes=[2, 14, 140, 140], elements=70,
                     span_bytes=140, logical_bytes=140)
        middle = copy.deepcopy(weight)
        middle.update(id=51, name="fixture.middle", op="VIEW", op_desc="VIEW",
                      shape=[7, 8, 1, 1], strides_bytes=[2, 14, 112, 112], elements=56,
                      span_bytes=112, logical_bytes=112, view_source=50, view_offset_bytes=8,
                      sources=[{"slot": 0, "tensor": 50}])
        weight.update(op="VIEW", op_desc="VIEW", view_source=51, view_offset_bytes=4,
                      sources=[{"slot": 0, "tensor": 51}])
        raw["tensors"].extend([owner, middle])
        return raw

    def with_matmul_chain(self):
        raw = copy.deepcopy(self.capture)
        weight = copy.deepcopy(raw["tensors"][0])
        weight.update(id=60, name="fixture.second_weight", shape=[5, 4, 1, 1],
                      elements=20, strides_bytes=[2, 10, 40, 40],
                      logical_bytes=40, span_bytes=40)
        output = copy.deepcopy(raw["tensors"][2])
        output.update(id=70, name="fixture.second_output", shape=[4, 3, 1, 1],
                      elements=12, strides_bytes=[4, 16, 48, 48],
                      logical_bytes=48, span_bytes=48,
                      sources=[{"slot": 0, "tensor": 60}, {"slot": 1, "tensor": 30}])
        raw["tensors"].extend([weight, output])
        raw["scheduled_nodes"] = [30, 70]
        return raw

    def test_nonsquare_dimensions_and_frozen_tensor_references(self):
        plan = lower_matmul(self.graph(), 30)
        # These unequal dimensions catch silently swapping GGML's inner-first axes.
        self.assertEqual((plan.m, plan.n, plan.k), (5, 3, 7))
        self.assertEqual((plan.a.tensor_id, plan.b.tensor_id, plan.c.tensor_id), (10, 20, 30))
        self.assertEqual(plan.tensor_id, 30)
        self.assertEqual(plan.a.shape, (7, 5, 1, 1))
        self.assertEqual((plan.a.storage_id, plan.a.offset_bytes), (10, 0))
        with self.assertRaises(FrozenInstanceError):
            plan.m = 999
        with self.assertRaises(FrozenInstanceError):
            plan.a.tensor_id = 999

    def test_decode_single_column(self):
        self.capture["metadata"].update(phase="decode", input_tokens=1)
        for tensor, width in ((self.capture["tensors"][1], 7), (self.capture["tensors"][2], 5)):
            tensor.update(shape=[width, 1, 1, 1], elements=width,
                          strides_bytes=[4, width * 4, width * 4, width * 4],
                          span_bytes=width * 4, logical_bytes=width * 4)
        graph = self.graph(phase="decode")
        result = compile_graph(graph)
        self.assertEqual(result.phase, "decode")
        self.assertEqual((result.plans[0].m, result.plans[0].n, result.plans[0].k), (5, 1, 7))

    def test_capture_descriptors_and_params_are_preserved_without_mutation(self):
        self.capture["tensors"][2]["op_params_i32"][0:2] = [17, -4]
        graph = self.graph()
        before = copy.deepcopy(graph.capture)
        plan = lower_matmul(graph, 30)
        self.assertEqual((plan.a.dtype, plan.b.dtype, plan.c.dtype), ("bf16", "f32", "f32"))
        self.assertEqual(plan.a.strides_bytes, (2, 14, 70, 70))
        self.assertEqual((plan.a.span_bytes, plan.a.logical_bytes), (70, 70))
        self.assertEqual(plan.op_params_i32, (17, -4) + (0,) * 14)
        self.assertEqual(graph.capture, before)

    def test_invalid_dimensions_and_broadcast_are_rejected(self):
        for index, shape in ((1, [8, 3, 1, 1]), (2, [6, 3, 1, 1]),
                             (0, [7, 5, 2, 1]), (0, [7, 5, 1]), (0, [0, 5, 1, 1])):
            with self.subTest(index=index, shape=shape):
                raw = copy.deepcopy(self.capture)
                raw["tensors"][index]["shape"] = shape
                self.rejected(raw)

    def test_missing_or_invalid_layout_is_rejected(self):
        changes = (("contiguous", False), ("contiguous", None),
                   ("strides_bytes", [2, -14, 70, 70]), ("strides_bytes", [2, 14]),
                   ("strides_bytes", None), ("span_bytes", -1), ("logical_bytes", -1))
        for field, value in changes:
            with self.subTest(field=field, value=value):
                raw = copy.deepcopy(self.capture)
                if value is None:
                    del raw["tensors"][0][field]
                else:
                    raw["tensors"][0][field] = value
                self.rejected(raw)

    def test_contiguous_flag_cannot_hide_inconsistent_strides(self):
        for strides in ([0, 0, 0, 0], [2, 12, 70, 70], [2, 16, 80, 80]):
            with self.subTest(strides=strides):
                raw = copy.deepcopy(self.capture)
                raw["tensors"][0]["strides_bytes"] = strides
                self.assertTrue(raw["tensors"][0]["contiguous"])
                self.rejected(raw)

    def test_source_format_and_byte_counts_must_describe_the_tensor(self):
        for patch in ({"span_bytes": 1, "logical_bytes": 1},
                      {"logical_bytes": 68}, {"span_bytes": 68},
                      {"dtype": "unrecognized_format"}):
            with self.subTest(patch=patch):
                raw = copy.deepcopy(self.capture)
                raw["tensors"][0].update(patch)
                # The declared BF16 tensor contains 35 two-byte elements.
                self.rejected(raw)

    def test_operation_description_cannot_override_a_different_operation(self):
        raw = copy.deepcopy(self.capture)
        raw["tensors"][2]["op"] = "ADD"
        self.assertEqual(raw["tensors"][2]["op_desc"], "MUL_MAT")
        self.rejected(raw)

    def test_missing_operand_and_unknown_selection_are_rejected(self):
        raw = copy.deepcopy(self.capture)
        raw["tensors"][2]["sources"] = [{"slot": 0, "tensor": 10}]
        self.rejected(raw)
        self.rejected(self.capture, tensor_id=999)

    def test_output_alias_of_an_input_is_rejected(self):
        raw = copy.deepcopy(self.capture)
        raw["tensors"][2].update(view_source=10, view_offset_bytes=0)
        self.rejected(raw)

    def test_nested_input_views_preserve_owner_and_cumulative_offset(self):
        plan = lower_matmul(self.graph(self.with_views()), 30)
        self.assertEqual((plan.a.tensor_id, plan.a.storage_id, plan.a.offset_bytes), (10, 50, 12))
        self.assertEqual(plan.a.shape, (7, 5, 1, 1))

    def test_invalid_view_offsets_are_rejected(self):
        for index, offset in ((0, -1), (0, 1000), (4, 1000)):
            with self.subTest(index=index, offset=offset):
                raw = self.with_views()
                raw["tensors"][index]["view_offset_bytes"] = offset
                self.rejected(raw)

    def test_view_must_fit_immediate_parent_even_when_root_has_space(self):
        raw = self.with_views()
        raw["tensors"][0]["view_offset_bytes"] = 50
        # Child occupies [58, 128) in a 140-byte owner, but [50, 120)
        # exceeds its immediate 112-byte parent view.
        self.assertLessEqual(8 + 50 + 70, 140)
        self.assertGreater(50 + 70, 112)
        self.rejected(raw)

    def test_phase_mismatch_is_rejected_by_api_and_cli(self):
        graph = self.graph(phase="decode")
        self.assertEqual(graph.meta["phase"], "prefill")
        with self.assertRaisesRegex(ValueError, "phase|metadata"):
            compile_graph(graph)
        result = self.cli(graph)
        self.assertEqual(result.returncode, 2)
        self.assertIn("prefill", result.stderr)

    def test_duplicate_tensor_and_schedule_ids_are_rejected(self):
        for duplicate in ("tensor", "schedule"):
            with self.subTest(duplicate=duplicate):
                raw = copy.deepcopy(self.capture)
                if duplicate == "tensor":
                    raw["tensors"].append(copy.deepcopy(raw["tensors"][0]))
                else:
                    raw["scheduled_nodes"].append(30)
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    compile_graph(self.graph(raw))

    def test_source_order_must_place_producers_before_consumers(self):
        raw = self.with_matmul_chain()
        ordered = compile_graph(self.graph(raw))
        self.assertEqual(ordered.diagnostics, ())
        self.assertEqual([(p.m, p.n, p.k) for p in ordered.plans], [(5, 3, 7), (4, 3, 5)])
        raw["scheduled_nodes"] = [70, 30]
        with self.assertRaisesRegex(ValueError, "topological"):
            compile_graph(self.graph(raw))

    def test_omitted_producers_do_not_become_external_inputs(self):
        for through_view in (False, True):
            with self.subTest(through_view=through_view):
                raw = self.with_matmul_chain()
                if through_view:
                    view = copy.deepcopy(raw["tensors"][2])
                    view.update(id=80, name="fixture.produced_view", op="VIEW", op_desc="VIEW",
                                view_source=30, view_offset_bytes=0,
                                sources=[{"slot": 0, "tensor": 30}])
                    raw["tensors"][-1]["sources"][1]["tensor"] = 80
                    raw["tensors"].append(view)
                # A selected region is allowed to reference a source producer
                # that remains in the complete captured schedule.
                selected = compile_graph(self.graph(raw), tensor_id=70)
                self.assertEqual(len(selected.plans), 1)
                self.assertEqual(selected.diagnostics, ())
                raw["scheduled_nodes"] = [70]
                with self.assertRaisesRegex(ValueError, "missing"):
                    compile_graph(self.graph(raw))

    def test_positive_shape_cannot_be_hidden_as_a_zero_element_node(self):
        raw = self.with_matmul_chain()
        raw["tensors"][-1]["elements"] = 0
        graph = self.graph(raw)
        self.assertEqual([node.tensor.id for node in graph.nodes], [30])
        with self.assertRaisesRegex(ValueError, "element count"):
            compile_graph(graph)

    def test_json_is_deterministic_and_decoded_identity_ignores_gzip(self):
        plain = compile_graph(self.graph())
        compressed = compile_graph(self.graph(compressed=True))
        self.assertEqual(plain.capture_sha256, decoded_capture_sha256(self.capture))
        self.assertEqual(plain.capture_sha256, compressed.capture_sha256)
        self.assertEqual(plain.to_json(), compressed.to_json())
        self.assertEqual(plain.to_json(), plain.to_json())
        result = json.loads(plain.to_json())
        self.assertEqual(result["schema"], "siliconbadgers.compiler-plan.v0")
        self.assertEqual(result["execution_status"], "symbolic_only")
        self.assertEqual(result["counts"], {"total": 1, "planned": 1, "unsupported": 0})

    def test_semantic_capture_change_changes_provenance(self):
        first = compile_graph(self.graph())
        self.capture["metadata"]["source"] += " revised provenance"
        second = compile_graph(self.graph())
        self.assertNotEqual(first.capture_sha256, second.capture_sha256)

    def test_every_active_node_is_planned_or_reported(self):
        graph = self.graph(self.with_unsupported())
        result = compile_graph(graph)
        self.assertEqual([plan.tensor_id for plan in result.plans], [30])
        self.assertEqual(len(result.diagnostics), 1)
        report = json.loads(result.to_json())
        self.assertEqual(report["counts"], {"total": 2, "planned": 1, "unsupported": 1})
        self.assertEqual(report["diagnostics"][0]["tensor_id"], 40)
        self.assertEqual(report["diagnostics"][0]["op"], "RELU")
        self.assertTrue(report["diagnostics"][0]["reason"])
        selected = compile_graph(graph, tensor_id=30)
        self.assertEqual(len(selected.plans), 1)
        self.assertEqual(selected.diagnostics, ())

    def test_cli_reports_partial_work_and_requires_explicit_allow_partial(self):
        graph = self.graph(self.with_unsupported())
        strict = self.cli(graph)
        permissive = self.cli(graph, "--allow-partial")
        self.assertEqual(strict.returncode, 2, strict.stderr)
        self.assertEqual(permissive.returncode, 0, permissive.stderr)
        self.assertEqual(json.loads(strict.stdout), json.loads(permissive.stdout))
        self.assertEqual(json.loads(strict.stdout)["counts"]["unsupported"], 1)

    def test_cli_selected_matmul_succeeds_and_unknown_id_fails(self):
        graph = self.graph(self.with_unsupported())
        selected = self.cli(graph, "--tensor-id", "30")
        unknown = self.cli(graph, "--tensor-id", "999")
        self.assertEqual(selected.returncode, 0, selected.stderr)
        self.assertEqual(json.loads(selected.stdout)["counts"]["planned"], 1)
        self.assertEqual(unknown.returncode, 2)

    @unittest.skipUnless(REAL_CAPTURE.is_file(), "committed Qwen capture is unavailable")
    def test_real_qwen_matmul_preserves_mixed_quantization_and_source_ids(self):
        graph = Graph(REAL_CAPTURE, "prefill")
        plan = lower_matmul(graph, 18)
        self.assertEqual((plan.m, plan.n, plan.k), (6144, 128, 2048))
        self.assertEqual((plan.a.tensor_id, plan.b.tensor_id, plan.c.tensor_id), (19, 4, 18))
        self.assertEqual(plan.a.dtype, graph.tensors[19].dtype)
        self.assertIn(plan.a.dtype.lower(), ("q4_k", "q5_k", "q6_k", "q8_0"))
        self.assertEqual(plan.a.logical_bytes, graph.tensors[19].nbytes)
        result = compile_graph(graph, tensor_id=18)
        self.assertEqual(result.capture_sha256, decoded_capture_sha256(graph.capture))
        self.assertEqual(json.loads(result.to_json())["execution_status"], "symbolic_only")


if __name__ == "__main__":
    unittest.main()
