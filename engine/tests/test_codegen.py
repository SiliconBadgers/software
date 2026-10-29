"""Symbolic compiler contracts; standard-library only, without tensor execution."""
import copy
from dataclasses import FrozenInstanceError, replace
import gzip
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ENGINE = Path(__file__).resolve().parents[1]
ROOT = ENGINE.parent
sys.path.insert(0, str(ENGINE))

from sbengine.codegen import Diagnostic, LoweringError, compile_graph, lower_matmul
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
                             (0, [7, 5, 2, 1]), (0, [0, 5, 1, 1])):
            with self.subTest(index=index, shape=shape):
                raw = copy.deepcopy(self.capture)
                tensor = raw["tensors"][index]
                element_bytes = 2 if tensor["dtype"] == "bf16" else 4
                strides = [element_bytes]
                for dimension in shape[:3]:
                    strides.append(strides[-1] * dimension)
                elements = math.prod(shape)
                tensor.update(shape=shape, elements=elements, strides_bytes=strides,
                              span_bytes=elements * element_bytes, logical_bytes=elements * element_bytes)
                # Internally consistent descriptors must reach the matmul/layout guard.
                with self.assertRaisesRegex(LoweringError, "incompatible|batched|positive"):
                    lower_matmul(self.graph(raw), 30)

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

    def test_changed_source_cannot_be_hashed_with_stale_graph_semantics(self):
        graph = self.graph()
        graph.capture["tensors"][2].update(op="ADD", op_desc="ADD")
        # Both raw fields now consistently say ADD, while the previously loaded
        # node still says MUL_MAT. Neither entry point may mix these snapshots.
        with self.assertRaises(ValueError):
            compile_graph(graph)
        with self.assertRaises(LoweringError):
            lower_matmul(graph, 30)

    def test_backing_must_agree_with_source_in_api_and_cli(self):
        raw = copy.deepcopy(self.capture)
        for tensor in raw["tensors"]:
            size = 2 if tensor["dtype"] == "bf16" else 4
            tensor.update(shape=[1] * 4, strides_bytes=[size] * 4, elements=1,
                          span_bytes=size, logical_bytes=size)
        unrelated = copy.deepcopy(raw["tensors"][1])
        unrelated.update(id=22, name="fixture.unrelated")
        view = copy.deepcopy(raw["tensors"][1])
        view.update(id=21, name="fixture.view", sources=[{"slot": 0, "tensor": 20}])
        raw["tensors"].extend([unrelated, view])
        raw["tensors"][2]["sources"][1]["tensor"] = 21
        raw["scheduled_nodes"] = [21, 30]
        for operation in ("VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE", "CPY"):
            sources = ([{"slot": 0, "tensor": 22}, {"slot": 1, "tensor": 20}]
                       if operation == "CPY" else [{"slot": 0, "tensor": 20}])
            view.update(op=operation, op_desc=operation, view_source=20,
                        sources=sources,
                        op_params_i32=([0, 1, 2, 3] + [0] * 12 if operation == "PERMUTE" else [0] * 16))
            self.assertEqual(lower_matmul(self.graph(raw), 30).b.storage_id, 20)
            for backing in (-1, 22):
                with self.subTest(operation=operation, backing=backing):
                    view["view_source"] = backing
                    graph = self.graph(raw)
                    with self.assertRaisesRegex(ValueError, "backing storage"):
                        compile_graph(graph)
                    for flags in ((), ("--allow-partial",)):
                        result = self.cli(graph, *flags)
                        self.assertEqual(result.returncode, 2, result.stderr)
                        self.assertEqual(result.stdout, "")
                        self.assertIn("backing storage", result.stderr)
                        self.assertNotIn("Traceback", result.stderr)

    def test_selectors_require_integer_tensor_ids(self):
        raw = copy.deepcopy(self.capture)
        raw["tensors"][2]["id"] = 1
        raw["scheduled_nodes"] = [1]
        graph = self.graph(raw)
        self.assertEqual(lower_matmul(graph, 1).tensor_id, 1)
        for selector in (1.0, True, "1", -1):
            with self.subTest(selector=selector):
                with self.assertRaises(ValueError):
                    compile_graph(graph, tensor_id=selector)
                with self.assertRaises(ValueError):
                    lower_matmul(graph, selector)

    def test_missing_operand_and_unknown_selection_are_rejected(self):
        raw = copy.deepcopy(self.capture)
        raw["tensors"][2]["sources"] = [{"slot": 0, "tensor": 10}]
        self.rejected(raw)
        self.rejected(self.capture, tensor_id=999)

    def test_output_alias_of_an_input_is_rejected(self):
        raw = copy.deepcopy(self.capture)
        raw["tensors"][2].update(view_source=20, view_offset_bytes=0)
        with self.assertRaisesRegex(LoweringError, "output aliases"):
            lower_matmul(self.graph(raw), 30)

    def test_nested_input_views_preserve_owner_and_cumulative_offset(self):
        plan = lower_matmul(self.graph(self.with_views()), 30)
        self.assertEqual((plan.a.tensor_id, plan.a.storage_id, plan.a.offset_bytes), (10, 50, 12))
        self.assertEqual(plan.a.shape, (7, 5, 1, 1))

    def test_forged_ancestor_spans_cannot_authorize_a_view(self):
        for ancestor in ("owner", "parent"):
            with self.subTest(ancestor=ancestor):
                raw = self.with_views()
                if ancestor == "owner":
                    raw["tensors"][3]["span_bytes"] = 1000
                    raw["tensors"][4]["span_bytes"] = 900
                    raw["tensors"][0]["view_offset_bytes"] = 500
                else:
                    raw["tensors"][4]["span_bytes"] = 130
                    raw["tensors"][0]["view_offset_bytes"] = 50
                # Inflated spans make the address inequalities pass, but the
                # ancestor shapes, strides and source formats cannot back them.
                self.rejected(raw)

    def test_phase_mismatch_is_rejected_by_api_and_cli(self):
        graph = self.graph(phase="decode")
        self.assertEqual(graph.meta["phase"], "prefill")
        with self.assertRaisesRegex(ValueError, "phase|metadata"):
            compile_graph(graph)
        result = self.cli(graph)
        self.assertEqual(result.returncode, 2)
        self.assertIn("phase", result.stderr)

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

    def test_deep_flattened_metadata_chain_retains_producer_in_api_and_cli(self):
        raw = self.with_matmul_chain()
        previous = 30
        view_ids = range(1000, 1000 + sys.getrecursionlimit() + 10)
        for tensor_id in view_ids:
            view = copy.deepcopy(raw["tensors"][2])
            view.update(id=tensor_id, name=f"fixture.view_{tensor_id}", op="VIEW", op_desc="VIEW",
                        view_source=30, sources=[{"slot": 0, "tensor": previous}])
            raw["tensors"].append(view)
            previous = tensor_id
        raw["tensors"][4]["sources"][1]["tensor"] = previous
        raw["scheduled_nodes"] = [30, *view_ids, 70]
        graph = self.graph(raw)
        report = json.loads(compile_graph(graph, tensor_id=70).to_json())
        self.assertEqual(report["plans"][0]["dependencies"], [30])
        self.assertEqual(report["source"]["external_dependency_tensor_ids"], [30])
        result = self.cli(graph, "--tensor-id", "70")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), report)

    def test_json_is_deterministic_and_decoded_identity_ignores_gzip(self):
        plain = compile_graph(self.graph())
        compressed = compile_graph(self.graph(compressed=True))
        self.assertEqual(plain.capture_sha256, decoded_capture_sha256(self.capture))
        self.assertEqual(plain.capture_sha256, compressed.capture_sha256)
        self.assertEqual(plain.to_json(), compressed.to_json())
        result = json.loads(plain.to_json())
        self.assertEqual(result["schema"], "siliconbadgers.compiler-plan.v0")
        self.assertEqual(result["execution_status"], "symbolic_only")
        self.assertEqual(result["counts"], {"total": 1, "planned": 1, "unsupported": 0})

    def test_semantic_capture_change_changes_provenance(self):
        first = compile_graph(self.graph())
        self.capture["metadata"]["source"] += " revised provenance"
        second = compile_graph(self.graph())
        self.assertNotEqual(first.capture_sha256, second.capture_sha256)

    def test_serialization_rejects_invalid_provenance_phase_and_scope(self):
        report = compile_graph(self.graph())
        for patch in ({"capture_sha256": "not-a-hash"}, {"capture_sha256": "f" * 63},
                      {"phase": "training"}, {"scope": "unknown"}):
            with self.subTest(patch=patch):
                with self.assertRaises(ValueError):
                    replace(report, **patch).to_json()

    def test_serialization_requires_exact_nonoverlapping_coverage(self):
        report = compile_graph(self.graph())
        plan = report.plans[0]
        patches = (
            {"plans": (plan, plan)},
            {"plans": ()},
            {"diagnostics": (Diagnostic(30, "MUL_MAT", "unsupported"),)},
            {"source_order": (30, 30)},
            {"source_order": (999,)},
        )
        for patch in patches:
            with self.subTest(patch=patch):
                with self.assertRaises(ValueError):
                    replace(report, **patch).to_json()

    def test_serialization_validates_diagnostic_fields(self):
        report = compile_graph(self.graph())
        for diagnostic in (Diagnostic(30, "", "unsupported"), Diagnostic(30, "MUL_MAT", ""),
                           Diagnostic(-1, "MUL_MAT", "unsupported"),
                           Diagnostic(30.0, "MUL_MAT", "unsupported")):
            with self.subTest(diagnostic=diagnostic):
                with self.assertRaises(ValueError):
                    replace(report, plans=(), diagnostics=(diagnostic,)).to_json()

    def test_caller_owned_lists_cannot_mutate_frozen_records(self):
        report = compile_graph(self.graph())
        original = report.plans[0]
        shape, strides = list(original.a.shape), list(original.a.strides_bytes)
        tensor = replace(original.a, shape=shape, strides_bytes=strides)
        params, dependencies = list(original.op_params_i32), list(original.dependencies)
        plan = replace(original, a=tensor, op_params_i32=params, dependencies=dependencies)
        plans, diagnostics, source_order = [plan], [], [30]
        frozen = replace(report, plans=plans, diagnostics=diagnostics, source_order=source_order)
        before = frozen.to_json()
        shape[0] = 999
        strides[0] = 999
        params[0] = 999
        dependencies.append(999)
        plans.clear()
        diagnostics.append(Diagnostic(30, "MUL_MAT", "injected"))
        source_order.clear()
        self.assertEqual(frozen.to_json(), before)

    def test_full_graph_report_names_operations_scope_and_coverage(self):
        result = compile_graph(self.graph(self.with_unsupported()))
        report = json.loads(result.to_json())
        self.assertEqual(report["scope"], "full_graph")
        self.assertEqual(report["counts"], {"total": 2, "planned": 1, "unsupported": 1})
        self.assertEqual(report["source"]["operation_tensor_ids"], [30, 40])
        self.assertEqual(report["source"]["external_dependency_tensor_ids"], [])
        self.assertEqual(report["coverage_by_op"], {
            "MUL_MAT": {"total": 1, "planned": 1, "unsupported": 0},
            "RELU": {"total": 1, "planned": 0, "unsupported": 1},
        })
        self.assertEqual(report["plans"][0]["operation"], "MUL_MAT")
        self.assertEqual(report["diagnostics"][0]["tensor_id"], 40)
        self.assertEqual(report["diagnostics"][0]["op"], "RELU")
        self.assertTrue(report["diagnostics"][0]["reason"])

    def test_selected_report_declares_dependencies_outside_the_selection(self):
        graph = self.graph(self.with_matmul_chain())
        selected = compile_graph(graph, tensor_id=70)
        report = json.loads(selected.to_json())
        self.assertEqual(report["scope"], "selected_operation")
        self.assertEqual(selected.source_order, (70,))
        self.assertEqual(report["source"]["operation_tensor_ids"], [70])
        self.assertEqual(report["source"]["external_dependency_tensor_ids"], [30])
        self.assertEqual(report["plans"][0]["dependencies"], [30])
        self.assertEqual(report["coverage_by_op"], {
            "MUL_MAT": {"total": 1, "planned": 1, "unsupported": 0},
        })
        full = json.loads(compile_graph(graph).to_json())
        self.assertEqual(full["source"]["external_dependency_tensor_ids"], [])

    def test_codegen_preserves_copy_dependencies_for_ordinary_input_storage(self):
        raw = copy.deepcopy(self.capture)
        incoming = copy.deepcopy(raw["tensors"][1])
        incoming.update(id=22, name="fixture.copy_source")
        copied = copy.deepcopy(raw["tensors"][1])
        copied.update(id=21, name="fixture.copy", op="CPY", op_desc="CPY", view_source=20,
                      sources=[{"slot": 0, "tensor": 22}, {"slot": 1, "tensor": 20}])
        raw["tensors"].extend([incoming, copied])
        raw["tensors"][2]["sources"][1]["tensor"] = 21
        raw["scheduled_nodes"] = [21, 30]
        graph = self.graph(raw)
        # The inherited analytical dependencies deliberately omit this ordinary
        # input-buffer write. Codegen must use the checked source dependency pass.
        self.assertEqual(graph.deps(), [[], []])
        full = compile_graph(graph)
        self.assertEqual(full.source_order, (21, 30))
        self.assertEqual([(d.tensor_id, d.op) for d in full.diagnostics], [(21, "CPY")])
        self.assertEqual([p.tensor_id for p in full.plans], [30])
        self.assertEqual(full.plans[0].dependencies, (21,))
        selected = json.loads(compile_graph(graph, tensor_id=30).to_json())
        self.assertEqual(selected["source"]["external_dependency_tensor_ids"], [21])

    def test_zero_element_owning_cast_is_skipped_by_api_and_cli(self):
        raw = copy.deepcopy(self.capture)
        source = copy.deepcopy(raw["tensors"][1])
        source.update(id=40, name="fixture.empty_input", shape=[0, 1, 1, 1],
                      strides_bytes=[4, 0, 0, 0], elements=0, span_bytes=0, logical_bytes=0)
        cast = copy.deepcopy(source)
        # ggml_cast owns its result and uses itself as the CPY destination.
        cast.update(id=41, name="fixture.empty_cast", op="CPY", op_desc="CPY",
                    dtype="f16", strides_bytes=[2, 0, 0, 0],
                    sources=[{"slot": 0, "tensor": 40}, {"slot": 1, "tensor": 41}])
        raw["tensors"].extend([source, cast])
        raw["scheduled_nodes"] = [41, 30]
        graph = self.graph(raw)
        for selected in (None, 30):
            with self.subTest(api_tensor_id=selected):
                report = compile_graph(graph, tensor_id=selected)
                self.assertEqual(report.source_order, (30,))
                self.assertEqual([plan.tensor_id for plan in report.plans], [30])
                self.assertEqual(report.diagnostics, ())
        for flags in ((), ("--tensor-id", "30"), ("--allow-partial",)):
            with self.subTest(cli_flags=flags):
                result = self.cli(graph, *flags)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)["counts"],
                                 {"total": 1, "planned": 1, "unsupported": 0})

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

    def test_malformed_cli_inputs_fail_without_output_or_traceback(self):
        for corruption in ("truncated_gzip", "deflate", "metadata"):
            with self.subTest(corruption=corruption):
                graph = self.graph(compressed=corruption != "metadata")
                if corruption == "metadata":
                    raw = copy.deepcopy(self.capture)
                    raw["metadata"] = []
                    graph.path.write_text(json.dumps(raw), encoding="utf-8")
                else:
                    data = bytearray(graph.path.read_bytes())
                    if corruption == "truncated_gzip":
                        del data[-8:]
                    else:
                        # Reserved DEFLATE block type after the ten-byte gzip header.
                        data[10] = (data[10] & 0xf8) | 7
                    graph.path.write_bytes(data)
                result = self.cli(graph)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertEqual(result.stdout, "")
                self.assertIn("metadata" if corruption == "metadata" else "error:", result.stderr)
                self.assertNotIn("Traceback", result.stderr)

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
