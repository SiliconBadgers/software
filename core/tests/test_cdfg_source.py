"""Strict source auditing for compiler consumers of the captured CDFG.

The fixture describes shapes and storage only. These checks do not execute a
model, evaluate tensor values, or establish support on an accelerator target.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "core"))
from cdfg import Graph  # noqa: E402
from cdfg.source import source_dependencies, storage_chain, validate_source  # noqa: E402


class BoundedLookupDict(dict):
    """Fail deterministically if a shared metadata path is expanded repeatedly."""

    def __init__(self, values, limit=20000):
        super().__init__(values)
        self.lookups = 0
        self.limit = limit

    def __getitem__(self, key):
        self.lookups += 1
        if self.lookups > self.limit:
            raise AssertionError("metadata traversal exceeded its lookup budget")
        return super().__getitem__(key)

    def get(self, key, default=None):
        self.lookups += 1
        if self.lookups > self.limit:
            raise AssertionError("metadata traversal exceeded its lookup budget")
        return super().get(key, default)


class SourceValidationTests(unittest.TestCase):
    def setUp(self):
        fixture = ROOT / "engine/tests/fixtures/compiler-matmul.json"
        self.capture = json.loads(fixture.read_text(encoding="utf-8"))
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "prefill.json"

    def graph(self, capture=None):
        self.path.write_text(json.dumps(self.capture if capture is None else capture), encoding="utf-8")
        return Graph(self.path, "prefill")

    @staticmethod
    def tensor(capture, tid):
        return next(t for t in capture["tensors"] if t["id"] == tid)

    def dependencies(self, graph):
        raw = validate_source(graph)
        deps = source_dependencies(graph, raw)
        return {node.tensor.id: [graph.nodes[i].tensor.id for i in deps[node.idx]]
                for node in graph.nodes}

    def copy_into_input(self):
        capture = deepcopy(self.capture)
        incoming = deepcopy(self.tensor(capture, 20))
        incoming.update(id=22, name="fixture.copy_source")
        copied = deepcopy(self.tensor(capture, 20))
        copied.update(id=21, name="fixture.copy", op="CPY", op_desc="CPY", view_source=20,
                      sources=[{"slot": 0, "tensor": 22}, {"slot": 1, "tensor": 20}])
        capture["tensors"].extend([incoming, copied])
        self.tensor(capture, 30)["sources"][1]["tensor"] = 21
        capture["scheduled_nodes"] = [21, 30]
        return capture

    def metadata_views(self, count=2, op="VIEW"):
        capture = self.copy_into_input()
        previous = 21
        ids = range(100, 100 + count)
        for tid in ids:
            view = deepcopy(self.tensor(capture, 20))
            view.update(id=tid, name=f"fixture.view_{tid}", op=op, op_desc=op,
                        view_source=20, sources=[{"slot": 0, "tensor": previous}])
            capture["tensors"].append(view)
            previous = tid
        self.tensor(capture, 30)["sources"][1]["tensor"] = previous
        capture["scheduled_nodes"] = [21, *ids, 30]
        return capture

    def test_metadata_backing_matches_resolved_source_not_immediate_id(self):
        for op in ("VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"):
            with self.subTest(op=op):
                graph = self.graph(self.metadata_views(op=op))
                raw = validate_source(graph)
                self.assertEqual(raw[101]["sources"][0]["tensor"], 100)
                self.assertEqual(storage_chain(raw, 101), (101, 20))
                self.assertEqual(source_dependencies(graph, raw), [[], [0]])

    def test_alias_operations_require_consistent_backing_and_exact_slots(self):
        for op in ("VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE", "CPY"):
            for corruption in ("own storage", "unrelated storage", "missing slot", "extra slot"):
                with self.subTest(op=op, corruption=corruption):
                    capture = self.copy_into_input() if op == "CPY" else self.metadata_views(op=op)
                    tensor = self.tensor(capture, 21 if op == "CPY" else 101)
                    if corruption == "own storage":
                        tensor["view_source"] = -1
                    elif corruption == "unrelated storage":
                        tensor["view_source"] = 22
                    elif corruption == "missing slot":
                        tensor["sources"].pop()
                    else:
                        tensor["sources"].append({"slot": len(tensor["sources"]), "tensor": 20})
                    with self.assertRaisesRegex(ValueError, "backing storage|operand slots"):
                        validate_source(self.graph(capture))

    def test_source_less_none_alias_remains_valid(self):
        capture = self.copy_into_input()
        alias = deepcopy(self.tensor(capture, 20))
        alias.update(id=23, name="fixture.external_alias", view_source=20, sources=[])
        capture["tensors"].append(alias)
        self.tensor(capture, 30)["sources"][1]["tensor"] = 23
        graph = self.graph(capture)
        raw = validate_source(graph)
        self.assertEqual(storage_chain(raw, 23), (23, 20))
        self.assertEqual(source_dependencies(graph, raw), [[], [0]])

    def test_alias_operation_requires_explicit_backing_even_if_unused(self):
        for op in ("VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"):
            with self.subTest(op=op):
                capture = deepcopy(self.capture)
                view = deepcopy(self.tensor(capture, 20))
                view.update(id=23, op=op, op_desc=op,
                            sources=[{"slot": 0, "tensor": 23}])
                capture["tensors"].append(view)
                with self.assertRaisesRegex(ValueError, "backing storage"):
                    validate_source(self.graph(capture))

    def test_owning_cast_source_storage_is_validated(self):
        for elements in (0, 1):
            with self.subTest(elements=elements):
                capture = deepcopy(self.capture)
                source = deepcopy(self.tensor(capture, 20))
                source.update(id=22, shape=[elements, 1, 1, 1], elements=elements,
                              strides_bytes=[4] + [4 * elements] * 3,
                              span_bytes=4 * elements, logical_bytes=4 * elements)
                cast = deepcopy(source)
                cast.update(id=21, op="CPY", op_desc="CPY", dtype="f16",
                            strides_bytes=[2] + [2 * elements] * 3,
                            span_bytes=2 * elements, logical_bytes=2 * elements,
                            sources=[{"slot": 0, "tensor": 22}, {"slot": 1, "tensor": 21}])
                capture["tensors"].extend([source, cast])
                capture["scheduled_nodes"] = [21, 30]
                graph = self.graph(capture)
                raw = validate_source(graph)
                self.assertEqual(storage_chain(raw, 21), (21,))
                if elements == 0:
                    self.assertEqual(source_dependencies(graph, raw), [[]])
                backing = deepcopy(cast)
                backing.update(id=23, op="NONE", op_desc="NONE", sources=[])
                capture["tensors"].append(backing)
                for corruption in ("external backing", "self source"):
                    with self.subTest(corruption=corruption):
                        cast["view_source"] = 23 if corruption == "external backing" else -1
                        cast["sources"][0]["tensor"] = 21 if corruption == "self source" else 22
                        with self.assertRaisesRegex(ValueError, "owning CPY|backing storage"):
                            validate_source(self.graph(capture))

    def test_copy_classification_cannot_bypass_alias_checks(self):
        for op, desc in (("CPY", "ADD"), ("ADD", "CPY")):
            with self.subTest(op=op, desc=desc):
                capture = self.copy_into_input()
                self.tensor(capture, 21).update(op=op, op_desc=desc)
                with self.assertRaisesRegex(ValueError, "operation classification"):
                    validate_source(self.graph(capture))

    def test_shape_metadata_and_copy_preserve_resolved_operand_offset(self):
        for op in ("RESHAPE", "PERMUTE", "TRANSPOSE", "CPY", "VIEW"):
            with self.subTest(op=op):
                capture = self.copy_into_input()
                for tensor in capture["tensors"]:
                    tensor.update(dtype="f32", shape=[1, 1, 1, 1], strides_bytes=[4, 4, 4, 4],
                                  elements=1, span_bytes=4, logical_bytes=4)
                self.tensor(capture, 20).update(shape=[2, 1, 1, 1], strides_bytes=[4, 8, 8, 8],
                                                elements=2, span_bytes=8, logical_bytes=8)
                source = deepcopy(self.tensor(capture, 21))
                source.update(id=23, op="VIEW", op_desc="VIEW", sources=[{"slot": 0, "tensor": 20}])
                capture["tensors"].append(source)
                inputs = [22, 23] if op == "CPY" else [20 if op == "VIEW" else 23]
                target = self.tensor(capture, 21)
                target.update(op=op, op_desc=op,
                              sources=[{"slot": slot, "tensor": tid} for slot, tid in enumerate(inputs)])
                capture["scheduled_nodes"] = [23, 21, 30]
                # Both zero-offset descriptors are valid; the root has space for
                # a four-byte shift, so bounds alone cannot detect the contradiction.
                validate_source(self.graph(capture))
                target["view_offset_bytes"] = 4
                if op == "VIEW":
                    validate_source(self.graph(capture))
                else:
                    with self.assertRaisesRegex(ValueError, "backing offset"):
                        validate_source(self.graph(capture))

    def test_metadata_producer_walk_exceeds_python_recursion_limit(self):
        count = sys.getrecursionlimit() + 50
        graph = self.graph(self.metadata_views(count))
        self.assertEqual(graph.metadata_nodes, count)
        self.assertEqual(self.dependencies(graph), {21: [], 30: [21]})

    def test_metadata_source_cycle_is_rejected_with_valid_storage(self):
        capture = self.metadata_views()
        self.tensor(capture, 100)["sources"][0]["tensor"] = 101
        graph = self.graph(capture)
        raw = validate_source(graph)
        with self.assertRaisesRegex(ValueError, "cycle in source descriptors"):
            source_dependencies(graph, raw)

    def test_copy_to_input_is_a_direct_producer_without_changing_analytical_deps(self):
        graph = self.graph(self.copy_into_input())
        self.assertEqual(graph.deps(), [[], []])
        self.assertEqual(self.dependencies(graph), {21: [], 30: [21]})
        self.assertEqual(graph.deps(), [[], []])

    def test_consumer_before_copy_producer_is_rejected(self):
        capture = self.copy_into_input()
        capture["scheduled_nodes"] = [30, 21]
        graph = self.graph(capture)
        with self.assertRaises(ValueError):
            self.dependencies(graph)

    def test_metadata_view_preserves_copy_producer(self):
        capture = self.copy_into_input()
        view = deepcopy(self.tensor(capture, 20))
        view.update(id=23, name="fixture.copy_view", op="VIEW", op_desc="VIEW", view_source=21,
                    sources=[{"slot": 0, "tensor": 21}])
        capture["tensors"].append(view)
        capture["scheduled_nodes"] = [21, 23, 30]
        self.tensor(capture, 30)["sources"][1]["tensor"] = 23
        graph = self.graph(capture)
        self.assertEqual(storage_chain(validate_source(graph), 23), (23, 21, 20))
        self.assertEqual(self.dependencies(graph), {21: [], 30: [21]})

    def test_compute_cannot_be_hidden_by_a_metadata_description(self):
        for op, desc in (("MUL_MAT", "NONE"), ("NONE", "MUL_MAT"), ("VIEW", "NONE"),
                         ("ADD", "VIEW"), ("VIEW", "ADD")):
            with self.subTest(op=op, desc=desc):
                capture = deepcopy(self.capture)
                self.tensor(capture, 30).update(op=op, op_desc=desc)
                with self.assertRaises(ValueError):
                    validate_source(self.graph(capture))

    def test_cached_graph_must_match_retained_capture(self):
        changes = (
            ("raw operation", lambda g: self.tensor(g.capture, 30).update(op="ADD", op_desc="ADD")),
            ("operand slots", lambda g: g.nodes[0].slots.update({1: g.tensors[10]})),
            ("schedule", lambda g: g.order.clear()),
            ("omitted active node", lambda g: g.nodes.clear()),
        )
        for label, change in changes:
            with self.subTest(change=label):
                graph = self.graph()
                change(graph)
                with self.assertRaises(ValueError):
                    validate_source(graph)

    def test_duplicate_tensor_and_schedule_ids_are_rejected(self):
        for field, value in (("tensors", self.capture["tensors"][0]), ("scheduled_nodes", 30)):
            with self.subTest(field=field):
                capture = deepcopy(self.capture)
                capture[field].append(deepcopy(value))
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    validate_source(self.graph(capture))

    def test_scheduled_ids_require_integers_and_exclude_booleans(self):
        for scheduled_id in (30.0, True):
            with self.subTest(scheduled_id=scheduled_id):
                capture = deepcopy(self.capture)
                if scheduled_id is True:
                    self.tensor(capture, 30)["id"] = 1
                capture["scheduled_nodes"] = [scheduled_id]
                with self.assertRaises(ValueError):
                    validate_source(self.graph(capture))

    def test_source_ids_require_integers_and_exclude_booleans(self):
        for source_id in (20.0, True):
            with self.subTest(source_id=source_id):
                capture = deepcopy(self.capture)
                if source_id is True:
                    self.tensor(capture, 20)["id"] = 1
                self.tensor(capture, 30)["sources"][1]["tensor"] = source_id
                with self.assertRaises(ValueError):
                    validate_source(self.graph(capture))

    def test_source_slots_require_integers_and_exclude_booleans(self):
        for slot in (1.0, True):
            with self.subTest(slot=slot):
                capture = deepcopy(self.capture)
                self.tensor(capture, 30)["sources"][1]["slot"] = slot
                with self.assertRaises(ValueError):
                    validate_source(self.graph(capture))

    def test_shape_and_element_count_are_validated_globally(self):
        changes = ({"shape": [7.0, 5, 1, 1]}, {"shape": [7, 5, True, 1]},
                   {"shape": [7, 5, 1]}, {"shape": [-7, 5, 1, 1]},
                   {"elements": 34}, {"elements": 0}, {"elements": 35.0})
        for change in changes:
            with self.subTest(change=change):
                capture = deepcopy(self.capture)
                # A weight is not a scheduled compute node; it still needs auditing.
                self.tensor(capture, 10).update(change)
                with self.assertRaises(ValueError):
                    validate_source(self.graph(capture))

    def test_legitimate_zero_element_compute_is_preserved_as_skipped(self):
        capture = deepcopy(self.capture)
        empty = deepcopy(self.tensor(capture, 30))
        empty.update(id=40, name="fixture.empty", shape=[5, 0, 1, 1], elements=0,
                     span_bytes=0, logical_bytes=0, sources=[])
        capture["tensors"].append(empty)
        capture["scheduled_nodes"].append(40)
        graph = self.graph(capture)
        self.assertEqual(graph.zero_nodes, 1)
        self.assertEqual(set(validate_source(graph)), {10, 20, 30, 40})
        self.assertEqual(self.dependencies(graph), {30: []})
        empty["shape"] = [5, 1, 1, 1]
        with self.assertRaisesRegex(ValueError, "element count"):
            validate_source(self.graph(capture))

    def test_alias_chain_checks_each_immediate_parent_bound(self):
        capture = deepcopy(self.capture)
        view = deepcopy(self.tensor(capture, 20))
        view.update(id=50, name="fixture.view", op="VIEW", op_desc="VIEW", view_source=20,
                    view_offset_bytes=0, shape=[7, 2, 1, 1], elements=14,
                    strides_bytes=[4, 28, 56, 56], span_bytes=56, logical_bytes=56,
                    sources=[{"slot": 0, "tensor": 20}])
        nested = deepcopy(view)
        nested.update(id=51, name="fixture.nested_view", view_source=50, view_offset_bytes=28,
                      shape=[7, 1, 1, 1], elements=7, strides_bytes=[4, 28, 28, 28],
                      span_bytes=28, logical_bytes=28, sources=[{"slot": 0, "tensor": 50}])
        capture["tensors"].extend([view, nested])
        raw = {t["id"]: t for t in capture["tensors"]}
        self.assertEqual(storage_chain(raw, 51), (51, 50, 20))
        # Offset 32 fits the 84-byte owner but exceeds the 56-byte parent.
        for tensor, field, value in ((nested, "view_offset_bytes", -1),
                                     (nested, "view_offset_bytes", 32),
                                     (view, "view_offset_bytes", 1000),
                                     (nested, "view_source", 50.0)):
            with self.subTest(tensor=tensor["id"], field=field, value=value):
                original = tensor[field]
                tensor[field] = value
                with self.assertRaises(ValueError):
                    storage_chain(raw, 51)
                tensor[field] = original

    def test_alias_cycle_is_rejected(self):
        raw = {tid: deepcopy(self.tensor(self.capture, 20)) for tid in (20, 30)}
        for tid, parent in ((20, 30), (30, 20)):
            raw[tid].update(id=tid, view_source=parent)
        with self.assertRaisesRegex(ValueError, "cycle"):
            storage_chain(raw, 30)

    def test_shared_metadata_traversal_is_bounded(self):
        capture = self.copy_into_input()
        previous = 21
        for tid in range(100, 140):
            view = deepcopy(self.tensor(capture, 20))
            view.update(id=tid, name=f"fixture.view_{tid}", op="VIEW", op_desc="VIEW",
                        view_source=previous, sources=[{"slot": 0, "tensor": previous}])
            capture["tensors"].append(view)
            previous = tid
        self.tensor(capture, 30)["sources"][1]["tensor"] = previous
        capture["scheduled_nodes"] = [21, *range(100, 140), 30]
        graph = self.graph(capture)
        raw = BoundedLookupDict(validate_source(graph))
        self.assertEqual(storage_chain(raw, 139), (*range(139, 99, -1), 21, 20))
        self.assertEqual(source_dependencies(graph, raw), [[], [0]])
        self.assertLess(raw.lookups, raw.limit)

    def test_ordinary_storage_owner_gets_raw_war_and_waw_ordering(self):
        capture = self.copy_into_input()
        for tid in (31, 32):
            reader = deepcopy(self.tensor(capture, 20))
            reader.update(id=tid, name=f"fixture.reader_{tid}", op="UNARY", op_desc="SILU",
                          sources=[{"slot": 0, "tensor": 20}])
            capture["tensors"].append(reader)
        writer = deepcopy(self.tensor(capture, 21))
        writer.update(id=24, name="fixture.second_copy")
        capture["tensors"].append(writer)
        capture["scheduled_nodes"] = [31, 21, 32, 24]
        graph = self.graph(capture)
        self.assertEqual(graph.deps(), [[], [], [], []])
        deps = self.dependencies(graph)
        self.assertIn(31, deps[21])  # WAR: copy waits for the first read.
        self.assertIn(21, deps[32])  # RAW: the next read sees the first copy.
        self.assertIn(21, deps[24])  # WAW: writes retain their captured order.
        self.assertIn(32, deps[24])  # WAR: the second copy waits for the next read.
        self.assertEqual(graph.deps(), [[], [], [], []])


if __name__ == "__main__":
    unittest.main(verbosity=2)
