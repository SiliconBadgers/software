"""Checked source facts for compilation, independent of target formats and costs.

The analytical Graph/deps contract stays unchanged. Compilation additionally
checks capture coherence and derives conservative dependencies for every storage
owner, including copies into ordinary input/activation buffers.
"""
import math

from .graph import METADATA_OPS, Tensor


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def storage_chain(tensors, tensor_id):
    """Return tensor-to-owner IDs, checking each relative view span and offset.

    These are captured byte extents. Consumers must validate source-format byte
    accounting before using them as allocated capacities.
    """
    chain, seen = [], set()
    while True:
        if tensor_id in seen:
            raise ValueError(f"tensor {tensor_id}: cycle in storage aliases")
        seen.add(tensor_id)
        chain.append(tensor_id)
        current = tensors[tensor_id]
        parent_id, offset = current["view_source"], current["view_offset_bytes"]
        if not _integer(parent_id, -1) or not _integer(offset):
            raise ValueError(f"tensor {tensor_id}: invalid storage identity or view offset")
        if not _integer(current["span_bytes"]):
            raise ValueError(f"tensor {tensor_id}: invalid captured span")
        if parent_id == -1:
            if offset:
                raise ValueError(f"tensor {tensor_id}: storage owner has a nonzero view offset")
            return tuple(chain)
        parent = tensors[parent_id]
        if not _integer(parent["span_bytes"]) or offset + current["span_bytes"] > parent["span_bytes"]:
            raise ValueError(f"tensor {tensor_id}: view exceeds its immediate parent's span")
        if current["dtype"] != parent["dtype"]:
            raise ValueError(f"tensor {tensor_id}: view changes its parent's source format")
        tensor_id = parent_id


def validate_source(graph):
    """Validate retained capture and cached Graph coherence; return captured tensors.

    Do not silently repair or reload a mutated Graph: provenance must describe
    the same source interpretation used by lowering.
    """
    if not isinstance(graph.capture["metadata"], dict):
        raise ValueError("capture metadata must be an object")
    tensors = graph.capture["tensors"]
    raw = {t["id"]: t for t in tensors}
    if len(raw) != len(tensors):
        raise ValueError("duplicate tensor IDs in capture")
    for t in tensors:
        if (not _integer(t["id"]) or len(t["shape"]) != 4
                or any(not _integer(v) for v in t["shape"])
                or not _integer(t["elements"]) or math.prod(t["shape"]) != t["elements"]):
            raise ValueError(f"tensor {t['id']}: invalid shape or captured element count")
        if any(not isinstance(t[key], str) or not t[key] for key in ("op", "op_desc", "dtype")):
            raise ValueError(f"tensor {t['id']}: operation and format names are required")
        if (t["op"] in METADATA_OPS or t["op_desc"] in METADATA_OPS) and t["op"] != t["op_desc"]:
            raise ValueError(f"tensor {t['id']}: inconsistent metadata operation classification")
        if not _integer(t["span_bytes"]) or not _integer(t["logical_bytes"]):
            raise ValueError(f"tensor {t['id']}: invalid captured byte extents")
        slots = []
        for source in t["sources"]:
            if not _integer(source["slot"]) or not _integer(source["tensor"]) or source["tensor"] not in raw:
                raise ValueError(f"tensor {t['id']}: invalid source identity or operand slot")
            slots.append(source["slot"])
        if len(set(slots)) != len(slots):
            raise ValueError(f"tensor {t['id']}: duplicate operand slots")
    order = graph.capture["scheduled_nodes"]
    if any(not _integer(tid) or tid not in raw for tid in order):
        raise ValueError("scheduled tensor IDs must be captured nonnegative integers")
    if len(set(order)) != len(order):
        raise ValueError("duplicate scheduled tensor IDs in capture")
    if graph.phase not in ("prefill", "decode"):
        raise ValueError("phase must be prefill or decode")
    if graph.capture["metadata"].get("phase", graph.phase) != graph.phase:
        raise ValueError(f"requested {graph.phase} but capture metadata has a different phase")
    expected = {tid: Tensor.from_capture(t) for tid, t in raw.items()}
    if (graph.tensors != expected or graph.meta != graph.capture["metadata"]
            or graph.order != order or any(not _integer(tid) for tid in graph.order)):
        raise ValueError("cached Graph and retained capture differ; reconstruct the source Graph")
    active = [(pos, tid) for pos, tid in enumerate(order)
              if raw[tid]["op_desc"] not in METADATA_OPS and raw[tid]["elements"]]
    if len(graph.nodes) != len(active):
        raise ValueError("cached active nodes differ from the retained capture")
    roots = {tid: storage_chain(raw, tid)[-1] for tid in raw}
    for idx, (node, (pos, tid)) in enumerate(zip(graph.nodes, active)):
        slots = {slot: expected[sid] for slot, sid in expected[tid].sources}
        srcs = tuple(dict.fromkeys(slots.values()))
        if (not _integer(node.idx) or not _integer(node.spos) or node.idx != idx or node.spos != pos
                or node.tensor != expected[tid] or node.slots != slots or node.srcs != srcs
                or node.dst_root != roots[tid] or node.src_roots != tuple(roots[t.id] for t in srcs)):
            raise ValueError(f"tensor {tid}: cached node differs from the retained capture")
    return raw


def source_dependencies(graph, tensors):
    """Conservative dataflow plus all-owner RAW/WAR/WAW in source order.

    Accept the mapping returned by validate_source(). Returned entries reference
    Graph node indexes. This is source ordering, not a hardware queue schedule.
    """
    indexes = {node.tensor.id: node.idx for node in graph.nodes}
    memo, visiting = {}, set()

    def producers(tid):
        if tid in memo:
            return memo[tid]
        if tid in visiting:
            raise ValueError(f"tensor {tid}: cycle in source descriptors")
        t = tensors[tid]
        if t["op_desc"] not in METADATA_OPS and t["elements"]:
            if tid not in indexes:
                raise ValueError(f"tensor {tid}: source computation is missing from captured order")
            result = {indexes[tid]}
        else:
            visiting.add(tid)
            result = set()
            parents = {s["tensor"] for s in t["sources"]}
            if t["view_source"] >= 0:
                parents.add(t["view_source"])
            for parent in parents:
                result.update(producers(parent))
            visiting.remove(tid)
        memo[tid] = frozenset(result)
        return memo[tid]

    dependencies, last_write, readers = [], {}, {}
    for node in graph.nodes:
        deps, read_roots = set(), set()
        for source, root in zip(node.srcs, node.src_roots):
            deps.update(producers(source.id))
            read_roots.add(root)
            if root in last_write:
                deps.add(last_write[root])
        if any(idx >= node.idx for idx in deps):
            raise ValueError(f"tensor {node.tensor.id}: captured order is not topological")
        for root in read_roots:
            readers.setdefault(root, set()).add(node.idx)
        written = node.dst_root
        deps.update(readers.get(written, set()) - {node.idx})
        if written in last_write:
            deps.add(last_write[written])
        readers[written] = set()
        last_write[written] = node.idx
        dependencies.append(sorted(deps))
    return dependencies
