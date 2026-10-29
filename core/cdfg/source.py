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
    classified_ops = METADATA_OPS | {"CPY"}
    for t in tensors:
        if (not _integer(t["id"]) or len(t["shape"]) != 4
                or any(not _integer(v) for v in t["shape"])
                or not _integer(t["elements"]) or math.prod(t["shape"]) != t["elements"]):
            raise ValueError(f"tensor {t['id']}: invalid shape or captured element count")
        if any(not isinstance(t[key], str) or not t[key] for key in ("op", "op_desc", "dtype")):
            raise ValueError(f"tensor {t['id']}: operation and format names are required")
        if (t["op"] in classified_ops or t["op_desc"] in classified_ops) and t["op"] != t["op_desc"]:
            raise ValueError(f"tensor {t['id']}: inconsistent metadata/copy operation classification")
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
    roots, offsets = {}, {}
    for tid in raw:
        chain = storage_chain(raw, tid)
        roots[tid] = chain[-1]
        offsets[tid] = sum(raw[parent]["view_offset_bytes"] for parent in chain)
    for tid, tensor in raw.items():
        if tensor["op_desc"] in {"VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE", "CPY"}:
            slot = 1 if tensor["op_desc"] == "CPY" else 0
            expected_slots = {0, 1} if slot == 1 else {0}
            slots = {s["slot"]: s["tensor"] for s in tensor["sources"]}
            if set(slots) != expected_slots:
                raise ValueError(f"tensor {tid}: {tensor['op_desc']} requires operand slots {sorted(expected_slots)}")
            if tensor["op_desc"] == "CPY" and slots[1] == tid:
                # ggml_cast owns its result; slot 1 names that destination.
                if tensor["view_source"] != -1 or slots[0] == tid:
                    raise ValueError(f"tensor {tid}: invalid owning CPY backing storage or input")
                continue
            if tensor["view_source"] < 0 or roots[tid] != roots[slots[slot]]:
                raise ValueError(f"tensor {tid}: {tensor['op_desc']} backing storage differs from operand {slot}")
            if tensor["op_desc"] != "VIEW" and offsets[tid] != offsets[slots[slot]]:
                raise ValueError(f"tensor {tid}: {tensor['op_desc']} backing offset differs from operand {slot}")
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
        stack = [(tid, None)]
        while stack:
            current, parents = stack.pop()
            if current in memo:
                continue
            if parents is not None:
                memo[current] = frozenset(dep for parent in parents for dep in memo[parent])
                visiting.remove(current)
                continue
            if current in visiting:
                raise ValueError(f"tensor {current}: cycle in source descriptors")
            tensor = tensors[current]
            if tensor["op_desc"] not in METADATA_OPS and tensor["elements"]:
                if current not in indexes:
                    raise ValueError(f"tensor {current}: source computation is missing from captured order")
                memo[current] = frozenset((indexes[current],))
            else:
                visiting.add(current)
                parents = {s["tensor"] for s in tensor["sources"]}
                if tensor["view_source"] >= 0:
                    parents.add(tensor["view_source"])
                # Revisit after every parent has a memoized producer set.
                stack.append((current, parents))
                stack.extend((parent, None) for parent in parents if parent not in memo)
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
