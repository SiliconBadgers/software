"""First lowering: an unbatched contiguous GGML MUL_MAT to logical C = A * B.

Use the canonical Graph and its preserved capture. Byte strides and quantized
formats stay as captured; this stage neither repacks nor changes arithmetic.
"""
import math

from cdfg import METADATA_OPS
from ..provenance import decoded_capture_sha256
from .plan import CompilationPlan, Diagnostic, LoweringError, MatmulPlan, TensorRef


def _raw_tensors(graph):
    tensors = graph.capture["tensors"]
    raw = {t["id"]: t for t in tensors}
    if len(raw) != len(tensors):
        raise ValueError("duplicate tensor IDs in capture")
    for tensor in tensors:
        shape = tensor["shape"]
        if (type(tensor["id"]) is not int or tensor["id"] < 0
                or len(shape) != 4 or any(type(v) is not int or v < 0 for v in shape)
                or type(tensor["elements"]) is not int or math.prod(shape) != tensor["elements"]):
            raise ValueError(f"tensor {tensor['id']}: invalid shape or captured element count")
    if len(set(graph.order)) != len(graph.order):
        raise ValueError("duplicate scheduled tensor IDs in capture")
    if graph.phase not in ("prefill", "decode"):
        raise ValueError("phase must be prefill or decode")
    captured_phase = graph.meta.get("phase", graph.phase)
    if captured_phase != graph.phase:
        raise ValueError(f"requested {graph.phase} but capture metadata says {captured_phase}")
    return raw


def _binding(raw, tensor_id):
    t = raw[tensor_id]
    if t.get("contiguous") is not True:
        raise ValueError(f"tensor {tensor_id}: non-contiguous or missing layout is not lowered yet")
    owner, offset, seen = tensor_id, 0, set()
    while True:
        if owner in seen:
            raise ValueError(f"tensor {tensor_id}: cycle in storage aliases")
        seen.add(owner)
        current = raw[owner]
        step = current["view_offset_bytes"]
        if type(step) is not int or step < 0:
            raise ValueError(f"tensor {owner}: invalid view offset")
        if current["view_source"] < 0:
            if step:
                raise ValueError(f"tensor {owner}: storage owner has a nonzero view offset")
            break
        parent = raw[current["view_source"]]
        if step + current["span_bytes"] > parent["span_bytes"]:
            raise ValueError(f"tensor {owner}: view exceeds its immediate parent's span")
        if current["dtype"] != parent["dtype"]:
            raise ValueError(f"tensor {owner}: view changes its parent's source format")
        offset += step
        owner = current["view_source"]
    ref = TensorRef(tensor_id, t["name"], owner, offset, raw[owner]["span_bytes"],
                    tuple(t["shape"]), t["dtype"], tuple(t["strides_bytes"]),
                    t["span_bytes"], t["logical_bytes"])
    ref.validate()
    if type(t["elements"]) is not int or math.prod(ref.shape) != t["elements"]:
        raise ValueError(f"tensor {tensor_id}: shape and captured element count differ")
    return ref


def _lower_node(graph, node, raw, dependencies):
    try:
        if node.op != "MUL_MAT" or node.tensor.op != "MUL_MAT":
            raise ValueError("operation lowering is not implemented")
        if len(node.tensor.sources) != 2 or set(node.slots) != {0, 1}:
            raise ValueError("MUL_MAT requires exactly operand slots 0 and 1")
        a, b, c = (_binding(raw, tid) for tid in
                   (node.slots[0].id, node.slots[1].id, node.tensor.id))
        plan = MatmulPlan(node.tensor.id, node.spos, a, b, c,
                          c.shape[0], c.shape[1], a.shape[0],
                          tuple(raw[node.tensor.id]["op_params_i32"]),
                          tuple(graph.nodes[idx].tensor.id for idx in dependencies[node.idx]))
        plan.validate()
        return plan
    except (KeyError, TypeError, ValueError) as error:
        raise LoweringError(node.tensor.id, node.op, str(error)) from error


def _source_dependencies(graph, raw):
    dependencies = graph.deps()
    active = {node.tensor.id for node in graph.nodes}
    # A selected region can use unselected producers from the source graph, but
    # an omitted source computation must not silently become an external input.
    def check_source(tid, seen):
        if tid in seen:
            raise ValueError(f"tensor {tid}: cycle in source descriptors")
        tensor = raw[tid]
        if tensor["op_desc"] not in METADATA_OPS and tensor["elements"]:
            if tid not in active:
                raise ValueError(f"tensor {tid}: source computation is missing from captured order")
            return
        for source in tensor["sources"]:
            check_source(source["tensor"], seen | {tid})
        if tensor["view_source"] >= 0:
            check_source(tensor["view_source"], seen | {tid})

    for node in graph.nodes:
        if any(type(idx) is not int or idx < 0 or idx >= node.idx for idx in dependencies[node.idx]):
            raise ValueError(f"tensor {node.tensor.id}: captured order is not topological")
        for source in node.srcs:
            check_source(source.id, set())
    return dependencies


def lower_matmul(graph, tensor_id):
    """Lower one active result tensor ID (not its position in the node list)."""
    try:
        raw = _raw_tensors(graph)
    except (KeyError, TypeError, ValueError) as error:
        raise LoweringError(tensor_id, "source", str(error)) from error
    node = next((node for node in graph.nodes if node.tensor.id == tensor_id), None)
    if node is None:
        raise LoweringError(tensor_id, "unknown", "not an active captured operation")
    return _lower_node(graph, node, raw, _source_dependencies(graph, raw))


def compile_graph(graph, tensor_id=None):
    """Account for every active node, or one selected node, without implicit fallback.

    Metadata/zero-size nodes remain in the shared CDFG/capture. This report counts
    active operations only; diagnostics are unsupported work, not executable host
    calls. Source dependencies are retained, not yet lowered to a queue schedule.
    """
    raw = _raw_tensors(graph)
    nodes = graph.nodes if tensor_id is None else [n for n in graph.nodes if n.tensor.id == tensor_id]
    if not nodes:
        raise ValueError("no active captured operations match the selection")
    plans, diagnostics = [], []
    dependencies = _source_dependencies(graph, raw)
    for node in nodes:
        try:
            plans.append(_lower_node(graph, node, raw, dependencies))
        except LoweringError as error:
            diagnostics.append(Diagnostic(error.tensor_id, error.op, error.reason))
    return CompilationPlan(decoded_capture_sha256(graph.capture), graph.phase,
                           tuple(plans), tuple(diagnostics))
