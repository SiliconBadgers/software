"""Lower checked source CDFGs to symbolic operations, preserving captured formats."""
from cdfg.source import source_dependencies, storage_chain, validate_source

from ..provenance import decoded_capture_sha256
from .plan import CompilationPlan, Diagnostic, LoweringError, MatmulPlan, TensorRef


def _binding(raw, tensor_id):
    chain = storage_chain(raw, tensor_id)
    owner, offset = chain[-1], 0
    # Validate every ancestor's actual format extent before trusting its span as
    # capacity. Ancestors can have more than two dimensions even for a 2D view.
    for tid in reversed(chain):
        t = raw[tid]
        if t.get("contiguous") is not True:
            raise ValueError(f"tensor {tid}: non-contiguous or missing layout is not lowered yet")
        offset += t["view_offset_bytes"]
        ref = TensorRef(tid, t["name"], owner, offset, raw[owner]["span_bytes"],
                        t["shape"], t["dtype"], t["strides_bytes"],
                        t["span_bytes"], t["logical_bytes"])
        ref.validate()
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
                          raw[node.tensor.id]["op_params_i32"],
                          (graph.nodes[idx].tensor.id for idx in dependencies[node.idx]))
        plan.validate()
        return plan
    except (KeyError, TypeError, ValueError) as error:
        raise LoweringError(node.tensor.id, node.op, str(error)) from error


def _selector(tensor_id):
    if type(tensor_id) is not int or tensor_id < 0:
        raise ValueError("selection must be a nonnegative integer tensor ID")


def lower_matmul(graph, tensor_id):
    """Lower one active result tensor ID (not its position in the node list)."""
    try:
        _selector(tensor_id)
        raw = validate_source(graph)
        dependencies = source_dependencies(graph, raw)
    except (KeyError, TypeError, ValueError) as error:
        raise LoweringError(tensor_id, "source", str(error)) from error
    node = next((node for node in graph.nodes if node.tensor.id == tensor_id), None)
    if node is None:
        raise LoweringError(tensor_id, "unknown", "not an active captured operation")
    return _lower_node(graph, node, raw, dependencies)


def compile_graph(graph, tensor_id=None):
    """Account for every active node, or one selected node, without implicit fallback.

    Metadata/zero-size nodes remain in the shared CDFG/capture. Diagnostics are
    unsupported work, not executable host calls. Dependencies preserve source
    ordering and storage hazards; they are not a hardware queue schedule.
    """
    if tensor_id is not None:
        _selector(tensor_id)
    raw = validate_source(graph)
    nodes = graph.nodes if tensor_id is None else [n for n in graph.nodes if n.tensor.id == tensor_id]
    if not nodes:
        raise ValueError("no active captured operations match the selection")
    plans, diagnostics = [], []
    dependencies = source_dependencies(graph, raw)
    for node in nodes:
        try:
            plans.append(_lower_node(graph, node, raw, dependencies))
        except LoweringError as error:
            diagnostics.append(Diagnostic(error.tensor_id, error.op, error.reason))
    return CompilationPlan(decoded_capture_sha256(graph.capture), graph.phase,
                           plans, diagnostics,
                           (node.tensor.id for node in nodes),
                           "full_graph" if tensor_id is None else "selected_operation")
