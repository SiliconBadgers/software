"""Small compiler records referencing shared CDFG identities, without a second graph.

Captured formats are provenance, not target capabilities. A symbolic plan has no
physical addresses, tile schedule, firmware IDs, binary encoding or cost model.
"""
from dataclasses import asdict, dataclass, field
import json
import math
import re

from ..formats import BLOCK, storage_bytes

SCHEMA = "siliconbadgers.compiler-plan.v0"


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


class LoweringError(ValueError):
    def __init__(self, tensor_id, op, reason):
        self.tensor_id, self.op, self.reason = tensor_id, op, reason
        super().__init__(f"tensor {tensor_id} ({op}): {reason}")


@dataclass(frozen=True)
class TensorRef:
    tensor_id: int
    name: str
    storage_id: int
    offset_bytes: int
    storage_span_bytes: int
    shape: tuple
    dtype: str
    strides_bytes: tuple
    span_bytes: int
    logical_bytes: int

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(self.shape))
        object.__setattr__(self, "strides_bytes", tuple(self.strides_bytes))

    def validate(self):
        if not all(_integer(v) for v in (self.tensor_id, self.storage_id, self.offset_bytes)):
            raise ValueError("tensor/storage IDs and offsets must be nonnegative integers")
        if not isinstance(self.name, str):
            raise ValueError("captured tensor name must be a string")
        if len(self.shape) != 4 or not all(_integer(v, 1) for v in self.shape):
            raise ValueError("expected four positive GGML dimensions")
        if len(self.strides_bytes) != 4 or not all(_integer(v) for v in self.strides_bytes):
            raise ValueError("four nonnegative captured byte strides are required")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise ValueError("captured dtype is required")
        if self.dtype not in BLOCK:
            raise ValueError(f"source storage format {self.dtype!r} is not described in formats.BLOCK")
        block_elements, block_bytes = BLOCK[self.dtype]
        if self.shape[0] % block_elements:
            raise ValueError("innermost dimension must contain whole source-format blocks")
        expected = [block_bytes, self.shape[0] // block_elements * block_bytes]
        for size in self.shape[1:3]:
            expected.append(expected[-1] * size)
        if self.strides_bytes != tuple(expected):
            raise ValueError("captured strides do not describe the supported contiguous layout")
        if not all(_integer(v, 1) for v in (self.span_bytes, self.logical_bytes, self.storage_span_bytes)):
            raise ValueError("positive captured byte extents are required")
        expected_bytes = storage_bytes(self.dtype, math.prod(self.shape))
        if self.logical_bytes != expected_bytes or self.span_bytes != expected_bytes:
            raise ValueError("captured byte extents differ from the source format and shape")
        if self.offset_bytes % block_bytes:
            raise ValueError("view offset does not start at a source-format block")
        if self.offset_bytes + self.span_bytes > self.storage_span_bytes:
            raise ValueError("view exceeds its storage owner's span")


@dataclass(frozen=True)
class MatmulPlan:
    tensor_id: int
    scheduled_position: int
    a: TensorRef
    b: TensorRef
    c: TensorRef
    m: int
    n: int
    k: int
    op_params_i32: tuple
    dependencies: tuple
    operation: str = field(default="MUL_MAT", init=False)

    def __post_init__(self):
        object.__setattr__(self, "op_params_i32", tuple(self.op_params_i32))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))

    def validate(self):
        if not _integer(self.tensor_id) or not _integer(self.scheduled_position):
            raise ValueError("result identity and scheduled position must be nonnegative integers")
        for tensor in (self.a, self.b, self.c):
            if not isinstance(tensor, TensorRef):
                raise ValueError("matmul operands must be captured TensorRef records")
            tensor.validate()
            if tensor.shape[2:] != (1, 1):
                raise ValueError("batched or broadcast matmul is not lowered yet")
        if not all(_integer(v, 1) for v in (self.m, self.n, self.k)):
            raise ValueError("M/N/K must be positive integers")
        if (self.a.shape[:2], self.b.shape[:2], self.c.shape[:2]) != (
                (self.k, self.m), (self.k, self.n), (self.m, self.n)):
            raise ValueError("incompatible MUL_MAT dimensions: expected A[K,M], B[K,N], C[M,N]")
        if self.tensor_id != self.c.tensor_id:
            raise ValueError("result identity differs from the captured output")
        if self.c.storage_id in (self.a.storage_id, self.b.storage_id):
            raise ValueError("matmul output aliases an input storage owner")
        if any(type(v) is not int or not -(2**31) <= v < 2**31 for v in self.op_params_i32):
            raise ValueError("operation parameters must be captured signed int32 values")
        if (len(set(self.dependencies)) != len(self.dependencies)
                or any(not _integer(v) or v == self.tensor_id for v in self.dependencies)):
            raise ValueError("invalid captured dependency tensor ID")


@dataclass(frozen=True)
class Diagnostic:
    tensor_id: int
    op: str
    reason: str

    def validate(self):
        if not _integer(self.tensor_id):
            raise ValueError("diagnostic tensor ID must be a nonnegative integer")
        if any(not isinstance(v, str) or not v.strip() for v in (self.op, self.reason)):
            raise ValueError("diagnostic operation and reason are required")


@dataclass(frozen=True)
class CompilationPlan:
    capture_sha256: str
    phase: str
    plans: tuple
    diagnostics: tuple
    source_order: tuple
    scope: str

    def __post_init__(self):
        for name in ("plans", "diagnostics", "source_order"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    def validate(self):
        if not isinstance(self.capture_sha256, str) or not re.fullmatch("[0-9a-f]{64}", self.capture_sha256):
            raise ValueError("decoded capture identity must be a SHA-256 hex digest")
        if self.phase not in ("prefill", "decode"):
            raise ValueError("phase must be prefill or decode")
        if self.scope not in ("full_graph", "selected_operation"):
            raise ValueError("invalid compilation scope")
        if (not self.source_order or any(not _integer(v) for v in self.source_order)
                or len(set(self.source_order)) != len(self.source_order)):
            raise ValueError("source operation IDs must be unique nonnegative integers")
        if self.scope == "selected_operation" and len(self.source_order) != 1:
            raise ValueError("selected-operation reports must cover exactly one operation")
        for records, cls in ((self.plans, MatmulPlan), (self.diagnostics, Diagnostic)):
            for record in records:
                if not isinstance(record, cls):
                    raise ValueError("invalid compiler report record")
                record.validate()
        covered = [p.tensor_id for p in self.plans] + [d.tensor_id for d in self.diagnostics]
        if len(set(covered)) != len(covered) or set(covered) != set(self.source_order):
            raise ValueError("plans and diagnostics must partition the selected source operations")
        rank = {tid: idx for idx, tid in enumerate(self.source_order)}
        if any(rank[a.tensor_id] >= rank[b.tensor_id] or a.scheduled_position >= b.scheduled_position
               for a, b in zip(self.plans, self.plans[1:])):
            raise ValueError("plans must preserve captured operation order")
        for plan in self.plans:
            for dependency in plan.dependencies:
                if dependency in rank:
                    if rank[dependency] >= rank[plan.tensor_id]:
                        raise ValueError("dependency must precede its consumer")
                elif self.scope == "full_graph":
                    raise ValueError("full-graph dependency is missing from source operations")

    def to_dict(self):
        self.validate()
        coverage = {}
        for records, field_name, op_name in ((self.plans, "planned", "operation"),
                                             (self.diagnostics, "unsupported", "op")):
            for record in records:
                counts = coverage.setdefault(getattr(record, op_name), {"total": 0, "planned": 0, "unsupported": 0})
                counts["total"] += 1
                counts[field_name] += 1
        external = sorted({dep for plan in self.plans for dep in plan.dependencies} - set(self.source_order))
        return {
            "schema": SCHEMA,
            "execution_status": "symbolic_only",
            "scope": self.scope,
            "source": {"decoded_capture_sha256": self.capture_sha256, "phase": self.phase,
                       "operation_tensor_ids": list(self.source_order),
                       "external_dependency_tensor_ids": external},
            "counts": {"total": len(self.plans) + len(self.diagnostics),
                       "planned": len(self.plans), "unsupported": len(self.diagnostics)},
            "coverage_by_op": coverage,
            "plans": [asdict(plan) for plan in self.plans],
            "diagnostics": [asdict(diagnostic) for diagnostic in self.diagnostics],
        }

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
