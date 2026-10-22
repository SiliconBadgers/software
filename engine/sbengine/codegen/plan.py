"""Small compiler records referencing shared CDFG identities, without a second graph.

Captured formats are provenance, not target capabilities. A symbolic plan has no
physical addresses, tile schedule, firmware IDs, binary encoding or cost model.
"""
from dataclasses import asdict, dataclass
import json
import math

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

    def validate(self):
        if not all(_integer(v) for v in (self.tensor_id, self.storage_id, self.offset_bytes)):
            raise ValueError("tensor/storage IDs and offsets must be nonnegative integers")
        if len(self.shape) != 4 or not all(_integer(v, 1) for v in self.shape):
            raise ValueError("expected four positive GGML dimensions")
        if self.shape[2:] != (1, 1):
            raise ValueError("batched or broadcast matmul is not lowered yet")
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

    def validate(self):
        for tensor in (self.a, self.b, self.c):
            tensor.validate()
        if not all(_integer(v, 1) for v in (self.m, self.n, self.k)):
            raise ValueError("M/N/K must be positive integers")
        if (self.a.shape[:2], self.b.shape[:2], self.c.shape[:2]) != (
                (self.k, self.m), (self.k, self.n), (self.m, self.n)):
            raise ValueError("incompatible MUL_MAT dimensions: expected A[K,M], B[K,N], C[M,N]")
        if self.tensor_id != self.c.tensor_id:
            raise ValueError("result identity differs from the captured output")
        if self.c.storage_id in (self.a.storage_id, self.b.storage_id):
            raise ValueError("matmul output aliases an input storage owner")
        if not _integer(self.scheduled_position):
            raise ValueError("scheduled position must be a nonnegative integer")
        if any(type(v) is not int or not -(2**31) <= v < 2**31 for v in self.op_params_i32):
            raise ValueError("operation parameters must be captured signed int32 values")
        if any(not _integer(v) or v == self.tensor_id for v in self.dependencies):
            raise ValueError("invalid captured dependency tensor ID")


@dataclass(frozen=True)
class Diagnostic:
    tensor_id: int
    op: str
    reason: str


@dataclass(frozen=True)
class CompilationPlan:
    capture_sha256: str
    phase: str
    plans: tuple
    diagnostics: tuple

    def to_dict(self):
        for plan in self.plans:
            plan.validate()
        return {
            "schema": SCHEMA,
            "execution_status": "symbolic_only",
            "source": {"decoded_capture_sha256": self.capture_sha256, "phase": self.phase},
            "counts": {"total": len(self.plans) + len(self.diagnostics),
                       "planned": len(self.plans), "unsupported": len(self.diagnostics)},
            "plans": [asdict(plan) for plan in self.plans],
            "diagnostics": [asdict(diagnostic) for diagnostic in self.diagnostics],
        }

    def to_json(self):
        return json.dumps(self.to_dict(), sort_keys=True, indent=2, allow_nan=False) + "\n"
