"""Source-faithful symbolic lowering; target commands and execution are later stages."""

from .lower import compile_graph, lower_matmul
from .plan import CompilationPlan, Diagnostic, LoweringError, MatmulPlan, TensorRef

__all__ = ["compile_graph", "lower_matmul", "CompilationPlan", "Diagnostic",
           "LoweringError", "MatmulPlan", "TensorRef"]
