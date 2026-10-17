"""Compatibility imports for the representation now maintained by core/cdfg/."""
from cdfg import (
    ACTIVATION_OPS, ELEMENTWISE_OPS, KNOWN_OPS, MEMORY_OPS, METADATA_OPS,
    Graph, Node, Tensor, capture_path, has_capture, layer_of,
)
from .workload import Workload, latest_run, load_workload
