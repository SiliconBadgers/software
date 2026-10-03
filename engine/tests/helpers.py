"""Shared test helpers: a tiny synthetic-graph builder and access to the real captured run (tests skip without it)."""
import json
import math
import sys
import tempfile
from pathlib import Path

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from sbengine import formats  # noqa: E402
from sbengine.graph import Graph, Workload, latest_run  # noqa: E402

RESULTS = ENGINE.parent / "results"


def real_run():
    try:
        return latest_run(RESULTS)
    except SystemExit:
        return None


class Builder:
    """Assemble a GGML-style captured graph in memory, then write it as prefill.json/decode.json."""

    def __init__(self):
        self.tensors, self.order = [], []

    def add(self, name, shape, dtype="f32", op="NONE", desc=None, src=(), view=-1, weight=False):
        shape4 = list(shape) + [1] * (4 - len(shape))
        elements = math.prod(shape4)
        nbytes = formats.storage_bytes(dtype, elements)
        tid = len(self.tensors)
        self.tensors.append({
            "id": tid, "name": name, "op": op, "op_desc": desc or op, "dtype": dtype, "shape": shape4,
            "strides_bytes": [], "elements": elements, "span_bytes": nbytes, "logical_bytes": nbytes,
            "contiguous": True, "flags": 0, "weight_buffer": weight, "buffer": "CPU",
            "sources": [{"slot": i, "tensor": s} for i, s in enumerate(src)], "view_source": view,
            "view_offset_bytes": 0, "op_params_i32": []})
        if op != "NONE":
            self.order.append(tid)
        return tid

    def write(self, directory, prompt_tokens=8, past_tokens=0, input_tokens=8, context_capacity=64):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        meta = {"prompt_tokens": prompt_tokens, "past_tokens": past_tokens, "input_tokens": input_tokens,
                "context_capacity": context_capacity, "flash_attention": False}
        doc = {"schema_version": 1, "metadata": meta, "scheduled_nodes": self.order, "tensors": self.tensors}
        for phase in ("prefill", "decode"):
            (directory / f"{phase}.json").write_text(json.dumps(doc), encoding="utf-8")
        return directory

    def graph(self, **meta):
        tmp = Path(tempfile.mkdtemp())
        self.write(tmp, **meta)
        return Graph(tmp / "prefill.json", "prefill")

    def workload(self, **meta):
        tmp = Path(tempfile.mkdtemp())
        self.write(tmp, **meta)
        return Workload("synthetic", tmp, Graph(tmp / "prefill.json", "prefill"), Graph(tmp / "decode.json", "decode"))
