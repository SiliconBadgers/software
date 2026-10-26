"""Published semantic DAGs, kept distinct from executed GGML captures.

No lowering, cost equations, schedule, or numerical execution is inferred here.
The source's symbolic shapes and composite arithmetic contracts remain intact.
"""
from collections import Counter
import math


class SemanticGraph:
    def __init__(self, raw):
        if not isinstance(raw, dict) or raw.get("schema") != "qwenquant.graph.v1":
            raise ValueError("unsupported semantic graph schema")
        if raw.get("kind") != "semantic_operator_dag":
            raise ValueError("expected a semantic operator DAG, not an execution capture")
        for name in ("architecture", "model", "revision", "source_weights_sha256"):
            if not isinstance(raw.get(name), str) or not raw[name]:
                raise ValueError(f"missing semantic graph {name}")
        for name in ("inputs", "state_inputs", "state_outputs", "tensor_manifest", "tensor_aliases", "contracts",
                     "config", "source_sha256"):
            if (not isinstance(raw.get(name), dict)
                    or any(not isinstance(key, str) or not key for key in raw[name])):
                raise ValueError(f"invalid semantic graph {name}")
        symbols = raw.get("symbols")
        if not isinstance(symbols, dict) or type(symbols.get("B")) is not int or symbols["B"] != 1:
            raise ValueError("only the declared batch-one semantic contract is supported")
        self.raw = raw
        self.tensors = raw["tensor_manifest"]
        for name, tensor in self.tensors.items():
            if not isinstance(tensor, dict):
                raise ValueError(f"invalid tensor descriptor: {name}")
            shape = tensor.get("shape")
            if not isinstance(shape, list) or not shape or any(type(d) is not int or d <= 0 for d in shape):
                raise ValueError(f"invalid static weight shape: {name}")
            dtype = tensor.get("source_dtype")
            layout = {"F16": (2, 1), "U8": (1, 24)}.get(dtype) if isinstance(dtype, str) else None
            if (layout is None or type(tensor.get("ggml_type")) is not int or type(tensor.get("bytes")) is not int
                    or tensor["ggml_type"] != layout[1] or tensor["bytes"] != math.prod(shape) * layout[0]):
                raise ValueError(f"unsupported/inconsistent tensor storage: {name}")
        aliases = raw["tensor_aliases"]
        for name in aliases:
            seen, target = set(), name
            while target in aliases:
                if target in seen or not isinstance(aliases[target], str):
                    raise ValueError(f"invalid/cyclic tensor alias: {name}")
                seen.add(target)
                target = aliases[target]
            if name in self.tensors or target not in self.tensors:
                raise ValueError(f"unresolved tensor alias: {name}")
        nodes = raw.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ValueError("semantic graph has no nodes")
        available = {name: None for name in (*raw["inputs"], *raw["state_inputs"])}
        self.dependencies, ids = {}, set()
        for node in nodes:
            if not isinstance(node, dict) or not isinstance(node.get("id"), str) or not node["id"] or node["id"] in ids:
                raise ValueError("invalid/duplicate semantic node identity")
            ident = node["id"]
            ids.add(ident)
            if not isinstance(node.get("op"), str) or not node["op"]:
                raise ValueError(f"invalid semantic operation: {ident}")
            for key in ("inputs", "outputs", "weights"):
                values = node.get(key, [] if key == "weights" else None)
                if not isinstance(values, list) or any(not isinstance(v, str) or not v for v in values):
                    raise ValueError(f"invalid {key}: {ident}")
            if not node["outputs"] or len(set(node["outputs"])) != len(node["outputs"]):
                raise ValueError(f"missing/duplicate outputs: {ident}")
            if any(v not in available for v in node["inputs"]):
                raise ValueError(f"unresolved/forward dependency: {ident}")
            if any(v in available for v in node["outputs"]):
                raise ValueError(f"duplicate tensor producer: {ident}")
            if any(v not in self.tensors and v not in aliases for v in node.get("weights", [])):
                raise ValueError(f"unresolved weight: {ident}")
            attrs = node.get("attrs", {})
            if not isinstance(attrs, dict) or ("contract" in attrs and
                    (not isinstance(attrs["contract"], str) or attrs["contract"] not in raw["contracts"])):
                raise ValueError(f"unknown arithmetic contract: {ident}")
            if node["op"] == "compact_int4_linear":
                self._check_pack(node, attrs)
            self.dependencies[ident] = sorted({available[v] for v in node["inputs"] if available[v] is not None})
            available.update(dict.fromkeys(node["outputs"], ident))
        outputs = raw.get("outputs")
        if not isinstance(outputs, list) or not outputs or any(not isinstance(v, str) or v not in available for v in outputs):
            raise ValueError("unresolved semantic graph outputs")
        if any(name not in available for name in raw["state_outputs"]):
            raise ValueError("unresolved persistent-state outputs")

    def _check_pack(self, node, attrs):
        shape, group = attrs.get("weight_shape"), attrs.get("group")
        contract = self.raw["contracts"].get("compact_int4_linear", {})
        if (not isinstance(shape, list) or len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape)
                or type(group) is not int or group <= 0 or not isinstance(contract, dict)
                or group != contract.get("group") or shape[1] % group or shape[1] % 2
                or (shape[1] // group) % 2):
            raise ValueError(f"invalid compact group/weight shape: {node['id']}")
        rows, columns = shape
        layouts = {"packed": ([rows, columns // 2], "U8"),
                   "scale_codes": ([rows, columns // group], "U8"),
                   "scale_bases": ([rows], "F16"), "zeros": ([rows, columns // group // 2], "U8")}
        for suffix, (expected, dtype) in layouts.items():
            names = [name for name in node.get("weights", []) if name.endswith(".int4_" + suffix)]
            if (len(names) != 1 or names[0] not in self.tensors or self.tensors[names[0]]["shape"] != expected
                    or self.tensors[names[0]]["source_dtype"] != dtype):
                raise ValueError(f"inconsistent compact {suffix} layout: {node['id']}")

    def inventory(self):
        return {"kind": self.raw["kind"], "nodes": len(self.raw["nodes"]),
                "operations": dict(sorted(Counter(n["op"] for n in self.raw["nodes"]).items())),
                "stored_tensors": len(self.tensors), "stored_tensor_bytes": sum(t["bytes"] for t in self.tensors.values()),
                "state_inputs": len(self.raw["state_inputs"]), "state_outputs": len(self.raw["state_outputs"]),
                "aliases": self.raw["tensor_aliases"], "execution_validated": False, "capture_compatible": False}
