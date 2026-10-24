"""Identity-checked semantic workload import using GGML's maintained GGUF reader.

Only reads the supplied artifact. Embedded source text is hashed, never executed.
Full output is new ignored evidence, not prefill/decode capture files.
"""
import hashlib
from importlib.metadata import version
import json
import math
from decimal import Decimal
from pathlib import Path
import sys

from common import CORE, REPO, sha256_of
from fetch import verify_model

sys.path.insert(0, str(CORE))
from cdfg.semantic import SemanticGraph


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def read_json(text):
    def bad(value):
        raise ValueError(f"nonfinite JSON value: {value}")
    def finite(value):
        number = float(value)
        if not math.isfinite(number) or (number == 0 and Decimal(value) != 0):
            raise ValueError(f"unrepresentable JSON number: {value}")
        return number
    try:
        return json.loads(text, object_pairs_hook=_unique, parse_constant=bad, parse_float=finite)
    except RecursionError as error:
        raise ValueError("JSON nesting exceeds the reader limit") from error


def inspect(path, experiment, artifact):
    if experiment.get("runtime", {}).get("status") != "inspection-only":
        raise ValueError("expected an inspection-only workload definition")
    if artifact not in ("model", "graph"):
        raise ValueError("artifact must be model or graph")
    identity = experiment["artifacts"][artifact]
    verify_model(path, identity)  # Before parsing or importing any optional reader.
    try:
        from gguf import GGUFReader
    except ImportError as error:
        raise ImportError("Install the locked model-inspection dependency group to read GGUF") from error
    if version("gguf") != "0.19.0":
        raise ValueError("use the locked gguf==0.19.0 reader from the model-inspection group")
    try:
        reader = GGUFReader(path, mode="r")
    except (ValueError, IndexError, TypeError, OverflowError, KeyError, RecursionError) as error:
        raise ValueError(f"invalid GGUF: {error}") from error

    def field(name, expected=str):
        value = reader.get_field(name)
        if value is None:
            raise ValueError(f"missing GGUF field: {name}")
        contents = value.contents()
        if type(contents) is not expected:
            raise ValueError(f"GGUF field {name} must be a scalar {expected.__name__}")
        return contents

    schema = field("qwenquant.schema_version", int)
    if field("GGUF.version", int) != 3 or schema != 1:
        raise ValueError("unsupported GGUF/custom metadata version")
    if field("general.architecture") != experiment["architecture"]:
        raise ValueError("GGUF architecture differs from workload definition")
    weighted = field("qwenquant.contains_weights", bool)
    if weighted != (artifact == "model"):
        raise ValueError("GGUF weight-payload role differs from requested artifact")
    graph_text = field("qwenquant.graph_json")
    graph = SemanticGraph(read_json(graph_text))
    raw = graph.raw
    if (raw["architecture"] != experiment["architecture"] or raw["model"] != experiment["upstream_model"]["repo"]
            or raw["revision"] != experiment["upstream_model"]["revision"]):
        raise ValueError("semantic graph source identity differs from workload definition")
    if hashlib.sha256(graph_text.encode()).hexdigest() != experiment["embedded_graph_sha256"]:
        raise ValueError("embedded semantic graph identity differs")
    if read_json(field("qwenquant.config_json")) != raw["config"]:
        raise ValueError("GGUF/graph configurations differ")
    recipe = read_json(field("qwenquant.recipe_json"))
    if (not isinstance(recipe, dict) or not isinstance(recipe.get("export"), dict)
            or recipe.get("model") != raw["model"] or recipe.get("revision") != raw["revision"]
            or recipe["export"].get("sha256") != raw["source_weights_sha256"]
            or field("qwenquant.source_weights_sha256") != raw["source_weights_sha256"]):
        raise ValueError("recipe/source-weight identity differs")
    bundle = read_json(field("qwenquant.runtime_sources_json"))
    if not isinstance(bundle, dict) or set(bundle) != set(raw["source_sha256"]):
        raise ValueError("embedded runtime source inventory differs")
    for name, source in bundle.items():
        if (not isinstance(source, dict) or not isinstance(source.get("text"), str)
                or hashlib.sha256(source["text"].encode()).hexdigest() != source.get("sha256")
                or source["sha256"] != raw["source_sha256"][name]):
            raise ValueError(f"embedded runtime source identity differs: {name}")
    actual = {}
    for tensor in reader.tensors:
        if tensor.name in actual:
            raise ValueError(f"duplicate stored tensor: {tensor.name}")
        actual[tensor.name] = {"shape": list(reversed([int(d) for d in tensor.shape])),
                               "ggml_type": int(tensor.tensor_type), "bytes": int(tensor.n_bytes)}
    expected = {name: {k: t[k] for k in ("shape", "ggml_type", "bytes")} for name, t in graph.tensors.items()}
    if actual != (expected if weighted else {}):
        raise ValueError("GGUF tensor directory differs from semantic manifest/payload role")
    result = {"schema": "siliconbadgers.semantic-import.v1", "evidence_kind": "semantic_model_import",
              "status": "format-verified", "source": {**experiment["source"], **identity, "artifact": artifact},
              "upstream_model": experiment["upstream_model"], "architecture": experiment["architecture"],
              "embedded_graph_sha256": experiment["embedded_graph_sha256"],
              "source_weights_sha256": raw["source_weights_sha256"], "runtime_source_sha256": raw["source_sha256"],
              "reader": "gguf==0.19.0", "tensor_payload_present": weighted,
              "implementation_sha256": {"import_workload.py": sha256_of(__file__),
                                         "cdfg/semantic.py": sha256_of(CORE / "cdfg/semantic.py")},
              "definition_sha256": hashlib.sha256(json.dumps(experiment, sort_keys=True, separators=(",", ":"),
                                                              allow_nan=False).encode()).hexdigest(),
              "numerical_execution_validated": False, "inventory": graph.inventory(),
              "limitations": ["Semantic DAG, not an executed GGML capture; no measured timings or finite-logit evidence.",
                              "Tensor/storage/source identity is checked; embedded code is not executed.",
                              "No numerical equivalence, target lowering, hardware mapping or performance is validated."]}
    return graph_text, result


def import_workload(path, experiment, artifact, out, root=REPO):
    out, boundary = Path(out).resolve(), Path(root).resolve() / "build" / "workloads"
    if boundary.resolve() != boundary or not out.is_relative_to(boundary) or out == boundary or out.exists():
        raise ValueError("choose a new output directory under build/workloads/; existing evidence is not overwritten")
    text, result = inspect(path, experiment, artifact)
    result["experiment"] = experiment["name"]
    out.mkdir(parents=True, exist_ok=False)
    (out / "semantic-graph.json").write_bytes(text.encode("utf-8"))
    (out / "import.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result
