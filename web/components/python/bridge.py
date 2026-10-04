"""Browser bridge to the preserved sbengine package."""
import copy
import json
from pathlib import Path

from sbengine.config import DEFAULTS, set_path, validate
from sbengine.graph import load_workload
from sbengine.model import evaluate

WORKLOADS = {}


def run(request_text):
    request = json.loads(request_text)
    capture = request["capture"]
    config = copy.deepcopy(DEFAULTS)
    for key, value in request["config"].items():
        if not key.startswith("_"):
            set_path(config, key, value)
    errors = validate(config)
    if errors:
        return json.dumps({"valid": False, "errors": errors})
    if capture not in WORKLOADS:
        WORKLOADS[capture] = load_workload(Path("/captures") / capture)
    workload = WORKLOADS[capture]
    report = evaluate(workload, config, detail=True)
    for name, phase in report["phases"].items():
        graph = workload.graph(name)
        dependencies = graph.deps()
        for row in phase["rows"]:
            node = graph.nodes[row["node"]]
            row["id"] = node.tensor.id
            row["dependencies"] = [graph.nodes[i].tensor.id for i in dependencies[node.idx]]
    return json.dumps(report, allow_nan=False)
