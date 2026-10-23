"""Captured workload pairing and discovery; public compatibility lives in graph.py."""
import json
from dataclasses import dataclass
from pathlib import Path

from cdfg import Graph, capture_path, has_capture

@dataclass
class Workload:
    name: str
    directory: Path
    prefill: Graph
    decode: Graph

    @property
    def meta(self):
        return self.prefill.meta

    def graph(self, phase):
        return self.prefill if phase == "prefill" else self.decode


def load_workload(directory):
    directory = Path(directory)
    graphs = {}
    for phase in ("prefill", "decode"):
        path = capture_path(directory, phase)
        if path is None:
            raise ValueError(f"missing captured phase: {directory.name}/{phase} in {directory}")
        graphs[phase] = Graph(path, phase)
    return Workload(directory.name, directory, graphs["prefill"], graphs["decode"])


def latest_run(results_dir):
    runs = []
    for manifest in Path(results_dir).glob("*/run-manifest.json"):
        if (manifest.parent / "graphs").is_dir():
            started = json.loads(manifest.read_text(encoding="utf-8")).get("run", {}).get("started_at", "")
            runs.append((started, manifest.parent))
    if not runs:
        raise SystemExit(f"No run with graphs/ under {results_dir}")
    return max(runs)[1]
