"""Reproduce the pre-refactor contract using maintained code, not new equations.

Full graphs, costs and schedules stay in ignored build/. Only the compact digest
fixture is reviewed into Git. This script never updates that fixture itself.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine"))
sys.path.insert(0, str(ROOT / "scripts"))

from check_boundaries import historical_files as preserved_historical_files  # noqa: E402

from sbengine import config  # noqa: E402
from sbengine.costs import Ctx, node_cost  # noqa: E402
from sbengine.graph import capture_path, has_capture, load_workload  # noqa: E402
from sbengine.model import evaluate  # noqa: E402

SOURCE = "0e06023d74c06cc12bb9e05d9e65102347ecc3f0"
RUN = ROOT / "results/2026-09-27_1839_windows-2"
FIXTURE = Path(__file__).with_name("fixtures") / "characterization-0e06023.json"
SWITCHES = {
    "fusion-none": {"memory.fusion": "none"},
    "fusion-groups": {"memory.fusion": "groups"},
    "direct-weights": {"memory.weight_path": "direct"},
    "serial-schedule": {"schedule.mode": "serial"},
    "no-within-op-overlap": {"schedule.within_op_overlap": False},
    "serial-recurrence": {"recurrent.lowering": "serial"},
    "matrix-recurrence-4x": {"recurrent.lowering": "matrix", "recurrent.matrix_penalty": 4.0},
    "uniform-int4-scenario": {"precision.weights": "uniform_int4"},
    "batch-2-scenario": {"batch": 2},
    "serial-softmax": {"attention.overlap_softmax": False},
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_unchanged(paths):
    subprocess.run(["git", "diff", "--quiet", "--no-ext-diff", "--ignore-submodules=dirty", "HEAD", "--", *paths],
                   cwd=ROOT, check=True)


def repository_file_digest(path):
    relative = Path(path).relative_to(ROOT).as_posix()
    require_unchanged([relative])
    return hashlib.sha256(subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=ROOT)).hexdigest()


def graph_record(graph):
    """Capture-derived structure only; no paths, cache internals or timing assumptions."""
    return {
        "phase": graph.phase, "metadata": graph.meta, "order": graph.order,
        "tensors": [asdict(t) for _, t in sorted(graph.tensors.items())],
        "nodes": [{"idx": n.idx, "tensor": n.tensor.id, "srcs": [s.id for s in n.srcs],
                   "slots": {k: t.id for k, t in n.slots.items()}, "dst_root": n.dst_root,
                   "src_roots": n.src_roots, "spos": n.spos} for n in graph.nodes],
        "storage": [[tid, graph.root(tid), graph.kind(graph.root(tid))] for tid in sorted(graph.tensors)],
        "deps": graph.deps(), "totals": graph.totals(),
        "metadata_nodes": graph.metadata_nodes, "zero_nodes": graph.zero_nodes,
    }


def evaluation_record(workload, cfg):
    result = evaluate(workload, cfg, detail=True)
    costs = {}
    for phase in ("prefill", "decode"):
        ctx = Ctx(cfg)
        # Use the maintained evaluator's state partition result; do not duplicate its equation.
        ctx.state_resident = result["phases"][phase]["state_resident_fraction"]
        costs[phase] = [asdict(node_cost(ctx, workload.graph(phase), n)) for n in workload.graph(phase).nodes]
    return {"evaluation": result, "raw_costs_before_fusion": costs}


def compact_evaluation(record):
    result = record["evaluation"]
    return [digest(result), digest(record["raw_costs_before_fusion"]),
            result["phases"]["prefill"]["seconds"], result["phases"]["decode"]["seconds"], result["valid"]]


def historical_files():
    """The original evidence subset; later research does not rewrite its digest."""
    return preserved_historical_files(ROOT)


def write_json(path, value, pretty=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2, allow_nan=False) if pretty else canonical(value).decode()
    path.write_text(data + "\n", encoding="utf-8")


def snapshot(out=None):
    """All registered capture pairs x all profiles, plus representative switches.

    Missing inputs fail rather than becoming a skip or an implicit latest-run selection.
    Optional output is deliberately restricted to this repository's ignored audit folder.
    """
    if out is not None:
        out = Path(out).resolve()
        if not out.is_relative_to(ROOT / "build/refactor-characterization"):
            raise ValueError("comparison output must stay under build/refactor-characterization/")
        if out.exists():
            raise ValueError("choose a new output folder; existing comparison evidence is not overwritten")
    names = sorted(p.name for p in (RUN / "graphs").iterdir() if has_capture(p))
    profiles = {p.stem: config.load_profile(p)[0] for p in sorted((ROOT / "engine/profiles").glob("*.json"))}
    history, gitlinks = historical_files()
    pinned_path = ROOT / "core/experiments/qwen35-2b-q4km.json"
    compact = {
        "schema_version": 1, "source_commit": SOURCE, "run": RUN.relative_to(ROOT).as_posix(),
        "pinned_experiment": json.loads(pinned_path.read_text(encoding="utf-8")),
        "pinned_experiment_repository_sha256": repository_file_digest(pinned_path),
        "run_manifest_repository_sha256": repository_file_digest(RUN / "run-manifest.json"),
        "historical_files": {"count": len(history), "sha256": digest(history)},
        "historical_gitlinks": gitlinks,
        "graph_columns": ["capture_sha256", "structure_sha256", "active_nodes", "tensors"],
        "case_columns": ["evaluation_sha256", "raw_costs_sha256", "prefill_s", "decode_s", "valid"],
        "profiles": profiles, "switches": SWITCHES, "graphs": {}, "cases": {},
    }
    for name in names:
        workload = load_workload(RUN / "graphs" / name)
        for phase in ("prefill", "decode"):
            graph = workload.graph(phase)
            record = graph_record(graph)
            compact["graphs"][f"{name}/{phase}"] = [file_digest(capture_path(workload.directory, phase)),
                                                      digest(record), len(graph.nodes), len(graph.tensors)]
            if out:
                write_json(out / "graphs" / name / f"{phase}.json", record)
        cases = dict(profiles)
        if name == "pp512-fa-on":
            for label, patch in SWITCHES.items():
                cfg = config.deep_merge(profiles["accel-balanced"], {})
                for key, value in patch.items():
                    config.set_path(cfg, key, value)
                cases[f"switch-{label}"] = config.make_config(base=cfg)
        for label, cfg in cases.items():
            record = evaluation_record(workload, cfg)
            compact["cases"][f"{name}/{label}"] = compact_evaluation(record)
            if out:
                write_json(out / "evaluations" / name / f"{label}.json", {"config": cfg, **record})
    if out:
        write_json(out / "historical-files.json", history, pretty=True)
        write_json(out / "local-historical-bytes.json", {name: file_digest(ROOT / name) for name in history}, pretty=True)
        write_json(out / "compact.json", compact, pretty=True)
    return compact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="new ignored comparison folder")
    parser.add_argument("--compare", type=Path, help="explicit compact reference (never rewritten)")
    args = parser.parse_args()
    got = snapshot(args.out)
    print(f"Recorded {len(got['graphs'])} phases and {len(got['cases'])} evaluations in {args.out}")
    if args.compare:
        expected = json.loads(args.compare.read_text(encoding="utf-8"))
        if expected != got:
            raise SystemExit("Characterization differs; inspect the full outputs before changing any fixture.")
        print("Exact compact comparison matches.")


if __name__ == "__main__":
    main()
