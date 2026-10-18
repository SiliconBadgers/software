"""Thin reproduction caller for existing grid evaluation and Pareto filtering."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "engine"))
from sbengine import config, sweep
from sbengine.graph import load_workload
from sbengine.provenance import validate_captures
from sbengine.trace import load_trace
from sweeps.specs import load_spec, validate_objective_values


def study(spec_path):
    spec_path = Path(spec_path).resolve()
    spec, baseline, cfg = load_spec(ROOT, spec_path)
    run = ROOT / baseline["run"]
    workload = load_workload(run / "graphs" / spec["workload"])
    validate_captures(workload, baseline)
    trace = load_trace(run, workload) if cfg["host"]["enabled"] else None
    records = sweep.run_sweep(workload, cfg, spec["axes"], trace)
    valid = [row for row in records if row["valid"]]
    objectives = [tuple(item) for item in spec["objectives"]]
    validate_objective_values(records, objectives)
    frontier = sweep.pareto(records, objectives)
    frontier_keys = list(dict.fromkeys([*spec["axes"], "cost", "prefill_s", "decode_s",
                                       *(metric for metric, _ in objectives)]))
    summary = {
        "id": spec["id"], "method": "existing sbengine.run_sweep + pareto", "baseline": baseline["id"],
        "baseline_record": spec["baseline"], "source_baseline": baseline["source_commit"],
        "workload": workload.name, "run": baseline["run"], "config_hash": config.config_hash(cfg),
        "axes": spec["axes"], "objectives": spec["objectives"],
        "fixed_assumptions": {path: config.get_path(cfg, path) for path in baseline["fixed_assumptions"]},
        "points": len(records), "valid_points": len(valid), "invalid_points": len(records) - len(valid),
        "frontier": [{key: row[key] for key in frontier_keys} for row in frontier],
        "records_sha256": hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "limitations": ["Analytical estimates under provisional assumptions; cost is a unitless proxy.",
                        "Feasibility flags do not establish physical fit or numerical quality.",
                        "No candidate is promoted and no globally optimal design is established."],
    }
    return spec, cfg, workload, records, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spec", type=Path)
    parser.add_argument("--out", type=Path, help="new output directory under build/sweeps/")
    args = parser.parse_args()
    spec, cfg, workload, records, summary = study(args.spec)
    out = (args.out or ROOT / "build/sweeps" / spec["id"]).resolve()
    if not out.is_relative_to(ROOT / "build/sweeps") or out == ROOT / "build/sweeps":
        raise ValueError("full study output must stay in a study folder under build/sweeps/")
    if out.exists() and any(out.iterdir()):
        raise ValueError("existing evidence is not overwritten; choose a new --out folder")
    command = " ".join(sys.argv)
    sweep.write_sweep(out, spec["id"], records, cfg, spec["axes"], workload, command)
    summary["reproduction"] = {"command": command, "git_head": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "spec_sha256": hashlib.sha256(args.spec.read_bytes()).hexdigest()}
    summary["reproduction"].update({
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "baseline_record_sha256": hashlib.sha256((ROOT / spec["baseline"]).read_bytes()).hexdigest(),
    })
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out / "records.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    print(f"{spec['id']}: {summary['points']} points, {summary['valid_points']} valid; output {out}")


if __name__ == "__main__":
    main()
