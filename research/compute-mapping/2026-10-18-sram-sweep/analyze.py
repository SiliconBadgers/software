"""SRAM study: run the engine/sweeps/sram specs and record how the L1 partition and traffic respond.

The sweep rows come from the existing study runner (engine/sweeps/run.py -> sbengine.run_sweep), so they are
identical to `python -B engine/sweeps/run.py engine/sweeps/sram/<spec>.json`. Each point is then re-evaluated
with detail=True to read the L1 partition, the HBM/L1 traffic split, activation residency, state residency and
the selected tiles, which the flat sweep record does not carry. A small robustness check re-ranks the capacity
and banking settings at other matrix-unit counts and workloads; it is a sensitivity check on the SRAM ranking,
not a compute study.

Usage, from the repository root:
    python -B research/compute-mapping/2026-10-18-sram-sweep/analyze.py
Writes points.csv, robustness.csv and summary.json next to this file.
"""
import copy
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "engine"))
from sbengine import VERSION, config, sweep
from sbengine.graph import load_workload
from sbengine.model import evaluate
from sweeps.run import study

SPECS = ["engine/sweeps/sram/capacity.json", "engine/sweeps/sram/bandwidth.json",
         "engine/sweeps/sram/partition.json", "engine/sweeps/sram/organization.json"]
ALL_AXES = sorted({axis for name in SPECS for axis in json.loads((ROOT / name).read_text())["axes"]})
# Representative operations whose selected tile is reported (first instance in schedule order).
PREFILL_MATMUL = "ffn_gate"       # 6144 x 2048 MLP expansion, the largest prefill matmul family
DECODE_MATMUL = "result_output"   # 248320 x 2048 vocabulary head, the largest decode matmul
# Robustness: the capacity and banking settings re-evaluated at other unit counts and workloads.
ROBUST_L1_MIB = [4, 8, 16, 32]
ROBUST_BANKS = [32, 64, 128, 256]
ROBUST_COUNTS = [2, 4, 8]
ROBUST_WORKLOADS = ["pp512-fa-on", "pp128-fa-on", "pp512-fa-off"]
MiB, GB = 2**20, 1e9


def first_note(rows, op, prefix=None):
    for row in rows:
        if row["op"] == op and (prefix is None or row["name"].startswith(prefix)):
            return row["note"]
    return ""


def partition(cfg):
    """The static L1 split the engine uses (costs.Ctx): tiles, resident activations, recurrent state."""
    l1, m = cfg["l1"], cfg["matrix"]
    total = l1["mib"] * MiB
    return {
        "tile_budget_per_unit_kib": total * l1["tile_fraction"] / max(1, m["count"]) / 1024,
        "activation_capacity_mib": total * (1 - l1["state_reserve"] - l1["tile_fraction"]) / MiB,
        "state_capacity_mib": total * l1["state_reserve"] / MiB,
        "l1_bandwidth_gbs": l1["banks"] * l1["bytes_per_bank_cycle"] * cfg["clock_mhz"] * 1e6 * l1["efficiency"] / GB,
    }


def detail(workload, cfg):
    """Partition, traffic split, residency and selected tiles for one configuration (GB = 1e9 bytes)."""
    result = evaluate(workload, cfg, detail=True)
    pf, dc = result["phases"]["prefill"], result["phases"]["decode"]
    out = partition(cfg)
    out.update({
        "state_resident_fraction": pf["state_resident_fraction"],
        "prefill_residency_hit_rate": pf["residency"]["hits"] / max(1, pf["residency"]["hits"] + pf["residency"]["misses"]),
        "prefill_peak_spilled_act_mib": pf["footprint"]["peak_spilled_activations"] / MiB,
        "prefill_weights_hbm_gb": pf["bytes"]["weights_hbm"] / GB,
        "prefill_act_hbm_gb": pf["bytes"]["act_hbm"] / GB,
        "prefill_kv_state_hbm_gb": pf["bytes"]["kv_state_hbm"] / GB,
        "prefill_l1_gb": pf["bytes"]["l1"] / GB,
        "prefill_l1_service_s": pf["l1_service_s"],
        "prefill_hbm_service_s": pf["hbm_service_s"],
        "decode_weights_hbm_gb": dc["bytes"]["weights_hbm"] / GB,
        "decode_kv_state_hbm_gb": dc["bytes"]["kv_state_hbm"] / GB,
        "decode_l1_gb": dc["bytes"]["l1"] / GB,
        "decode_l1_service_s": dc["l1_service_s"],
        "decode_hbm_service_s": dc["hbm_service_s"],
        "hbm_required_gib": max(pf["footprint"]["hbm_required"], dc["footprint"]["hbm_required"]) / 2**30,
        "prefill_matmul_tile": first_note(pf["rows"], "MUL_MAT", PREFILL_MATMUL),
        "decode_matmul_tile": first_note(dc["rows"], "MUL_MAT", DECODE_MATMUL),
        "prefill_attention_tile": first_note(pf["rows"], "FLASH_ATTN_EXT"),
    })
    return out


def run_specs():
    rows, studies = [], []
    for name in SPECS:
        spec, cfg, workload, records, summary = study(ROOT / name)
        for (assignment, point_cfg), record in zip(sweep.grid(cfg, spec["axes"]), records):
            keep = {k: record.get(k, "") for k in ("valid", "errors", "cost", "dsp", "prefill_s", "decode_s",
                                                   "prefill_binding", "decode_binding", "config_hash")}
            if "config_hash" not in record:
                # Rejected by config validation before evaluation (no record hash, nothing to detail).
                rows.append({"study": spec["id"], **{k: assignment.get(k, "") for k in ALL_AXES}, **keep,
                             **partition(point_cfg)})
                continue
            if record["config_hash"] != config.config_hash(point_cfg):
                raise RuntimeError(f"{spec['id']}: grid order differs from the study records")
            rows.append({"study": spec["id"], **{k: assignment.get(k, "") for k in ALL_AXES},
                         **keep, **detail(workload, point_cfg)})
        studies.append({key: summary[key] for key in ("id", "baseline", "workload", "config_hash", "axes", "objectives",
                                                      "points", "valid_points", "invalid_points", "frontier",
                                                      "records_sha256")}
                       | {"spec": name, "spec_sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()})
    return rows, studies


def robustness(base, run):
    rows = []
    settings = ([("l1.mib", v) for v in ROBUST_L1_MIB] + [("l1.banks", v) for v in ROBUST_BANKS])
    for wl_name in ROBUST_WORKLOADS:
        workload = load_workload(ROOT / run / "graphs" / wl_name)
        for count in ROBUST_COUNTS:
            for path, value in settings:
                cfg = copy.deepcopy(base)
                cfg["matrix"]["count"] = count
                config.set_path(cfg, path, value)
                record = sweep.summarize(evaluate(workload, cfg))
                rows.append({"workload": wl_name, "matrix.count": count, "axis": path, "value": value,
                             "l1.mib": cfg["l1"]["mib"], "l1.banks": cfg["l1"]["banks"],
                             **{k: record[k] for k in ("valid", "errors", "cost", "prefill_s", "decode_s",
                                                       "prefill_binding", "decode_binding")}})
    return rows


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")   # same bytes on every OS
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    rows, studies = run_specs()
    baseline = json.loads((ROOT / json.loads((ROOT / SPECS[0]).read_text())["baseline"]).read_text())
    base, _ = config.load_profile(ROOT / baseline["profile"])
    robust = robustness(base, baseline["run"])
    write_csv(HERE / "points.csv", rows)
    write_csv(HERE / "robustness.csv", robust)
    summary = {
        "method": "engine/sweeps/run.py study() per spec, then sbengine.evaluate(detail=True) per point",
        "engine_version": VERSION, "baseline": baseline["id"], "run": baseline["run"],
        "studies": studies,
        "robustness": {"workloads": ROBUST_WORKLOADS, "matrix_count": ROBUST_COUNTS, "l1_mib": ROBUST_L1_MIB,
                       "l1_banks": ROBUST_BANKS,
                       "note": "Re-ranks capacity and banking settings; not a compute or array-shape study."},
        "reproduction": {"command": "python -B research/compute-mapping/2026-10-18-sram-sweep/analyze.py",
                         "git_head": git("rev-parse", "HEAD"),
                         "uncommitted_inputs": bool(git("status", "--porcelain", "--", "engine", "core",
                                                        Path(__file__).relative_to(ROOT).as_posix())),
                         "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "limitations": ["Analytical estimates under the provisional characterized-v1 baseline; cost is a unitless proxy.",
                        "Compute (matrix.*) is held at the baseline except in the labelled robustness check.",
                        "The cost proxy prices L1 capacity and bank count but not bank width (bytes_per_bank_cycle).",
                        "Feasibility flags do not establish physical fit, banking/routing, area, power or timing closure."],
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"{len(rows)} sweep points, {sum(r['valid'] is True for r in rows)} valid; {len(robust)} robustness points")


if __name__ == "__main__":
    main()
