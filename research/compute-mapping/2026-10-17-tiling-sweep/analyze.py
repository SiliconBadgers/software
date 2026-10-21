"""Tiling study: run the engine/sweeps/tiling specs and record which tiles the engine chose.

The sweep rows come from the existing study runner (engine/sweeps/run.py -> sbengine.run_sweep), so they are
identical to `python -B engine/sweeps/run.py engine/sweeps/tiling/<spec>.json`. Each point is then re-evaluated
with detail=True to read the selected matmul / fused-attention tiles and the traffic split, which the flat sweep
record does not carry. A small robustness check re-ranks equal-PE array shapes at other L1 sizes and at the
128-token prompt; it is a sensitivity check on the tiling ranking, not an SRAM study.

Every evaluated workload, including the robustness workloads loaded outside the study runner, is checked against
the baseline record's decoded capture SHA-256 before use, so a substituted or edited capture fails instead of
being reported under the pinned workload name. The base configuration comes from the same validated spec loader.

Usage, from the repository root:
    python -B research/compute-mapping/2026-10-17-tiling-sweep/analyze.py [--out build/sweeps/<name>]
Writes points.csv, robustness.csv and summary.json to a new folder under the ignored build/sweeps/ (default
build/sweeps/tiling-analysis-v1); an existing non-empty folder is never overwritten. The compact summary is
recorded in docs/sweeps/tiling-v1.json; its row hashes pin the full candidate output.
"""
import argparse
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
from sbengine.costs import Ctx, node_cost
from sbengine.graph import load_workload
from sbengine.model import evaluate
from sbengine.provenance import validate_captures
from sweeps.run import study
from sweeps.specs import load_spec

SPECS = ["engine/sweeps/tiling/array-shape.json", "engine/sweeps/tiling/matmul-tile.json",
         "engine/sweeps/tiling/attention-tile.json"]
ALL_AXES = sorted({axis for name in SPECS for axis in json.loads((ROOT / name).read_text())["axes"]})
OUTPUT_ROOT = ROOT / "build/sweeps"
DEFAULT_OUT = OUTPUT_ROOT / "tiling-analysis-v1"
# Representative operations whose selected tile is reported (first instance in schedule order).
PREFILL_MATMUL = "ffn_gate"       # 6144 x 2048 MLP expansion, the largest prefill matmul family
DECODE_MATMUL = "result_output"   # 248320 x 2048 vocabulary head, the largest decode matmul
# Equal-PE shapes re-ranked by the robustness check (count, rows, cols); all hold 4,096 or 2,048 PEs.
ROBUST_SHAPES = [(1, 64, 64), (2, 32, 64), (2, 64, 32), (4, 16, 64), (4, 32, 32), (4, 64, 16), (8, 16, 32),
                 (8, 32, 16), (16, 16, 16), (1, 32, 64), (1, 64, 32), (2, 32, 32), (4, 16, 32), (4, 32, 16),
                 (8, 16, 16)]
ROBUST_L1_MIB = [4, 8, 16]
ROBUST_WORKLOADS = ["pp512-fa-on", "pp128-fa-on"]


def first_note(rows, op, prefix=None):
    for row in rows:
        if row["op"] == op and (prefix is None or row["name"].startswith(prefix)):
            return row["note"]
    return ""


def matmul_feed(workload, cfg):
    """Prefill matmul L1->array feed bytes split by operand, read from the engine's own per-node cost terms.

    With every prefill matmul a weight x activation product and requantization fused, the weight feed is the
    node's fixed L1 term and the activation feed is its per-activation feed term (costs.cost_matmul).
    """
    if "requant" not in cfg["matrix"]["caps"]:
        raise ValueError("unfused requantization adds epilogue bytes to the fixed L1 term")
    g, ctx = workload.graph("prefill"), Ctx(cfg)
    weight = activation = 0.0
    for n in g.nodes:
        if n.op != "MUL_MAT":
            continue
        if (g.kind(g.root(n.slots[0].id)), g.kind(g.root(n.slots[1].id))) != ("weight", "activation"):
            raise ValueError(f"{n.tensor.name}: feed split assumes a weight x activation matmul")
        cost = node_cost(ctx, g, n)
        weight += cost.l1_fixed
        activation += sum(cost.feed.values())
    return weight, activation


def detail(workload, cfg):
    """Selected tiles and the traffic split for one configuration (GB = 1e9 bytes)."""
    result = evaluate(workload, cfg, detail=True)
    pf, dc = result["phases"]["prefill"], result["phases"]["decode"]
    m = cfg["matrix"]
    matmuls = [row for row in pf["rows"] if row["op"] == "MUL_MAT"]
    weight_feed, activation_feed = matmul_feed(workload, cfg)
    return {
        "total_pe": m["count"] * m["rows"] * m["cols"],
        "tile_budget_per_unit_kib": cfg["l1"]["mib"] * 1024 * cfg["l1"]["tile_fraction"] / max(1, m["count"]),
        "prefill_matmul_tile": first_note(pf["rows"], "MUL_MAT", PREFILL_MATMUL),
        "decode_matmul_tile": first_note(dc["rows"], "MUL_MAT", DECODE_MATMUL),
        "prefill_attention_tile": first_note(pf["rows"], "FLASH_ATTN_EXT"),
        "prefill_weights_hbm_gb": pf["bytes"]["weights_hbm"] / 1e9,
        "prefill_act_hbm_gb": pf["bytes"]["act_hbm"] / 1e9,
        "prefill_l1_gb": pf["bytes"]["l1"] / 1e9,
        "decode_weights_hbm_gb": dc["bytes"]["weights_hbm"] / 1e9,
        "decode_l1_gb": dc["bytes"]["l1"] / 1e9,
        "prefill_l1_service_s": pf["l1_service_s"],
        "prefill_matrix_busy_s": pf["pool_busy"].get("Matrix", 0.0),
        # Per-operation compute vs shared-L1 service, summed over prefill matmuls: an operation is L1-bound when
        # its L1 term exceeds its compute term, which is what moves a shape off the matrix-bound regime.
        "prefill_matmul_compute_s": sum(row["compute_s"] for row in matmuls),
        "prefill_matmul_l1_s": sum(row["l1_s"] for row in matmuls),
        "prefill_matmul_l1_bound_s": sum(row["duration_s"] for row in matmuls if row["dominant"] == "l1"),
        # Output-stationary feed ~ M N K (w / cols + a / rows): weight feed follows columns, activation feed rows.
        "prefill_matmul_weight_feed_gb": weight_feed / 1e9,
        "prefill_matmul_activation_feed_gb": activation_feed / 1e9,
    }


def checked(workload, baseline, captures):
    """Require both phases of an already loaded workload to match the baseline's decoded capture identity."""
    validate_captures(workload, baseline)
    for phase in ("prefill", "decode"):
        key = f"{workload.name}/{phase}"
        captures[key] = baseline["decoded_capture_sha256"][key]
    return workload


def pinned_workload(baseline, name, captures):
    return checked(load_workload(ROOT / baseline["run"] / "graphs" / name), baseline, captures)


def run_specs(baseline, captures):
    rows, studies = [], []
    for name in SPECS:
        if load_spec(ROOT, ROOT / name)[1]["id"] != baseline["id"]:
            raise ValueError(f"{name}: all tiling specs must share one baseline")
        spec, cfg, workload, records, summary = study(ROOT / name)
        # study() validated these captures; re-check the same object that detail() re-evaluates below.
        checked(workload, baseline, captures)
        for (assignment, point_cfg), record in zip(sweep.grid(cfg, spec["axes"]), records):
            if record["config_hash"] != config.config_hash(point_cfg):
                raise RuntimeError(f"{spec['id']}: grid order differs from the study records")
            keep = {k: record[k] for k in ("valid", "errors", "cost", "dsp", "prefill_s", "decode_s",
                                           "prefill_binding", "decode_binding", "config_hash")}
            rows.append({"study": spec["id"], **{k: assignment.get(k, "") for k in ALL_AXES},
                         **keep, **detail(workload, point_cfg)})
        studies.append({key: summary[key] for key in ("id", "baseline", "workload", "config_hash", "axes", "objectives",
                                                      "points", "valid_points", "invalid_points", "frontier",
                                                      "records_sha256")}
                       | {"spec": name, "spec_sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()})
    return rows, studies


def robustness(base, baseline, captures):
    rows = []
    for wl_name in ROBUST_WORKLOADS:
        workload = pinned_workload(baseline, wl_name, captures)
        for mib in ROBUST_L1_MIB:
            for count, r, c in ROBUST_SHAPES:
                cfg = copy.deepcopy(base)
                cfg["matrix"].update(count=count, rows=r, cols=c)
                cfg["l1"]["mib"] = mib
                record = sweep.summarize(evaluate(workload, cfg))
                rows.append({"workload": wl_name, "l1.mib": mib, "matrix.count": count, "matrix.rows": r,
                             "matrix.cols": c, "total_pe": count * r * c,
                             **{k: record[k] for k in ("valid", "errors", "prefill_s", "decode_s",
                                                       "prefill_binding", "decode_binding")}})
    return rows


def l1_size_control(base, baseline, captures):
    """Separate the larger-L1 prefill increase into causal attention granularity and everything else.

    A larger L1 enlarges the fused-attention query tile; causal skipping works at tile granularity, so fewer,
    larger tiles skip less work. Dense attention (causal_skip off) removes that effect from the comparison.
    """
    workload = pinned_workload(baseline, "pp512-fa-on", captures)
    rows = []
    for count, r, c in [(1, 64, 64), (1, 32, 64), (1, 64, 32)]:
        for causal in (True, False):
            seconds = {}
            for mib in (8, 16):
                cfg = copy.deepcopy(base)
                cfg["matrix"].update(count=count, rows=r, cols=c)
                cfg["l1"]["mib"] = mib
                cfg["attention"]["causal_skip"] = causal
                phase = evaluate(workload, cfg, detail=True)["phases"]["prefill"]
                attention = sum(row["duration_s"] for row in phase["rows"] if row["op"] == "FLASH_ATTN_EXT")
                seconds[mib] = (phase["seconds"], attention)
            rows.append({"shape": [count, r, c], "causal_skip": causal,
                         "prefill_increase_8_to_16_mib_s": seconds[16][0] - seconds[8][0],
                         "attention_increase_8_to_16_mib_s": seconds[16][1] - seconds[8][1]})
    return rows


def rows_sha256(rows):
    """Content hash of full candidate rows, independent of CSV formatting and line endings."""
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def analysis():
    """Full candidate rows and the compact summary, without reproduction metadata or any file output."""
    # Same validated loader as the study runner: resolved profile whose hash matches the baseline record.
    _, baseline, base = load_spec(ROOT, ROOT / SPECS[0])
    captures = {}
    rows, studies = run_specs(baseline, captures)
    robust = robustness(base, baseline, captures)
    control = l1_size_control(base, baseline, captures)
    summary = {
        "method": "engine/sweeps/run.py study() per spec, then sbengine.evaluate(detail=True) per point",
        "engine_version": VERSION, "baseline": baseline["id"], "run": baseline["run"],
        "studies": studies,
        "points": {"rows": len(rows), "valid": sum(row["valid"] for row in rows), "sha256": rows_sha256(rows)},
        "robustness": {"workloads": ROBUST_WORKLOADS, "l1_mib": ROBUST_L1_MIB, "shapes": ROBUST_SHAPES,
                       "rows": len(robust), "sha256": rows_sha256(robust),
                       "note": "Re-ranks equal-PE shapes; not an SRAM design study."},
        "l1_size_control": {"workload": "pp512-fa-on", "rows": control,
                            "note": "Model accounting check (causal tile granularity), not a physical SRAM penalty."},
        # Every capture evaluated above, validated against the baseline record before use.
        "validated_decoded_capture_sha256": dict(sorted(captures.items())),
        "limitations": ["Analytical estimates under the provisional characterized-v1 baseline; cost is a unitless proxy.",
                        "SRAM (l1.*) is held at the baseline except in the labelled robustness check.",
                        "Feasibility flags do not establish physical fit, timing closure or numerical quality."],
    }
    return rows, robust, summary


def output_folder(path):
    """A new study folder under build/sweeps/; existing evidence is never overwritten (as engine/sweeps/run.py)."""
    out = Path(path).resolve()
    if not out.is_relative_to(OUTPUT_ROOT) or out == OUTPUT_ROOT:
        raise ValueError("full tiling output must stay in a study folder under build/sweeps/")
    if out.exists() and any(out.iterdir()):
        raise ValueError("existing evidence is not overwritten; choose a new --out folder")
    return out


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="new output folder under build/sweeps/")
    args = parser.parse_args()
    out = output_folder(args.out)
    rows, robust, summary = analysis()
    summary["reproduction"] = {
        "command": "python -B research/compute-mapping/2026-10-17-tiling-sweep/analyze.py",
        "git_head": git("rev-parse", "HEAD"),
        # Source paths only; capture identity is the validated hashes above.
        "uncommitted_source_paths": bool(git("status", "--porcelain", "--", "engine", "core",
                                             Path(__file__).relative_to(ROOT).as_posix())),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "points.csv", rows)
    write_csv(out / "robustness.csv", robust)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"{len(rows)} sweep points, {summary['points']['valid']} valid; {len(robust)} robustness points; "
          f"output {out}")


if __name__ == "__main__":
    main()
