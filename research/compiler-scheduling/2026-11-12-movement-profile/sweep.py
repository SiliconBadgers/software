"""Profile one symbolic movement plan per distinct matmul shape in two captures."""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "engine"), str(ROOT / "core")]

from cdfg import Graph
from sbengine.codegen.lower import compile_graph
from sbengine.codegen.movement import MovementPlan, StubTiledMatmul, TargetMemory, round_up
from sbengine.codegen.movement_check import check

DEFAULT_CAPTURES = ROOT / "results/2026-09-27_1839_windows-2/graphs/pp128-fa-on"
MIB = 1024 * 1024
OUTPUT = ROOT / "build/compiler/movement-profile/2026-11-12"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_sram(plan, tile_m, target):
    """Match the planner's resident B/C and two A-panel allocations."""
    row = target.row_bytes
    return (round_up(plan.k * plan.n * target.b_bytes, row)
            + round_up(plan.m * plan.n * target.c_bytes, row)
            + target.a_slots * round_up(tile_m * plan.k * target.a_bytes, row))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=DEFAULT_CAPTURES)
    parser.add_argument("--sram-bytes", type=int, default=8 * MIB)
    parser.add_argument("--tile", default="8,16,8", help="symbolic M,N,K tile limits")
    args = parser.parse_args()
    tm, tn, tk = (int(part) for part in args.tile.split(","))
    if min(tm, tn, tk, args.sram_bytes) < 1:
        parser.error("tile dimensions and SRAM budget must be positive")

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                       text=True).strip()
    rows = []
    captures = {}
    for phase in ("prefill", "decode"):
        capture = args.captures / f"{phase}.json.gz"
        graph = Graph(str(capture), phase)
        compiled = compile_graph(graph)
        matmul_nodes = sum(node.op == "MUL_MAT" for node in graph.nodes)
        if len(compiled.plans) != matmul_nodes:
            raise RuntimeError(f"{phase}: lowered {len(compiled.plans)} of {matmul_nodes} matmuls")
        groups = defaultdict(list)
        for plan in compiled.plans:
            groups[(plan.m, plan.n, plan.k)].append(plan)
        captures[phase] = {"path": capture.relative_to(ROOT).as_posix(),
                           "sha256_compressed": sha256(capture),
                           "matmul_instances": matmul_nodes,
                           "distinct_shapes": len(groups)}
        for m, n, k in sorted(groups):
            representative = groups[(m, n, k)][0]
            target = TargetMemory(sram_bytes=args.sram_bytes)
            needed = required_sram(representative, tm, target)
            fits = needed <= args.sram_bytes
            # If a shape exceeds the assumed budget, use just enough capacity
            # to report its hypothetical counts and mark that plan as unfit.
            if not fits:
                target = TargetMemory(sram_bytes=needed)
            print(f"{phase}: M={m} N={n} K={k}, instances={len(groups[(m,n,k)])}",
                  file=sys.stderr, flush=True)
            start = time.monotonic()
            movement = MovementPlan(StubTiledMatmul(representative, tm, tn, tk), target)
            checked = check(movement)
            if not checked["passed"]:
                raise RuntimeError(f"{phase} {(m,n,k)}: {checked['violations'][:3]}")
            transfers = movement.transfers
            row = {"phase": phase, "m": m, "n": n, "k": k,
                   "instances": len(groups[(m, n, k)]),
                   "sample_tensor_id": representative.tensor_id,
                   "sample_name": representative.c.name,
                   "sram_used_bytes": movement.sram_used,
                   "fits_budget": fits,
                   "matrix_commands": movement.matrix_count,
                   "transfer_commands": len(transfers),
                   "hbm_to_sram_bytes": sum(x.length_bytes for x in transfers
                                            if x.direction == "hbm_to_sram"),
                   "sram_to_hbm_bytes": sum(x.length_bytes for x in transfers
                                            if x.direction == "sram_to_hbm"),
                   "checker_passed": checked["passed"],
                   "elapsed_seconds": round(time.monotonic() - start, 2)}
            rows.append(row)
            print(f"  checked {row['matrix_commands']:,} MATRIX commands in "
                  f"{row['elapsed_seconds']:.2f}s", file=sys.stderr, flush=True)

    totals = {}
    for phase in ("prefill", "decode"):
        subset = [row for row in rows if row["phase"] == phase]
        totals[phase] = {
            "matmul_instances": sum(row["instances"] for row in subset),
            "matrix_commands": sum(row["instances"] * row["matrix_commands"] for row in subset),
            "transfer_commands": sum(row["instances"] * row["transfer_commands"] for row in subset),
            "hbm_to_sram_bytes": sum(row["instances"] * row["hbm_to_sram_bytes"] for row in subset),
            "sram_to_hbm_bytes": sum(row["instances"] * row["sram_to_hbm_bytes"] for row in subset),
            "peak_sram_bytes": max(row["sram_used_bytes"] for row in subset),
            "unfit_instances": sum(row["instances"] for row in subset if not row["fits_budget"]),
        }

    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "siliconbadgers.movement-sweep.v0",
               "planner_revision": revision,
               "python_version": sys.version.split()[0],
               "capture": captures,
               "target": {"sram_bytes": args.sram_bytes, "tile_m": tm,
                          "tile_n": tn, "tile_k": tk, "row_bytes": 512,
                          "a_bytes": 1, "b_bytes": 1, "c_bytes": 4,
                          "a_slots": 2, "reuse_after": "inputs_released",
                          "block_scaling": False},
               "rows": rows, "weighted_totals": totals}
    (OUTPUT / "summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    with (OUTPUT / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
