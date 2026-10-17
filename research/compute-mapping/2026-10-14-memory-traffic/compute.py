"""Memory-traffic estimates for the Qwen3.5-2B mapping track (2026-10-14).

Standard library only. Reads the frozen op-dependency-map tables and prints
four logical-byte studies:

1. INT32 partial-sum traffic across K tiles, compared with Abhinav Gummadi's
   2026-10-12 Timeloop baseline (constants below, from 34f09ed on
   abhinav/timeloop-accelergy-qwen35).
2. DeltaNet recurrent/conv state traffic per decode step versus weight traffic,
   by batch size and SRAM-resident fraction.
3. Matrix-to-vector handoff bytes for the FFN and output head under three
   handoff levels.
4. Decode weight-feed demand versus SRAM port width per candidate array.

Every number is a logical byte count from captured tensor shapes. None is a
measured SRAM/HBM transfer, cycle count or hardware result.

Usage, from the repository root:
    python research/compute-mapping/2026-10-14-memory-traffic/compute.py
"""

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEPMAP = os.path.join(HERE, "..", "2026-10-02-op-dependency-map", "out")
MIB = 1024 ** 2

# Precision assumptions (architecture compute-team-decisions, 70ee075).
ACC_BYTES = 4          # INT32 accumulation / partial sums
ACT_BYTES = 1          # INT8 activations after requantization
WEIGHT_BYTES = {"INT8": 1.0, "INT4": 0.5}

# Abhinav Gummadi's Timeloop baseline (research/timeloop-accelergy-qwen35/REPORT.md,
# 34f09ed on abhinav/timeloop-accelergy-qwen35).
# K groups are the Timeloop mapping's K split; c_updates are its reported C
# update counts, which this script reproduces as k_groups * M * N.
TIMELOOP_CASES = [
    # name, M, K, N, k_groups, timeloop_c_updates, timeloop_a_reads
    ("ffn_gate/up decode", 6144, 2048, 1, 2, 12288, 12582912),
    ("ffn_gate/up pp128 prefill", 6144, 2048, 128, 2, 1572864, 12582912),
    ("ffn_out decode", 2048, 6144, 1, 6, 12288, 12582912),
    ("ffn_out pp128 prefill", 2048, 6144, 128, 6, 1572864, 12582912),
    ("output head", 248320, 2048, 1, 2, 496640, 508559360),
]
ARRAY_K = 8  # 8x8 candidate RTL (adam/systolic-array 4926042) sums 8 products

# Candidate arrays and SRAM port widths (Software meeting, 2026-10-11).
ARRAYS = [(8, 8), (16, 32), (32, 64)]  # (side, port bytes/cycle)


def load_nodes(run):
    with open(os.path.join(DEPMAP, run, "nodes.csv"), newline="") as f:
        return list(csv.DictReader(f))


def load_summary(run):
    with open(os.path.join(DEPMAP, run, "summary.json")) as f:
        return json.load(f)


def weight_count(nodes):
    return sum(int(r["macs"]) for r in nodes if r["category"] == "weight_matrix"
               and int(r["N"] or 0) == 1)


def find(nodes, name):
    for r in nodes:
        if r["name"] == name:
            return int(r["M"]), int(r["K"]), int(r["N"])
    raise KeyError(name)


def partial_sums(checks):
    print("== 1. INT32 partial sums across K tiles ==")
    print("keep = accumulators held in the array (clear_acc/write_output flags)")
    print("timeloop = spill per Timeloop K group; per8 = spill every 8-wide K tile")
    rows = []
    for name, M, K, N, kg, c_tl, a_tl in TIMELOOP_CASES:
        out = M * N
        if kg * out != c_tl:
            checks.append(f"FAIL {name}: k_groups*M*N {kg * out} != Timeloop {c_tl}")
        else:
            checks.append(f"ok   {name}: k_groups*M*N matches Timeloop C updates")
        weights = a_tl  # Timeloop A reads, 8-bit words = bytes
        keep = out * ACC_BYTES
        spill_tl = (kg + (kg - 1)) * out * ACC_BYTES
        t = K // ARRAY_K
        spill_8 = (t + (t - 1)) * out * ACC_BYTES
        rows.append(dict(case=name, weight_bytes=weights, keep=keep,
                         spill_timeloop=spill_tl, spill_per8=spill_8))
        print(f"  {name:27s} weights {weights / 1e6:8.2f} MB | keep {keep / 1e6:7.3f} MB"
              f" | timeloop {spill_tl / 1e6:7.3f} MB ({spill_tl / weights * 100:6.1f}%)"
              f" | per8 {spill_8 / 1e6:8.1f} MB ({spill_8 / weights * 100:7.1f}%)")
    return rows


def deltanet_state(summary, weights):
    print("\n== 2. DeltaNet state per decode step ==")
    dn = summary["layer_types"]["deltanet"]
    per_layer_r = dn["state_read_bytes_per_layer"]
    per_layer_w = dn["state_write_bytes_per_layer"]
    layers = len(dn["layers"])
    state = per_layer_r * layers
    print(f"  {layers} layers x {per_layer_r} B = {state / MIB:.2f} MiB read, "
          f"{per_layer_w * layers / MIB:.2f} MiB written per sequence")
    rows = []
    for batch in (1, 8, 32):
        for keep in (0.0, 0.5, 1.0):
            hbm = (per_layer_r + per_layer_w) * layers * batch * (1 - keep)
            sram = state * batch * keep
            row = dict(batch=batch, keep=keep, hbm_bytes=hbm, sram_bytes=sram)
            for prec, b in WEIGHT_BYTES.items():
                row[f"vs_{prec}"] = hbm / (weights * b)
            rows.append(row)
            print(f"  batch {batch:2d} keep {int(keep * 100):3d}%: HBM {hbm / 1e6:8.1f} MB,"
                  f" SRAM {sram / MIB:6.1f} MiB, vs INT8 weights {row['vs_INT8'] * 100:5.1f}%,"
                  f" vs INT4 {row['vs_INT4'] * 100:5.1f}%")
    return rows


def handoffs(decode_nodes, prefill_nodes):
    print("\n== 3. Matrix-to-vector handoff (per layer FFN, output head) ==")
    print("L0 INT32 through SRAM; L1 requant at matrix write (command spec); L2 direct")
    rows = []
    for label, nodes in (("decode", decode_nodes), ("pp512 prefill", prefill_nodes)):
        Mg, Kg, N = find(nodes, "ffn_gate-0")
        Md, Kd, _ = find(nodes, "ffn_out-0")
        mat_out = 2 * Mg * N          # gate + up outputs
        vec_out = Kd * N              # SwiGLU output feeds ffn_down
        l0 = mat_out * (ACC_BYTES * 2) + vec_out * ACT_BYTES * 2
        l1 = mat_out * ACT_BYTES * 2 + vec_out * ACT_BYTES * 2
        l2 = vec_out * ACT_BYTES * 2
        w = 2 * Mg * Kg + Md * Kd
        rows.append(dict(case=f"ffn {label}", L0=l0, L1=l1, L2=l2, weight_bytes=w))
        print(f"  ffn {label:13s}: L0 {l0 / 1e6:7.3f} MB  L1 {l1 / 1e6:7.3f}  L2 {l2 / 1e6:7.3f}"
              f"  | INT8 weights {w / 1e6:.1f} MB (L0 {l0 / w * 100:.1f}%, L1 {l1 / w * 100:.1f}%,"
              f" L2 {l2 / w * 100:.1f}%)")
    Mh, Kh, Nh = find(decode_nodes, "result_output")
    out = Mh * Nh
    rows.append(dict(case="output head", L0=out * ACC_BYTES * 2, L1=out * ACT_BYTES * 2,
                     L2=0, weight_bytes=Mh * Kh))
    print(f"  output head       : L0 {out * 8 / 1e6:7.3f} MB  L1 {out * 2 / 1e6:7.3f}  L2 0"
          f"  | INT8 weights {Mh * Kh / 1e6:.0f} MB (prefill projects only the last token)")
    return rows


def decode_feed():
    print("\n== 4. Decode weight feed vs SRAM port (N=1, each weight used once) ==")
    rows = []
    for side, port in ARRAYS:
        for prec, b in WEIGHT_BYTES.items():
            need = side * side * b
            util = min(1.0, port / need)
            rows.append(dict(array=f"{side}x{side}", port=port, precision=prec,
                             need_bytes_per_cycle=need, util_bound=util))
            print(f"  {side}x{side} port {port:2d} B/cyc {prec}: needs {need:5.0f} B/cyc"
                  f" -> utilization bound {util * 100:5.1f}%")
    print("  HBM 460 GB/s (F2 setting in explorer MODEL.md) = "
          + ", ".join(f"{460e9 / f:.0f} B/cyc at {f / 1e6:.0f} MHz"
                      for f in (250e6, 500e6, 1e9)))
    return rows


def main():
    checks = []
    decode = load_nodes("pp128-decode")
    prefill512 = load_nodes("pp512-prefill")
    summary = load_summary("pp128-decode")

    weights = weight_count(decode)
    expected = 1881276416
    checks.append(("ok   " if weights == expected else "FAIL ")
                  + f"decode weight-matrix elements {weights} (expected {expected})")

    result = dict(
        weights_elements=weights,
        partial_sums=partial_sums(checks),
        deltanet_state=deltanet_state(summary, weights),
        handoffs=handoffs(decode, prefill512),
        decode_feed=decode_feed(),
    )
    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump(result, f, indent=1)

    print("\n== checks ==")
    for c in checks:
        print("  " + c)
    if any(c.startswith("FAIL") for c in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
