"""HBM latency / head-of-line study for the command-format progress counters (2026-10-25).

Standard library only. Models the Memory queue of the top-level command format
(rtl codex/command-format-spec, docs/command-format.md) issuing HBM transfers
over 32 AXI3 ports, and compares two ways of making transfer progress visible
to the Matrix and Vector queues:

* counter: the spec's single wrapping counter per queue. Milestones publish in
  command order, so a milestone cannot become visible before every earlier
  Memory milestone has (first row of j+1 waits for completion of j).
* scoreboard: per-command completion bits. A milestone is visible as soon as it
  happens.

Both schemes see the same per-burst latency samples (common random numbers),
so the difference between them is the cost of in-order publication.

Model (all times in 450 MHz AXI cycles, the F2 HBM IP clock):
* A transfer is split into AXI bursts of BURST_BYTES (16 beats x 32 B). Bursts
  are striped round-robin over the 32 ports. A port returns one 32 B beat per
  cycle, so the peak is 32 * 32 = 1024 B/cycle (460.8 GB/s).
* The Memory engine issues bursts in queue order, at most `issue_per_cycle`
  per cycle. A read burst needs a slot in the shared reorder buffer
  (`rob_slots` x BURST_BYTES, shared by all 32 ports per Zeb, 2026-10-25).
  Slots retire in issue order (`drain="in_order"`) or on arrival
  (`drain="arrival"`).
* Read data starts at issue + latency sample, then occupies the port for 16
  beats. Stores send their beats on the same port data bus (an HBM
  pseudo-channel is shared by reads and writes) and complete at the B
  response; they do not take reorder-buffer slots.
* A transfer waits for its wait targets before any of its bursts issue, and a
  blocked transfer blocks everything behind it (in-order issue).
* Consumers (Matrix, Vector) run their commands in order. A command starts
  when its transfer's milestone is visible: the first SRAM row for streaming
  consumers, completion otherwise. A streaming consumer cannot finish before
  its data has physically arrived.

Every latency here is a modeling assumption. The `prior` distribution uses
public Alveo U280 HBM measurements (Shuhai, Wang et al., FCCM 2020: 55/62/69
cycle page hit/closed/miss with the switch on, periodic refresh stalls) and
the F2 IP's 75 % page-hit setting. It is not a measurement of the F2 AMD
model; see README.md for the calibration plan.

Usage, from the repository root:
    python research/compiler-scheduling/2026-10-25-hbm-counter-hazard/sim.py [--out DIR] [--force] [--quick]

All checks run before anything is written. If a check fails the script exits
non-zero and leaves existing output untouched. Results go to the ignored
build/hbm-counter/2026-10-25/ folder; an existing results.json is replaced
only with --force.
"""

import argparse
import collections
import heapq
import json
import math
import multiprocessing
import os
import random
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DEFAULT_OUT = os.path.join(ROOT, "build", "hbm-counter", "2026-10-25")

PORTS = 32                 # AXI3 ports on the two-stack F2 HBM IP
BEAT_BYTES = 32            # 256-bit data bus
BURST_BEATS = 16           # AXI3 maximum burst
BURST_BYTES = BEAT_BYTES * BURST_BEATS
PEAK_BYTES_PER_CYCLE = PORTS * BEAT_BYTES
AXI_MHZ = 450
MIB = 1024 ** 2

# Workload sizes from the 2026-10-14 memory-traffic study (software PR #22):
# 19.27 MiB of DeltaNet state per sequence over 18 layers, and 1,881,276,416
# decode weight elements of which 508,559,360 are the output head.
STATE_BYTES_PER_LAYER = int(19.27 * MIB / 18)
WEIGHT_BYTES_PER_LAYER_INT4 = int((1881276416 - 508559360) / 24 * 0.5)


# --------------------------------------------------------------------------
# Latency models (first read beat after AR issue, AXI cycles)
# --------------------------------------------------------------------------
# A model returns (base, stall). The base part queues behind earlier bursts
# on the same port's data bus. The stall part (refresh, bank conflict) delays
# only that burst: other IDs on the port may return around it, as AXI allows
# for different transaction IDs.

def fixed_latency(cycles):
    def sample(rng, port):
        return cycles, 0.0
    sample.label = f"fixed({cycles})"
    return sample


def prior_latency(stall_scale=1.0, page_hit=0.75, refresh_prob=0.04):
    """Public-prior mixture. stall_scale stretches the refresh/stall tail."""
    hit, miss = 55, 69               # Shuhai Table VI, nearest mini-switch
    stall_max = 72 * stall_scale     # tRFCsb 160 ns = 72 cycles at 450 MHz

    def sample(rng, port):
        base = hit if rng.random() < page_hit else miss
        stall = rng.uniform(0, stall_max) if rng.random() < refresh_prob else 0.0
        return base, stall
    sample.label = f"prior(stall x{stall_scale:g})"
    return sample


# --------------------------------------------------------------------------
# Workloads
# --------------------------------------------------------------------------

Transfer = collections.namedtuple(
    "Transfer", "kind size consumer stream compute waits")
# kind: "load" (HBM->SRAM) or "store" (SRAM->HBM)
# consumer: "matrix", "vector" or None. A load with a consumer creates the
#   next command in that consumer queue.
# stream: consumer may start at the first-row milestone
# compute: consumer busy cycles for the command
# waits: list of (queue, command_index) whose completion must be visible
#   before this transfer issues (buffer reuse or producer dependency)


def single_stream(n_tiles, tile_bytes, consume_rate, buffers=2, stream=False):
    """One consumer reading n_tiles weight tiles through `buffers` SRAM buffers.

    consume_rate is Matrix bytes per AXI cycle; compute = tile_bytes / rate.
    """
    out = []
    for n in range(n_tiles):
        waits = [("matrix", n - buffers)] if n >= buffers else []
        out.append(Transfer("load", tile_bytes, "matrix", stream,
                            tile_bytes / consume_rate, waits))
    return out


def decode_mix(layers, tile_bytes, matrix_rate, vector_rate,
               weight_bytes=WEIGHT_BYTES_PER_LAYER_INT4,
               state_bytes=STATE_BYTES_PER_LAYER, batch=1,
               buffers=2, stream=False):
    """DeltaNet decode layers sharing one Memory queue.

    Per layer, in Memory-queue order: load recurrent state (Vector), load the
    layer's weights as tiles (Matrix), store the updated state after the
    Vector command finishes. This is the compiler order where state is
    prefetched first; other orders are a scheduling choice to compare.
    """
    out = []
    n_matrix = 0
    for layer in range(layers):
        sbytes = state_bytes * batch
        waits = [("vector", layer - 1)] if layer >= 1 else []
        out.append(Transfer("load", sbytes, "vector", False,
                            sbytes / vector_rate, waits))
        n_tiles = max(1, math.ceil(weight_bytes / tile_bytes))
        for _ in range(n_tiles):
            w = [("matrix", n_matrix - buffers)] if n_matrix >= buffers else []
            out.append(Transfer("load", tile_bytes, "matrix", stream,
                                tile_bytes / matrix_rate, w))
            n_matrix += 1
        out.append(Transfer("store", sbytes, None, False, 0.0,
                            [("vector", layer)]))
    return out


# --------------------------------------------------------------------------
# Simulator
# --------------------------------------------------------------------------

def simulate(transfers, scheme, latency, rob_slots=64, drain="in_order",
             issue_per_cycle=4, row_bytes=4096, seed=1):
    if scheme not in ("counter", "scoreboard"):
        raise ValueError(scheme)
    if drain not in ("in_order", "arrival"):
        raise ValueError(drain)
    rng = random.Random(seed)
    bus_free = [0.0] * PORTS   # one data bus per port, shared by reads and writes
    rob = collections.deque() if drain == "in_order" else []
    last_retire = 0.0
    issue_t = -1.0 / issue_per_cycle
    port_cursor = 0
    last_pub = 0.0
    queues = {"matrix": [], "vector": []}   # (start, end) per command
    queue_end = {"matrix": 0.0, "vector": 0.0}
    read_bytes = 0
    publish_delay = 0.0
    milestones = []   # (event, published) for each Memory milestone

    for t in transfers:
        ready = 0.0
        for q, idx in t.waits:
            if idx < 0:
                continue
            if idx >= len(queues[q]):
                raise ValueError(f"wait on future {q} command {idx}")
            ready = max(ready, queues[q][idx][1])
        n_bursts = max(1, math.ceil(t.size / BURST_BYTES))
        row_bursts = min(n_bursts, max(1, math.ceil(row_bytes / BURST_BYTES)))
        row_event = done_event = 0.0
        for i in range(n_bursts):
            nbytes = min(BURST_BYTES, t.size - i * BURST_BYTES)
            beats = max(1, math.ceil(nbytes / BEAT_BYTES))
            at = max(issue_t + 1.0 / issue_per_cycle, ready)
            port = port_cursor % PORTS
            port_cursor += 1
            base, stall = latency(rng, port)
            if t.kind == "load":
                if len(rob) >= rob_slots:
                    free = rob.popleft() if drain == "in_order" else heapq.heappop(rob)
                    at = max(at, free)
                start = max(at + base, bus_free[port])
                bus_free[port] = start + beats
                finish = start + beats + stall
                if drain == "in_order":
                    last_retire = max(last_retire, finish)
                    commit = last_retire
                    rob.append(commit)
                else:
                    commit = finish
                    heapq.heappush(rob, commit)
                read_bytes += nbytes
            else:
                wstart = max(at, bus_free[port])
                bus_free[port] = wstart + beats
                commit = wstart + beats + base + stall
                if i < row_bursts:
                    row_event = max(row_event, at)
            if t.kind == "load" and i < row_bursts:
                row_event = max(row_event, commit)
            done_event = max(done_event, commit)
            issue_t = at

        if scheme == "counter":
            row_pub = max(row_event, last_pub)
            done_pub = max(done_event, row_pub)
            last_pub = done_pub
        else:
            row_pub, done_pub = row_event, done_event
        publish_delay += (row_pub - row_event) + (done_pub - done_event)
        milestones.append((row_event, row_pub))
        milestones.append((done_event, done_pub))

        if t.consumer:
            q = t.consumer
            dep = row_pub if t.stream else done_pub
            start = max(queue_end[q], dep)
            end = start + t.compute
            if t.stream:
                end = max(end, done_event + 1)
            queues[q].append((start, end))
            queue_end[q] = end

    makespan = max([m[1] for m in milestones] + list(queue_end.values()))
    busy = {q: sum(e - s for s, e in cmds) for q, cmds in queues.items()}
    return {
        "makespan_cycles": makespan,
        "makespan_us": makespan / AXI_MHZ,
        "read_bytes": read_bytes,
        "hbm_read_util": read_bytes / (makespan * PEAK_BYTES_PER_CYCLE) if makespan else 0.0,
        "matrix_busy": busy["matrix"] / makespan if makespan else 0.0,
        "vector_busy": busy["vector"] / makespan if makespan else 0.0,
        "publish_delay_cycles": publish_delay,
        "milestones": len(milestones),
    }


def compare(transfers, latency, **kw):
    c = simulate(transfers, "counter", latency, **kw)
    s = simulate(transfers, "scoreboard", latency, **kw)
    loss = c["makespan_cycles"] / s["makespan_cycles"] - 1.0
    return {"counter": c, "scoreboard": s, "counter_loss": loss}


# --------------------------------------------------------------------------
# Checks (must all pass before anything is written)
# --------------------------------------------------------------------------

def run_checks():
    failures = []

    def check(name, ok, detail=""):
        if not ok:
            failures.append(f"{name}: {detail}")

    # 1. Fixed latency, one consumer: both schemes identical.
    wl = single_stream(64, 64 * 1024, consume_rate=512)
    r = compare(wl, fixed_latency(60), rob_slots=64)
    check("fixed latency equality", r["counter"]["makespan_cycles"] ==
          r["scoreboard"]["makespan_cycles"], str(r["counter_loss"]))

    # 2. One in-order consumer: no counter loss under any latency, with or
    #    without streaming (structural argument in README).
    for stream in (False, True):
        wl = single_stream(64, 64 * 1024, consume_rate=512, stream=stream)
        r = compare(wl, prior_latency(stall_scale=16), rob_slots=32)
        check(f"single consumer equality stream={stream}",
              abs(r["counter_loss"]) < 1e-12, str(r["counter_loss"]))

    # 3. Bandwidth ceiling: never above peak; near peak with a deep buffer.
    wl = single_stream(32, 1 * MIB, consume_rate=1e9)
    s = simulate(wl, "scoreboard", fixed_latency(60), rob_slots=1024,
                 issue_per_cycle=8)
    check("bandwidth <= peak", s["hbm_read_util"] <= 1.0 + 1e-9,
          str(s["hbm_read_util"]))
    check("bandwidth near peak", s["hbm_read_util"] > 0.95,
          str(s["hbm_read_util"]))

    # 4. Little's law with a small buffer: throughput = slots * 512 / (L + 16).
    L, slots = 100, 8
    s = simulate(single_stream(8, 4 * MIB, consume_rate=1e9), "scoreboard",
                 fixed_latency(L), rob_slots=slots, issue_per_cycle=8)
    expect = slots * BURST_BYTES / (L + BURST_BEATS) / PEAK_BYTES_PER_CYCLE
    check("little's law", abs(s["hbm_read_util"] / expect - 1) < 0.02,
          f"{s['hbm_read_util']:.4f} vs {expect:.4f}")

    # 5. Determinism with a fixed seed.
    wl = decode_mix(1, 256 * 1024, 512, 256, weight_bytes=4 * MIB)
    a = compare(wl, prior_latency(), seed=7)
    b = compare(wl, prior_latency(), seed=7)
    check("determinism", a == b)

    # 6. Shared Memory queue can only lose under the counter, never gain.
    r = compare(wl, prior_latency(stall_scale=4), rob_slots=32)
    check("counter never faster", r["counter_loss"] >= -1e-12,
          str(r["counter_loss"]))
    return failures


# --------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------

VARIANTS = {
    # name: (drain, scheme)
    "spec": ("in_order", "counter"),        # command-format spec as written
    "ordered_scoreboard": ("in_order", "scoreboard"),
    "arrival_counter": ("arrival", "counter"),
    "ideal": ("arrival", "scoreboard"),     # per-command bits, no ordered retire
}


def run_variants(transfers, latency, **kw):
    out = {}
    for name, (drain, scheme) in VARIANTS.items():
        out[name] = simulate(transfers, scheme, latency, drain=drain, **kw)
    ideal = out["ideal"]["makespan_cycles"]
    out["loss_vs_ideal"] = {k: out[k]["makespan_cycles"] / ideal - 1.0 for k in VARIANTS}
    out["counter_only_loss"] = (out["spec"]["makespan_cycles"]
                                / out["ordered_scoreboard"]["makespan_cycles"] - 1.0)
    return out


# Layers are scaled down for sweep speed with the weight:state ratio kept.
# Loss ratios are per-layer steady-state quantities, so the scale mostly
# changes absolute times.
SWEEP_SCALE = 0.25

LATENCIES = {"fixed62": fixed_latency(62), "prior": prior_latency(1),
             "prior_x4": prior_latency(4), "prior_x16": prior_latency(16)}


def _sweep_point(cfg):
    batch, stream, rate, tile, rob, lname = cfg
    wl = decode_mix(3, tile, matrix_rate=rate * PEAK_BYTES_PER_CYCLE,
                    vector_rate=0.25 * PEAK_BYTES_PER_CYCLE,
                    weight_bytes=int(WEIGHT_BYTES_PER_LAYER_INT4 * SWEEP_SCALE),
                    state_bytes=int(STATE_BYTES_PER_LAYER * SWEEP_SCALE),
                    batch=batch, stream=stream)
    r = run_variants(wl, LATENCIES[lname], rob_slots=rob)
    row = {"batch": batch, "stream": stream, "matrix_rate_vs_peak": rate,
           "tile_kib": tile // 1024, "rob_slots": rob,
           "rob_kib": rob * BURST_BYTES // 1024, "latency": lname,
           "counter_only_loss": r["counter_only_loss"]}
    for k in VARIANTS:
        row[f"util_{k}"] = r[k]["hbm_read_util"]
        row[f"loss_{k}"] = r["loss_vs_ideal"][k]
    return row


def sweep(quick=False):
    rob_list = [32, 64, 128, 256, 512]
    tiles = [16 * 1024, 64 * 1024, 256 * 1024, 1 * MIB]
    lats = [(k, v) for k, v in LATENCIES.items() if k != "prior_x4"]
    # Matrix consumption relative to HBM peak: below 1 means the array is the
    # bottleneck (PR #22 decode finding), above 1 means HBM is.
    rates = [0.25, 4.0]
    batches, streams = (1, 32), (False, True)
    if quick:
        rob_list, tiles, rates = [32, 128, 512], [64 * 1024, 1 * MIB], [4.0]
        lats, batches = lats[:2], (1,)
    configs = [(b, st, ra, ti, ro, ln) for b in batches for st in streams
               for ra in rates for ti in tiles for ro in rob_list
               for ln, _ in lats]
    if quick:
        return [_sweep_point(c) for c in configs]
    with multiprocessing.Pool() as pool:
        return pool.map(_sweep_point, configs, chunksize=4)


def write_outputs(out, rows, force):
    os.makedirs(out, exist_ok=True)
    target = os.path.join(out, "results.json")
    if os.path.exists(target) and not force:
        raise FileExistsError(f"{target} exists; pass --force to replace it")
    fd, tmp = tempfile.mkstemp(dir=out, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump({"rows": rows}, f, indent=1)
    os.replace(tmp, target)
    csv_path = os.path.join(out, "sweep.csv")
    with open(csv_path, "w") as f:
        keys = list(rows[0])
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")
    return target


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true", help="small sweep for a smoke run")
    args = p.parse_args(argv)
    failures = run_checks()
    if failures:
        for f in failures:
            print("CHECK FAILED", f, file=sys.stderr)
        return 1
    print("all checks passed")
    if os.path.exists(os.path.join(args.out, "results.json")) and not args.force:
        print(f"{args.out}/results.json exists; pass --force to replace it", file=sys.stderr)
        return 1
    rows = sweep(quick=args.quick)
    try:
        path = write_outputs(args.out, rows, args.force)
    except FileExistsError as e:
        print(e, file=sys.stderr)
        return 1
    print(f"wrote {len(rows)} rows to {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
