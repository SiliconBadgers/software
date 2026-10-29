"""Independent ordering check for a data-movement plan (issue 28, work item E).

Reads only the emitted commands, the tiles they claim to implement and the
target facts. It does not reuse the planner's buffer bookkeeping, so a planner
bug shows up as a violation instead of being repeated. Checked:

* every transfer moves whole SRAM rows inside the SRAM budget and the HBM extent
* every MATRIX input was loaded by a transfer that has completed before the
  command may start (its Memory wait covers 2j+2), and that transfer brought
  the right HBM bytes to the addresses the command reads
* no later load overwrites an input before the reader's release milestone
* the output store waits for every overlapping MATRIX write (3j+3), and the
  store maps each written tile to the right HBM bytes
* MATRIX writes never land in a range that a transfer loads
* the two queues can always make progress (no wait cycle)
"""
from bisect import bisect_left, bisect_right
from collections import defaultdict

from .movement import REUSE_EVENTS

MAX_REPORTED = 50


def _overlap(a0, a1, b0, b1):
    return a0 < b1 and b0 < a1


class _Result:
    def __init__(self):
        self.violations, self.total = [], 0

    def add(self, kind, detail):
        self.total += 1
        if len(self.violations) < MAX_REPORTED:
            self.violations.append({"kind": kind, "detail": detail})


def check(movement):
    """Return {"passed", "violations", "violation_count", "observed"}."""
    target, plan, tiled = movement.target, movement.plan, movement.tiled
    row, sram = target.row_bytes, target.sram_bytes
    release = REUSE_EVENTS[target.reuse_after]
    elem = {"a": target.a_bytes, "b": target.b_bytes, "c": target.c_bytes}
    pitch = {"a": plan.k * elem["a"], "b": plan.n * elem["b"], "c": plan.n * elem["c"]}
    tensor = {"a": plan.m * plan.k * elem["a"], "b": plan.k * plan.n * elem["b"],
              "c": plan.m * plan.n * elem["c"]}
    result = _Result()
    transfers = list(movement.transfers)
    for op, size in tensor.items():
        if movement.hbm_bytes.get(op, 0) < size:
            result.add("bounds", f"declared HBM buffer for {op} is smaller than the tensor")

    # Transfer shape, alignment and bounds.
    loads_by_range = defaultdict(list)   # (start, end) -> sorted load indices
    stores = []
    for j, x in enumerate(transfers):
        if x.index != j:
            result.add("order", f"transfer {j} carries index {x.index}")
        if x.wait_vector:
            result.add("deadlock", f"transfer {j} waits on the Vector queue, which is empty")
        if x.sram_address % row or x.length_bytes % row or x.length_bytes <= 0:
            result.add("alignment", f"transfer {j} is not whole {row} B rows: {x}")
        if x.sram_address < 0 or x.sram_address + x.length_bytes > sram:
            result.add("bounds", f"transfer {j} leaves the {sram} B SRAM: {x}")
        extent = movement.hbm_bytes.get(x.operand, 0)
        if x.operand not in tensor or x.hbm_offset < 0 or x.hbm_offset + x.length_bytes > extent:
            result.add("bounds", f"transfer {j} leaves operand {x.operand}'s padded HBM buffer")
        span = (x.sram_address, x.sram_address + x.length_bytes)
        if x.direction == "hbm_to_sram":
            loads_by_range[span].append(j)
        elif x.direction == "sram_to_hbm":
            stores.append(j)
        else:
            result.add("order", f"transfer {j} has unknown direction {x.direction!r}")
    ranges = list(loads_by_range)

    def source_load(lo, hi, limit):
        """Latest load index <= limit whose destination overlaps [lo, hi)."""
        best = None
        for span in ranges:
            if _overlap(lo, hi, *span):
                idx = loads_by_range[span]
                pos = bisect_right(idx, limit) - 1
                if pos >= 0 and (best is None or idx[pos] > best):
                    best = idx[pos]
        return best

    def next_load(lo, hi, after):
        """Earliest load index > after whose destination overlaps [lo, hi)."""
        best = None
        for span in ranges:
            if _overlap(lo, hi, *span):
                idx = loads_by_range[span]
                pos = bisect_right(idx, after)
                if pos < len(idx) and (best is None or idx[pos] < best):
                    best = idx[pos]
        return best

    def footprint(base, rows, cols, operand):
        return base, base + (rows - 1) * pitch[operand] + cols * elem[operand]

    def check_read(i, cmd, operand, base, rows, cols, hbm_element):
        lo, hi = footprint(base, rows, cols, operand)
        limit = (cmd.wait_memory - 2) // 2 if cmd.wait_memory >= 2 else -1
        j = source_load(lo, hi, limit)
        if j is None:
            result.add("raw", f"matrix {i} reads {operand} [{lo}, {hi}) before any covering "
                       f"load completes (wait_memory={cmd.wait_memory})")
            return
        x = transfers[j]
        if x.operand != operand or not (x.sram_address <= lo and hi <= x.sram_address + x.length_bytes):
            result.add("raw", f"matrix {i} reads {operand} [{lo}, {hi}) but the latest completed "
                       f"load there is transfer {j} ({x.operand})")
            return
        expected = x.sram_address + hbm_element * elem[operand] - x.hbm_offset
        if expected != base:
            result.add("data", f"matrix {i} reads {operand} at {base}, but transfer {j} put the "
                       f"tile's first element at {expected}")
        later = next_load(lo, hi, j)
        if later is not None and transfers[later].wait_matrix < 3 * i + release:
            result.add("war", f"transfer {later} may overwrite {operand} [{lo}, {hi}) before "
                       f"matrix {i} reaches 3j+{release} (waits {transfers[later].wait_matrix}, "
                       f"needs {3 * i + release})")

    # Walk the Matrix queue once, advancing the Memory queue as its waits allow.
    memory_next, memory_counter = 0, 0
    writers = defaultdict(int)          # store index -> latest overlapping writer milestone
    observed = {"matrix": 0, "loads": sum(len(v) for v in loads_by_range.values()),
                "stores": len(stores), "max_memory_lead": 0}
    tiles = tiled.tiles()
    for i, cmd in enumerate(movement.matrix_commands()):
        tile = next(tiles, None)
        if tile is None:
            result.add("order", f"matrix {i} has no matching tile")
            break
        if cmd.index != i or (cmd.m, cmd.n, cmd.k) != (tile.m, tile.n, tile.k) \
                or (cmd.clear_acc, cmd.write_output) != (tile.clear_acc, tile.write_output) \
                or cmd.transpose_a or cmd.transpose_b:
            result.add("order", f"matrix {i} does not match tile {tile}")
        if cmd.wait_vector:
            result.add("deadlock", f"matrix {i} waits on the Vector queue, which is empty")
        # Matrix commands 0..i-1 have published all three milestones.
        while memory_next < len(transfers) and transfers[memory_next].wait_matrix <= 3 * i:
            memory_next += 1
            memory_counter = 2 * memory_next
        if memory_counter < cmd.wait_memory:
            result.add("deadlock", f"matrix {i} waits for Memory {cmd.wait_memory}, but transfer "
                       f"{memory_next} first needs Matrix {transfers[memory_next].wait_matrix}")
            break
        observed["max_memory_lead"] = max(observed["max_memory_lead"],
                                          memory_next - (cmd.wait_memory // 2))
        check_read(i, cmd, "a", cmd.a_base, tile.m, tile.k, tile.m0 * plan.k + tile.k0)
        check_read(i, cmd, "b", cmd.b_base, tile.k, tile.n, tile.k0 * plan.n + tile.n0)
        lo, hi = footprint(cmd.c_base, tile.m, tile.n, "c")
        if lo < 0 or hi > sram:
            result.add("bounds", f"matrix {i} writes outside SRAM")
        for span in ranges:
            if _overlap(lo, hi, *span):
                result.add("waw", f"matrix {i} writes C into [{lo}, {hi}), a load destination")
                break
        if cmd.write_output:
            hbm = (tile.m0 * plan.n + tile.n0) * elem["c"]
            covered = False
            for s in stores:
                x = transfers[s]
                if _overlap(lo, hi, x.sram_address, x.sram_address + x.length_bytes):
                    writers[s] = max(writers[s], 3 * i + 3)
                    if x.sram_address + hbm - x.hbm_offset == cmd.c_base:
                        covered = True
            if not covered:
                result.add("data", f"matrix {i} output at {cmd.c_base} is not stored to its HBM bytes")
        observed["matrix"] = i + 1
    else:
        if next(tiles, None) is not None:
            result.add("order", "tiles remain after the last matrix command")
        while memory_next < len(transfers):
            if transfers[memory_next].wait_matrix > 3 * observed["matrix"]:
                result.add("deadlock", f"transfer {memory_next} waits for Matrix "
                           f"{transfers[memory_next].wait_matrix}, beyond the last command")
                break
            memory_next += 1
    for s in stores:
        if transfers[s].wait_matrix < writers[s]:
            result.add("store", f"transfer {s} stores C before its last writer's 3j+3 "
                       f"(waits {transfers[s].wait_matrix}, needs {writers[s]})")
    return {"passed": result.total == 0, "violation_count": result.total,
            "violations": result.violations, "observed": observed,
            "reuse_after": target.reuse_after}
