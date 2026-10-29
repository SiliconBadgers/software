"""Data movement and ordering for tiled matmuls (issue 28, work item E).

Turns a tiled matmul into the command format's TRANSFER and MATRIX records:
SRAM buffer assignment, HBM <-> SRAM transfers, and the queue-progress wait
targets that keep every input valid and stop a buffer from being reused while
it is still being read or written. `movement_check.py` verifies the result
independently.

Everything here is a symbolic plan. Addresses are SRAM byte offsets and HBM
offsets inside host-prepared target buffers; nothing is encoded or executed.
Target facts that are still open are explicit `TargetMemory` fields, and each
report lists the assumptions it used.

Command-format milestones (rtl codex/command-format-spec, docs/command-format.md),
for the j-th command of a queue, counting from zero:
  Memory  2j+1 first SRAM row, 2j+2 transfer complete
  Matrix  3j+1 SRAM inputs released, 3j+2 next K tile may launch,
          3j+3 accumulation finished and output writes completed
A wait target of zero disables the wait. Counters are treated as unbounded
integers; wraparound is not modeled yet.
"""
from collections import namedtuple
from dataclasses import dataclass
import argparse
import hashlib
import json
import sys
import zlib

from .plan import MatmulPlan

SCHEMA = "siliconbadgers.compiler-movement.v0"
MIB = 1024 * 1024

# Release events a new load may wait on before overwriting an input buffer.
REUSE_EVENTS = {
    # 3j+1 means every Matrix SRAM read has finished (confirmed by Zeb, 2026-10-29).
    "inputs_released": 1,
    # 3j+3 waits for the output writes too; kept as a conservative option.
    "complete": 3,
}


class MovementError(ValueError):
    """The plan cannot be built under the given target assumptions."""


@dataclass(frozen=True)
class TargetMemory:
    """Open target facts, kept configurable until B settles them.

    Defaults: 512 B SRAM rows on the proposed banked path and full-row transfers
    (B update, 2026-10-28). The 8 MiB SRAM size is not a confirmed usable
    budget. Element sizes are target storage formats, not the captured source
    formats: INT8 weights and activations, INT8 requantized output.
    """
    sram_bytes: int = 8 * MIB
    row_bytes: int = 512
    a_bytes: int = 1
    b_bytes: int = 1
    c_bytes: int = 1
    a_slots: int = 2
    layout: str = "row_major"
    reuse_after: str = "inputs_released"

    def validate(self):
        for name in ("sram_bytes", "row_bytes", "a_bytes", "b_bytes", "c_bytes", "a_slots"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise MovementError(f"{name} must be a positive integer")
        if self.sram_bytes % self.row_bytes:
            raise MovementError("SRAM size must be a whole number of rows")
        if self.layout != "row_major":
            raise MovementError(f"layout {self.layout!r} is not supported yet")
        if self.reuse_after not in REUSE_EVENTS:
            raise MovementError(f"reuse_after must be one of {sorted(REUSE_EVENTS)}")

    def assumptions(self):
        return [
            f"SRAM budget {self.sram_bytes} B (configurable; 8 MiB is not confirmed)",
            f"{self.row_bytes} B SRAM rows; transfers move whole rows only",
            "row-major logical layout in HBM target buffers and SRAM: A M x K, B K x N, C M x N",
            f"target element bytes A={self.a_bytes} B={self.b_bytes} C={self.c_bytes}; "
            "the host prepares operands in these formats, captured formats are provenance only",
            "HBM target buffers are padded to whole SRAM rows",
            f"an input buffer is overwritten only after its readers reach milestone "
            f"3j+{REUSE_EVENTS[self.reuse_after]} ({self.reuse_after})",
            "C is stored after the last writer's 3j+3, which waits until the output writes "
            "are committed to SRAM (confirmed by Zeb, 2026-10-29)",
        ]


# --------------------------------------------------------------------------
# Tiles. The fields follow the TiledMatmul / MatmulTile shape agreed with D
# (Anu, 2026-10-28): the plan, maximum tile sizes and an iterator in run order
# with k innermost. StubTiledMatmul stands in until D's tiler lands.
# --------------------------------------------------------------------------

MatmulTile = namedtuple("MatmulTile", "tensor_id m0 n0 k0 m n k clear_acc write_output")


@dataclass(frozen=True)
class StubTiledMatmul:
    plan: MatmulPlan
    tile_m: int = 8
    tile_n: int = 16
    tile_k: int = 8

    def tiles(self):
        p = self.plan
        for m0 in range(0, p.m, self.tile_m):
            for n0 in range(0, p.n, self.tile_n):
                for k0 in range(0, p.k, self.tile_k):
                    yield MatmulTile(p.tensor_id, m0, n0, k0,
                                     min(self.tile_m, p.m - m0), min(self.tile_n, p.n - n0),
                                     min(self.tile_k, p.k - k0),
                                     k0 == 0, k0 + self.tile_k >= p.k)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------

Transfer = namedtuple(
    "Transfer",
    "index direction operand hbm_offset sram_address length_bytes wait_matrix wait_vector")
Matrix = namedtuple(
    "Matrix",
    "index a_base b_base c_base m n k transpose_a transpose_b clear_acc write_output "
    "wait_memory wait_vector")
Buffer = namedtuple("Buffer", "name operand sram_address size_bytes row_pitch_bytes")


def round_up(value, multiple):
    return -(-value // multiple) * multiple


def memory_done(j):
    return 2 * j + 2


def matrix_milestone(i, event):
    return 3 * i + event


class MovementPlan:
    """Buffers, the Memory queue and a re-iterable Matrix queue for one matmul."""

    def __init__(self, tiled, target=TargetMemory()):
        target.validate()
        self.tiled, self.target, self.plan = tiled, target, tiled.plan
        p, t = self.plan, target
        for name in ("tile_m", "tile_n", "tile_k"):
            if type(getattr(tiled, name)) is not int or getattr(tiled, name) < 1:
                raise MovementError(f"{name} must be a positive integer")
        self.pitch = {"a": p.k * t.a_bytes, "b": p.n * t.b_bytes, "c": p.n * t.c_bytes}
        self.tensor_bytes = {"a": p.m * p.k * t.a_bytes, "b": p.k * p.n * t.b_bytes,
                             "c": p.m * p.n * t.c_bytes}
        b_size = round_up(self.tensor_bytes["b"], t.row_bytes)
        c_size = round_up(self.tensor_bytes["c"], t.row_bytes)
        panel = round_up(tiled.tile_m * self.pitch["a"], t.row_bytes)
        address = 0
        self.buffers = []
        for name, operand, size in [("b", "b", b_size), ("c", "c", c_size)] + [
                (f"a{slot}", "a", panel) for slot in range(t.a_slots)]:
            self.buffers.append(Buffer(name, operand, address, size, self.pitch[operand]))
            address += size
        if address > t.sram_bytes:
            raise MovementError(
                f"needs {address} B of SRAM (B and C resident, {t.a_slots} A panels) but the "
                f"budget is {t.sram_bytes} B; streaming B or C is not implemented")
        self.sram_used = address
        self.panel_bytes = panel
        self._index_panels()
        self.transfers = self._transfers()
        # Host-prepared HBM target buffers, padded so whole-row transfers stay inside.
        self.hbm_bytes = {op: max([self.tensor_bytes[op]] + [
            x.hbm_offset + x.length_bytes for x in self.transfers if x.operand == op])
            for op in ("a", "b", "c")}

    def _index_panels(self):
        """One pass over the tiles: last Matrix command reading each A panel."""
        tm = self.tiled.tile_m
        self.panel_last, self.matrix_count = [], 0
        previous = -1
        last_writer = -1
        for i, tile in enumerate(self.tiled.tiles()):
            self._check_tile(tile)
            panel = tile.m0 // tm
            if panel < previous or panel > previous + 1:
                raise MovementError("tiles must visit A panels (m blocks) in order, each once")
            if panel == previous + 1:
                self.panel_last.append(i)
            else:
                self.panel_last[panel] = i
            previous = panel
            if tile.write_output:
                last_writer = i
            self.matrix_count = i + 1
        if self.matrix_count == 0 or last_writer < 0:
            raise MovementError("the tiled matmul has no tiles or never writes its output")
        self.last_writer = last_writer

    def _check_tile(self, tile):
        p, tm = self.plan, self.tiled.tile_m
        if tile.tensor_id != p.tensor_id:
            raise MovementError("tile belongs to a different matmul")
        if (tile.m0 % tm or tile.m < 1 or tile.n < 1 or tile.k < 1
                or tile.m0 + tile.m > p.m or tile.n0 + tile.n > p.n or tile.k0 + tile.k > p.k
                or tile.m > tm):
            raise MovementError(f"tile outside the matmul or crossing an A panel: {tile}")

    def _slot(self, panel):
        return self.buffers[2 + panel % self.target.a_slots]

    def _transfers(self):
        t, p = self.target, self.plan
        release = REUSE_EVENTS[t.reuse_after]
        out = [Transfer(0, "hbm_to_sram", "b", 0, self.buffers[0].sram_address,
                        self.buffers[0].size_bytes, 0, 0)]
        for panel in range(len(self.panel_last)):
            wait = 0
            if panel >= t.a_slots:
                wait = matrix_milestone(self.panel_last[panel - t.a_slots], release)
            slot = self._slot(panel)
            out.append(Transfer(len(out), "hbm_to_sram", "a",
                                panel * self.tiled.tile_m * self.pitch["a"],
                                slot.sram_address, slot.size_bytes, wait, 0))
        c = self.buffers[1]
        out.append(Transfer(len(out), "sram_to_hbm", "c", 0, c.sram_address, c.size_bytes,
                            matrix_milestone(self.last_writer, 3), 0))
        return out

    def matrix_commands(self):
        """Matrix queue in run order; a fresh generator on every call."""
        p, t, tm = self.plan, self.target, self.tiled.tile_m
        b, c = self.buffers[0], self.buffers[1]
        for i, tile in enumerate(self.tiled.tiles()):
            panel = tile.m0 // tm
            slot = self._slot(panel)
            a_base = slot.sram_address + (tile.m0 - panel * tm) * self.pitch["a"] + tile.k0 * t.a_bytes
            b_base = b.sram_address + tile.k0 * self.pitch["b"] + tile.n0 * t.b_bytes
            c_base = c.sram_address + tile.m0 * self.pitch["c"] + tile.n0 * t.c_bytes
            # Panel loads are Memory commands 1.., after the resident B load.
            yield Matrix(i, a_base, b_base, c_base, tile.m, tile.n, tile.k, False, False,
                         tile.clear_acc, tile.write_output, memory_done(1 + panel), 0)

    def digest(self):
        """SHA-256 over the full command stream, for determinism checks."""
        h = hashlib.sha256()
        for command in self.transfers:
            h.update(repr(tuple(command)).encode())
        for command in self.matrix_commands():
            h.update(repr(tuple(command)).encode())
        return h.hexdigest()

    def summary(self, check=None, sample=4):
        t = self.target
        loaded = sum(x.length_bytes for x in self.transfers if x.direction == "hbm_to_sram")
        stored = sum(x.length_bytes for x in self.transfers if x.direction == "sram_to_hbm")
        matrix = self.matrix_commands()
        return {
            "schema": SCHEMA,
            "execution_status": "symbolic_only",
            "tensor_id": self.plan.tensor_id,
            "dimensions": {"m": self.plan.m, "n": self.plan.n, "k": self.plan.k},
            "tiles": {"tile_m": self.tiled.tile_m, "tile_n": self.tiled.tile_n,
                      "tile_k": self.tiled.tile_k, "tiler": type(self.tiled).__name__},
            "target": {name: getattr(t, name) for name in TargetMemory.__dataclass_fields__},
            "assumptions": t.assumptions(),
            "buffers": [b._asdict() for b in self.buffers],
            "sram_used_bytes": self.sram_used,
            "hbm_buffer_bytes": self.hbm_bytes,
            "counts": {"transfers": len(self.transfers), "matrix": self.matrix_count,
                       "a_panels": len(self.panel_last)},
            "bytes": {"hbm_to_sram": loaded, "sram_to_hbm": stored},
            "command_digest_sha256": self.digest(),
            "transfers_head": [x._asdict() for x in self.transfers[:sample]],
            "matrix_head": [next(matrix)._asdict() for _ in range(min(sample, self.matrix_count))],
            "check": check,
        }


def main(argv=None):
    """Plan and check data movement for one captured matmul."""
    from cdfg import Graph
    from .lower import lower_matmul
    from .movement_check import check

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("capture", help="captured phase JSON or JSON.gz")
    parser.add_argument("--phase", required=True, choices=("prefill", "decode"))
    parser.add_argument("--tensor-id", type=int, required=True)
    parser.add_argument("--tile", default="8,16,8", help="tile_m,tile_n,tile_k (stub tiler)")
    parser.add_argument("--sram-bytes", type=int, default=TargetMemory.sram_bytes)
    parser.add_argument("--reuse-after", choices=sorted(REUSE_EVENTS), default="inputs_released")
    parser.add_argument("--commands-out", help="write every command as JSON lines here")
    args = parser.parse_args(argv)
    try:
        tm, tn, tk = (int(v) for v in args.tile.split(","))
        plan = lower_matmul(Graph(args.capture, args.phase), args.tensor_id)
        target = TargetMemory(sram_bytes=args.sram_bytes, reuse_after=args.reuse_after)
        movement = MovementPlan(StubTiledMatmul(plan, tm, tn, tk), target)
        result = check(movement)
    except (EOFError, OSError, KeyError, TypeError, ValueError, zlib.error) as error:
        parser.error(str(error))
    if args.commands_out:
        with open(args.commands_out, "w", encoding="utf-8") as handle:
            for command in movement.transfers:
                handle.write(json.dumps({"queue": "memory", **command._asdict()}) + "\n")
            for command in movement.matrix_commands():
                handle.write(json.dumps({"queue": "matrix", **command._asdict()}) + "\n")
    sys.stdout.write(json.dumps(movement.summary(result), indent=2, sort_keys=True) + "\n")
    if not result["passed"]:
        print(f"{len(result['violations'])} ordering violations", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
