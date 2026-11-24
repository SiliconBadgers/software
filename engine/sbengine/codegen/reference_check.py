"""Execute tiled matmul plans on values and compare with the reference (issue 28, item C).

A tile is any object with `m0, n0, k0, m, n, k, clear_acc, write_output`, such
as D's `MatmulTile`. Offsets and sizes are elements of the logical C = A * B.
This module reads only the tiles and the operands; it does not reuse a tiler's
bookkeeping, so it can catch tiling mistakes that the tiler itself would repeat.

Two ways of combining K tiles:

* `matrix_accumulate`: the command format's Matrix unit. `clear_acc` starts a
  new output tile, later K tiles add into the retained accumulator and
  `write_output` writes the accumulated tile to C. Each C element must be
  written exactly once, after its full K range.
* `host_combine`: the SoC experiment. Every tile is standalone (clear and write),
  and the host adds each INT32 partial sum into an exact total.

Accumulator width and overflow behavior are not specified for the proposed unit.
Any accumulator or partial sum outside INT32 is reported instead of being wrapped
or saturated. Output requantization is not modeled; C holds integer sums.
"""
from collections import namedtuple
from dataclasses import dataclass, field

from .reference import INT32_MAX, INT32_MIN, _sums, check_operands, int_matmul

MODES = ("matrix_accumulate", "host_combine")
Violation = namedtuple("Violation", "tile kind detail")


@dataclass(frozen=True)
class TileLimits:
    """Largest tile the target accepts. Defaults are the SoC test format
    (0xffff0001: 1 <= M,K <= 8, 1 <= N <= 16); the agreed target comes from item B."""
    m: int = 8
    n: int = 16
    k: int = 8


@dataclass
class TileExecution:
    mode: str
    shape: tuple
    sums: list
    violations: list = field(default_factory=list)
    tiles: int = 0
    writes: int = 0

    @property
    def passed(self):
        return not self.violations

    def summary(self):
        kinds = {}
        for v in self.violations:
            kinds[v.kind] = kinds.get(v.kind, 0) + 1
        return {"mode": self.mode, "shape": dict(zip("mnk", self.shape)), "tiles": self.tiles,
                "writes": self.writes, "passed": self.passed, "violations_by_kind": kinds,
                "first_violations": [v._asdict() for v in self.violations[:10]]}


def _fields(tile):
    try:
        values = tuple(getattr(tile, name) for name in ("m0", "n0", "k0", "m", "n", "k"))
        flags = (tile.clear_acc, tile.write_output)
    except AttributeError as error:
        raise TypeError(f"tile is missing {error.name!r}") from error
    if any(type(v) is not int for v in values) or any(type(f) is not bool for f in flags):
        raise TypeError("tile offsets/sizes must be ints and flags must be bools")
    return values + flags


def _merge_interval(intervals, k0, k):
    """Add [k0, k0 + k) to a sorted disjoint list; return False on overlap."""
    stop = k0 + k
    for start, end in intervals:
        if k0 < end and start < stop:
            return False
    intervals.append((k0, stop))
    intervals.sort()
    return True


def _covers(intervals, k):
    position = 0
    for start, end in intervals:
        if start != position:
            return False
        position = end
    return position == k


def execute_tiles(a_q, b_q, tiles, *, mode="matrix_accumulate", limits=TileLimits()):
    """Run `tiles` over INT8 operands (A rows, B columns) and record what C receives.

    Returns a TileExecution. `sums[m][n]` is None where nothing was written.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    check_operands(a_q, b_q)
    M, N, K = len(a_q), len(b_q), len(a_q[0])
    run = TileExecution(mode, (M, N, K), [[None] * N for _ in range(M)])
    violation = lambda i, kind, detail: run.violations.append(Violation(i, kind, detail))
    acc = None  # matrix_accumulate: {"region", "values", "k", "written", "tile"}
    covered = [[[] for _ in range(N)] for _ in range(M)]  # host_combine K intervals

    def discard_check(index):
        if acc is not None and not acc["written"]:
            violation(index, "accumulator_discarded",
                      f"accumulator from tile {acc['tile']} was never written")

    for index, tile in enumerate(tiles):
        run.tiles += 1
        m0, n0, k0, m, n, k, clear, write = _fields(tile)
        if min(m, n, k) < 1 or min(m0, n0, k0) < 0:
            violation(index, "invalid_tile", "sizes must be positive and offsets nonnegative")
            continue
        if m0 + m > M or n0 + n > N or k0 + k > K:
            violation(index, "out_of_bounds", f"tile exceeds C/K extent {M}x{N}x{K}")
            continue
        if m > limits.m or n > limits.n or k > limits.k:
            violation(index, "exceeds_tile_limit",
                      f"{m}x{n}x{k} exceeds {limits.m}x{limits.n}x{limits.k}")
        block = _sums(a_q[m0:m0 + m], b_q[n0:n0 + n], k0, k0 + k)
        if any(not INT32_MIN <= v <= INT32_MAX for row in block for v in row):
            violation(index, "accumulator_overflow", "a K-tile partial sum exceeds INT32")

        if mode == "host_combine":
            if not (clear and write):
                violation(index, "not_standalone", "host_combine tiles must clear and write")
            for i in range(m):
                for j in range(n):
                    if not _merge_interval(covered[m0 + i][n0 + j], k0, k):
                        violation(index, "k_overlap", f"C[{m0 + i}][{n0 + j}] K range repeated")
                    current = run.sums[m0 + i][n0 + j] or 0
                    run.sums[m0 + i][n0 + j] = current + block[i][j]
            run.writes += 1
            continue

        region = (m0, n0, m, n)
        if clear:
            discard_check(index)
            acc = {"region": region, "values": [[0] * n for _ in range(m)], "k": [],
                   "written": False, "tile": index}
        elif acc is None:
            violation(index, "accumulate_without_clear", "no accumulator was started")
            continue
        elif acc["region"] != region:
            violation(index, "accumulator_region_mismatch",
                      f"continues {acc['region']} with output region {region}")
            continue
        if not _merge_interval(acc["k"], k0, k):
            violation(index, "k_overlap", f"K range [{k0}, {k0 + k}) added twice")
        for i in range(m):
            for j in range(n):
                acc["values"][i][j] += block[i][j]
                if not INT32_MIN <= acc["values"][i][j] <= INT32_MAX:
                    violation(index, "accumulator_overflow", f"accumulator exceeds INT32 at ({i}, {j})")
        if write:
            if not _covers(acc["k"], K):
                violation(index, "incomplete_k", f"written after K ranges {acc['k']} of [0, {K})")
            for i in range(m):
                for j in range(n):
                    if run.sums[m0 + i][n0 + j] is not None:
                        violation(index, "duplicate_write", f"C[{m0 + i}][{n0 + j}] written again")
                    run.sums[m0 + i][n0 + j] = acc["values"][i][j]
            acc["written"] = True
            run.writes += 1
    discard_check(None)

    if mode == "host_combine":
        for i in range(M):
            for j in range(N):
                if run.sums[i][j] is not None and not _covers(covered[i][j], K):
                    violation(None, "incomplete_k", f"C[{i}][{j}] K ranges {covered[i][j]} of [0, {K})")
    missing = [(i, j) for i in range(M) for j in range(N) if run.sums[i][j] is None]
    if missing:
        violation(None, "missing_output", f"{len(missing)} C elements never written, first {missing[0]}")
    return run


def check_tiles(a_q, b_q, tiles, *, plan=None, mode="matrix_accumulate", limits=TileLimits(),
                expected=None):
    """Execute tiles and compare C with the untiled reference sums.

    `plan` (a MatmulPlan or anything with m, n, k) must match the operand shape.
    `expected` overrides the reference sums, for example a committed fixture.
    """
    M, N, K = len(a_q), len(b_q), len(a_q[0])
    if plan is not None and (plan.m, plan.n, plan.k) != (M, N, K):
        raise ValueError(f"plan is {plan.m}x{plan.n}x{plan.k} but operands are {M}x{N}x{K}")
    run = execute_tiles(a_q, b_q, tiles, mode=mode, limits=limits)
    reference = int_matmul(a_q, b_q) if expected is None else expected
    wrong = [(i, j) for i in range(M) for j in range(N)
             if run.sums[i][j] is not None and run.sums[i][j] != reference[i][j]]
    if wrong:
        i, j = wrong[0]
        run.violations.append(Violation(None, "value_mismatch",
                                        f"{len(wrong)} C elements differ, first C[{i}][{j}] = "
                                        f"{run.sums[i][j]}, expected {reference[i][j]}"))
    return run
