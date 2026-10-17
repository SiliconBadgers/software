"""Per-node cost equations. See docs/EQUATIONS.md for the derivation of every formula here.

A node's cost is split into:
  * compute time on one unit pool (matrix / vector / recurrent / DMA / scalar / host),
  * traffic that does not depend on what else is in L1: weights, KV cache, recurrent state and inputs streamed
    from HBM, and L1<->unit feed bytes,
  * activation operands (`reads`) and results (`writes`), whose HBM traffic depends on residency and is
    decided later by memory.py from the whole schedule.
Conventions: shapes are GGML order (ne0 innermost); B is the batch (independent sequences); bytes are
storage bytes under the configured precision policy.
"""
import math
from dataclasses import dataclass, field

from . import formats
from .mapping import (FUSIBLE_CONSUMERS, FUSIBLE_PRODUCERS, VECTOR_OPS,
                      _candidates, _tile_plan, _gdn_dims, map_node, vector_or_scalar)




@dataclass
class Cost:
    pool: str = None                 # None => no implementation (error set)
    compute_s: float = 0.0
    aux_busy: dict = field(default_factory=dict)   # extra pool busy time when an op uses two pools at once
    macs: float = 0.0                # MACs executed by this op (MUL_MAT, SSM_CONV, fused attention, chunked GDN)
    arith: float = 0.0               # arithmetic operations (2 per MAC, plus vector ops)
    w_hbm: float = 0.0               # weight bytes read from HBM, including re-streaming
    p_hbm_r: float = 0.0             # persistent KV/state bytes read from HBM
    p_hbm_w: float = 0.0
    in_hbm: float = 0.0              # external inputs (indices, positions, mask when not generated on-chip)
    bypass: float = 0.0              # HBM bytes that stream to the units without crossing L1 (weight_path=direct)
    l1_fixed: float = 0.0            # L1<->unit bytes not tied to a fusible activation operand
    feed: dict = field(default_factory=dict)      # activation root -> L1->unit bytes
    out: dict = field(default_factory=dict)       # activation root -> unit->L1 bytes
    reads: list = field(default_factory=list)     # (root, one-pass bytes, stream factor if not L1-resident)
    writes: list = field(default_factory=list)    # (root, bytes)
    note: str = ""
    error: str = None


class Ctx:
    """Everything derived from the config that cost equations need."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.B = cfg["batch"]
        self.f = cfg["clock_mhz"] * 1e6
        p = cfg["precision"]
        self.act_b, self.kv_b, self.state_b = p["act_bits"] / 8, p["kv_bits"] / 8, p["state_bits"] / 8
        l1 = cfg["l1"]
        self.l1_bytes = l1["mib"] * 2**20
        self.tile_budget = self.l1_bytes * l1["tile_fraction"]
        self.act_cap = self.l1_bytes * (1 - l1["state_reserve"] - l1["tile_fraction"])
        self.state_cap = self.l1_bytes * l1["state_reserve"]
        self.l1_bw = l1["banks"] * l1["bytes_per_bank_cycle"] * self.f * l1["efficiency"]
        self.hbm_bw = cfg["hbm"]["gbs"] * 1e9 * cfg["hbm"]["efficiency"]
        self.launch_s = cfg["launch_cycles"] / self.f
        self.sync_s = cfg["sync_cycles"] / self.f
        self.state_resident = 0.0    # set per graph by model.py (needs the graph's state size)

    # -- storage policy ------------------------------------------------------------------------------------
    def weight_bytes(self, g, t):
        p = self.cfg["precision"]
        if p["weights"] == "captured" or g.root(t.id) not in g.matrix_weight_roots:
            return float(t.nbytes)
        bits = 4 if p["weights"] == "uniform_int4" else 8
        return t.elements * bits / 8 + math.ceil(t.elements / p["group_size"]) * p["scale_bytes"]

    def weight_class(self, g, t):
        p = self.cfg["precision"]["weights"]
        if p == "captured" or g.root(t.id) not in g.matrix_weight_roots:
            return formats.rate_class(t.dtype)
        return "w4" if p == "uniform_int4" else "w8"

    def tbytes(self, g, t):
        r = g.root(t.id)
        kind = g.kind(r)
        if kind == "weight":
            return self.weight_bytes(g, t)
        if kind == "kv":
            return t.elements * self.kv_b * self.B
        if kind == "state":
            return t.elements * self.state_b * self.B
        if kind in ("input", "mask"):
            return float(t.nbytes)
        bits = 32 if t.name == "result_output" else self.cfg["precision"]["act_bits"]
        return t.elements * bits / 8 * self.B

    # -- unit time equations ---------------------------------------------------------------------------------
    def matrix_time(self, M, N, K, G, rate):
        """Cycles on the systolic-style array; M x N output, reduction K, G independent groups.

        GEMM: tiles = ceil(M/rows) ceil(N/cols) G; cycles = ceil(tiles/units) K/rate (+ fill/drain).
        GEMV lane reuse (N < cols and cap 'gemv'): the narrow N dimension is folded into reduction parallelism.
        With matrix.pipelined_tiles the (rows+cols-2) fill/drain is paid once per op, else once per tile wave.
        """
        m = self.cfg["matrix"]
        units, rows, cols = m["count"], m["rows"], m["cols"]
        if units <= 0 or rate <= 0:
            return math.inf
        tiles_m, tiles_n = math.ceil(M / rows), math.ceil(N / cols)
        if N < cols and "gemv" in m["caps"]:
            active = min(units, max(1, tiles_m * G))
            cycles = math.ceil(M * N * K * G / (active * rows * cols * rate)) + rows + cols
        else:
            waves = math.ceil(tiles_m * tiles_n * G / units)
            fill = rows + cols - 2
            cycles = waves * (K / rate) + (fill if m["pipelined_tiles"] else waves * fill)
        return cycles / (self.f * m["efficiency"])

    def vector_time(self, ops, special):
        v = self.cfg["vector"]
        return (ops + special * v["special_cycles"]) / (v["count"] * v["lanes"] * self.f * v["efficiency"])

    def scalar_time(self, ops, special):
        s = self.cfg["scalar"]
        return (ops * s["cycles_per_op"] + special * s["cycles_per_special"]) / (s["count"] * s["ipc"] * s["mhz"] * 1e6)

    def vec_or_scalar(self, ops, special, caps):
        """Compatibility entry point; selection is owned by mapping."""
        return vector_or_scalar(self, ops, special, caps)

    def dma_time(self, nbytes):
        d = self.cfg["dma"]
        return nbytes / (d["count"] * d["bytes_per_cycle"] * self.f)

    # -- matmul tiling ---------------------------------------------------------------------------------------
    def tile_plan(self, M, N, K, a_bpe, x_bpe, a_total, x_total):
        m = self.cfg["matrix"]
        return _tile_plan(M, N, K, a_bpe, x_bpe, a_total, x_total, m["count"], m["rows"], m["cols"], m["tile_k"],
                          m["tile_mode"], m["legacy_tile_n"], self.tile_budget)





# ------------------------------------------------------------------------------------------------------------
def _operands(ctx, g, n):
    """Classify a node's sources: stream totals plus activation reads / L1 feed."""
    out = {"w": 0.0, "p": 0.0, "in": 0.0, "reads": [], "feed": {}, "feed_other": 0.0}
    for t in n.srcs:
        r = g.root(t.id)
        if r == n.dst_root:
            continue
        kind, b = g.kind(r), ctx.tbytes(g, t)
        if kind == "weight":
            out["w"] += b
            out["feed_other"] += b
        elif kind == "kv":
            out["p"] += b
            out["feed_other"] += b
        elif kind == "state":
            out["p"] += b * (1 - ctx.state_resident)
            out["feed_other"] += b
        elif kind == "mask":
            if not ctx.cfg["attention"]["mask_onchip"]:
                out["in"] += b
                out["feed_other"] += b
        elif kind == "input":
            out["in"] += b
            out["feed_other"] += b
        else:
            out["reads"].append((r, b, 1))
            out["feed"][r] = out["feed"].get(r, 0.0) + b
    return out


def _finish_elementwise(ctx, g, n, c, op):
    """Shared tail for streaming ops: operands read once, one result written."""
    c.w_hbm, c.p_hbm_r, c.in_hbm = op["w"], op["p"], op["in"]
    c.reads, c.feed, c.l1_fixed = op["reads"], op["feed"], op["feed_other"]
    out_b = ctx.tbytes(g, n.tensor)
    c.writes = [(n.dst_root, out_b)]
    c.out = {n.dst_root: out_b}
    return c


def _mapped_cost(n, mapped):
    if (mapped.node, mapped.tensor) != (n.idx, n.tensor.id):
        raise ValueError("mapping does not reference this captured operation")
    return Cost(pool=mapped.pool, compute_s=mapped.compute_s, macs=mapped.macs, arith=mapped.arith,
                aux_busy=dict(mapped.aux_busy), l1_fixed=mapped.extra_l1, note=mapped.note, error=mapped.error)


def cost_vector(ctx, g, n, mapped=None):
    c = _mapped_cost(n, mapped or map_node(ctx, g, n))
    return _finish_elementwise(ctx, g, n, c, _operands(ctx, g, n))


def cost_matmul(ctx, g, n, mapped=None):
    mapped = mapped or map_node(ctx, g, n)
    c = _mapped_cost(n, mapped)
    t, m = n.tensor, ctx.cfg["matrix"]
    f = mapped.facts
    M, N, K, G = (f[k] for k in ("M", "N", "K", "G"))
    a_root, x_root, a_kind = (f[k] for k in ("a_root", "x_root", "a_kind"))
    a_total, x_total, a_bpe, x_bpe = (f[k] for k in ("a_total", "x_total", "a_bpe", "x_bpe"))
    nr, mr = mapped.tile["nr"], mapped.tile["mr"]
    rows, cols = m["rows"], m["cols"]
    if ctx.cfg["memory"]["l1_feed"] == "array":
        # output-stationary array: every output tile streams K slices of both operands past the PEs
        tiles = math.ceil(M / rows) * math.ceil(N / cols) * G
        feed_a = tiles * K * min(rows, M) * a_bpe
        feed_x = tiles * K * min(cols, N) * x_bpe
    else:                              # legacy: reuse counted at L1-tile level only
        feed_a = a_total * nr
        feed_x = x_total * mr
    if a_kind == "weight":
        c.w_hbm = a_total * nr
        if ctx.cfg["memory"]["weight_path"] == "direct":
            c.bypass = c.w_hbm          # weights go HBM -> PE staging registers, never through L1
        else:
            c.l1_fixed += feed_a
    elif a_kind in ("kv", "state"):
        c.p_hbm_r = a_total * nr * (1.0 if a_kind == "kv" else 1 - ctx.state_resident)
        c.l1_fixed += feed_a
    else:
        c.reads.append((a_root, a_total, nr))
        c.feed[a_root] = feed_a
    if g.kind(x_root) == "activation":
        c.reads.append((x_root, x_total, mr))
        c.feed[x_root] = c.feed.get(x_root, 0.0) + feed_x
    else:
        c.in_hbm += x_total
        c.l1_fixed += feed_x
    out_b = ctx.tbytes(g, t)
    c.writes, c.out = [(n.dst_root, out_b)], {n.dst_root: out_b}
    return c


def cost_flash(ctx, g, n, mapped=None):
    mapped = mapped or map_node(ctx, g, n)
    c = _mapped_cost(n, mapped)
    if not mapped.facts:  # original early return for an infeasible attention tile
        return c
    nq, avg_keys, d, Hkv, B, scores = (mapped.facts[k] for k in ("nq", "avg_keys", "d", "Hkv", "B", "scores"))
    at, q = ctx.cfg["attention"], n.slots[0]
    kv_bytes = nq * avg_keys * d * Hkv * 2 * ctx.kv_b * B
    c.p_hbm_r = kv_bytes
    c.l1_fixed += kv_bytes + scores * (8 + 2 * ctx.act_b)      # K/V feed + score hand-off through L1
    if 3 in n.slots and g.kind(g.root(n.slots[3].id)) == "mask" and not at["mask_onchip"]:
        c.in_hbm += ctx.tbytes(g, n.slots[3])
    q_root = g.root(q.id)
    q_b = ctx.tbytes(g, q)
    c.reads = [(q_root, q_b, 1)]
    c.feed = {q_root: q_b}
    out_b = ctx.tbytes(g, n.tensor)
    c.writes, c.out = [(n.dst_root, out_b)], {n.dst_root: out_b}
    return c





def cost_gdn(ctx, g, n, mapped=None):
    mapped = mapped or map_node(ctx, g, n)
    c = _mapped_cost(n, mapped)
    if not mapped.facts:  # no legal existing recurrence option
        return c
    S, H, T, B, state = (mapped.facts[k] for k in ("S", "H", "T", "B", "state"))
    _finish_elementwise(ctx, g, n, c, _operands(ctx, g, n))
    c.l1_fixed += mapped.extra_l1
    out_b = S * H * T * B * ctx.act_b + state * ctx.state_b
    c.writes, c.out = [(n.dst_root, out_b)], {n.dst_root: out_b}
    return c


def cost_conv(ctx, g, n, mapped=None):
    c = _mapped_cost(n, mapped or map_node(ctx, g, n))
    return _finish_elementwise(ctx, g, n, c, _operands(ctx, g, n))


def cost_memory(ctx, g, n, mapped=None):
    mapped = mapped or map_node(ctx, g, n)
    c = _mapped_cost(n, mapped)
    out_b, dst_kind, index_bytes = (mapped.facts[k] for k in ("out_b", "dst_kind", "index_bytes"))
    c.in_hbm = float(index_bytes)
    if n.op == "GET_ROWS":
        table = n.slots[0]
        troot = g.root(table.id)
        tkind = g.kind(troot)
        c.writes = [(n.dst_root, out_b)]
        if tkind == "weight":
            c.w_hbm = n.tensor.elements * ctx.B * ctx.weight_bytes(g, table) / table.elements
        elif tkind in ("state", "kv"):
            c.p_hbm_r = out_b * ((1 - ctx.state_resident) if tkind == "state" else 1.0)
        else:                                            # gather from an activation held in L1
            c.reads, c.feed, c.out = [(troot, out_b, 1)], {troot: out_b}, {n.dst_root: out_b}
    elif n.op == "SET_ROWS" or (n.op == "CPY" and dst_kind in ("kv", "state")):
        src = n.slots[0]
        sroot = g.root(src.id)
        bytes_per_el = ctx.kv_b if dst_kind == "kv" else ctx.state_b
        c.p_hbm_w = src.elements * bytes_per_el * ctx.B * ((1 - ctx.state_resident) if dst_kind == "state" else 1.0)
        if g.kind(sroot) == "activation":
            c.reads = [(sroot, ctx.tbytes(g, src), 1)]   # egress bytes are the p_hbm_w; no separate unit-side feed
    else:                                                # CPY / CONT / CONCAT inside L1: read + write on L1 ports
        _finish_elementwise(ctx, g, n, c, _operands(ctx, g, n))
    return c


def node_cost(ctx, g, n, mapped=None):
    mapped = mapped or map_node(ctx, g, n)
    op = n.op
    if op == "MUL_MAT":
        return cost_matmul(ctx, g, n, mapped)
    if op == "FLASH_ATTN_EXT":
        return cost_flash(ctx, g, n, mapped)
    if op == "GATED_DELTA_NET":
        return cost_gdn(ctx, g, n, mapped)
    if op == "SSM_CONV":
        return cost_conv(ctx, g, n, mapped)
    if op in VECTOR_OPS:
        return cost_vector(ctx, g, n, mapped)
    if op in ("GET_ROWS", "SET_ROWS", "CPY", "CONT", "CONCAT"):
        return cost_memory(ctx, g, n, mapped)
    return _mapped_cost(n, mapped)
