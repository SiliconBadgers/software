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
from functools import lru_cache

from . import formats
from .graph import ACTIVATION_OPS, ELEMENTWISE_OPS

# desc -> (basic ops per element, special calls per element ("rows": one per row), required vector caps)
VECTOR_OPS = {
    "RMS_NORM": (3.0, "rows", ("elementwise", "reduce")),
    "SOFT_MAX": (5.0, 1.0, ("elementwise", "reduce", "exp")),
    "SILU": (3.0, 1.0, ("elementwise", "exp")),
    "SIGMOID": (3.0, 1.0, ("elementwise", "exp")),
    "SOFTPLUS": (2.0, 2.0, ("elementwise", "exp")),
    "SWIGLU": (4.0, 1.0, ("elementwise", "exp")),
    "ROPE": (3.0, 0.5, ("elementwise", "rope")),
    "MUL": (1.0, 0.0, ("elementwise",)),
    "ADD": (1.0, 0.0, ("elementwise",)),
    "SCALE": (1.0, 0.0, ("elementwise",)),
}
# Consumers that can be applied to a producer's result before it is stored (elementwise epilogues).
FUSIBLE_CONSUMERS = ELEMENTWISE_OPS | ACTIVATION_OPS | {"ROPE"}
FUSIBLE_PRODUCERS = {"MUL_MAT", "RMS_NORM", "SOFT_MAX", "ROPE"} | ELEMENTWISE_OPS | ACTIVATION_OPS


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
        v, s = self.cfg["vector"], self.cfg["scalar"]
        if v["count"] > 0 and all(c in v["caps"] for c in caps):
            return "Vector", self.vector_time(ops, special)
        if s["count"] > 0:
            return "Scalar", self.scalar_time(ops, special)
        return None, math.inf

    def dma_time(self, nbytes):
        d = self.cfg["dma"]
        return nbytes / (d["count"] * d["bytes_per_cycle"] * self.f)

    # -- matmul tiling ---------------------------------------------------------------------------------------
    def tile_plan(self, M, N, K, a_bpe, x_bpe, a_total, x_total):
        m = self.cfg["matrix"]
        return _tile_plan(M, N, K, a_bpe, x_bpe, a_total, x_total, m["count"], m["rows"], m["cols"], m["tile_k"],
                          m["tile_mode"], m["legacy_tile_n"], self.tile_budget)


def _candidates(dim, base):
    out = {dim}
    v = 1
    while v <= dim:
        out.add(v)
        v *= 2
    v = base
    while v <= dim:
        out.add(v)
        v *= 2
    return sorted(out)


@lru_cache(maxsize=None)
def _tile_plan(M, N, K, a_bpe, x_bpe, a_total, x_total, units, rows, cols, tile_k, mode, legacy_n, budget):
    """Choose the L1 tile (m x n, K slab kt) for one matmul. Returns dict(m, n, nr, mr, fits).

    Footprint (double-buffered operands, INT32 accumulators): 2 (m kt a + kt n x + m n 4) <= budget / units.
    nr = ceil(N/n) is how many times the A operand (weights / K,V) is re-streamed; mr = ceil(M/m) for X.
    "search": minimise A_total ceil(N/n) + X_total ceil(M/m) over power-of-two / array-multiple tiles.
    "fixed" (legacy): m = 4 rows, n = legacy_n, k = tile_k, halved until it fits.
    """
    cap = budget / max(1, units)

    def foot(m, n, kt):
        return 2 * (m * kt * a_bpe + kt * n * x_bpe + m * n * 4)

    kt = min(K, tile_k)
    if foot(1, 1, 1) > cap:
        return {"m": 1, "n": 1, "nr": N, "mr": M, "fits": False}
    if mode == "fixed":
        m, n, k = min(M, rows * 4), min(N, legacy_n), kt
        while foot(m, n, k) > cap and (m > 1 or n > 1 or k > 1):
            if n >= m and n > 1:
                n = max(1, n // 2)
            elif m > 1:
                m = max(1, m // 2)
            else:
                k = max(1, k // 2)
        return {"m": m, "n": n, "nr": math.ceil(N / n), "mr": math.ceil(M / m), "fits": foot(m, n, k) <= cap}
    best = None
    for m in _candidates(M, rows):
        for n in _candidates(N, cols):
            if foot(m, n, kt) > cap:
                continue
            traffic = a_total * math.ceil(N / n) + x_total * math.ceil(M / m)
            key = (traffic, -(m * n))
            if best is None or key < best[0]:
                best = (key, m, n)
    if best is None:                    # only a sub-array tile fits (kt too large): shrink the K slab
        return {"m": 1, "n": 1, "nr": N, "mr": M, "fits": True}
    _, m, n = best
    return {"m": m, "n": n, "nr": math.ceil(N / n), "mr": math.ceil(M / m), "fits": True}


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


def cost_vector(ctx, g, n):
    ops_pe, special, caps = VECTOR_OPS[n.op]
    E = n.tensor.elements * ctx.B
    special_calls = E / max(1, n.tensor.shape[0]) if special == "rows" else E * special
    basic = ops_pe * E
    pool, t = ctx.vec_or_scalar(basic, special_calls, caps)
    c = Cost(pool=pool, compute_s=t, arith=basic)
    if pool is None:
        c.error = f"{n.op}: no vector or scalar implementation"
    return _finish_elementwise(ctx, g, n, c, _operands(ctx, g, n))


def cost_matmul(ctx, g, n):
    t, a, x = n.tensor, n.slots[0], n.slots[1]
    B, m = ctx.B, ctx.cfg["matrix"]
    M, Ncols, K = t.shape[0], t.shape[1], a.shape[0]
    dst_groups, src_groups = t.shape[2] * t.shape[3], a.shape[2] * a.shape[3]
    a_root, x_root = g.root(a.id), g.root(x.id)
    a_kind = g.kind(a_root)
    is_weight = a_kind == "weight"
    macs = float(M * Ncols * dst_groups * K * B)
    a_total, x_total = ctx.tbytes(g, a), ctx.tbytes(g, x)
    a_bpe = a_total / (a.elements if is_weight else a.elements * B)
    x_bpe = x_total / (x.elements * B)
    if is_weight:                      # tokens (and batch) form the N dimension; weights are shared across the batch
        N, G = Ncols * B, dst_groups
        cls = ctx.weight_class(g, a)
    else:                              # per-head groups; query heads sharing a KV head widen N
        bcast = max(1, dst_groups // max(1, src_groups))
        N, G = Ncols * bcast, src_groups * B
        cls = "aa"
    rate = m["rates"].get(cls, 0)
    cap = "gemm" if is_weight else "attention"
    c = Cost(macs=macs, arith=2 * macs)
    if m["count"] > 0 and cap in m["caps"] and rate > 0:
        c.pool = "Matrix"
        c.compute_s = ctx.matrix_time(M, N, K, G, rate)
    else:
        c.pool, c.compute_s = ctx.vec_or_scalar(2 * macs, 0, ("gemm", "elementwise"))
        if c.pool is None:
            c.error = f"MUL_MAT ({cls}): no matrix, software-GEMM vector or scalar implementation"
    E = t.elements * B
    if c.pool == "Matrix" and "requant" not in m["caps"]:     # unfused requantisation epilogue on the vector side
        pool, et = ctx.vec_or_scalar(3 * E, 0, ("quant", "elementwise"))
        if pool is not None:
            c.compute_s += et
            c.l1_fixed += E * (4 + ctx.act_b)
    plan = ctx.tile_plan(M, N, K, a_bpe, x_bpe, a_total, x_total)
    if not plan["fits"]:
        c.error = "L1 tile budget too small for a minimal matmul tile"
    nr, mr = plan["nr"], plan["mr"]
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
    c.note = f"{cls} tile {plan['m']}x{plan['n']} nr={nr} mr={mr}"
    return c


def cost_flash(ctx, g, n):
    """Fused attention: QK^T, online softmax and AV in one streamed pass over K/V; scores never reach HBM."""
    at, m, B = ctx.cfg["attention"], ctx.cfg["matrix"], ctx.B
    q, k = n.slots[0], n.slots[1]
    d, Tq, Hq = q.shape[0], q.shape[1], q.shape[2]
    Tk, Hkv = k.shape[1], k.shape[2]
    gq = max(1, Hq // Hkv)
    past = g.meta["past_tokens"]
    live = past + Tq
    keys_read = Tk if at["kv_padding"] == "captured" else live
    # q tile: rows of Q (incl. the gq heads sharing a KV head) held in L1 while K/V stream past
    cap = ctx.tile_budget / max(1, m["count"])
    kt = at["kv_tile"]
    fixed = 2 * 2 * kt * d * ctx.kv_b
    per_row = d * ctx.act_b + d * 4 + kt * 4
    c = Cost()
    if cap - fixed < per_row:
        c.error = "L1 tile budget too small for a fused-attention tile"
        return c
    q_rows_tile = max(1, int((cap - fixed) // per_row))
    tok_tile = max(1, q_rows_tile // gq)
    nq = math.ceil(Tq / tok_tile)
    avg_keys = keys_read
    if at["causal_skip"]:
        avg_keys = min(keys_read, past + Tq * (nq + 1) / (2 * nq))
    avg_keys = max(1, math.ceil(avg_keys))
    macs = 2.0 * d * Tq * Hq * B * avg_keys
    scores = float(Hq * Tq * B * avg_keys)
    c.macs, c.arith = macs, 2 * macs + 5 * scores
    rate = m["rates"].get("aa", 0)
    vec_pool, vec_t = ctx.vec_or_scalar(5 * scores, scores, ("elementwise", "reduce", "exp"))
    if m["count"] > 0 and "attention" in m["caps"] and rate > 0 and vec_pool is not None:
        qk = ctx.matrix_time(avg_keys, Tq * gq, d, Hkv * B, rate)
        av = ctx.matrix_time(d, Tq * gq, avg_keys, Hkv * B, rate)
        mat_t = qk + av
        c.pool = "Matrix"
        c.compute_s = max(mat_t, vec_t) if at["overlap_softmax"] else mat_t + vec_t
        c.aux_busy = {vec_pool: vec_t}
    else:
        pool, tt = ctx.vec_or_scalar(2 * macs + 5 * scores, scores, ("gemm", "elementwise", "reduce", "exp"))
        c.pool, c.compute_s = pool, tt
        if pool is None:
            c.error = "FLASH_ATTN_EXT: no attention-capable matrix unit or fallback"
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
    c.note = f"fused q_tile={tok_tile} nq={nq} avg_keys={avg_keys}"
    return c


def _gdn_dims(g, n):
    v = n.slots[2]
    return v.shape[0], v.shape[1], g.meta["input_tokens"]      # S, H, T


def cost_gdn(ctx, g, n):
    """Gated delta rule. Serial: (7 S^2 + 3 S) ops per head-token on a recurrence unit, vector or scalar.
    Chunked (prefill only): 3 C S^2 + 5 C^2 S + C^3/3 MACs per head per chunk, matrix-mapped.
    Matrix: the captured serial update on the matrix arrays, 3 S^2 32-bit MACs per head-token (two state-vector
    products and the outer-product update) at `matrix_penalty` W8xA16-equivalent PE cycles each, with the decay,
    scalar and exponential tail on the vector side. Same equation as the 2026-10-03 recurrence study's
    shared-matrix path (explorer/engine.js); works for decode as well as prefill."""
    S, H, T = _gdn_dims(g, n)
    B, r, m = ctx.B, ctx.cfg["recurrent"], ctx.cfg["matrix"]
    state = float(S * S * H * B)
    work = (7 * S * S + 3 * S) * H * T * B
    special = float(H * T * B)
    options = []   # (pool, seconds, extra L1 bytes beyond one pass over the operands, note, matrix MACs)
    aux = {}
    if r["lowering"] == "matrix" and m["count"] > 0 and m["rates"].get("w8", 0) > 0:
        macs = 3.0 * S * S * H * T * B
        t_mat = macs * r["matrix_penalty"] / (m["count"] * m["rows"] * m["cols"] * m["rates"]["w8"] * ctx.f * m["efficiency"])
        vpool, t_vec = ctx.vec_or_scalar((S * S + 3 * S) * H * T * B, special, ("elementwise", "exp"))
        if vpool is not None:      # state re-read 5x per token through L1, as on the vector path
            options.append(("Matrix", t_mat + t_vec, state * ctx.state_b * (5 * T - 1), f"serial/matrix x{r['matrix_penalty']:g}", macs))
            # Matrix and vector time are added serially and the per-token tail is interleaved with the products,
            # so the vector side is held for the whole compute interval, not only its first t_vec seconds. The
            # scheduler has no explicit phases; until it does this is the reservation that cannot be overlapped.
            aux = {vpool: t_mat + t_vec}
    if r["lowering"] in ("serial", "auto"):
        if r["count"] > 0:
            active = min(r["count"], H * B)
            t = (work + special * ctx.cfg["vector"]["special_cycles"]) / (active * r["lanes"] * ctx.f * r["efficiency"])
            fits = r["scratch_kib"] * 1024 >= S * S * ctx.state_b
            extra = 0.0 if fits else state * ctx.state_b * (5 * T - 1)
            options.append(("Recurrent", t, extra, "serial/dedicated" + ("" if fits else " (scratch < state: multipass)"), 0.0))
        else:
            pool, t = ctx.vec_or_scalar(work, special, ("recurrent", "elementwise", "reduce", "exp"))
            if pool is not None:   # state re-read 5x and re-written 2x per token through L1
                options.append((pool, t, state * ctx.state_b * (5 * T - 1) + 2 * state * T * ctx.state_b, f"serial/{pool.lower()}", 0.0))
    if r["lowering"] in ("chunked", "auto") and T > 1 and m["count"] > 0 and "gemm" in m["caps"] and m["rates"].get("aa", 0) > 0:
        C = min(r["chunk"], T)
        macs = H * B * math.ceil(T / C) * (3 * C * S * S + 5 * C * C * S + C ** 3 / 3)
        t_mat = macs / (m["count"] * m["rows"] * m["cols"] * m["rates"]["aa"] * ctx.f * m["efficiency"])
        vpool, t_vec = ctx.vec_or_scalar(12 * S * H * T * B, special, ("elementwise", "exp"))
        if vpool is not None:      # gating / decay elementwise work runs on the vector side, added serially
            options.append(("Matrix", t_mat + t_vec, 0.0, f"chunked C={C}", macs))
    c = Cost(arith=work)
    if not options:
        c.error = "GATED_DELTA_NET: no recurrence, vector, scalar or matrix implementation"
        return c
    pool, seconds, extra, note, macs = min(options, key=lambda o: o[1])
    c.pool, c.compute_s, c.note = pool, seconds, note
    if note.startswith("serial/matrix"):
        c.aux_busy = aux
    if macs:
        c.macs, c.arith = macs, 2 * macs
    _finish_elementwise(ctx, g, n, c, _operands(ctx, g, n))
    c.l1_fixed += extra
    out_b = S * H * T * B * ctx.act_b + state * ctx.state_b     # sequence output plus the updated state
    c.writes, c.out = [(n.dst_root, out_b)], {n.dst_root: out_b}
    return c


def cost_conv(ctx, g, n):
    """Depthwise causal conv (SSM_CONV): macs = elements * kernel taps."""
    m = ctx.cfg["matrix"]
    K = n.slots[1].shape[0]
    macs = float(n.tensor.elements * K * ctx.B)
    c = Cost(macs=macs, arith=2 * macs)
    if m["count"] > 0 and "conv" in m["caps"] and m["rates"].get("aa", 0) > 0:
        c.pool = "Matrix"
        c.compute_s = macs / (m["count"] * m["rows"] * m["cols"] * m["rates"]["aa"] * ctx.f * m["efficiency"]) * m["conv_penalty"]
    else:
        c.pool, c.compute_s = ctx.vec_or_scalar(2 * macs, 0, ("conv", "elementwise"))
        if c.pool is None:
            c.error = "SSM_CONV: no matrix, vector or scalar implementation"
    return _finish_elementwise(ctx, g, n, c, _operands(ctx, g, n))


def cost_memory(ctx, g, n):
    """GET_ROWS / SET_ROWS / CPY / CONT / CONCAT on the DMA engines (vector fallback without DMA).

    Data that enters L1 from HBM is charged once as ingress (via the HBM byte count) and data that leaves L1 for
    HBM once as egress, so the unit-side copy of that same byte stream is not charged a second time."""
    out_b = ctx.tbytes(g, n.tensor)
    dst_kind = g.kind(n.dst_root)
    index_bytes = sum(ctx.tbytes(g, s) for s in n.srcs if g.kind(g.root(s.id)) == "input")
    c = Cost(in_hbm=float(index_bytes))
    if n.op == "GET_ROWS":
        table = n.slots[0]
        troot = g.root(table.id)
        tkind = g.kind(troot)
        moved = out_b
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
        moved = src.elements * bytes_per_el * ctx.B
        c.p_hbm_w = moved * ((1 - ctx.state_resident) if dst_kind == "state" else 1.0)
        if g.kind(sroot) == "activation":
            c.reads = [(sroot, ctx.tbytes(g, src), 1)]   # egress bytes are the p_hbm_w; no separate unit-side feed
    else:                                                # CPY / CONT / CONCAT inside L1: read + write on L1 ports
        _finish_elementwise(ctx, g, n, c, _operands(ctx, g, n))
        moved = out_b
    # DMA rate (bytes per engine-cycle) is counted against bytes read plus bytes written, as in the explorer
    if ctx.cfg["dma"]["count"] > 0:
        c.pool, c.compute_s = "DMA", ctx.dma_time(2 * moved)
    else:
        c.pool, c.compute_s = ctx.vec_or_scalar(2 * moved / 4, 0, ("elementwise",))
        if c.pool is None:
            c.error = f"{n.op}: no DMA, vector or scalar implementation"
    return c


def node_cost(ctx, g, n):
    op = n.op
    if op == "MUL_MAT":
        return cost_matmul(ctx, g, n)
    if op == "FLASH_ATTN_EXT":
        return cost_flash(ctx, g, n)
    if op == "GATED_DELTA_NET":
        return cost_gdn(ctx, g, n)
    if op == "SSM_CONV":
        return cost_conv(ctx, g, n)
    if op in VECTOR_OPS:
        return cost_vector(ctx, g, n)
    if op in ("GET_ROWS", "SET_ROWS", "CPY", "CONT", "CONCAT"):
        return cost_memory(ctx, g, n)
    return Cost(error=f"{op}: no cost formula")
