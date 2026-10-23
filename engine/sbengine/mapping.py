"""Existing selection, tiling and lowering, extracted from Raghav Iyer's costs.

Candidate compute prices are needed to compare recurrence lowerings; unit-service
primitives stay on Ctx. Records reference captured IDs, not a copied graph.
Capability priority, tie-breaking and known incomplete composite behavior are preserved.
"""
import math
from dataclasses import dataclass, field
from functools import lru_cache

from cdfg import ACTIVATION_OPS, ELEMENTWISE_OPS
from . import groups

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
class Mapping:
    node: int
    tensor: int
    pool: str = None
    compute_s: float = 0.0
    macs: float = 0.0
    arith: float = 0.0
    aux_busy: dict = field(default_factory=dict)
    extra_l1: float = 0.0
    tile: dict = field(default_factory=dict)
    facts: dict = field(default_factory=dict)
    note: str = ""
    error: str = None


def vector_or_scalar(ctx, ops, special, caps):
    v, s = ctx.cfg["vector"], ctx.cfg["scalar"]
    if v["count"] > 0 and all(c in v["caps"] for c in caps):
        return "Vector", ctx.vector_time(ops, special)
    if s["count"] > 0:
        return "Scalar", ctx.scalar_time(ops, special)
    return None, math.inf


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


def map_vector(ctx, g, n):
    ops_pe, special, caps = VECTOR_OPS[n.op]
    E = n.tensor.elements * ctx.B
    special_calls = E / max(1, n.tensor.shape[0]) if special == "rows" else E * special
    basic = ops_pe * E
    pool, t = ctx.vec_or_scalar(basic, special_calls, caps)
    c = Mapping(n.idx, n.tensor.id, pool=pool, compute_s=t, arith=basic)
    if pool is None:
        c.error = f"{n.op}: no vector or scalar implementation"
    return c


def map_matmul(ctx, g, n):
    t, a, x = n.tensor, n.slots[0], n.slots[1]
    B, m = ctx.B, ctx.cfg["matrix"]
    M, Ncols, K = t.shape[0], t.shape[1], a.shape[0]
    dst_groups, src_groups = t.shape[2] * t.shape[3], a.shape[2] * a.shape[3]
    a_root, x_root = g.root(a.id), g.root(x.id)
    a_kind = g.kind(a_root)
    is_weight = a_kind == "weight"
    macs = float(g.macs_of(n) * B)
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
    c = Mapping(n.idx, n.tensor.id, macs=macs, arith=2 * macs)
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
            c.extra_l1 += E * (4 + ctx.act_b)
    plan = ctx.tile_plan(M, N, K, a_bpe, x_bpe, a_total, x_total)
    if not plan["fits"]:
        c.error = "L1 tile budget too small for a minimal matmul tile"
    c.tile = plan
    c.facts = dict(M=M, N=N, K=K, G=G, a_root=a_root, x_root=x_root, a_kind=a_kind,
                   a_total=a_total, x_total=x_total, a_bpe=a_bpe, x_bpe=x_bpe)
    c.note = f"{cls} tile {plan['m']}x{plan['n']} nr={plan['nr']} mr={plan['mr']}"
    return c


def map_flash(ctx, g, n):
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
    c = Mapping(n.idx, n.tensor.id)
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
    c.facts = dict(nq=nq, avg_keys=avg_keys, d=d, Hkv=Hkv, B=B, scores=scores)
    c.note = f"fused q_tile={tok_tile} nq={nq} avg_keys={avg_keys}"
    return c


def _gdn_dims(g, n):
    v = n.slots[2]
    return v.shape[0], v.shape[1], g.meta["input_tokens"]      # S, H, T


def map_gdn(ctx, g, n):
    """Gated delta rule. Serial: (7 S^2 + 3 S) ops per head-token on a recurrence unit, vector or scalar.
    Chunked (prefill only): 3 C S^2 + 5 C^2 S + C^3/3 MACs per head per chunk, matrix-mapped.
    Matrix: the captured serial update on the matrix arrays, 3 S^2 32-bit MACs per head-token (two state-vector
    products and the outer-product update) at `matrix_penalty` W8xA16-equivalent PE cycles each, with the decay,
    scalar and exponential tail on the vector side. Compute follows the 2026-10-03 recurrence study's
    shared-matrix path (explorer/engine.js); works for decode as well as prefill. It needs the `gemm` capability,
    like the chunked lowering. State traffic is the vector path's: the arrays have no local state storage, so the
    state is read 5x and written 2x per token through L1 (the study's path charges the reads only)."""
    S, H, T = _gdn_dims(g, n)
    B, r, m = ctx.B, ctx.cfg["recurrent"], ctx.cfg["matrix"]
    state = float(S * S * H * B)
    work = (7 * S * S + 3 * S) * H * T * B
    special = float(H * T * B)
    options = []   # (pool, seconds, extra L1 bytes beyond one pass over the operands, note, matrix MACs)
    aux = {}
    if r["lowering"] == "matrix" and m["count"] > 0 and "gemm" in m["caps"] and m["rates"].get("w8", 0) > 0:
        macs = 3.0 * S * S * H * T * B
        t_mat = macs * r["matrix_penalty"] / (m["count"] * m["rows"] * m["cols"] * m["rates"]["w8"] * ctx.f * m["efficiency"])
        vpool, t_vec = ctx.vec_or_scalar((S * S + 3 * S) * H * T * B, special, ("elementwise", "exp"))
        if vpool is not None:      # state re-read 5x and re-written 2x per token through L1, as on the vector path
            traffic = state * ctx.state_b * (5 * T - 1) + 2 * state * T * ctx.state_b
            options.append(("Matrix", t_mat + t_vec, traffic, f"serial/matrix x{r['matrix_penalty']:g}", macs))
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
    c = Mapping(n.idx, n.tensor.id, arith=work)
    if not options:
        c.error = "GATED_DELTA_NET: no recurrence, vector, scalar or matrix implementation"
        return c
    pool, seconds, extra, note, macs = min(options, key=lambda o: o[1])
    c.pool, c.compute_s, c.note = pool, seconds, note
    if note.startswith("serial/matrix"):
        c.aux_busy = aux
    if macs:
        c.macs, c.arith = macs, 2 * macs
    c.extra_l1 = extra
    c.facts = dict(S=S, H=H, T=T, B=B, state=state)
    return c


def map_conv(ctx, g, n):
    """Depthwise causal conv (SSM_CONV): macs = elements * kernel taps."""
    m = ctx.cfg["matrix"]
    K = n.slots[1].shape[0]
    macs = float(g.macs_of(n) * ctx.B)
    c = Mapping(n.idx, n.tensor.id, macs=macs, arith=2 * macs)
    if m["count"] > 0 and "conv" in m["caps"] and m["rates"].get("aa", 0) > 0:
        c.pool = "Matrix"
        c.compute_s = macs / (m["count"] * m["rows"] * m["cols"] * m["rates"]["aa"] * ctx.f * m["efficiency"]) * m["conv_penalty"]
    else:
        c.pool, c.compute_s = ctx.vec_or_scalar(2 * macs, 0, ("conv", "elementwise"))
        if c.pool is None:
            c.error = "SSM_CONV: no matrix, vector or scalar implementation"
    return c


def map_memory(ctx, g, n):
    out_b = ctx.tbytes(g, n.tensor)
    dst_kind = g.kind(n.dst_root)
    index_bytes = sum(ctx.tbytes(g, s) for s in n.srcs if g.kind(g.root(s.id)) == "input")
    if n.op == "SET_ROWS" or (n.op == "CPY" and dst_kind in ("kv", "state")):
        bytes_per_el = ctx.kv_b if dst_kind == "kv" else ctx.state_b
        moved = n.slots[0].elements * bytes_per_el * ctx.B
    else:
        moved = out_b
    c = Mapping(n.idx, n.tensor.id, facts=dict(out_b=out_b, dst_kind=dst_kind, index_bytes=index_bytes))
    if ctx.cfg["dma"]["count"] > 0:
        c.pool, c.compute_s = "DMA", ctx.dma_time(2 * moved)
    else:
        c.pool, c.compute_s = ctx.vec_or_scalar(2 * moved / 4, 0, ("elementwise",))
        if c.pool is None:
            c.error = f"{n.op}: no DMA, vector or scalar implementation"
    return c


def fused_roots(g, policy):
    """Roots whose producer->consumer edge is fused, so the intermediate is never stored.

    "chains": the result has exactly one consumer node and that consumer is an elementwise epilogue of a compute
    producer. Optimistic (the accelerator must support the epilogue).
    "groups": the result stays inside one anchored fusion group (groups.py): any single-consumer edge within a
    layer, around at most one compute anchor, including prologues (a state read feeding the recurrence), cache
    writes and the whole attention core. More optimistic than "chains"; an upper bracket on what fusion removes."""
    if policy == "groups":
        return set(groups.analyze(g).fused_roots)
    if policy != "chains":
        return set()
    fused = set()
    for root, prod_idx in g.producer.items():
        consumers = set(g.consumers.get(root, []))
        if len(consumers) != 1:
            continue
        c, p = g.nodes[next(iter(consumers))], g.nodes[prod_idx]
        if c.op in FUSIBLE_CONSUMERS and p.op in FUSIBLE_PRODUCERS:
            fused.add(root)
    return fused


def map_node(ctx, g, n):
    op = n.op
    if op == "MUL_MAT":
        return map_matmul(ctx, g, n)
    if op == "FLASH_ATTN_EXT":
        return map_flash(ctx, g, n)
    if op == "GATED_DELTA_NET":
        return map_gdn(ctx, g, n)
    if op == "SSM_CONV":
        return map_conv(ctx, g, n)
    if op in VECTOR_OPS:
        return map_vector(ctx, g, n)
    if op in ("GET_ROWS", "SET_ROWS", "CPY", "CONT", "CONCAT"):
        return map_memory(ctx, g, n)
    return Mapping(n.idx, n.tensor.id, error=f"{op}: no cost formula")
