"""Activation traffic: which intermediate tensors stay in L1 and which round-trip through HBM.

Zeb's explorer charged every op's inputs and outputs to L1 and added a spill proxy
`2 * max(0, working_set - L1)` per op. Here the whole schedule is used: a tensor is written once, stays in
L1 while it fits and is still needed, and only leaves when it must (farthest-next-use eviction, i.e. Belady's
rule, which is optimal for equal-size objects and a good heuristic for variable sizes). Elementwise epilogues
on a single-consumer result can be fused so the intermediate is never stored at all.

Both modes report the HBM bytes each node reads/writes for activations, and the peak bytes of activations
that live in HBM (a real liveness footprint, replacing the explorer's peak-working-set proxy).
"""
import bisect
import math

from . import groups
from .costs import FUSIBLE_CONSUMERS, FUSIBLE_PRODUCERS


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


def drop_fused(costs, fused):
    """Remove fused roots from every node's reads/feed/writes/out (they are never stored)."""
    if not fused:
        return
    for c in costs:
        c.reads = [x for x in c.reads if x[0] not in fused]
        c.feed = {r: b for r, b in c.feed.items() if r not in fused}
        c.writes = [x for x in c.writes if x[0] not in fused]
        c.out = {r: b for r, b in c.out.items() if r not in fused}


def simulate_residency(costs, capacity):
    """Returns (hbm_read[i], hbm_write[i], peak_spilled_bytes, stats) for activation tensors."""
    n = len(costs)
    sizes, uses = {}, {}
    for i, c in enumerate(costs):
        for r, b in c.writes:
            sizes[r] = b
        for r, _, _ in c.reads:
            uses.setdefault(r, []).append(i)
    hbm_r, hbm_w = [0.0] * n, [0.0] * n
    resident = {}            # root -> dirty (no HBM copy yet)
    on_hbm = set()           # live roots that have an HBM copy and are not resident
    used = 0.0
    peak = 0.0
    hits = misses = 0

    def next_use(root, i):
        lst = uses.get(root)
        if not lst:
            return math.inf
        k = bisect.bisect_right(lst, i)
        return lst[k] if k < len(lst) else math.inf

    def evict(root, i):
        nonlocal used
        dirty = resident.pop(root)
        used -= sizes[root]
        if next_use(root, i) < math.inf:
            if dirty:
                hbm_w[i] += sizes[root]
            on_hbm.add(root)

    def make_room(need, i, pinned):
        while used + need > capacity:
            victim, best = None, -1.0
            for r in resident:
                if r in pinned:
                    continue
                nu = next_use(r, i)
                if nu > best:
                    victim, best = r, nu
            if victim is None:
                return False
            evict(victim, i)
        return True

    for i, c in enumerate(costs):
        pinned = {r for r, _, _ in c.reads} | {r for r, _ in c.writes}
        for r, b, factor in c.reads:
            if r in resident:
                hits += 1
                continue
            misses += 1
            size = sizes.setdefault(r, b)          # a result no node wrote (an errored op) is sized by its first read
            if size <= capacity:
                hbm_r[i] += size                      # bring the whole tensor in
                if make_room(size, i, pinned):
                    resident[r] = False
                    used += size
            else:
                hbm_r[i] += b * factor                # too large to keep: streamed straight from HBM
        for r, b in c.writes:
            size = sizes[r]
            if next_use(r, i) == math.inf:            # graph output or dead result: leaves the chip
                hbm_w[i] += b
                continue
            if size <= capacity and make_room(size, i, pinned):
                resident[r] = True
                used += size
            else:
                hbm_w[i] += b
                on_hbm.add(r)
        for r in [x for x in resident if next_use(x, i) == math.inf]:
            used -= sizes[r]
            del resident[r]
        for r in [x for x in on_hbm if next_use(x, i) == math.inf]:
            on_hbm.discard(r)
        peak = max(peak, sum(sizes[r] for r in on_hbm))
    return hbm_r, hbm_w, peak, {"hits": hits, "misses": misses}


def per_op_spill(costs, capacity):
    """Legacy (explorer) proxy: every activation read/write goes through L1 and each op adds
    2 * max(0, working_set - capacity) HBM bytes; matmul and recurrence carry no proxy spill."""
    spill = []
    peak = 0.0
    for c in costs:
        read = sum(b for _, b, _ in c.reads) + c.w_hbm + c.p_hbm_r + c.in_hbm
        write = sum(b for _, b in c.writes)
        working = 0.0 if c.pool == "Matrix" or "chunked" in c.note or "serial" in c.note else read + write
        s = 2 * max(0.0, working - capacity)
        spill.append(s)
        peak = max(peak, working)
    return spill, peak
