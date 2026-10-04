"""Whole-graph timing: serial sum, dependency-aware list schedule, and a lower bound.

Zeb's explorer summed op times in captured order (no overlap between ops). With the captured dependency graph
we can bound the answer from both sides:

  serial       = sum of op durations (the explorer's result; an upper bound if units never overlap)
  list         = in-order issue per unit pool, ops wait for their dependencies (+ a cross-pool hand-off), and can
                 overlap ops on different pools; never below the total L1 or HBM service time. It can exceed the
                 serial sum by the hand-off costs, which the serial sum does not include
  lower_bound  = max(critical path, busiest pool, total L1 service, total HBM service)

Pools are exclusive (all units of a pool cooperate on one op) and L1/HBM are treated as shared resources whose
total service time is a floor, so the estimate cannot beat either memory system.
"""


def schedule(durations, pools, deps, l1_s, hbm_s, sync_s, aux_busy):
    """aux_busy[i] = {pool: seconds} for ops that also occupy a second pool (fused attention: matrix + softmax on vector)."""
    n = len(durations)
    finish, pool_free, cp = [0.0] * n, {}, [0.0] * n
    busy = {}
    for i in range(n):
        ready = crit = 0.0
        for d in deps[i]:
            hop = sync_s if pools[d] != pools[i] else 0.0
            ready = max(ready, finish[d] + hop)
            crit = max(crit, cp[d] + hop)
        start = max(ready, pool_free.get(pools[i], 0.0), *(pool_free.get(p, 0.0) for p in aux_busy[i]))
        finish[i] = start + durations[i]
        pool_free[pools[i]] = finish[i]
        cp[i] = crit + durations[i]
        busy[pools[i]] = busy.get(pools[i], 0.0) + durations[i]
        for pool, t in aux_busy[i].items():
            pool_free[pool] = max(pool_free.get(pool, 0.0), start + t)
            busy[pool] = busy.get(pool, 0.0) + t
    l1_total, hbm_total = sum(l1_s), sum(hbm_s)
    listed = max(max(finish, default=0.0), l1_total, hbm_total)
    lower = max(max(cp, default=0.0), max(busy.values(), default=0.0), l1_total, hbm_total)
    return {"serial": sum(durations), "list": listed, "lower_bound": min(lower, listed),
            "critical_path": max(cp, default=0.0), "pool_busy": busy, "l1_total": l1_total, "hbm_total": hbm_total}
