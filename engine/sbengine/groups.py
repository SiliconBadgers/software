"""Anchored fusion groups and parallel projection groups on a captured graph.

Port of the grouping rule in research/compute-mapping/2026-10-02-op-dependency-map/build_map.py (Adrian Luo,
PR #10) onto this engine's graph loader, so the dependency map and the engine read one graph and the rule can be
selected with `memory.fusion = "groups"`. tests/test_groups.py checks that it reproduces that study's committed
group assignment on the four 2026-09-24 captures.

  parallel group   weight MUL_MATs in one layer that read the same single activation (they could be stacked
                   into one taller multiply)
  anchor unit      the compute resource a fused kernel would run on: a parallel group, the attention core of a
                   layer (QK and AV, or the captured FLASH_ATTN_EXT), or one matrix / conv / recurrent op
  fusion group     producers merged into their consumer when the producer has exactly one consumer in the same
                   layer and the merged group would still hold at most one anchor unit. Vector and memory ops
                   merge in schedule order, so an op between two anchors joins the earlier one as its epilogue.

FLASH_ATTN_EXT is treated as an attention-core anchor; the original study only covered flash-attention-off
graphs, so that case is an extension of its rule, not a reproduction.

Groups are candidates the dependencies allow. They do not claim that a kernel or hardware unit implements them.
"""
import re
from dataclasses import dataclass, field

from .graph import MEMORY_OPS

ANCHOR_CATEGORIES = {"weight_matrix", "act_matrix", "conv", "recurrent"}
_LAYER = re.compile(r"-(\d+)\b|_l(\d+)\b")


@dataclass
class Groups:
    layer: list                 # per node: layer number, "pre" (embedding) or "post" (output head)
    category: list              # per node: weight_matrix | act_matrix | conv | recurrent | memory | vector
    unit: list                  # per node: anchor unit id (P<layer>.<k>, A<layer>, U<spos>) or None
    fusion_of: list             # per node: fusion group id F<layer>.<k>
    parallel: dict = field(default_factory=dict)   # id -> {"layer", "input" (node idx), "members" [node idx]}
    fusion: dict = field(default_factory=dict)     # id -> {"layer", "members" [node idx], "units" [unit id]}
    fused_roots: set = field(default_factory=set)  # activation roots that stay inside one fusion group


def _layers(g):
    """Layer of every node, carried forward over the whole schedule (views included) from tensor names."""
    count = g.meta.get("layers")
    last = f"l_out-{count - 1}" if count else None
    current, by_tid = "pre", {}
    for tid in g.order:
        name = g.tensors[tid].name
        m = _LAYER.search(name)
        if m:
            current = int(m.group(1) or m.group(2))
        by_tid[tid] = current
        if name == last:
            current = "post"
    return [by_tid[n.tensor.id] for n in g.nodes]


def _category(g, n):
    op = n.tensor.op
    if op == "MUL_MAT":
        weight = 0 in n.slots and g.kind(g.root(n.slots[0].id)) == "weight"
        return "weight_matrix" if weight else "act_matrix"
    if op == "FLASH_ATTN_EXT":
        return "act_matrix"
    if op == "SSM_CONV":
        return "conv"
    if op == "GATED_DELTA_NET":
        return "recurrent"
    return "memory" if op in MEMORY_OPS else "vector"


def analyze(g):
    """Groups for one captured phase graph (cached on the graph)."""
    if "_groups" in g.__dict__:
        return g.__dict__["_groups"]
    nodes = g.nodes
    layer = _layers(g)
    category = [_category(g, n) for n in nodes]
    preds = [sorted({g.producer[r] for r in n.src_roots if r in g.producer and g.producer[r] != n.idx}) for n in nodes]
    consumers = [set() for _ in nodes]
    for n in nodes:
        for p in preds[n.idx]:
            consumers[p].add(n.idx)

    # parallel groups: weight multiplies in one layer that share their only activation input
    by_input = {}
    for n in nodes:
        if category[n.idx] == "weight_matrix" and len(preds[n.idx]) == 1:
            by_input.setdefault((layer[n.idx], preds[n.idx][0]), []).append(n.idx)
    unit, parallel, counter = [None] * len(nodes), {}, {}
    for (lyr, src), members in sorted(by_input.items(), key=lambda kv: kv[0][1]):
        if len(members) < 2:
            continue
        gid = f"P{lyr}.{counter.get(lyr, 0)}"
        counter[lyr] = counter.get(lyr, 0) + 1
        parallel[gid] = {"layer": lyr, "input": src, "members": members}
        for m in members:
            unit[m] = gid
    for n in nodes:
        if unit[n.idx] is None and category[n.idx] in ANCHOR_CATEGORIES:
            unit[n.idx] = f"A{layer[n.idx]}" if category[n.idx] == "act_matrix" else f"U{n.spos}"

    # fusion groups: merge a single-consumer producer into its consumer, at most one anchor unit per group
    group = list(range(len(nodes)))
    members = {}
    units = {}
    for n in nodes:
        i = n.idx
        members[i] = [i]
        units[i] = {unit[i]} if unit[i] else set()
        for p in preds[i]:
            if consumers[p] != {i} or layer[p] != layer[i]:
                continue
            gp, gn = group[p], group[i]
            if gp == gn or len(units[gp] | units[gn]) > 1:
                continue
            for m in members[gp]:
                group[m] = gn
            members[gn].extend(members.pop(gp))
            units[gn] |= units.pop(gp)
    fusion_of, fusion, counter = [None] * len(nodes), {}, {}
    for root in sorted(members, key=lambda r: min(members[r])):
        mem = sorted(members[root])
        lyr = layer[mem[0]]
        gid = f"F{lyr}.{counter.get(lyr, 0)}"
        counter[lyr] = counter.get(lyr, 0) + 1
        for m in mem:
            fusion_of[m] = gid
        fusion[gid] = {"layer": lyr, "members": mem, "units": sorted(units[root])}

    fused = set()
    for root, p in g.producer.items():
        c = consumers[p]
        if len(c) == 1 and fusion_of[next(iter(c))] == fusion_of[p]:
            fused.add(root)
    out = Groups(layer, category, unit, fusion_of, parallel, fusion, fused)
    g.__dict__["_groups"] = out
    return out


def summary(g):
    """Counts per layer type, for the CLI and for comparison with the dependency-map study."""
    gr = analyze(g)
    types = {}
    for lyr in dict.fromkeys(gr.layer):
        cats = {gr.category[i] for i, x in enumerate(gr.layer) if x == lyr}
        kind = ("embedding" if lyr == "pre" else "output_head" if lyr == "post" else "deltanet" if "recurrent" in cats
                else "full_attention" if "act_matrix" in cats else "other")
        types.setdefault(kind, []).append(lyr)
    per_type = {}
    for kind, layers in types.items():
        first = layers[0]
        per_type[kind] = {
            "layers": len(layers),
            "ops_per_layer": sum(1 for x in gr.layer if x == first),
            "parallel_groups_per_layer": sum(1 for p in gr.parallel.values() if p["layer"] == first),
            "fusion_groups_per_layer": sum(1 for f in gr.fusion.values() if f["layer"] == first),
        }
    return {"nodes": len(g.nodes), "parallel_groups": len(gr.parallel), "fusion_groups": len(gr.fusion),
            "fused_edges": len(gr.fused_roots), "layer_types": per_type}
