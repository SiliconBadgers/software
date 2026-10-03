#!/usr/bin/env python3
"""Operation dependency map for captured Qwen3.5-2B llama.cpp graphs.

Reads graph JSON captured by experiments/llama-cpp/2026-09-24-qwen35-2b
(src/capture_graph.cpp) and writes, for each graph:

  nodes.csv            every operation that does work, with layer, category,
                       shapes, weights, persistent-state access and MACs
  edges.csv            data dependencies after collapsing metadata-only views,
                       plus ordering edges for in-place cache/state writes
  chain_<type>.csv     operation chain of one representative layer per type
  parallel_groups.csv  weight matrix multiplies that read the same activation
  fusion_groups.csv    single-consumer fusion candidates around at most one
                       compute anchor
  handoffs.csv         cross-category edges, split into fused and remaining
  summary.json         input hash, counts, per-layer-type totals, validations

Standard library only. Shapes are in ggml order (ne[0] innermost).
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
GRAPHS = REPO / "experiments/llama-cpp/2026-09-24-qwen35-2b/graphs"
DEFAULT_GRAPHS = [GRAPHS / "pp128/decode.json"]

METADATA_OPS = {"VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"}
MEMORY_OPS = {"CPY", "SET_ROWS", "GET_ROWS", "CONCAT", "CONT"}
# A fusion group may contain at most one anchor unit: the compute resource
# that would execute the fused kernel.
ANCHOR_CATEGORIES = {"weight_matrix", "act_matrix", "conv", "recurrent"}
LAYER_RE = re.compile(r"-(\d+)\b|_l(\d+)\b")


def rel(name: str) -> str:
    """Layer-independent tensor name."""
    if re.fullmatch(r"node_\d+", name):
        return ""
    name = re.sub(r"blk\.\d+\.", "", name)
    name = re.sub(r"_l\d+\b", "_lN", name)
    return re.sub(r"-\d+\b", "", name)


def fmt_shape(shape: list[int]) -> str:
    dims = list(shape)
    while len(dims) > 2 and dims[-1] == 1:
        dims.pop()
    return "x".join(str(d) for d in dims)


class Edge:
    __slots__ = ("src", "dst", "kind", "bytes", "via")

    def __init__(self, src: int, dst: int, kind: str, nbytes: int, via: str = ""):
        self.src, self.dst, self.kind, self.bytes, self.via = src, dst, kind, nbytes, via


class Graph:
    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        self.path = path
        self.sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        self.meta = data["metadata"]
        self.t = {t["id"]: t for t in data["tensors"]}
        self.schedule = data["scheduled_nodes"]
        self.pos = {tid: i for i, tid in enumerate(self.schedule)}
        self.base = {tid: self._base(tid) for tid in self.t}
        self.active = [tid for tid in self.schedule if self._does_work(tid)]
        self.active_set = set(self.active)
        self.layer = self._assign_layers()
        self.category = {n: self._category(n) for n in self.active}
        self.state_buffers = {self.base[n] for n in self.active if self.writes_state(n)}
        self._producers: dict[int, set[int]] = {}
        self._build_edges()
        self.layer_type = self._layer_types()

    # --- step 1: nodes and edges -------------------------------------------

    def _does_work(self, tid: int) -> bool:
        t = self.t[tid]
        return t["op"] not in METADATA_OPS and t["op"] != "NONE" and t["elements"] > 0

    def _base(self, tid: int) -> int:
        while self.t[tid].get("view_source", -1) != -1:
            tid = self.t[tid]["view_source"]
        return tid

    def is_leaf(self, tid: int) -> bool:
        return self.t[tid]["op"] == "NONE"

    def is_weight(self, tid: int) -> bool:
        return bool(self.t[self.base[tid]]["weight_buffer"])

    def writes_state(self, tid: int) -> bool:
        """True when the node's output is an in-place view of a persistent input buffer."""
        b = self.base[tid]
        return b != tid and self.is_leaf(b) and not self.t[b]["weight_buffer"]

    def producers(self, tid: int) -> set[int]:
        """Active nodes or leaves that supply the data seen through tensor tid."""
        if tid in self._producers:
            return self._producers[tid]
        t = self.t[tid]
        if tid in self.active_set or self.is_leaf(tid):
            out = {tid}
        elif t["op"] in METADATA_OPS:
            out = self.producers(t["sources"][0]["tensor"])
        else:  # zero-sized compute node: pass its inputs through
            out = set()
            for s in t.get("sources", []):
                out |= self.producers(s["tensor"])
        self._producers[tid] = out
        return out

    def _assign_layers(self) -> dict[int, object]:
        last = f"l_out-{self.meta['layers'] - 1}"
        layer: dict[int, object] = {}
        current: object = "pre"
        for tid in self.schedule:
            name = self.t[tid]["name"]
            m = LAYER_RE.search(name)
            if m:
                current = int(m.group(1) or m.group(2))
            layer[tid] = current
            if name == last:
                current = "post"
        return layer

    def _category(self, n: int) -> str:
        op = self.t[n]["op"]
        if op == "MUL_MAT":
            return "weight_matrix" if self.is_weight(self.t[n]["sources"][0]["tensor"]) else "act_matrix"
        if op == "SSM_CONV":
            return "conv"
        if op == "GATED_DELTA_NET":
            return "recurrent"
        if op in MEMORY_OPS:
            return "memory"
        return "vector"

    def _layer_types(self) -> dict[object, str]:
        cats: dict[object, set[str]] = collections.defaultdict(set)
        for n in self.active:
            cats[self.layer[n]].add(self.category[n])
        types = {}
        for layer, c in cats.items():
            if layer == "pre":
                types[layer] = "embedding"
            elif layer == "post":
                types[layer] = "output_head"
            elif "recurrent" in c:
                types[layer] = "deltanet"
            elif "act_matrix" in c:
                types[layer] = "full_attention"
            else:
                types[layer] = "other"
        return types

    def _build_edges(self) -> None:
        self.data_edges: list[Edge] = []
        self.leaf_inputs: dict[int, list[tuple[int, int]]] = collections.defaultdict(list)
        for n in self.active:
            dest = self.base[n] if self.writes_state(n) else None
            acc: dict[int, int] = {}
            for s in self.t[n].get("sources", []):
                via = s["tensor"]
                if dest is not None and self.base[via] == dest:
                    continue  # the buffer being written, not an input
                for p in sorted(self.producers(via)):
                    if p in self.active_set:
                        acc[p] = acc.get(p, 0) + self.t[via]["logical_bytes"]
                    else:
                        self.leaf_inputs[n].append((p, via))
            for p, nbytes in acc.items():
                self.data_edges.append(Edge(p, n, "data", nbytes))

        self.preds: dict[int, list[Edge]] = collections.defaultdict(list)
        self.consumers: dict[int, set[int]] = collections.defaultdict(set)
        for e in self.data_edges:
            self.preds[e.dst].append(e)
            self.consumers[e.src].add(e.dst)

        # Ordering edges for in-place writes into persistent buffers (KV cache,
        # conv and recurrent state). These are not visible as source edges.
        self.state_edges: list[Edge] = []
        events: dict[int, list[tuple[int, str, int, int]]] = collections.defaultdict(list)
        for n in self.active:
            if self.writes_state(n):
                events[self.base[n]].append((self.pos[n], "w", n, self.written_bytes(n)))
            for leaf, via in self.leaf_inputs[n]:
                if leaf in self.state_buffers:
                    events[leaf].append((self.pos[n], "r", n, self.t[via]["logical_bytes"]))
        for buf, evs in events.items():
            name = rel(self.t[buf]["name"])
            last_write, reads = None, []
            for _, kind, n, nbytes in sorted(evs):
                if kind == "r":
                    if last_write is not None:
                        self.state_edges.append(Edge(last_write, n, "state_raw", nbytes, name))
                    reads.append(n)
                else:
                    for r in reads:
                        self.state_edges.append(Edge(r, n, "state_war", nbytes, name))
                    if last_write is not None:
                        self.state_edges.append(Edge(last_write, n, "state_waw", nbytes, name))
                    last_write, reads = n, []

        self.state_preds: dict[int, list[Edge]] = collections.defaultdict(list)
        for e in self.state_edges:
            self.state_preds[e.dst].append(e)

    def written_bytes(self, n: int) -> int:
        """Logical bytes of the value a state writer stores (not the destination view)."""
        return sum(e.bytes for e in self.preds[n])

    def state_read_bytes(self, n: int) -> int:
        return sum(self.t[via]["logical_bytes"] for leaf, via in self.leaf_inputs[n]
                   if leaf in self.state_buffers)

    def leaf_kind(self, leaf: int) -> str:
        if self.t[leaf]["weight_buffer"]:
            return "W"
        if leaf in self.state_buffers:
            return "S"
        return "I"

    def gemm(self, n: int) -> tuple[int, int, int, int, int]:
        """(M, K, N, batch, MACs) for MUL_MAT; MACs only for SSM_CONV."""
        t = self.t[n]
        op = t["op"]
        if op == "MUL_MAT":
            k = self.t[t["sources"][0]["tensor"]]["shape"][0]
            m, nn, b2, b3 = t["shape"]
            return m, k, nn, b2 * b3, m * nn * b2 * b3 * k
        if op == "SSM_CONV":
            width = self.t[t["sources"][1]["tensor"]]["shape"][0]
            return 0, 0, 0, 0, t["elements"] * width
        return 0, 0, 0, 0, 0

    # --- step 5: parallel groups and fusion ---------------------------------

    def find_parallel_groups(self) -> None:
        by_input: dict[tuple[object, int], list[int]] = collections.defaultdict(list)
        for n in self.active:
            if self.category[n] != "weight_matrix":
                continue
            acts = [e.src for e in self.preds[n]]
            if len(acts) == 1:
                by_input[(self.layer[n], acts[0])].append(n)
        self.parallel: dict[str, dict] = {}
        self.unit: dict[int, str | None] = {}
        counter: dict[object, int] = collections.defaultdict(int)
        for (layer, src), members in sorted(by_input.items(), key=lambda kv: self.pos[kv[0][1]]):
            if len(members) < 2:
                continue
            gid = f"P{layer}.{counter[layer]}"
            counter[layer] += 1
            self.parallel[gid] = {"layer": layer, "input": src, "members": members}
            for m in members:
                self.unit[m] = gid
        for n in self.active:
            if n in self.unit:
                continue
            cat = self.category[n]
            if cat == "act_matrix":
                self.unit[n] = f"A{self.layer[n]}"  # QK and AV form one attention core
            elif cat in ANCHOR_CATEGORIES:
                self.unit[n] = f"U{n}"
            else:
                self.unit[n] = None

    def find_fusion_groups(self) -> None:
        group: dict[int, int] = {}
        members: dict[int, list[int]] = {}
        units: dict[int, set[str]] = {}
        for n in self.active:
            group[n], members[n] = n, [n]
            units[n] = {self.unit[n]} if self.unit[n] else set()
            for p in sorted({e.src for e in self.preds[n]}, key=self.pos.get):
                if self.consumers[p] != {n} or self.layer[p] != self.layer[n]:
                    continue
                gp, gn = group[p], group[n]
                if gp == gn or len(units[gp] | units[gn]) > 1:
                    continue
                for m in members[gp]:
                    group[m] = gn
                members[gn].extend(members.pop(gp))
                units[gn] |= units.pop(gp)
        self.fusion_of: dict[int, str] = {}
        self.fusion: dict[str, dict] = {}
        counter: dict[object, int] = collections.defaultdict(int)
        for root in sorted(members, key=lambda g: min(self.pos[m] for m in members[g])):
            mem = sorted(members[root], key=self.pos.get)
            layer = self.layer[mem[0]]
            gid = f"F{layer}.{counter[layer]}"
            counter[layer] += 1
            for m in mem:
                self.fusion_of[m] = gid
            self.fusion[gid] = {"layer": layer, "members": mem, "units": sorted(units[root])}

    def fusion_label(self, mem: list[int], unit_ids: list[str]) -> str:
        anchors = [m for m in mem if self.category[m] in ANCHOR_CATEGORIES]
        if not anchors:
            cats = {self.category[m] for m in mem}
            return "memory only" if cats == {"memory"} else "vector/memory only"
        u = unit_ids[0]
        if u.startswith("P"):
            kind = "parallel weight_matrix group"
        elif u.startswith("A"):
            kind = "attention core"
        else:
            kind = self.category[anchors[0]]
        # Prologue ops feed an anchor; epilogue ops consume an anchor's result;
        # side inputs join the epilogue without passing through an anchor.
        inside = set(mem)
        succ: dict[int, list[int]] = collections.defaultdict(list)
        pred: dict[int, list[int]] = collections.defaultdict(list)
        for m in mem:
            for e in self.preds[m]:
                if e.src in inside:
                    succ[e.src].append(m)
                    pred[m].append(e.src)

        def reach(start: list[int], adj: dict[int, list[int]]) -> set[int]:
            seen, stack = set(), list(start)
            while stack:
                for x in adj[stack.pop()]:
                    if x not in seen:
                        seen.add(x)
                        stack.append(x)
            return seen

        before, after = reach(anchors, pred), reach(anchors, succ)
        others = [m for m in mem if m not in anchors]
        if any(m in before for m in others):
            kind += " + prologue"
        if any(m in after for m in others):
            kind += " + epilogue"
        if any(m not in before and m not in after for m in others):
            kind += " + side input"
        return kind


# --- output -----------------------------------------------------------------


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def describe_inputs(g: Graph, n: int, step: dict[int, int], external: str | None = None) -> str:
    parts = []
    for e in sorted(g.preds[n], key=lambda e: g.pos[e.src]):
        if e.src in step:
            parts.append(f"#{step[e.src]}")
        else:
            parts.append(external or f"<{rel(g.t[e.src]['name']) or g.t[e.src]['op']}")
    for leaf, _ in g.leaf_inputs[n]:
        t = g.t[leaf]
        parts.append(f"{g.leaf_kind(leaf)}:{rel(t['name'])}[{fmt_shape(t['shape'])} {t['dtype']}]")
    return "; ".join(parts)


def describe_consumers(g: Graph, n: int, step: dict[int, int], external: str | None = None) -> str:
    out = []
    for c in sorted(g.consumers[n], key=g.pos.get):
        out.append(f"#{step[c]}" if c in step else external or f">{g.layer_type[g.layer[c]]}")
    return "; ".join(dict.fromkeys(out))


def describe_state(g: Graph, n: int, step: dict[int, int]) -> str:
    parts = []
    if g.writes_state(n):
        parts.append(f"writes {rel(g.t[g.base[n]]['name'])}")
    for leaf, _ in g.leaf_inputs[n]:
        if leaf in g.state_buffers:
            parts.append(f"reads {rel(g.t[leaf]['name'])}")
    for e in g.state_preds[n]:
        if e.src in step:
            parts.append(f"{e.kind} after #{step[e.src]}")
    return "; ".join(parts)


def layer_rows(g: Graph, layer: object) -> tuple[list[list], list[tuple]]:
    """Chain rows for one layer, plus a signature that ignores which outside
    node feeds or consumes the layer (embedding vs previous layer, next layer
    vs output head)."""
    nodes = [n for n in g.active if g.layer[n] == layer]
    step = {n: i for i, n in enumerate(nodes)}
    rows, sig = [], []
    for n in nodes:
        t = g.t[n]
        m, k, nn, b, macs = g.gemm(n)
        inputs = describe_inputs(g, n, step)
        consumers = describe_consumers(g, n, step)
        state = describe_state(g, n, step)
        gemm = f"M={m} K={k} N={nn}" + (f" batch={b}" if b > 1 else "") if t["op"] == "MUL_MAT" else ""
        rows.append([
            step[n], g.pos[n], t["op"], g.category[n], rel(t["name"]), fmt_shape(t["shape"]),
            t["dtype"], t["logical_bytes"], gemm, macs, inputs, consumers, state,
            g.unit[n] if g.unit[n] and g.unit[n].startswith("P") else "",
            g.fusion_of[n].split(".", 1)[1],
        ])
        sig.append((t["op"], g.category[n], tuple(t["shape"]), t["dtype"],
                    describe_inputs(g, n, step, "<outside"), describe_consumers(g, n, step, ">outside"),
                    state))
    return rows, sig


def build(graph_path: Path, out_dir: Path) -> bool:
    g = Graph(graph_path)
    g.find_parallel_groups()
    g.find_fusion_groups()
    out_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # nodes.csv
    rows = []
    for n in g.active:
        t = g.t[n]
        m, k, nn, b, macs = g.gemm(n)
        weights = [f"{rel(g.t[l]['name'])}[{fmt_shape(g.t[l]['shape'])} {g.t[l]['dtype']}]"
                   for l, _ in g.leaf_inputs[n] if g.leaf_kind(l) == "W"]
        rows.append([g.pos[n], n, t["name"], t["op"], g.category[n], g.layer[n],
                     g.layer_type[g.layer[n]], fmt_shape(t["shape"]), t["dtype"], t["logical_bytes"],
                     m or "", k or "", nn or "", b or "", macs, "; ".join(weights),
                     describe_state(g, n, {}), g.unit[n] or "", g.fusion_of[n]])
    write_csv(out_dir / "nodes.csv",
              ["sched_pos", "tensor_id", "name", "op", "category", "layer", "layer_type", "shape_ggml",
               "dtype", "out_bytes", "M", "K", "N", "batch", "macs", "weights", "state", "unit",
               "fusion_group"], rows)

    # edges.csv
    rows = []
    for e in g.data_edges + g.state_edges:
        rows.append([g.pos[e.src], g.t[e.src]["name"], g.category[e.src], g.layer[e.src],
                     g.pos[e.dst], g.t[e.dst]["name"], g.category[e.dst], g.layer[e.dst],
                     e.kind, e.bytes, e.via])
    rows.sort(key=lambda r: (r[4], r[0]))
    write_csv(out_dir / "edges.csv",
              ["src_pos", "src_name", "src_category", "src_layer", "dst_pos", "dst_name",
               "dst_category", "dst_layer", "kind", "bytes", "state_buffer"], rows)

    # chain_<type>.csv: first layer of each type, checked against the others
    by_type: dict[str, list[object]] = collections.defaultdict(list)
    for layer in dict.fromkeys(g.layer[n] for n in g.active):
        by_type[g.layer_type[layer]].append(layer)
    type_summary = {}
    for ltype, layers in by_type.items():
        rows, ref = layer_rows(g, layers[0])
        write_csv(out_dir / f"chain_{ltype}.csv",
                  ["step", "sched_pos", "op", "category", "name", "shape_ggml", "dtype", "out_bytes",
                   "gemm", "macs", "inputs", "consumers", "state", "parallel_group", "fusion_group"], rows)
        differing = [str(l) for l in layers[1:] if layer_rows(g, l)[1] != ref]
        check(f"{ltype}: all {len(layers)} layers structurally identical", not differing,
              f"differs from layer {layers[0]}: {', '.join(differing)}" if differing else "")
        type_summary[ltype] = {"layers": [str(l) for l in layers], "representative": str(layers[0]),
                               "active_ops_per_layer": len(rows)}

    # parallel_groups.csv
    rows = []
    for gid, pg in g.parallel.items():
        mem = sorted(pg["members"], key=g.pos.get)
        gm = [g.gemm(m) for m in mem]
        wbytes = sum(g.t[l]["logical_bytes"] for m in mem for l, _ in g.leaf_inputs[m] if g.leaf_kind(l) == "W")
        rows.append([gid, pg["layer"], g.layer_type[pg["layer"]], rel(g.t[pg["input"]]["name"]),
                     "; ".join(rel(g.t[m]["name"]) or g.t[m]["op"] for m in mem),
                     "; ".join(str(x[0]) for x in gm), sum(x[0] for x in gm),
                     "; ".join(sorted({str(x[1]) for x in gm})), gm[0][2], wbytes, sum(x[4] for x in gm)])
    write_csv(out_dir / "parallel_groups.csv",
              ["group", "layer", "layer_type", "shared_input", "members", "rows_each", "rows_total",
               "K", "N", "weight_bytes", "macs"], rows)

    # fusion_groups.csv
    rows = []
    for gid, fg in g.fusion.items():
        mem = fg["members"]
        inside = set(mem)
        internal = [e for e in g.data_edges if e.src in inside and e.dst in inside]
        crossings = [e for e in internal if g.category[e.src] != g.category[e.dst]]
        fg["label"] = g.fusion_label(mem, fg["units"]) if len(mem) > 1 else "single op"
        fg["internal_bytes"] = sum(e.bytes for e in internal)
        fg["handoffs_removed"] = len(crossings)
        rows.append([gid, fg["layer"], g.layer_type[fg["layer"]], len(mem), fg["label"],
                     " -> ".join(g.t[m]["op"] for m in mem),
                     " -> ".join(rel(g.t[m]["name"]) or "_" for m in mem),
                     "/".join(g.category[m] for m in mem), "; ".join(fg["units"]),
                     int(any(g.writes_state(m) for m in mem)),
                     int(any(l in g.state_buffers for m in mem for l, _ in g.leaf_inputs[m])),
                     len(internal), fg["internal_bytes"], fg["handoffs_removed"]])
    write_csv(out_dir / "fusion_groups.csv",
              ["group", "layer", "layer_type", "n_ops", "pattern", "ops", "names", "categories", "units",
               "writes_state", "reads_state", "internal_edges", "internal_bytes", "handoffs_removed"], rows)

    # handoffs.csv: edges whose endpoints have different categories. One tensor
    # sent to several ops of the same fusion group counts once.
    agg: dict[tuple, list[int]] = collections.defaultdict(lambda: [0, 0])
    seen_handoffs: set[tuple[int, str]] = set()
    for e in g.data_edges:
        a, b = g.category[e.src], g.category[e.dst]
        if a == b or (e.src, g.fusion_of[e.dst]) in seen_handoffs:
            continue
        seen_handoffs.add((e.src, g.fusion_of[e.dst]))
        if g.layer[e.src] != g.layer[e.dst]:
            status = "inter-layer"
        elif g.fusion_of[e.src] == g.fusion_of[e.dst]:
            status = "fused"
        else:
            status = "remaining"
        key = (g.layer_type[g.layer[e.dst]], a, b, status)
        agg[key][0] += 1
        agg[key][1] += e.bytes
    rows = []
    for (ltype, a, b, status), (count, nbytes) in sorted(agg.items()):
        n_layers = len(by_type[ltype])
        rows.append([ltype, a, b, status, count, n_layers, round(count / n_layers, 2), nbytes])
    write_csv(out_dir / "handoffs.csv",
              ["layer_type", "from_category", "to_category", "status", "count_total", "layers",
               "count_per_layer", "bytes_total"], rows)

    # per-type totals
    for ltype, info in type_summary.items():
        layers = set(by_type[ltype])
        n_layers = len(layers)
        nodes = [n for n in g.active if g.layer[n] in layers]
        fgs = [fg for fg in g.fusion.values() if fg["layer"] in layers]
        handoff = {s: sum(c for (lt, _, _, st), (c, _) in agg.items() if lt == ltype and st == s)
                   for s in ("fused", "remaining", "inter-layer")}
        patterns = collections.Counter(fg["label"] for fg in fgs)
        info.update({
            "macs_per_layer": sum(g.gemm(n)[4] for n in nodes) // n_layers,
            "categories_per_layer": {c: v // n_layers for c, v in
                                     collections.Counter(g.category[n] for n in nodes).items()},
            "parallel_groups_per_layer": sum(1 for pg in g.parallel.values() if pg["layer"] in layers) // n_layers,
            "fusion_groups_per_layer": len(fgs) // n_layers,
            "fusion_patterns_per_layer": {k: v // n_layers for k, v in sorted(patterns.items())},
            "cross_category_edges_per_layer": {
                "before_fusion": (handoff["fused"] + handoff["remaining"]) / n_layers,
                "after_fusion": handoff["remaining"] / n_layers,
            },
            "fused_internal_bytes_per_layer": sum(fg["internal_bytes"] for fg in fgs) // n_layers,
            "state_read_bytes_per_layer": sum(g.state_read_bytes(n) for n in nodes) // n_layers,
            "state_write_bytes_per_layer": sum(g.written_bytes(n) for n in nodes
                                               if g.writes_state(n)) // n_layers,
        })

    # validations
    n_meta = sum(1 for tid in g.schedule if g.t[tid]["op"] in METADATA_OPS)
    n_zero = len(g.schedule) - n_meta - len(g.active)
    check("scheduled nodes = active + metadata-only + zero-sized",
          all(g.t[tid]["op"] in METADATA_OPS or tid in g.active_set or g.t[tid]["elements"] == 0
              for tid in g.schedule), f"{len(g.active)} + {n_meta} + {n_zero} = {len(g.schedule)}")
    backwards = [e for e in g.data_edges + g.state_edges if g.pos[e.src] >= g.pos[e.dst]]
    check("captured schedule respects every data and state edge", not backwards,
          f"{len(backwards)} backward edges" if backwards else "")
    covered = collections.Counter(m for fg in g.fusion.values() for m in fg["members"])
    check("every active op is in exactly one fusion group",
          set(covered) == g.active_set and set(covered.values()) == {1})
    check("every fusion group has at most one anchor unit",
          all(len(fg["units"]) <= 1 for fg in g.fusion.values()))
    unordered = [rel(g.t[b]["name"]) for b in g.state_buffers
                 if not any(e.via == rel(g.t[b]["name"]) for e in g.state_edges)]
    check("every written state buffer has ordering edges", not unordered, ", ".join(sorted(set(unordered))))
    macs = collections.Counter()
    for n in g.active:
        if g.t[n]["op"] in ("MUL_MAT", "SSM_CONV"):
            macs[g.t[n]["op"]] += g.gemm(n)[4]
    ref_path = graph_path.with_name(graph_path.stem + ".summary.json")
    if ref_path.exists():
        ref = json.loads(ref_path.read_text(encoding="utf-8"))["macs_by_supported_op"]
        check("MACs match the capture's summary.json", dict(macs) == ref, f"ours {dict(macs)} vs {ref}")

    try:
        shown = graph_path.resolve().relative_to(REPO).as_posix()
    except ValueError:
        shown = str(graph_path)
    summary = {
        "input": shown,
        "input_sha256": g.sha256,
        "metadata": {k: g.meta[k] for k in ("phase", "input_tokens", "past_tokens", "context_capacity",
                                            "flash_attention", "kv_dtype", "backend", "layers")},
        "counts": {
            "scheduled_nodes": len(g.schedule),
            "active_ops": len(g.active),
            "metadata_only_skipped": n_meta,
            "zero_sized_skipped": n_zero,
            "data_edges": len(g.data_edges),
            "state_edges": dict(collections.Counter(e.kind for e in g.state_edges)),
            "state_buffers": len(g.state_buffers),
            "parallel_groups": len(g.parallel),
            "fusion_groups": len(g.fusion),
            "macs": dict(macs),
        },
        "layer_types": type_summary,
        "validations": checks,
        "method_notes": [
            "Metadata-only ops (VIEW/RESHAPE/PERMUTE/TRANSPOSE) and zero-sized nodes are collapsed; "
            "their edges pass through to the real producer.",
            "State edges order in-place CPY/SET_ROWS writes against reads of the same persistent buffer; "
            "whole-buffer granularity, so they are conservative.",
            "Parallel groups: weight MUL_MATs in one layer that read the same activation.",
            "Fusion groups: producer->consumer merges where the producer has exactly one consumer in the "
            "same layer, with at most one anchor unit (a parallel group, the QK/AV attention core, or a "
            "single matrix/conv/recurrent op). Vector ops are merged in schedule order, so an op between "
            "two anchors joins the earlier one. Labels: prologue ops feed the anchor, epilogue ops consume "
            "its result, side inputs join the epilogue without passing through the anchor.",
            "Handoffs count one tensor sent to one fusion group once, even if several ops in it read it.",
            "State read bytes are the logical size of the views read; state write bytes are the size of the "
            "value written (source dtype), not of the destination cache view.",
            "Bytes are logical tensor sizes from the capture, not measured memory traffic.",
        ],
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=2) + "\n")

    print(f"{shown}: {len(g.active)} active ops, {len(g.data_edges)} data + {len(g.state_edges)} state edges, "
          f"{len(g.parallel)} parallel groups, {len(g.fusion)} fusion groups -> {out_dir}")
    for c in checks:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']}" + (f"  ({c['detail']})" if c["detail"] else ""))
    return all(c["pass"] for c in checks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("graphs", nargs="*", type=Path, help="graph JSON files (default: pp128/decode.json)")
    ap.add_argument("--out-dir", type=Path, default=HERE / "out",
                    help="output root; each graph writes <workload>-<phase>/ inside it")
    args = ap.parse_args()
    out_root = args.out_dir.resolve()
    protected = (REPO / "experiments").resolve()
    if out_root == protected or protected in out_root.parents:
        print(f"refusing to write under {protected}; recorded experiments stay unchanged", file=sys.stderr)
        return 2
    ok = True
    for path in args.graphs or DEFAULT_GRAPHS:
        ok &= build(path, out_root / f"{path.parent.name}-{path.stem}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
