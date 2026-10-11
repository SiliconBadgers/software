"""Anchored fusion groups (sbengine/groups.py): the rule on small synthetic graphs, and a reproduction of the
2026-10-02 dependency-map study's committed tables on the four 2026-09-24 captures it was built from."""
import csv
import unittest
from functools import lru_cache

from helpers import ENGINE, Builder, real_run
from sbengine import config, groups
from sbengine.graph import Graph, load_workload
from sbengine.memory import fused_roots
from sbengine.model import evaluate

REPO = ENGINE.parent
CAPTURES = REPO / "experiments/llama-cpp/2026-09-24-qwen35-2b/graphs"
STUDY = REPO / "research/compute-mapping/2026-10-02-op-dependency-map/out"
RUN = real_run()


class Rule(unittest.TestCase):
    def _layer(self):
        """One layer: a norm feeding two weight multiplies, each with an epilogue, joined by an add."""
        b = Builder()
        w1, w2 = b.add("blk.0.a.weight", [4, 8], weight=True), b.add("blk.0.b.weight", [4, 8], weight=True)
        x = b.add("inp", [4, 2], op="ROPE")
        norm = b.add("norm-0", [4, 2], op="RMS_NORM", src=(x,))
        a = b.add("a-0", [8, 2], op="MUL_MAT", src=(w1, norm))
        c = b.add("b-0", [8, 2], op="MUL_MAT", src=(w2, norm))
        a2 = b.add("a_act-0", [8, 2], op="UNARY", desc="SILU", src=(a,))
        c2 = b.add("b_act-0", [8, 2], op="UNARY", desc="SIGMOID", src=(c,))
        out = b.add("l_out-0", [8, 2], op="MUL", src=(a2, c2))
        return b.graph(), {"norm": norm, "a": a, "b": c, "a_act": a2, "b_act": c2, "out": out}

    def test_parallel_group_and_single_anchor_fusion(self):
        g, t = self._layer()
        gr = groups.analyze(g)
        idx = {n.tensor.id: n.idx for n in g.nodes}
        self.assertEqual([p["members"] for p in gr.parallel.values()], [[idx[t["a"]], idx[t["b"]]]])
        # both multiplies share one anchor unit, so both epilogues and the join end up in one group with them
        group = gr.fusion_of[idx[t["a"]]]
        for name in ("b", "a_act", "b_act", "out"):
            self.assertEqual(gr.fusion_of[idx[t[name]]], group, name)
        # the norm feeds two consumers, so it stays materialised and outside the group
        self.assertNotEqual(gr.fusion_of[idx[t["norm"]]], group)
        self.assertEqual(gr.fused_roots, {t["a"], t["b"], t["a_act"], t["b_act"]})
        self.assertEqual(fused_roots(g, "groups"), gr.fused_roots)

    def test_two_anchors_do_not_merge(self):
        b = Builder()
        w1, w2 = b.add("blk.0.a.weight", [4, 8], weight=True), b.add("blk.0.b.weight", [8, 8], weight=True)
        x = b.add("inp", [4, 2], op="ROPE")
        a = b.add("a-0", [8, 2], op="MUL_MAT", src=(w1, x))
        mid = b.add("mid-0", [8, 2], op="UNARY", desc="SILU", src=(a,))
        c = b.add("b-0", [8, 2], op="MUL_MAT", src=(w2, mid))
        g = b.graph()
        gr = groups.analyze(g)
        idx = {n.tensor.id: n.idx for n in g.nodes}
        self.assertEqual(gr.fusion_of[idx[mid]], gr.fusion_of[idx[a]])        # epilogue of the earlier anchor
        self.assertNotEqual(gr.fusion_of[idx[c]], gr.fusion_of[idx[a]])       # a second anchor starts a new group
        self.assertEqual(gr.parallel, {})                                       # different inputs: not stackable
        self.assertNotIn(mid, gr.fused_roots)                                   # the hand-off between them is stored

    def test_fusion_does_not_cross_layers(self):
        b = Builder()
        x = b.add("inp", [4, 2], op="ROPE")
        a = b.add("l_out-0", [4, 2], op="SCALE", src=(x,))
        c = b.add("norm-1", [4, 2], op="RMS_NORM", src=(a,))
        g = b.graph()
        gr = groups.analyze(g)
        idx = {n.tensor.id: n.idx for n in g.nodes}
        self.assertEqual((gr.layer[idx[a]], gr.layer[idx[c]]), (0, 1))
        self.assertNotEqual(gr.fusion_of[idx[a]], gr.fusion_of[idx[c]])


@unittest.skipUnless((CAPTURES / "pp128/decode.json").is_file() and (STUDY / "pp128-decode/nodes.csv").is_file(),
                     "2026-09-24 captures or the dependency-map study tables are not present")
class ReproducesTheDependencyMapStudy(unittest.TestCase):
    """The engine's loader and groups.py against the tables committed by build_map.py (PR #10)."""

    @staticmethod
    @lru_cache(maxsize=None)
    def load(workload, phase):
        g = Graph(CAPTURES / workload / f"{phase}.json", phase)
        out = STUDY / f"{workload}-{phase}"
        with (out / "nodes.csv").open(encoding="utf-8", newline="") as f:
            nodes = {int(r["sched_pos"]): r for r in csv.DictReader(f)}
        with (out / "edges.csv").open(encoding="utf-8", newline="") as f:
            edges = {(int(r["src_pos"]), int(r["dst_pos"])) for r in csv.DictReader(f)}
        return g, nodes, edges

    CASES = [(w, p) for w in ("pp128", "pp512") for p in ("prefill", "decode")]

    def test_same_operations(self):
        for case in self.CASES:
            g, nodes, _ = self.load(*case)
            self.assertEqual([n.spos for n in g.nodes], sorted(nodes), case)

    def test_same_dependencies_including_state_ordering(self):
        """Data edges plus read-after-write, write-after-read and write-after-write on the caches and state."""
        for case in self.CASES:
            g, _, edges = self.load(*case)
            mine = {(g.nodes[d].spos, n.spos) for n, deps in zip(g.nodes, g.deps()) for d in deps}
            self.assertEqual(mine, edges, case)

    def test_same_layers_categories_and_groups(self):
        for case in self.CASES:
            g, nodes, _ = self.load(*case)
            gr = groups.analyze(g)
            for n in g.nodes:
                row = nodes[n.spos]
                where = (case, n.spos, n.tensor.name)
                self.assertEqual(str(gr.layer[n.idx]), row["layer"], where)
                self.assertEqual(gr.category[n.idx], row["category"], where)
                self.assertEqual(gr.fusion_of[n.idx], row["fusion_group"], where)
                unit = gr.unit[n.idx] or ""
                if row["unit"].startswith("U"):          # single-op anchors are named by tensor id in the study
                    self.assertTrue(unit.startswith("U"), where)
                else:
                    self.assertEqual(unit, row["unit"], where)


@unittest.skipIf(RUN is None, "no captured run under results/")
class OnTheEngineCaptures(unittest.TestCase):
    def test_groups_keep_at_least_what_chains_keep(self):
        """Observed on these captures, not a property of the two rules in general."""
        for name in ("pp512-fa-off", "pp512-fa-on"):
            w = load_workload(RUN / "graphs" / name)
            for phase in ("prefill", "decode"):
                g = w.graph(phase)
                self.assertLessEqual(fused_roots(g, "chains"), fused_roots(g, "groups"), (name, phase))

    def test_flash_attention_is_one_anchor_per_layer(self):
        g = load_workload(RUN / "graphs" / "pp512-fa-on").graph("decode")
        gr = groups.analyze(g)
        flash = [n.idx for n in g.nodes if n.op == "FLASH_ATTN_EXT"]
        self.assertEqual(len(flash), 6)
        self.assertEqual({gr.unit[i] for i in flash}, {f"A{gr.layer[i]}" for i in flash})

    def test_more_fusion_never_adds_activation_traffic(self):
        w = load_workload(RUN / "graphs" / "pp512-fa-off")
        base = config.load_profile("accel-balanced")[0]
        act = {}
        for policy in ("none", "chains", "groups"):
            r = evaluate(w, config.make_config({"memory": {"fusion": policy}}, base=base))
            self.assertTrue(r["valid"], policy)
            act[policy] = r["phases"]["prefill"]["bytes"]["act_hbm"]
        self.assertGreater(act["none"], act["chains"])
        self.assertGreater(act["chains"], act["groups"])


if __name__ == "__main__":
    unittest.main()
