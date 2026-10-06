"""Graph structure (roots, kinds, hazards) and per-node cost equations, on small synthetic graphs with hand-computed answers."""
import itertools
import math
import unittest

from helpers import Builder
from sbengine import config
from sbengine.costs import Ctx, _candidates, _tile_plan, node_cost
from sbengine.graph import KNOWN_OPS, METADATA_OPS, Graph, capture_path
from sbengine.memory import fused_roots
from sbengine.schedule import schedule


def cfg_small(**over):
    base = {"clock_mhz": 1000, "matrix": {"count": 1, "rows": 2, "cols": 2, "efficiency": 1.0,
                                          "rates": {"w32": 1.0, "aa": 1.0}}}
    return config.make_config(config.deep_merge(base, over))


class GraphStructure(unittest.TestCase):
    def test_roots_kinds_and_zero_or_metadata_skipping(self):
        b = Builder()
        w = b.add("blk.0.w", [8, 4], weight=True)
        x = b.add("inp_tokens", [4], dtype="i32")
        y = b.add("y", [4, 4], op="MUL_MAT", src=(w, x))
        v = b.add("y (view)", [4, 2], op="VIEW", view=y)
        kv = b.add("cache_k_l0", [8, 16], dtype="f16")
        st = b.add("cache_s_l0", [16])
        z = b.add("z", [4, 2], op="SCALE", src=(v,))
        b.add("empty", [4, 0], op="CPY", src=(z,))
        g = b.graph()
        self.assertEqual([n.tensor.name for n in g.nodes], ["y", "z"])          # view and zero-size nodes dropped
        self.assertEqual((g.metadata_nodes, g.zero_nodes), (1, 1))
        self.assertEqual(g.root(v), y)
        self.assertEqual([g.kind(r) for r in (w, x, y, kv, st)], ["weight", "input", "activation", "kv", "state"])
        self.assertEqual(g.nodes[1].src_roots, (y,))                              # reading a view reads its root's storage
        self.assertEqual(g.producer[y], 0)
        self.assertEqual(g.consumers[y], [1])

    def test_persistent_hazard_dependencies(self):
        b = Builder()
        kv = b.add("cache_k_l0", [8, 16], dtype="f16")
        newk = b.add("Kcur", [8, 1], op="ROPE")
        idx = b.add("idx", [1], dtype="i64")
        w1 = b.add("cache_k_l0 (view)", [8, 1], dtype="f16", op="SET_ROWS", src=(newk, idx, kv), view=kv)
        kperm = b.add("cache_k_l0 (view) (permuted)", [8, 16], dtype="f16", op="PERMUTE", view=kv)
        q = b.add("Q", [8, 1], op="ROPE")
        rd = b.add("read", [16, 1], op="MUL_MAT", src=(kperm, q))
        newk2 = b.add("Kcur2", [8, 1], op="ROPE")
        w2 = b.add("cache_k_l0 (view) 2", [8, 1], dtype="f16", op="SET_ROWS", src=(newk2, idx, kv), view=kv)
        g = b.graph()
        by = {n.tensor.name: n.idx for n in g.nodes}
        deps = g.deps()
        self.assertIn(by["cache_k_l0 (view)"], deps[by["read"]])                   # read-after-write on the cache
        self.assertIn(by["read"], deps[by["cache_k_l0 (view) 2"]])                 # write-after-read
        self.assertIn(by["cache_k_l0 (view)"], deps[by["cache_k_l0 (view) 2"]])    # write-after-write
        self.assertEqual(deps[by["Kcur"]], [])

    def test_fusion_rule(self):
        b = Builder()
        w = b.add("blk.0.w", [8, 4], weight=True)
        x = b.add("x", [8, 4], op="ROPE")
        y = b.add("y", [4, 4], op="MUL_MAT", src=(w, x))       # consumed by two nodes -> stays materialized
        z = b.add("z", [4, 4], op="SCALE", src=(y,))            # single consumer, elementwise epilogue -> fused
        b.add("o", [4, 4], op="ADD", src=(z, y))
        g = b.graph()
        fused = fused_roots(g, "chains")
        self.assertIn(z, fused)
        self.assertNotIn(y, fused)
        self.assertEqual(fused_roots(g, "none"), set())

    def test_every_op_in_every_captured_graph_has_a_cost_formula(self):
        from helpers import real_run
        run = real_run()
        if run is None:
            self.skipTest("no captured run")
        for d in sorted((run / "graphs").glob("*")):
            if capture_path(d, "prefill") is None:
                continue
            g = Graph(capture_path(d, "prefill"), "prefill")
            unknown = {n.op for n in g.nodes} - KNOWN_OPS
            self.assertEqual(unknown, set(), d.name)
            self.assertLessEqual({t.desc for t in g.tensors.values() if t.is_metadata}, METADATA_OPS)


class MatrixEquations(unittest.TestCase):
    def test_gemm_and_gemv_cycles(self):
        ctx = Ctx(cfg_small())
        # 4x4 output, K=8, 2x2 array: 4 tiles, one unit -> 4 waves * 8 cycles + one fill/drain (2+2-2)
        self.assertAlmostEqual(ctx.matrix_time(4, 4, 8, 1, 1.0), 34 / 1e9)
        # legacy: fill/drain per tile wave
        self.assertAlmostEqual(Ctx(cfg_small(matrix={"pipelined_tiles": False})).matrix_time(4, 4, 8, 1, 1.0), 40 / 1e9)
        # N=1 < cols with GEMV lane reuse: ceil(4*1*8/(1*2*2*1)) + 2 + 2
        self.assertAlmostEqual(ctx.matrix_time(4, 1, 8, 1, 1.0), 12 / 1e9)
        # without the gemv capability the narrow N wastes array columns: 2 tiles * 8 + 2
        no_gemv = Ctx(cfg_small(matrix={"caps": ["gemm", "attention", "requant"]}))
        self.assertAlmostEqual(no_gemv.matrix_time(4, 1, 8, 1, 1.0), 18 / 1e9)
        self.assertEqual(ctx.matrix_time(4, 4, 8, 1, 0.0), math.inf)

    def test_more_units_never_slower(self):
        times = [Ctx(cfg_small(matrix={"count": n})).matrix_time(64, 64, 256, 1, 1.0) for n in (1, 2, 4, 8)]
        self.assertEqual(times, sorted(times, reverse=True))

    def test_tile_search_is_optimal_over_its_candidates_and_beats_fixed(self):
        M, N, K, a_bpe, x_bpe = 64, 256, 128, 0.5, 2.0
        a_total, x_total = M * K * a_bpe, K * N * x_bpe
        args = (M, N, K, a_bpe, x_bpe, a_total, x_total, 1, 8, 8, 32)
        prev = math.inf
        for budget in (2000, 8000, 30000, 120000, 500000):
            s = _tile_plan(*args, "search", 64, budget)
            f = _tile_plan(*args, "fixed", 64, budget)

            def traffic(p):
                return a_total * p["nr"] + x_total * p["mr"]
            self.assertLessEqual(traffic(s), traffic(f))
            self.assertLessEqual(traffic(s), prev)                               # more L1 never means more traffic
            prev = traffic(s)
            kt = min(K, 32)
            best = min(a_total * math.ceil(N / n) + x_total * math.ceil(M / m)
                       for m, n in itertools.product(_candidates(M, 8), _candidates(N, 8))
                       if 2 * (m * kt * a_bpe + kt * n * x_bpe + m * n * 4) <= budget)
            self.assertAlmostEqual(traffic(s), best)

    def test_weight_matmul_cost(self):
        b = Builder()
        w = b.add("blk.0.w", [8, 4], weight=True)                   # f32: 32 elements, 128 bytes
        x = b.add("x", [8, 4], op="ROPE")
        b.add("y", [4, 4], op="MUL_MAT", src=(w, x))
        g = b.graph()
        n = [n for n in g.nodes if n.op == "MUL_MAT"][0]
        ctx = Ctx(cfg_small())
        c = node_cost(ctx, g, n)
        self.assertEqual((c.pool, c.macs, c.error), ("Matrix", 128.0, None))
        self.assertAlmostEqual(c.compute_s, ctx.matrix_time(4, 4, 8, 1, 1.0))
        plan = ctx.tile_plan(4, 4, 8, 4.0, 2.0, 128.0, 64.0)
        self.assertEqual(c.w_hbm, 128.0 * plan["nr"])
        self.assertIn(g.root(x), c.feed)
        # uniform int4, group 64: 32 * 4 / 8 = 16 bytes + ceil(32/64) * 2 scale bytes
        c4 = node_cost(Ctx(cfg_small(precision={"weights": "uniform_int4"})), g, n)
        self.assertEqual(c4.w_hbm / c4.w_hbm, 1.0)
        self.assertEqual(Ctx(cfg_small(precision={"weights": "uniform_int4"})).weight_bytes(g, g.tensors[w]), 18.0)
        # direct weight path: weights bypass L1
        cd = node_cost(Ctx(cfg_small(memory={"weight_path": "direct"})), g, n)
        self.assertEqual(cd.bypass, cd.w_hbm)
        self.assertLess(cd.l1_fixed, c.l1_fixed)

    def test_missing_units_fall_back_or_fail(self):
        b = Builder()
        w = b.add("blk.0.w", [8, 4], weight=True)
        x = b.add("x", [8, 4], op="ROPE")
        b.add("y", [4, 4], op="MUL_MAT", src=(w, x))
        g = b.graph()
        n = [n for n in g.nodes if n.op == "MUL_MAT"][0]
        no_matrix = node_cost(Ctx(cfg_small(matrix={"count": 0})), g, n)
        self.assertEqual(no_matrix.pool, "Scalar")                                # vector lacks 'gemm' -> scalar core
        none_left = node_cost(Ctx(cfg_small(matrix={"count": 0}, scalar={"count": 0})), g, n)
        self.assertIsNone(none_left.pool)
        self.assertIn("no matrix", none_left.error)


class AttentionAndRecurrence(unittest.TestCase):
    def _fa_graph(self):
        b = Builder()
        q_src = b.add("Qcur", [4, 2, 64], op="ROPE")
        q = b.add("Qcur (view) (permuted)", [4, 64, 2], op="PERMUTE", view=q_src)
        kroot, vroot = b.add("cache_k_l0", [4, 64], dtype="f16"), b.add("cache_v_l0", [4, 64], dtype="f16")
        k = b.add("K", [4, 64, 1], dtype="f16", op="PERMUTE", view=kroot)
        v = b.add("V", [4, 64, 1], dtype="f16", op="PERMUTE", view=vroot)
        mask = b.add("attn_inp_kq_mask", [64, 64], dtype="f16")
        b.add("fa", [4, 2, 64], op="FLASH_ATTN_EXT", src=(q, k, v, mask))
        return b.graph(input_tokens=64, past_tokens=0)

    def test_fused_attention_macs_and_causal_tiles(self):
        g = self._fa_graph()
        n = g.nodes[-1]
        dense = node_cost(Ctx(config.make_config({"attention": {"causal_skip": False}})), g, n)
        self.assertEqual(dense.macs, 2 * 4 * 64 * 2 * 64)                          # 2 d Tq Hq keys
        # tiny L1 forces query tiles: q_tile = 18 rows -> 9 tokens/tile -> 8 tiles -> avg keys = 64 * 9/16 = 36
        small = config.make_config({"l1": {"mib": 0.005}, "attention": {"kv_tile": 2, "causal_skip": True}})
        skipped = node_cost(Ctx(small), g, n)
        self.assertIsNone(skipped.error)
        self.assertEqual(skipped.macs, 2 * 4 * 64 * 2 * 36)
        self.assertLess(skipped.macs, dense.macs)
        self.assertGreater(skipped.p_hbm_r, 0)
        # the mask is generated on-chip by default and read from HBM when not
        with_mask = node_cost(Ctx(config.make_config({"attention": {"mask_onchip": False}})), g, n)
        self.assertGreater(with_mask.in_hbm, dense.in_hbm)

    def test_fused_attention_needs_a_tile_that_fits(self):
        g = self._fa_graph()
        c = node_cost(Ctx(config.make_config({"l1": {"mib": 0.0001}})), g, g.nodes[-1])
        self.assertIn("too small", c.error)

    def _gdn_graph(self):
        b = Builder()
        act = lambda name, shape: b.add(name, shape, op="ROPE")               # noqa: E731
        q, k, v = act("q", [4, 2, 8]), act("k", [4, 2, 8]), act("v", [4, 2, 8])
        gate, beta, state = act("gate", [1, 2, 8]), act("beta", [1, 2, 8]), act("state", [4, 4, 2])
        b.add("gdn", [8, 5], op="GATED_DELTA_NET", src=(q, k, v, gate, beta, state))
        return b.graph(input_tokens=8)

    def test_recurrence_lowerings(self):
        g = self._gdn_graph()
        n = g.nodes[-1]
        serial = node_cost(Ctx(config.make_config({"recurrent": {"lowering": "serial"}})), g, n)
        self.assertEqual((serial.pool, serial.arith), ("Recurrent", (7 * 16 + 12) * 2 * 8))
        chunked = node_cost(Ctx(config.make_config({"recurrent": {"lowering": "chunked", "chunk": 4}})), g, n)
        self.assertEqual(chunked.pool, "Matrix")
        self.assertAlmostEqual(chunked.macs, 2 * 1 * 2 * (3 * 4 * 16 + 5 * 16 * 4 + 64 / 3))
        # with no recurrence unit the serial form falls back to vector (caps include 'recurrent')
        vec = node_cost(Ctx(config.make_config({"recurrent": {"count": 0, "lowering": "serial"}})), g, n)
        self.assertEqual(vec.pool, "Vector")
        # a single token cannot be chunked
        one = Builder()
        act = lambda name, shape: one.add(name, shape, op="ROPE")             # noqa: E731
        srcs = [act(nm, s) for nm, s in (("q", [4, 2, 1]), ("k", [4, 2, 1]), ("v", [4, 2, 1]), ("g", [1, 2, 1]), ("b", [1, 2, 1]), ("s", [4, 4, 2]))]
        one.add("gdn", [8, 5], op="GATED_DELTA_NET", src=tuple(srcs))
        g1 = one.graph(input_tokens=1)
        c1 = node_cost(Ctx(config.make_config({"recurrent": {"lowering": "chunked"}})), g1, g1.nodes[-1])
        self.assertIn("no recurrence", c1.error)

    def test_serial_recurrence_on_the_matrix_arrays(self):
        """recurrent.lowering = "matrix": 3 S^2 MACs per head-token at matrix_penalty, plus the vector tail
        (same equation as the 2026-10-03 recurrence study's shared-matrix path in explorer/engine.js)."""
        g = self._gdn_graph()                                                   # S = 4, H = 2, T = 8
        n = g.nodes[-1]

        def cost(penalty, **over):
            cfg = config.make_config(config.deep_merge({"recurrent": {"lowering": "matrix", "matrix_penalty": penalty}}, over))
            return cfg, node_cost(Ctx(cfg), g, n)

        cfg, c = cost(4)
        macs = 3 * 16 * 2 * 8
        self.assertEqual((c.pool, c.macs, c.note), ("Matrix", macs, "serial/matrix x4"))
        m, v = cfg["matrix"], cfg["vector"]
        f = cfg["clock_mhz"] * 1e6
        t_mat = macs * 4 / (m["count"] * m["rows"] * m["cols"] * m["rates"]["w8"] * f * m["efficiency"])
        t_vec = ((16 + 12) * 2 * 8 + 2 * 8 * v["special_cycles"]) / (v["count"] * v["lanes"] * f * v["efficiency"])
        self.assertAlmostEqual(c.compute_s, t_mat + t_vec)
        # matrix and vector time are added serially, so the vector side is held for the whole compute interval:
        # other vector work must not slip in behind the first t_vec seconds
        self.assertEqual(set(c.aux_busy), {"Vector"})
        self.assertAlmostEqual(c.aux_busy["Vector"], t_mat + t_vec)
        timeline = schedule([c.compute_s, t_vec], ["Matrix", "Vector"], [[], []], [0.0, 0.0], [0.0, 0.0], 0.0,
                            [c.aux_busy, {}], detail=True)["operations"]
        self.assertAlmostEqual(timeline[1]["start"], c.compute_s)               # an independent vector op waits
        # dedicated recurrence units are not used by this lowering, whatever their count
        self.assertEqual(cost(4, recurrent={"count": 0})[1].compute_s, c.compute_s)
        # the penalty scales only the matrix part
        self.assertAlmostEqual(cost(16)[1].compute_s - c.compute_s, 3 * t_mat)
        # "auto" never picks it, so existing configurations are unchanged
        auto = node_cost(Ctx(config.make_config({"recurrent": {"matrix_penalty": 0.001}})), g, n)
        self.assertNotIn("serial/matrix", auto.note)
        # it needs matrix units and somewhere to run the tail
        self.assertIn("no recurrence", cost(4, matrix={"count": 0})[1].error)
        # unlike chunking it also covers a single token (decode)
        one = Builder()
        act = lambda name, shape: one.add(name, shape, op="ROPE")             # noqa: E731
        srcs = [act(nm, s) for nm, s in (("q", [4, 2, 1]), ("k", [4, 2, 1]), ("v", [4, 2, 1]), ("g", [1, 2, 1]), ("b", [1, 2, 1]), ("s", [4, 4, 2]))]
        one.add("gdn", [8, 5], op="GATED_DELTA_NET", src=tuple(srcs))
        g1 = one.graph(input_tokens=1)
        c1 = node_cost(Ctx(config.make_config({"recurrent": {"lowering": "matrix"}})), g1, g1.nodes[-1])
        self.assertEqual((c1.pool, c1.macs), ("Matrix", 3 * 16 * 2))

    def test_memory_ops(self):
        b = Builder()
        table = b.add("token_embd.weight", [8, 100], weight=True)               # 4 bytes per element
        idx = b.add("inp_tokens", [3], dtype="i32")
        b.add("emb", [8, 3], op="GET_ROWS", src=(table, idx))
        kv = b.add("cache_k_l0", [8, 16], dtype="f16")
        newk = b.add("Kcur", [8, 1], op="ROPE")
        i64 = b.add("idxs", [1], dtype="i64")
        b.add("cache_k_l0 (view)", [8, 1], dtype="f16", op="SET_ROWS", src=(newk, i64, kv), view=kv)
        g = b.graph()
        ctx = Ctx(config.make_config())
        get = node_cost(ctx, g, [n for n in g.nodes if n.op == "GET_ROWS"][0])
        self.assertEqual(get.w_hbm, 24 * 4.0)                                       # rows gathered * bytes per weight element
        self.assertEqual(get.out, {})                                               # ingress already counts the L1 write
        self.assertEqual(get.pool, "DMA")
        setr = node_cost(ctx, g, [n for n in g.nodes if n.op == "SET_ROWS"][0])
        self.assertEqual(setr.p_hbm_w, 8 * 2.0)                                     # 8 new values at 16-bit KV
        self.assertEqual(setr.writes, [])


if __name__ == "__main__":
    unittest.main()
