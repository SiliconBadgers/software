"""Model behaviour on the real captured graphs. Skipped when results/ has no run."""
import json
import unittest
from functools import lru_cache

from helpers import real_run
from sbengine import config, cpu_validate
from sbengine.graph import load_workload
from sbengine.model import evaluate
from sbengine.trace import load_trace

RUN = real_run()


@lru_cache(maxsize=None)
def wl(name):
    return load_workload(RUN / "graphs" / name)


def balanced(**over):
    return config.make_config(config.deep_merge(config.load_profile("accel-balanced")[0], over))


@unittest.skipIf(RUN is None, "no captured run under results/")
class Accounting(unittest.TestCase):
    def test_mac_counts_match_the_capture_summaries(self):
        if not (RUN / "graphs" / "pp128-fa-on" / "prefill.summary.json").is_file():
            self.skipTest("graph summaries are local-only (not committed); run `analyze` on the run to create them")
        for name in ("pp128-fa-on", "pp128-fa-off", "pp512-fa-on", "pp512-fa-off", "prompt-code-fa-on"):
            r = evaluate(wl(name), balanced())
            for phase in ("prefill", "decode"):
                summary = json.loads((RUN / "graphs" / name / f"{phase}.summary.json").read_text(encoding="utf-8"))["macs_by_supported_op"]
                got = r["phases"][phase]["macs_by_op"]
                for op, macs in summary.items():
                    self.assertEqual(got[op], macs, f"{name} {phase} {op}")

    def test_fused_attention_does_the_same_matmul_work_as_the_unfused_capture(self):
        """Dense fused attention (no causal skipping, captured padding) must equal the QK and AV matmuls that the
        separately captured fa-off graph contains: an independent check on the attention accounting."""
        cfg = balanced(attention={"causal_skip": False, "kv_padding": "captured"})
        for size in ("pp512", "pp8192"):
            on, off = evaluate(wl(f"{size}-fa-on"), cfg), evaluate(wl(f"{size}-fa-off"), cfg)
            for phase in ("prefill", "decode"):
                fused = on["phases"][phase]["macs_by_op"]["FLASH_ATTN_EXT"]
                unfused = off["phases"][phase]["macs_by_op"]["MUL_MAT"] - on["phases"][phase]["macs_by_op"]["MUL_MAT"]
                self.assertEqual(fused, unfused, f"{size} {phase}")

    def test_causal_skipping_only_removes_work(self):
        dense = evaluate(wl("pp512-fa-on"), balanced(attention={"causal_skip": False}))
        skip = evaluate(wl("pp512-fa-on"), balanced(attention={"causal_skip": True}))
        d, s = (x["phases"]["prefill"]["macs_by_op"]["FLASH_ATTN_EXT"] for x in (dense, skip))
        self.assertLess(s, d)
        self.assertGreater(s, 0.5 * d)                   # at least the causal half of the work remains

    # engine.js of the 2026-09-24 explorer (origin/main 11d36ed) run in Node with configs/recommended-{balanced,efficient}.json,
    # prompt = context = n, on its own fa-off graphs. Values in milliseconds: (prefill, decode).
    EXPLORER = {("balanced", 128): (313.102, 21.996), ("balanced", 512): (1241.269, 22.167),
                ("efficient", 128): (985.139, 22.613), ("efficient", 512): (3951.660, 22.783)}

    def test_legacy_switches_reproduce_the_explorer_on_its_own_graphs(self):
        """With every switchable equation set back to the explorer's version this engine lands on its numbers: prefill
        to <0.1%, decode to <1.5% (the residual is the weight-scale bytes now counted in the L1 feed and the place where
        recurrent-state traffic is charged; see docs/EQUATIONS.md)."""
        legacy = config.load_profile("legacy-equations")[0]
        for (design, n), (prefill_ms, decode_ms) in self.EXPLORER.items():
            over = {} if design == "balanced" else {"matrix": {"count": 1}, "recurrent": {"count": 0}}
            r = evaluate(wl(f"pp{n}-fa-off"), config.make_config(over, base=legacy))
            self.assertAlmostEqual(r["phases"]["prefill"]["seconds"] * 1e3, prefill_ms, delta=prefill_ms * 0.001, msg=f"{design} {n} prefill")
            self.assertAlmostEqual(r["phases"]["decode"]["seconds"] * 1e3, decode_ms, delta=decode_ms * 0.015, msg=f"{design} {n} decode")


@unittest.skipIf(RUN is None, "no captured run under results/")
class Invariants(unittest.TestCase):
    def seconds(self, name="pp512-fa-on", **over):
        r = evaluate(wl(name), balanced(**over))
        return r["phases"]["prefill"]["seconds"], r["phases"]["decode"]["seconds"], r

    def test_more_of_a_resource_is_never_slower(self):
        base = self.seconds()
        for over in ({"hbm": {"gbs": 920}}, {"clock_mhz": 600}, {"l1": {"banks": 128}}, {"dma": {"count": 4}},
                     {"memory": {"weight_path": "direct"}}):
            faster = self.seconds(**over)
            self.assertLessEqual(faster[0], base[0] * 1.0001, over)
            self.assertLessEqual(faster[1], base[1] * 1.0001, over)

    def test_bounds_are_ordered(self):
        for name in ("pp128-fa-on", "pp512-fa-off"):
            r = evaluate(wl(name), balanced())
            for p in r["phases"].values():
                b = p["bounds"]
                self.assertLessEqual(b["lower_bound"], b["list"] + 1e-12)
                self.assertGreaterEqual(b["list"], b["critical_path"] - 1e-12)

    def test_weight_policy_orders_footprints(self):
        w = {p: self.seconds(precision={"weights": p})[2]["phases"]["decode"]["footprint"]["weights"]
             for p in ("uniform_int4", "captured", "uniform_int8")}
        self.assertLess(w["uniform_int4"], w["captured"])
        self.assertLess(w["captured"], w["uniform_int8"])

    def test_searched_tiles_never_stream_more_weights_than_the_fixed_heuristic(self):
        fixed = self.seconds(matrix={"tile_mode": "fixed"})[2]["phases"]["prefill"]["bytes"]["weights_hbm"]
        search = self.seconds(matrix={"tile_mode": "search"})[2]["phases"]["prefill"]["bytes"]["weights_hbm"]
        self.assertLessEqual(search, fixed)

    def test_weights_are_shared_across_a_batch_but_activations_and_kv_scale(self):
        one = self.seconds()[2]["phases"]["decode"]
        two = self.seconds(batch=2)[2]["phases"]["decode"]
        # matmul weights are shared; the embedding gather reads one more row per extra sequence (2048 values, ~1.7 kB)
        self.assertAlmostEqual(one["bytes"]["weights_hbm"], two["bytes"]["weights_hbm"], delta=1e-4 * one["bytes"]["weights_hbm"])
        self.assertAlmostEqual(two["footprint"]["kv"], 2 * one["footprint"]["kv"])
        self.assertAlmostEqual(two["macs_by_op"]["MUL_MAT"], 2 * one["macs_by_op"]["MUL_MAT"])

    def test_infeasible_designs_are_flagged(self):
        self.assertIn("DSP", " ".join(self.seconds(matrix={"count": 64})[2]["errors"]))
        self.assertIn("HBM footprint", " ".join(self.seconds(hbm={"gib": 0.5})[2]["errors"]))
        r = self.seconds(matrix={"count": 0}, scalar={"count": 0})[2]
        self.assertFalse(r["valid"])
        self.assertTrue(any("no matrix" in e for e in r["errors"]))
        slow = self.seconds(matrix={"count": 0})                                  # scalar fallback: valid but far slower
        self.assertTrue(slow[2]["valid"])
        self.assertGreater(slow[1], 50 * self.seconds()[1])

    def test_deterministic(self):
        a, b = evaluate(wl("pp128-fa-on"), balanced()), evaluate(wl("pp128-fa-on"), balanced())
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_host_fallback_uses_measured_cpu_time(self):
        w = wl("pp128-fa-on")
        trace = load_trace(RUN, w)
        if trace is None:
            self.skipTest("no op trace for pp128-fa-on")
        over = {"matrix": {"count": 0}, "scalar": {"count": 0}}
        without = evaluate(w, balanced(**over))
        self.assertFalse(without["valid"])
        with_host = evaluate(w, balanced(host={"enabled": True}, **over), trace)
        self.assertTrue(with_host["valid"], with_host["errors"])
        self.assertIn("Host", with_host["phases"]["prefill"]["pool_busy"])
        self.assertGreater(with_host["phases"]["prefill"]["seconds"], 1.0)         # measured CPU matmuls take seconds


@unittest.skipIf(RUN is None, "no captured run under results/")
class CpuValidation(unittest.TestCase):
    def test_held_out_prompts_are_predicted_from_the_accounting(self):
        res = cpu_validate.run(RUN)
        prefill = [abs(r["error_pct"]) for r in res["heldout"] if r["phase"] == "prefill"]
        decode = [abs(r["error_pct"]) for r in res["heldout"] if r["phase"] == "decode"]
        if not prefill:
            self.skipTest("no traces")
        self.assertLess(sum(prefill) / len(prefill), 10.0)
        self.assertLess(sum(decode) / len(decode), 20.0)
        # a prompt that the profiled run split into two micro-batches is not comparable and must be excluded
        self.assertNotIn(("prompt-json_records-fa-on", "prefill"), {(r["workload"], r["phase"]) for r in res["heldout"]})


class CpuValidationWithoutTraces(unittest.TestCase):
    """A fresh clone has no operation traces (they are local-only). The check must say it did not run."""

    def test_missing_graphs_and_traces_are_not_checked(self):
        import contextlib
        import io
        import tempfile
        from pathlib import Path
        from sbengine import cli
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "some-run"
            (run / "graphs").mkdir(parents=True)
            res = cpu_validate.run(run)
            self.assertFalse(res["checked"])
            self.assertEqual(res["heldout"], [])
            self.assertIn("Nothing was fitted", res["not_checked_reason"])
            self.assertEqual([(s["workload"], s["reason"]) for s in res["not_checked"]],
                             [("pp128-fa-on", "no captured graph"), ("pp512-fa-on", "no captured graph")])
            text = cpu_validate.report_markdown(res, "some-run")
            self.assertIn("**Status: NOT CHECKED.**", text)
            self.assertIn("| pp128-fa-on | prefill and decode | no captured graph |", text)
            self.assertNotIn("Measured, ms", text)                               # no empty result tables
            self.assertNotIn("Held-out", text)
            # the command reports it and does not exit successfully
            report = run / "engine" / "validate-cpu.md"
            with contextlib.redirect_stdout(io.StringIO()) as printed:
                self.assertEqual(cli.main(["validate-cpu", "--run", str(run), "--report", str(report)]), 1)
            self.assertIn("NOT CHECKED on some-run", printed.getvalue())
            self.assertIn("**Status: NOT CHECKED.**", report.read_text(encoding="utf-8"))

    @unittest.skipIf(RUN is None, "no captured run under results/")
    def test_a_run_without_its_traces_is_labelled_not_checked(self):
        """The committed run keeps its graphs but not its traces; with the traces present locally it is checked."""
        res = cpu_validate.run(RUN)
        text = cpu_validate.report_markdown(res, RUN.name)
        self.assertEqual(res["checked"], bool(res["heldout"]))
        if res["checked"]:
            self.assertIn("**Status: checked.**", text)
            self.assertIsNone(res["not_checked_reason"])
        else:
            self.assertIn("**Status: NOT CHECKED.**", text)
            self.assertIn("no operation trace", {s["reason"] for s in res["not_checked"]})
            self.assertEqual(res["train_fit"], [])
            self.assertNotIn("Measured, ms", text)

    def test_no_prediction_without_training_traces(self):
        """Held-out traces alone are not a check: a fit on no data would predict zero for every operation."""
        from unittest import mock
        sample = {("prompt-a-fa-on", "prefill"): {"samples": [("concat", {"bytes": 1.0, "one": 1.0}, 0.5)],
                                                  "metadata_seconds": 0.0, "tokens": 10}}

        def fake_collect(run_dir, names, not_checked=None):
            return {} if "pp128-fa-on" in names else sample

        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(cpu_validate, "collect", fake_collect):
            (Path(temp) / "graphs" / "prompt-a-fa-on").mkdir(parents=True)
            res = cpu_validate.run(Path(temp))
        self.assertFalse(res["checked"])
        self.assertEqual(res["heldout"], [])
        self.assertIn("Nothing was fitted", res["not_checked_reason"])


class CpuValidationReport(unittest.TestCase):
    def test_report_lists_measured_and_predicted_time_per_operation_class(self):
        """The Markdown report needs no traces to test: it only formats run()'s result."""
        def row(workload, phase, tokens, classes):
            measured, predicted = sum(m for m, _ in classes.values()), sum(p for _, p in classes.values())
            return {"workload": workload, "phase": phase, "tokens": tokens, "measured_s": measured, "predicted_s": predicted,
                    "error_pct": 100 * (predicted / measured - 1),
                    "by_class": {k: {"measured_s": m, "predicted_s": p} for k, (m, p) in classes.items()}}
        res = {"train": ["pp128-fa-on", "pp512-fa-on"],
               "train_fit": [row("pp128-fa-on", "prefill", 128, {"concat": (0.080, 0.219), "matmul[q4_K]": (0.500, 0.500)})],
               "heldout": [row("prompt-a-fa-on", "prefill", 200, {"concat": (0.100, 0.300), "matmul[q4_K]": (0.900, 0.900)}),
                           row("prompt-b-fa-on", "prefill", 300, {"concat": (0.200, 0.400), "matmul[q4_K]": (1.800, 1.700)}),
                           row("prompt-a-fa-on", "decode", 200, {"matmul[q4_K]": (0.030, 0.031)})]}
        text = cpu_validate.report_markdown(res, "some-run", top=1)
        self.assertIn("# CPU accounting check: some-run", text)
        self.assertIn("**Status: checked.** 3 held-out graph phases were compared.", text)
        self.assertIn("--run ../results/some-run", text)                         # the command is run from engine/
        self.assertIn("| held out | prompt-b-fa-on | prefill | 300 | 2,000.0 | 2,100.0 | +5.0% |", text)
        # classes are summed over the held-out graphs and ordered by the size of the difference
        self.assertIn("| `concat` | 300.0 | 700.0 | +400.0 | +13.3% |", text)
        self.assertIn("| 1 other classes | 2,700.0 | 2,600.0 | -100.0 | -3.3% |", text)
        self.assertIn("| **All** | **3,000.0** | **3,300.0** | **+300.0** | **+10.0%** |", text)
        self.assertIn("Held-out prefill: mean absolute error 12.5%, max 20.0% (n = 2).", text)
        self.assertIn("## Training graph pp128-fa-on, prefill, by operation class", text)
        self.assertIn("| `concat` | 80.0 | 219.0 | +139.0 |", text)


if __name__ == "__main__":
    unittest.main()
