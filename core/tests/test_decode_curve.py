"""Decode time versus context length, and the continuation-length settings around it. No model needed."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "analysis"))
import decode_curve as dc  # noqa: E402
from policy import load_experiment  # noqa: E402

RECORDED = CORE.parent / "experiments" / "llama-cpp" / "2026-09-22" / "results"
CLI = CORE / "runner" / "cli.py"


def measurement(n, steps, ms, rep=0):
    """A harness measurement record whose step j costs ms(context) at context n + j + 1."""
    return {"kind": "measurement", "prompt_tokens": n, "rep": rep, "prefill_us": 1.0,
            "decode_us": [1000 * ms(n + j + 1) for j in range(steps)]}


class Curve(unittest.TestCase):
    def test_linear_growth_is_recovered(self):
        c = dc.curve([measurement(100, 64, lambda ctx: 10 + 0.005 * ctx)])          # 5 ms per 1000 tokens
        self.assertAlmostEqual(c["slope_ms_per_1000_context_tokens"], 5.0, places=6)
        self.assertAlmostEqual(c["fit_intercept_ms"], 10.0, places=6)
        self.assertAlmostEqual(c["fit_r_squared"], 1.0)
        self.assertGreater(c["growth_ratio"], 1.0)
        self.assertEqual((c["first_context_tokens"], c["last_context_tokens"]), (101, 164))

    def test_flat_cost_has_no_slope_and_ratio_one(self):
        c = dc.curve([measurement(100, 32, lambda ctx: 12.0)])
        self.assertAlmostEqual(c["slope_ms_per_1000_context_tokens"], 0.0, places=9)
        self.assertAlmostEqual(c["growth_ratio"], 1.0)

    def test_median_over_repetitions_ignores_one_spike(self):
        quiet = [measurement(50, 16, lambda ctx: 10.0, rep=r) for r in range(2)]
        spike = measurement(50, 16, lambda ctx: 10.0, rep=2)
        spike["decode_us"][5] = 500_000.0
        c = dc.curve(quiet + [spike])
        self.assertAlmostEqual(c["per_step_ms"][5], 10.0)
        self.assertEqual(c["repetitions"], 3)

    def test_too_few_steps_gives_no_slope(self):
        self.assertIsNone(dc.curve([measurement(50, 2, lambda ctx: 10.0)])["slope_ms_per_1000_context_tokens"])
        one = dc.curve([measurement(50, 1, lambda ctx: 10.0)])
        self.assertEqual((one["steps"], one["window_steps"]), (1, 1))

    def test_span_fraction_shows_when_a_slope_cannot_mean_anything(self):
        short = dc.curve([measurement(2048, 32, lambda ctx: 12.0)])
        long = dc.curve([measurement(128, 256, lambda ctx: 12.0)])
        self.assertLess(short["context_span_fraction"], 0.02)
        self.assertGreater(long["context_span_fraction"], 1.5)

    def test_bins_partition_the_steps(self):
        for steps in (5, 16, 100, 257):
            c = dc.curve([measurement(10, steps, lambda ctx: float(ctx))])
            self.assertEqual(sum(b["steps"] for b in c["bins"]), steps)
            self.assertEqual(len(c["bins"]), min(dc.MAX_BINS, steps))
            self.assertEqual(c["bins"][0]["first_context"], 11)
            self.assertEqual(c["bins"][-1]["last_context"], 10 + steps)
            for a, b in zip(c["bins"], c["bins"][1:]):
                self.assertEqual(b["first_context"], a["last_context"] + 1)

    def test_repetitions_must_agree_on_steps(self):
        with self.assertRaises(ValueError):
            dc.curve([measurement(10, 4, lambda ctx: 1.0), measurement(10, 5, lambda ctx: 1.0, rep=1)])

    def test_compact_drops_the_arrays(self):
        c = dc.compact(dc.curve([measurement(10, 8, lambda ctx: 1.0)]))
        self.assertNotIn("per_step_ms", c)
        self.assertNotIn("context_tokens", c)
        self.assertIn("bins", c)


class Run(unittest.TestCase):
    def write_run(self, directory):
        for name in ("cpu-baseline", "cpu-profile"):
            rows = [{"kind": "metadata"}, measurement(64, 8, lambda ctx: 5.0), measurement(128, 8, lambda ctx: 6.0)]
            (directory / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def test_instrumented_runs_are_excluded(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_run(Path(tmp))
            curves = dc.summarize_run(Path(tmp), load_experiment())
            self.assertEqual({c["run"] for c in curves}, {"cpu-baseline"})
            self.assertEqual([c["prompt_tokens"] for c in curves], [64, 128])

    def test_outputs_have_one_csv_row_per_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_run(Path(tmp))
            curves = dc.summarize_run(Path(tmp), load_experiment())
            dc.write_outputs(Path(tmp), curves)
            with (Path(tmp) / "decode-curve.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(len(rows) - 1, 16)
            self.assertEqual(len(json.loads((Path(tmp) / "decode-curve.json").read_text())["curves"]), 2)

    def test_plot_writes_a_figure_only_when_there_is_a_curve(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.write_run(Path(tmp))
            curves = dc.summarize_run(Path(tmp), load_experiment())
            self.assertTrue(dc.plot(Path(tmp), curves).is_file())
            self.assertIsNone(dc.plot(Path(tmp), [dict(c, steps=1) for c in curves]))

    @unittest.skipUnless(RECORDED.is_dir(), "recorded 2026-09-22 results not present")
    def test_recorded_baseline_has_32_step_curves(self):
        curves = dc.summarize_run(RECORDED, load_experiment())
        self.assertGreaterEqual(len(curves), 6)
        self.assertTrue(all(c["steps"] == 32 for c in curves))
        # 32 steps barely move the context, which is exactly why longer continuations are needed
        by = {(c["run"], c["prompt_tokens"]): c for c in curves}
        self.assertLess(by[("cpu-baseline", 2048)]["context_span_fraction"], 0.02)


class ContinuationSettings(unittest.TestCase):
    def dry_run(self, *args):
        return subprocess.run([sys.executable, str(CLI), "profile", "--dry-run", *args],
                              capture_output=True, text=True, timeout=120)

    def test_too_long_a_continuation_is_rejected_before_anything_runs(self):
        result = self.dry_run("--lengths", "8192", "--decode-steps", "2000")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("1308", result.stderr + result.stdout)

    def test_the_exact_limit_is_accepted(self):
        self.assertEqual(self.dry_run("--lengths", "8192", "--decode-steps", "1308").returncode, 0)

    def test_long_continuation_flag_uses_the_configured_length(self):
        steps = load_experiment()["workload"]["long_continuation_steps"]
        result = self.dry_run("--lengths", "128", "--long-continuation", "--no-graphs", "--no-prompt-graphs",
                              "--no-prompt-timings", "--no-trace")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertRegex(result.stdout, rf" {steps} 128 ")

    def test_explicit_decode_steps_beat_the_long_flag(self):
        result = self.dry_run("--lengths", "128", "--long-continuation", "--decode-steps", "40", "--no-graphs",
                              "--no-prompt-graphs", "--no-prompt-timings", "--no-trace")
        self.assertRegex(result.stdout, r" 40 128 ")


if __name__ == "__main__":
    unittest.main(verbosity=2)
