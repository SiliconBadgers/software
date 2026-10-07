"""Reference selection, historical identity, dataset boundaries and speedup controls."""
import json
import csv
import gzip
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "analysis"))
sys.path.insert(0, str(CORE / "runner"))
import backend_speedup
import datasets
import measure
import validate
import cli
import plot
from policy import load_experiment
import numpy as np
import importlib.util

# Python's startup site module otherwise shadows the dashboard generator.
spec = importlib.util.spec_from_file_location("result_site", CORE / "analysis" / "site.py")
result_site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(result_site)


class Provenance(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def manifest(self, data):
        (self.root / "run-manifest.json").write_text(json.dumps(data), encoding="utf-8")

    def test_split_run_without_explicit_roles_uses_build_identity(self):
        self.manifest({"config": {"backends": ["cpu", "cuda-cpu", "cuda"]},
                       "build": {"requested": {"GGML_CUDA": "OFF"}},
                       "cuda_build": {"requested": {"GGML_CUDA": "ON"}}})
        self.assertEqual(datasets.cpu_reference(self.root)["name"], "cpu")
        self.assertEqual([r["role"] for r in datasets.read(self.root)],
                         ["cpu_reference", "cuda_build_cpu_control", "accelerated_execution"])
        self.assertEqual(datasets.read(self.root / "analysis"), datasets.read(self.root))
        self.assertEqual(datasets.read(self.root / "prompts" / "alpha"), datasets.read(self.root))

    def test_contradictory_explicit_reference_is_rejected(self):
        self.manifest({"build": {"requested": {"GGML_CUDA": "ON"}}, "datasets": [
            {"name": "cpu", "role": "cpu_reference", "build": "baseline"}]})
        with self.assertRaisesRegex(ValueError, "contradicts"):
            datasets.read(self.root)

    def test_split_build_existence_does_not_prove_cpu_identity(self):
        self.manifest({"config": {"backends": ["cpu", "cuda"]}, "build": None, "cuda_build": {}})
        with self.assertRaisesRegex(ValueError, "refusing to guess"):
            datasets.cpu_reference(self.root)

    def test_cuda_baseline_is_control_even_with_separate_cuda_build(self):
        self.manifest({"config": {"backends": ["cpu", "cuda"]},
                       "build": {"requested": {"GGML_CUDA": "ON"}}, "cuda_build": {}})
        self.assertEqual(datasets.read(self.root)[0]["role"], "cuda_build_cpu_control")
        with self.assertRaisesRegex(ValueError, "refusing to guess"):
            datasets.cpu_reference(self.root)

    def test_unknown_files_do_not_become_cpu_reference(self):
        (self.root / "cpu-baseline.jsonl").touch()
        with self.assertRaisesRegex(ValueError, "refusing to guess"):
            datasets.cpu_reference(self.root)

    def test_legacy_package_manifest_preserves_reference(self):
        legacy = self.root / "results"
        legacy.mkdir()
        (self.root / "manifest.json").write_text(json.dumps({"baseline_backends": ["CPU", "Metal"]}))
        self.assertEqual(datasets.cpu_reference(legacy)["build"], "legacy_baseline")

    def test_historical_cuda_cannot_enter_profile_validation(self):
        self.manifest({"config": {"backends": ["cpu", "cuda"]},
                       "build": {"requested": {"GGML_CUDA": "ON"}}})
        with self.assertRaisesRegex(ValueError, "refusing to guess"):
            validate.validate(self.root, {"model": {"vocabulary_size": 5}}, [128], True)

    def test_longest_prefix_and_trace_roles(self):
        rows = [{"name": "cpu", "role": "cpu_reference", "build": "baseline"},
                {"name": "cuda", "role": "accelerated_execution", "build": "cuda"},
                {"name": "cuda-cpu", "role": "cuda_build_cpu_control", "build": "cuda"},
                {"name": "cuda-cpu-no-host", "role": "cuda_build_cpu_no_host_control", "build": "cuda"}]
        self.assertEqual(datasets.for_run(rows, "cuda-cpu-no-host-8k")["name"], "cuda-cpu-no-host")
        self.assertEqual(datasets.for_run(rows, "cpu-profile")["role"], "cpu_trace")
        self.assertEqual(datasets.for_run(rows, "cpu-8k-control")["role"], "unknown")

    def test_speedups_include_each_control_and_exclude_trace(self):
        rows = [{"name": "cpu", "role": "cpu_reference", "build": "baseline"},
                {"name": "cuda-cpu", "role": "cuda_build_cpu_control", "build": "cuda"},
                {"name": "cuda-cpu-no-host", "role": "cuda_build_cpu_no_host_control", "build": "cuda"},
                {"name": "cuda", "role": "accelerated_execution", "build": "cuda"}]
        summaries = [{"run": name, "prompt_tokens": 128, "prefill_seconds": pre, "decode_ms": dec}
                     for name, pre, dec in [("cpu-baseline", 8, 6), ("cuda-cpu-baseline", 10, 4),
                                            ("cuda-cpu-no-host-baseline", 9, 5),
                                            ("cuda-baseline", 2, 1), ("cpu-profile", 100, 100)]]
        result = backend_speedup.compare(summaries, rows)
        self.assertEqual(len(result), 3)
        self.assertEqual([(r["prefill_speedup"], r["decode_speedup"]) for r in result],
                         [(4, 6), (5, 4), (4.5, 5)])

    def test_dashboard_keeps_roles_and_builds_for_historical_run(self):
        self.manifest({"config": {"backends": ["cpu", "cuda"]},
                       "build": {"requested": {"GGML_CUDA": "ON"}}})
        analysis = self.root / "analysis"
        analysis.mkdir()
        (analysis / "summary.csv").write_text(
            "run,prompt_tokens,prefill_tps,decode_ms,decode_tps\ncpu-baseline,128,100,10,100\n")
        row, = result_site.performance_rows(self.root, datasets.read(self.root))
        self.assertEqual(row["dataset_role"], "cuda_build_cpu_control")
        self.assertEqual(row["build"], "baseline")

    def test_dashboard_without_derived_graphs_does_not_overwrite_existing_page(self):
        output = self.root / "index.html"
        output.write_text("existing graph dashboard", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "No derived graph summaries"):
            result_site.build(self.root, output)
        self.assertEqual(output.read_text(encoding="utf-8"), "existing graph dashboard")

    def test_fallback_reports_failure_instead_of_speedup(self):
        with self.assertRaisesRegex(ValueError, "fell back to CPU"):
            measure.check_fallback("cuda", {"cpu_fallback": True})
        measure.check_fallback("cpu", {"cpu_fallback": False})

    def test_reader_fallback_overrides_requested_role_across_backends_and_sources(self):
        for backend in ("cuda", "metal"):
            for evidence in ({"cpu_fallback": True},
                             {"cpu_fallback": False, "cases": [
                                 {"file": f"{backend}-baseline.jsonl", "actual": {"cpu_fallback": False}},
                                 {"file": f"{backend}-8k.jsonl", "actual": {"cpu_fallback": True}}]}):
                with self.subTest(backend=backend, evidence=evidence):
                    self.manifest({"datasets": [{"name": backend, "role": "accelerated_execution",
                                                 "build": "baseline", **evidence}]})
                    row, = datasets.read(self.root)
                    self.assertEqual(row["role"], "cpu_fallback")
                    self.assertEqual(row["requested_role"], "accelerated_execution")
                    self.assertIn("CPU fallback", datasets.label(row))
                    self.assertEqual(datasets.read(self.root / "analysis"), [row])

    def test_compressed_raw_placement_prevents_filename_only_acceleration(self):
        with gzip.open(self.root / "cuda-baseline.jsonl.gz", "wt", encoding="utf-8") as stream:
            stream.write(json.dumps({"kind": "metadata", "backend": {"cpu_fallback": True}}) + "\n")
        row, = datasets.read(self.root)
        self.assertEqual(row["role"], "cpu_fallback")
        self.assertEqual(datasets.label(row), "CPU fallback (CUDA requested)")

    def test_raw_fallback_overrides_inferred_manifest_without_confusing_cpu_control(self):
        self.manifest({"config": {"backends": ["cuda-cpu", "cuda"]}})
        for name, fallback in (("cuda-cpu", False), ("cuda", True)):
            (self.root / f"{name}-baseline.jsonl").write_text(json.dumps(
                {"kind": "metadata", "backend": {"cpu_fallback": fallback}}) + "\n", encoding="utf-8")
        control, fallback = datasets.read(self.root)
        self.assertEqual(control["role"], "cuda_build_cpu_control")
        self.assertEqual(fallback["role"], "cpu_fallback")
        self.assertNotIn("CUDA measured", datasets.report_subtitle([fallback], "Host · CUDA measured"))
        self.assertEqual(datasets.report_subtitle([control], "Host"), "Host")

    def test_prompt_placement_is_scoped_to_that_prompt(self):
        for name in ("alpha", "beta"):
            (self.root / "prompts" / name).mkdir(parents=True)
        self.manifest({"datasets": [{"name": "cuda", "role": "accelerated_execution", "build": "cuda"}],
                       "prompt_backend_placements": [
                           {"file": "prompts/alpha/cuda-baseline.jsonl", "actual": {"cpu_fallback": True}},
                           {"file": "prompts/beta/cuda-baseline.jsonl", "actual": {"cpu_fallback": False}}]})
        self.assertEqual(datasets.read(self.root)[0]["role"], "accelerated_execution")
        self.assertEqual(datasets.read(self.root / "prompts" / "alpha")[0]["role"], "cpu_fallback")
        self.assertEqual(datasets.read(self.root / "prompts" / "beta")[0]["role"], "accelerated_execution")

    def test_speedups_require_accelerated_role_and_reject_unnormalized_fallback(self):
        summaries = [{"run": name, "prompt_tokens": 128, "prefill_seconds": pre, "decode_ms": pre}
                     for name, pre in (("cpu-baseline", 4), ("cuda-baseline", 1))]
        for identity in ({"role": "cpu_fallback"}, {"role": "unknown"},
                         {"role": "accelerated_execution", "cpu_fallback": True},
                         {"role": "accelerated_execution", "cases": [{"actual": {"cpu_fallback": True}}]}):
            with self.subTest(identity=identity):
                rows = [{"name": "cpu", "role": "cpu_reference"}, {"name": "cuda", **identity}]
                self.assertEqual(backend_speedup.compare(summaries, rows), [])

    def test_failed_run_reanalysis_relabels_reports_and_replaces_stale_speedups(self):
        experiment = load_experiment()
        experiment["model"]["vocabulary_size"] = 5
        data = {"run": {"status": "failed"}, "experiment": {"name": "qwen35-2b-q4km"},
                "config": {"prompt_lengths": [128], "trace": False, "backends": ["cpu", "cuda"]},
                "build": {"requested": {"GGML_CUDA": "OFF"}}, "plot_subtitle": "Test host · CUDA measured",
                "datasets": [{"name": "cpu", "role": "cpu_reference", "build": "baseline"},
                             {"name": "cuda", "role": "accelerated_execution", "build": "cuda"}]}
        self.manifest(data)
        for name, duration in (("cpu", 4), ("cuda", 1)):
            records = [{"kind": "metadata", "backend": {"cpu_fallback": False}},
                       {"kind": "measurement", "prompt_tokens": 128, "rep": 0,
                        "prefill_us": duration * 1_000_000, "decode_us": [duration * 1000] * 2}]
            (self.root / f"{name}-baseline.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
            for phase in ("prefill", "decode"):
                (self.root / f"{name}-baseline-p128-{phase}.f32").write_bytes(np.arange(5, dtype="<f4").tobytes())
        args = SimpleNamespace(run_dir=self.root)
        with patch.object(cli, "load_experiment", return_value=experiment):
            cli.analyze_command(args)
        out = self.root / "analysis"
        self.assertEqual(len(json.loads((out / "backend-speedups.json").read_text())), 1)
        # An observed fallback in one case defeats even a contradictory false aggregate/raw flag.
        data["datasets"][1].update(cpu_fallback=False, cases=[
            {"file": "cuda-baseline.jsonl", "actual": {"cpu_fallback": True}}])
        self.manifest(data)
        figures = []
        close = plot.plt.close

        def inspect_figure(fig):
            figures.append(([ax.get_legend_handles_labels()[1] for ax in fig.axes],
                            [text.get_text() for text in fig.texts]))
            close(fig)

        with patch.object(cli, "load_experiment", return_value=experiment), \
             patch.object(plot.plt, "close", side_effect=inspect_figure):
            cli.analyze_command(args)
        self.assertEqual(json.loads((out / "backend-speedups.json").read_text()), [])
        with (out / "backend-speedups.csv").open(newline="") as stream:
            self.assertEqual(list(csv.DictReader(stream)), [])
        self.assertTrue(any("CPU fallback (CUDA requested)" in labels for axes, _ in figures for labels in axes))
        self.assertFalse(any("CUDA" in labels for axes, _ in figures for labels in axes))
        self.assertFalse(any("CUDA measured" in text for _, texts in figures for text in texts))
        curve = next(row for row in json.loads((out / "decode-curve.json").read_text())["curves"]
                     if row["run"] == "cuda-baseline")
        self.assertTrue(curve["cpu_fallback"])
        checks = json.loads((out / "cpu-cuda-logit-check.json").read_text())
        self.assertTrue(all(row["cpu_fallback"] for row in checks))
        dashboard = result_site.performance_rows(self.root, datasets.read(self.root))
        self.assertEqual(next(row for row in dashboard if row["run"] == "cuda-baseline")["dataset_role"], "cpu_fallback")
        manifest = json.loads((self.root / "run-manifest.json").read_text())
        self.assertEqual(manifest["run"]["status"], "failed")
        self.assertEqual(manifest["analysis"]["backend_speedups"], 0)

    def test_baseline_preserves_actual_placement_when_fallback_raises(self):
        ctx = SimpleNamespace(state={}, repetitions=1)
        plan = [("cuda", "cuda", 99, False, "accelerated_execution")]
        actual = {"requested_ngl": 99, "actual_accelerator": False, "cpu_fallback": True,
                  "accelerator_model_bytes": 0, "model_buffers": []}
        with patch.object(measure, "backends", return_value=plan), \
             patch.object(measure, "build_dir"), patch.object(measure, "exe_path"), \
             patch.object(measure, "_cases", return_value=[("baseline", [128])]), \
             patch.object(measure, "_measure", return_value=actual):
            with self.assertRaisesRegex(ValueError, "fell back to CPU"):
                measure.baseline(ctx, "model")
        row, = ctx.state["datasets"]
        self.assertTrue(row["cpu_fallback"])
        self.assertEqual(row["cases"], [{"file": "cuda-baseline.jsonl", "actual": actual}])

    def test_old_harness_metadata_is_not_silently_accepted(self):
        path = self.root / "old.jsonl"
        path.write_text(json.dumps({"kind": "metadata", "ngl": 99}) + "\n")
        with self.assertRaisesRegex(ValueError, "actual backend placement"):
            measure.read_backend_metadata(path)


if __name__ == "__main__":
    unittest.main()
