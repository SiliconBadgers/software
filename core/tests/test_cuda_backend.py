"""Optional CUDA runner behavior using synthetic files; no GPU or model required."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import json
import os
import sys
import tempfile
import unittest

import numpy as np

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "analysis"))
sys.path.insert(0, str(CORE / "runner"))

import cli  # noqa: E402
import build  # noqa: E402
import hostinfo  # noqa: E402
import measure  # noqa: E402
import platform_profile  # noqa: E402
import plot  # noqa: E402
import datasets  # noqa: E402
import validate  # noqa: E402


class CudaRunner(unittest.TestCase):
    def test_default_is_cpu_reference_only(self):
        args = cli.build_parser().parse_args(["profile", "--smoke"])
        self.assertEqual(args.cuda, "off")
        ctx = SimpleNamespace(state={"metal": False, "cuda_mode": "off"}, skip_metal=False)
        self.assertEqual(measure.backends(ctx), [("cpu", "baseline", 0, False, "cpu_reference")])

    def test_cuda_compare_separates_reference_and_cuda_builds(self):
        args = cli.build_parser().parse_args(["profile", "--cuda", "on", "--smoke"])
        self.assertEqual(args.cuda, "on")
        ctx = SimpleNamespace(state={"metal": False, "cuda_mode": "on"}, skip_metal=False)
        self.assertEqual(measure.backends(ctx), [
            ("cpu", "baseline", 0, False, "cpu_reference"),
            ("cuda", "cuda", 99, False, "accelerated_execution"),
        ])

    def test_cuda_control_is_diagnostic_opt_in(self):
        args = cli.build_parser().parse_args(["profile", "--cuda", "on", "--cuda-controls", "--smoke"])
        ctx = cli.make_context(args)
        ctx.state.update(metal=False, cuda_mode=ctx.cuda)
        self.assertEqual([row[0] for row in measure.backends(ctx)], ["cpu", "cuda-cpu", "cuda"])
        self.assertTrue(ctx.run_trace)
        self.assertTrue(ctx.run_graphs)

    def test_diagnostic_flags_require_cpu_cuda_comparison(self):
        for flag in ("--cuda-controls", "--cuda-no-host-control"):
            for mode in ("off", "only"):
                with self.subTest(flag=flag, mode=mode):
                    args = cli.build_parser().parse_args(["profile", "--cuda", mode, flag, "--smoke"])
                    with self.assertRaisesRegex(SystemExit, "requires --cuda on"):
                        cli.make_context(args)

    def test_cuda_only_is_a_gpu_only_fast_path(self):
        args = cli.build_parser().parse_args(["profile", "--cuda", "only", "--smoke", "--dry-run"])
        ctx = cli.make_context(args)
        ctx.state["cuda_mode"] = ctx.cuda
        self.assertEqual(measure.backends(ctx), [("cuda", "cuda", 99, False, "accelerated_execution")])
        self.assertFalse(ctx.run_trace)
        self.assertFalse(ctx.run_graphs)
        self.assertFalse(ctx.run_prompt_graphs)
        self.assertFalse(ctx.run_prompt_timings)

    def test_cuda_and_trace_sources_are_separate(self):
        ctx = SimpleNamespace(work=Path("work"))
        self.assertEqual(build.source_dir(ctx, "baseline"), Path("work/llama.cpp"))
        self.assertEqual(build.source_dir(ctx, "cuda"), Path("work/llama.cpp"))
        self.assertEqual(build.source_dir(ctx, "profile"), Path("work/llama.cpp-profile"))

    def test_no_host_control_is_explicit_and_keeps_cuda_execution(self):
        ctx = SimpleNamespace(state={"metal": False, "cuda_mode": "on"}, skip_metal=False,
                              cuda_no_host_control=True)
        self.assertEqual([row[0] for row in measure.backends(ctx)],
                         ["cpu", "cuda-cpu", "cuda-cpu-no-host", "cuda"])
        self.assertEqual(measure.backends(ctx)[2][2:4], (0, True))

    def test_metal_is_enabled_only_for_the_baseline_build(self):
        ctx = SimpleNamespace(state={"metal": True})
        self.assertEqual(build.backend_options(ctx, "baseline"), {"GGML_METAL": "ON", "GGML_CUDA": "OFF"})
        self.assertEqual(build.backend_options(ctx, "profile"), {"GGML_METAL": "OFF", "GGML_CUDA": "OFF"})
        self.assertEqual(build.backend_options(ctx, "cuda"), {"GGML_METAL": "OFF", "GGML_CUDA": "ON"})

    def test_cpu_plot_label_distinguishes_legacy_control_from_cpu_reference(self):
        legacy = [{"run": "cpu-baseline"}, {"run": "cuda-baseline"}]
        split = legacy + [{"run": "cuda-cpu-baseline"}]
        self.assertEqual(plot.cpu_label(legacy), "CPU (build not identified)")
        self.assertEqual(plot.cpu_label(split), "CPU (build not identified)")
        self.assertEqual(plot.cpu_label([{"run": "cpu-baseline"}]), "CPU (build not identified)")

    def test_cpu_cuda_logit_comparison(self):
        experiment = {
            "model": {"vocabulary_size": 5},
            "policy": {"separate_case_prefixes": {}}
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "run-manifest.json").write_text(json.dumps({
                "build": {"requested": {"GGML_CUDA": "OFF"}},
                "datasets": [
                    {"name": "cpu", "role": "cpu_reference", "build": "baseline"},
                    {"name": "cuda", "role": "accelerated_execution", "build": "cuda"},
                ]}), encoding="utf-8")
            cpu = np.array([0, 1, 2, 3, 4], dtype="<f4")
            cuda = np.array([0, 1, 2, 3.25, 4], dtype="<f4")
            for phase in ("prefill", "decode"):
                (root / f"cpu-baseline-p128-{phase}.f32").write_bytes(cpu.tobytes())
                (root / f"cuda-baseline-p128-{phase}.f32").write_bytes(cuda.tobytes())
            checks = validate.compare_backend(root, experiment, [128], "cuda")
        self.assertEqual(len(checks), 2)
        self.assertTrue(all(row["both_finite"] for row in checks))
        self.assertTrue(all(row["cpu_top_token"] == row["cuda_top_token"] == 4 for row in checks))
        self.assertTrue(all(row["max_abs_difference"] == 0.25 for row in checks))

    def test_historical_cuda_build_cpu_is_not_mislabeled_as_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "run-manifest.json").write_text(json.dumps({
                "config": {"backends": ["cpu", "cuda"]},
                "build": {"requested": {"GGML_CUDA": "ON"}},
                "cuda_build": None,
            }), encoding="utf-8")
            rows = datasets.read(root)
            self.assertEqual(rows[0]["role"], "cuda_build_cpu_control")
            with self.assertRaisesRegex(ValueError, "refusing to guess"):
                datasets.cpu_reference(root)

    def test_host_manifest_omits_cuda_when_not_requested(self):
        host = hostinfo.collect("test", {"c": None, "cxx": None})
        self.assertNotIn("cuda_compiler", host["tools"])
        self.assertNotIn("cuda_device", host)

    def test_cuda_compiler_honors_toolkit_environment(self):
        with tempfile.TemporaryDirectory() as folder:
            compiler = Path(folder) / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
            compiler.parent.mkdir()
            compiler.touch()
            with mock.patch.dict(os.environ, {"CUDA_PATH": folder}, clear=False), \
                    mock.patch.object(platform_profile.shutil, "which", return_value=None):
                self.assertEqual(platform_profile.cuda_compiler(), str(compiler))


if __name__ == "__main__":
    unittest.main(verbosity=2)
