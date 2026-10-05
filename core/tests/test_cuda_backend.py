"""Optional CUDA runner behavior using synthetic files; no GPU or model required."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import os
import sys
import tempfile
import unittest

import numpy as np

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "analysis"))
sys.path.insert(0, str(CORE / "runner"))

import cli  # noqa: E402
import hostinfo  # noqa: E402
import measure  # noqa: E402
import platform_profile  # noqa: E402
import validate  # noqa: E402


class CudaRunner(unittest.TestCase):
    def test_cuda_is_explicit_and_cpu_stays_first(self):
        args = cli.build_parser().parse_args(["profile", "--cuda", "on", "--smoke"])
        self.assertEqual(args.cuda, "on")
        ctx = SimpleNamespace(state={"metal": False, "cuda": True}, skip_metal=False)
        self.assertEqual(measure.backends(ctx), [("cpu", 0), ("cuda", 99)])

    def test_cpu_cuda_logit_comparison(self):
        experiment = {
            "model": {"vocabulary_size": 5},
            "policy": {"separate_case_prefixes": {}}
        }
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
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
