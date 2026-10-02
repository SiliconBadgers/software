"""The core analysis must reproduce the recorded 2026-09-22 outputs from its raw evidence.

Needs no llama.cpp, model or compiler: only Python and numpy. Run from anywhere:
    python core/tests/test_regression_0922.py
"""
import gzip
import json
import os
import shutil
from pathlib import Path
import sys
import tempfile
import unittest

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "analysis"))
RECORDED = CORE.parent / "experiments" / "llama-cpp" / "2026-09-22" / "results"

import graph, ops, summarize, validate  # noqa: E402


def load(path):
    path = Path(path)
    if not path.exists():
        path = Path(str(path) + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return json.load(stream)


def text(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", newline="") as stream:
        return stream.read().replace("\r\n", "\n")


@unittest.skipUnless(RECORDED.is_dir(), "recorded 2026-09-22 results not present")
class Regression0922(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.out = Path(cls.tmp.name)
        common = ["--results-dir", str(RECORDED), "--output-dir", str(cls.out)]
        summarize.main(common)
        ops.main(common)
        validate.main(common)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def same_json(self, name):
        self.assertEqual(load(self.out / name), load(RECORDED / name), name)

    def test_summary_json(self):
        self.same_json("summary.json")

    def test_operation_summary_json(self):
        self.same_json("operation-summary.json")

    def test_matrix_weight_inventory(self):
        self.same_json("matrix-weight-inventory.json")

    def test_operator_shapes(self):
        self.same_json("operator-shapes.json")

    def test_profile_validation(self):
        self.same_json("profile-output-validation.json")

    def test_backend_logit_check(self):
        self.same_json("cpu-metal-logit-check.json")

    def test_csv_outputs(self):
        for name in ("summary.csv", "operation-summary.csv"):
            self.assertEqual(text(self.out / name), text(RECORDED / name), name)

    def test_outputs_refuse_recorded_directory(self):
        with self.assertRaises(ValueError):
            summarize.main(["--results-dir", str(RECORDED), "--output-dir", str(RECORDED)])


# Zeb's captures live on an unmerged branch; point SB_ZEB_GRAPHS at an extracted copy to run this early.
ZEB = Path(os.environ.get("SB_ZEB_GRAPHS") or
           CORE.parent / "experiments" / "llama-cpp" / "2026-09-24-qwen35-2b" / "graphs" / "pp128")


@unittest.skipUnless((ZEB / "prefill.json").is_file(),
                     "Zeb Taylor's 2026-09-24 graph captures not present (unmerged import branch)")
class GraphRegression(unittest.TestCase):
    """core/analysis/graph.py must reproduce the recorded analyzer outputs from the same captures."""

    def test_pp128_outputs_identical(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "pp128", Path(tmp) / "out"
            src.mkdir()
            for phase in ("prefill", "decode"):
                (src / f"{phase}.json").write_bytes((ZEB / f"{phase}.json").read_bytes())
            graph.analyze_directory(src, out)
            for name in ("prefill.ops.csv", "decode.ops.csv", "prefill.summary.json", "decode.summary.json",
                         "prefill.dot", "decode.dot", "prefill.layer0.dot", "prefill.layer3.dot",
                         "decode.layer0.dot", "decode.layer3.dot", "REPORT.md"):
                a, b = text(out / name), text(ZEB / name)
                if name == "REPORT.md" and shutil.which("dot") is None:
                    # Only the closing sentence about rendered SVGs may differ without Graphviz.
                    a, b = ([l for l in x.splitlines() if not l.startswith("Raw JSON contains")] for x in (a, b))
                self.assertEqual(a, b, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
