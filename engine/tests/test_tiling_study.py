"""The tiling study's robustness path validates capture identity like the study runner does."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from helpers import ENGINE
from sweeps.specs import load_spec

ROOT = ENGINE.parent
SCRIPT = ROOT / "research/compute-mapping/2026-10-17-tiling-sweep/analyze.py"


def load_analysis():
    spec = importlib.util.spec_from_file_location("tiling_analysis", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RobustnessCaptureIdentity(unittest.TestCase):
    """Substitutions go into a temporary copy of the run; the committed results are never touched."""

    def setUp(self):
        self.analysis = load_analysis()
        _, baseline, self.base = load_spec(ROOT, ROOT / self.analysis.SPECS[0])
        self.source = ROOT / baseline["run"] / "graphs"
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        for name in self.analysis.ROBUST_WORKLOADS:
            shutil.copytree(self.source / name, self.run / "graphs" / name)
        self.baseline = copy.deepcopy(baseline)
        self.baseline["run"] = str(self.run)  # absolute, so ROOT / run resolves to the temporary copy
        self.analysis.ROBUST_SHAPES = [(4, 32, 32)]
        self.analysis.ROBUST_L1_MIB = [8]

    def capture(self, workload, phase):
        return json.loads(gzip.decompress((self.source / workload / f"{phase}.json.gz").read_bytes()))

    def override(self, workload, phase, capture):
        (self.run / "graphs" / workload / f"{phase}.json").write_text(json.dumps(capture), encoding="utf-8")

    def robustness(self):
        captures = {}
        rows = self.analysis.robustness(self.base, self.baseline, captures)
        return rows, captures

    def test_pinned_captures_are_validated_and_recorded(self):
        rows, captures = self.robustness()
        self.assertEqual(len(rows), len(self.analysis.ROBUST_WORKLOADS))
        expected = {f"{w}/{p}" for w in self.analysis.ROBUST_WORKLOADS for p in ("prefill", "decode")}
        self.assertEqual(set(captures), expected)
        for key in expected:
            self.assertEqual(captures[key], self.baseline["decoded_capture_sha256"][key])

    def test_equivalent_plain_json_is_accepted(self):
        self.override("pp128-fa-on", "prefill", self.capture("pp128-fa-on", "prefill"))
        self.robustness()

    def test_substituted_capture_fails_in_either_phase(self):
        for phase in ("prefill", "decode"):
            with self.subTest(phase=phase):
                self.override("pp128-fa-on", phase, self.capture("pp512-fa-on", phase))
                with self.assertRaisesRegex(ValueError, f"capture identity differs: pp128-fa-on/{phase}"):
                    self.robustness()
                (self.run / "graphs" / "pp128-fa-on" / f"{phase}.json").unlink()

    def test_edited_operation_parameter_fails(self):
        capture = self.capture("pp128-fa-on", "prefill")
        tensor = next(t for t in capture["tensors"] if t.get("op_params_i32"))
        tensor["op_params_i32"][0] += 1
        self.override("pp128-fa-on", "prefill", capture)
        with self.assertRaisesRegex(ValueError, "capture identity differs: pp128-fa-on/prefill"):
            self.robustness()


class OutputAndRecordedSummary(unittest.TestCase):
    def setUp(self):
        self.analysis = load_analysis()

    def test_output_stays_in_a_new_build_sweeps_folder(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.analysis.OUTPUT_ROOT = root / "build/sweeps"
            fresh = root / "build/sweeps/tiling-analysis"
            self.assertEqual(self.analysis.output_folder(fresh), fresh)
            fresh.mkdir(parents=True)
            self.assertEqual(self.analysis.output_folder(fresh), fresh)  # existing but empty
            (fresh / "points.csv").write_text("kept\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not overwritten"):
                self.analysis.output_folder(fresh)
            self.assertEqual((fresh / "points.csv").read_text(encoding="utf-8"), "kept\n")
            for outside in (root / "build/sweeps", root / "research", root / "build/elsewhere"):
                with self.subTest(path=outside), self.assertRaisesRegex(ValueError, "under build/sweeps"):
                    self.analysis.output_folder(outside)

    def test_analysis_reproduces_the_recorded_compact_summary(self):
        recorded = json.loads((ROOT / "docs/sweeps/tiling-v1.json").read_text(encoding="utf-8"))
        *_, summary = self.analysis.analysis()
        recorded.pop("reproduction")
        self.assertEqual(json.loads(json.dumps(summary)), recorded)


if __name__ == "__main__":
    unittest.main()
