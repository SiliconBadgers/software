"""Exact pre-refactor regression contract, not a hardware-accuracy certification."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from characterization import FIXTURE, ROOT, RUN, SWITCHES, graph_record, snapshot
from helpers import real_run
from sbengine.graph import Graph, Workload, capture_path, has_capture, layer_of, load_workload, latest_run


class Characterization(unittest.TestCase):
    def test_all_pinned_captures_profiles_and_switches_match(self):
        expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
        got = snapshot()
        self.assertEqual(len(got["graphs"]), 44)
        self.assertEqual(len(got["cases"]), 22 * 3 + len(SWITCHES))
        for key in expected:
            with self.subTest(section=key):
                self.assertEqual(got[key], expected[key], "Inspect full comparison outputs; do not auto-update the fixture.")

    def test_plain_and_compressed_capture_contract(self):
        source = RUN / "graphs/pp512-fa-on"
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for phase in ("prefill", "decode"):
                captured = capture_path(source, phase)
                compressed = directory / f"{phase}.json.gz"
                compressed.write_bytes(captured.read_bytes())
                self.assertEqual(capture_path(directory, phase), compressed)
                plain = directory / f"{phase}.json"
                plain.write_bytes(gzip.decompress(compressed.read_bytes()))
                self.assertEqual(capture_path(directory, phase), plain)  # existing plain-file precedence
                self.assertEqual(graph_record(Graph(plain, phase)), graph_record(Graph(compressed, phase)))
            self.assertTrue(has_capture(directory))
            workload = load_workload(directory)
            self.assertIsInstance(workload, Workload)
            self.assertIs(workload.graph("prefill"), workload.prefill)
            self.assertIs(workload.graph("decode"), workload.decode)

    def test_test_selection_is_pinned_but_latest_api_still_exists(self):
        self.assertEqual(real_run(), RUN)
        self.assertTrue(callable(latest_run))
        self.assertEqual(layer_of("blk.12.attn_q.weight"), 12)

    def test_output_is_contained_and_existing_evidence_is_not_overwritten(self):
        with self.assertRaisesRegex(ValueError, "must stay under"):
            snapshot(ROOT / "engine/tests/fixtures")
        parent = ROOT / "build/refactor-characterization"
        parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=parent) as temp:
            with self.assertRaisesRegex(ValueError, "existing comparison evidence"):
                snapshot(Path(temp))


if __name__ == "__main__":
    unittest.main()
