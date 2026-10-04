"""The prompt set is well-formed, and the per-prompt graph comparison groups graphs correctly."""
import json
from pathlib import Path
import re
import sys
import tempfile
import unittest

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "runner"))
sys.path.insert(0, str(CORE / "analysis"))
import capture  # noqa: E402
import graph  # noqa: E402

PROMPTS_JSON = CORE / "prompts" / "prompts.json"


class PromptSet(unittest.TestCase):
    def setUp(self):
        self.listing = json.loads(PROMPTS_JSON.read_text(encoding="utf-8"))["prompts"]

    def test_there_are_eight_prompts_with_unique_safe_ids(self):
        ids = [p["id"] for p in self.listing]
        self.assertEqual(len(ids), 8)
        self.assertEqual(len(set(ids)), 8)
        for prompt_id in ids:
            self.assertRegex(prompt_id, r"^[a-z0-9_]+$")
            # the folder name the pipeline creates must be recognised by the comparison
            self.assertTrue(graph.PROMPT_DIR.match(f"prompt-{prompt_id}-fa-on"))

    def test_files_are_present_utf8_and_lf(self):
        for entry in self.listing:
            data = (PROMPTS_JSON.parent / entry["file"]).read_bytes()
            self.assertGreater(len(data), 100, entry["file"])
            data.decode("utf-8")
            self.assertNotIn(b"\r", data, entry["file"])

    def test_loader_returns_every_prompt(self):
        self.assertEqual([p[0] for p in capture.load_prompt_set(PROMPTS_JSON)], [p["id"] for p in self.listing])

    def test_prompts_differ(self):
        texts = [(PROMPTS_JSON.parent / p["file"]).read_text(encoding="utf-8") for p in self.listing]
        self.assertEqual(len(set(texts)), len(texts))


def write_graph(directory, tokens, shapes, extra_op=False):
    """A tiny stand-in for a captured graph: enough for fingerprints() and compare_prompt_graphs()."""
    directory.mkdir(parents=True)
    tensors = [{"id": i, "op_desc": op, "dtype": "f32", "shape": shape}
               for i, (op, shape) in enumerate([("MUL_MAT", shapes[0]), ("ADD", shapes[1])])]
    if extra_op:
        tensors.append({"id": 2, "op_desc": "SILU", "dtype": "f32", "shape": shapes[1]})
    order = [t["id"] for t in tensors]
    for phase in ("prefill", "decode"):
        (directory / f"{phase}.json").write_text(json.dumps({"tensors": tensors, "scheduled_nodes": order}))
        (directory / f"{phase}.summary.json").write_text(json.dumps({
            "metadata": {"prompt_tokens": tokens, "context_capacity": 4 * tokens},
            "scheduled_nodes": len(order), "macs_by_supported_op": {"MUL_MAT": tokens * 1000}}))


class PromptGraphComparison(unittest.TestCase):
    def test_grouping(self):
        with tempfile.TemporaryDirectory() as run:
            g = Path(run) / "graphs"
            write_graph(g / "prompt-a-fa-on", 100, ([8, 100, 1, 1], [8, 100, 1, 1]))
            write_graph(g / "prompt-b-fa-on", 100, ([8, 100, 1, 1], [8, 100, 1, 1]))   # same shapes as a
            write_graph(g / "prompt-c-fa-on", 200, ([8, 200, 1, 1], [8, 200, 1, 1]))   # same ops, other shapes
            write_graph(g / "prompt-d-fa-on", 300, ([8, 300, 1, 1], [8, 300, 1, 1]), extra_op=True)
            write_graph(g / "not-a-prompt-fa-on", 1, ([1, 1, 1, 1], [1, 1, 1, 1]))     # ignored
            result = graph.compare_prompt_graphs(run)
            groups = result["groups"]["flash_attention_on"]
            self.assertEqual(sorted(map(sorted, groups["structure_groups"])), [["a", "b"], ["c"], ["d"]])
            self.assertEqual(sorted(map(sorted, groups["op_sequence_groups"])), [["a", "b", "c"], ["d"]])
            self.assertEqual({r["prompt_id"] for r in result["prompts"]}, {"a", "b", "c", "d"})
            self.assertTrue((g / "prompt-comparison.json").is_file())
            self.assertIn("Distinct structures: 3", (g / "prompt-comparison.md").read_text(encoding="utf-8"))

    def test_run_without_prompt_graphs_returns_none(self):
        with tempfile.TemporaryDirectory() as run:
            (Path(run) / "graphs").mkdir()
            self.assertIsNone(graph.compare_prompt_graphs(run))

    def test_flash_attention_settings_are_compared_separately(self):
        with tempfile.TemporaryDirectory() as run:
            g = Path(run) / "graphs"
            write_graph(g / "prompt-a-fa-on", 100, ([8, 100, 1, 1], [8, 100, 1, 1]))
            write_graph(g / "prompt-a-fa-off", 100, ([8, 100, 1, 1], [8, 100, 1, 1]), extra_op=True)
            groups = graph.compare_prompt_graphs(run)["groups"]
            self.assertEqual(set(groups), {"flash_attention_on", "flash_attention_off"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
