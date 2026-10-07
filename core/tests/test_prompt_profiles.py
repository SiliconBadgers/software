"""prompt-profiles.json: one row per prompt built from that prompt's timing, trace and logit files.

Uses small synthetic files, so it needs neither the model nor a compiler.
"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "analysis"))
sys.path.insert(0, str(CORE / "tests"))
import prompts  # noqa: E402
import test_harness_equivalence as harness  # noqa: E402
from policy import load_experiment  # noqa: E402

N, STEPS = 50, 2
VOCAB = load_experiment()["model"]["vocabulary_size"]


def jsonl(path, rows):
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def timings(directory, name, reps):
    meta = {"kind": "metadata", "ngl": 0, "threads": 6, "reps": reps, "decode_steps": STEPS, "layers": 24,
            "vocabulary": VOCAB, "load_us": 1, "mode": "teacher_forced_shared_continuation", "n_ubatch": 512,
            "flash_attention": "on", "prompt_tokens": N, "continuation_tokens": STEPS}
    rows = [meta] + [{"kind": "measurement", "prompt_tokens": N, "rep": r, "prefill_us": 2_000_000 + 200_000 * r,
                      "decode_us": [70_000, 72_000]} for r in range(reps)]
    jsonl(directory / f"{name}.jsonl", rows)


def record(phase, op, elapsed, weight="blk.0.ffn_up.weight"):
    return {"graph_id": 0, "phase": phase, "node_index": 0, "n_threads": 6, "graph_status": 0, "elapsed_us": elapsed,
            "tensor": {"name": "out", "op": op, "dtype": "f32", "ne": [4, 5, 1, 1], "nbytes": 80, "is_view": False},
            "src": [{"name": weight, "dtype": "q4_K", "ne": [3, 4, 1, 1], "nbytes": 8, "is_view": False}, None],
            "fused_successors": []}


def logits(directory, prefix, value=0.0, tweak=False):
    data = np.full(VOCAB, value, dtype="<f4")
    if tweak:
        data[7] += 1.0
    for phase in ("prefill", "decode"):
        (directory / f"{prefix}-p{N}-{phase}.f32").write_bytes(data.tobytes())


def make_prompt_dir(run, prompt_id, traced=True, tweak=False):
    (run / "run-manifest.json").write_text(json.dumps({
        "config": {"backends": ["cpu"]}, "build": {"requested": {"GGML_CUDA": "OFF"}}
    }), encoding="utf-8")
    directory = run / "prompts" / prompt_id
    directory.mkdir(parents=True)
    timings(directory, "cpu-baseline", 2)
    logits(directory, "cpu-baseline")
    if traced:
        timings(directory, "cpu-profile", 1)
        logits(directory, "cpu-profile", tweak=tweak)
        jsonl(directory / "cpu-op-trace.jsonl", [
            record(f"p{N}_r0_prefill", "MUL_MAT", 100),
            record(f"p{N}_r0_decode0", "MUL_MAT", 30),
            record(f"p{N}_r0_decode1", "GET_ROWS", 10)])
    return directory


def make_prompt_set(folder, ids):
    folder.mkdir(parents=True, exist_ok=True)
    entries = []
    for prompt_id in ids:
        (folder / f"{prompt_id}.txt").write_text(f"text of {prompt_id}\n", encoding="utf-8")
        entries.append({"id": prompt_id, "file": f"{prompt_id}.txt", "description": f"the {prompt_id} prompt"})
    (folder / "prompts.json").write_text(json.dumps({"prompts": entries}), encoding="utf-8")
    return folder / "prompts.json"


class PromptProfiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = Path(self.tmp.name) / "run"
        self.run.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def test_row_has_timing_trace_and_checks(self):
        make_prompt_dir(self.run, "alpha")
        result = prompts.build(self.run, prompt_set=make_prompt_set(Path(self.tmp.name) / "set", ["alpha"]))
        (row,) = result["prompts"]
        self.assertEqual((row["prompt_id"], row["prompt_tokens"]), ("alpha", N))
        self.assertEqual(row["continuation"]["steps"], STEPS)
        self.assertAlmostEqual(row["timing"]["cpu"]["prefill_seconds"], 2.1)           # median of 2.0, 2.2
        self.assertAlmostEqual(row["timing"]["cpu"]["decode_ms"], 71.0)
        curve = row["timing"]["cpu"]["decode_curve"]                                     # per-step cost vs context
        self.assertEqual((curve["steps"], curve["first_context_tokens"], curve["last_context_tokens"]), (STEPS, N + 1, N + STEPS))
        self.assertNotIn("per_step_ms", curve)
        trace = row["trace"]
        self.assertEqual(trace["prefill"]["operation_calls"], {"MUL_MAT": 1})
        self.assertAlmostEqual(trace["prefill"]["groups_percent"]["MLP projections"], 100.0)
        self.assertAlmostEqual(trace["decode"]["timed_interval_us_per_token"], (30 + 10) / STEPS)
        self.assertAlmostEqual(trace["decode"]["groups_percent"]["MLP projections"], 75.0)
        self.assertIn("prefill_timed_over_untraced", trace["profiler_overhead_check"])
        self.assertIs(row["checks"]["traced_vs_untraced_logits_bit_identical"], True)
        self.assertTrue(row["prompt_sha256"])
        self.assertEqual(row["files"]["cpu_trace"], "prompts/alpha/cpu-op-trace.jsonl")

    def test_changed_logits_are_recorded_not_hidden(self):
        make_prompt_dir(self.run, "alpha", tweak=True)
        (row,) = prompts.build(self.run, prompt_set=make_prompt_set(Path(self.tmp.name) / "set", ["alpha"]))["prompts"]
        self.assertIs(row["checks"]["traced_vs_untraced_logits_bit_identical"], False)

    def test_untraced_prompt_has_no_trace_block(self):
        make_prompt_dir(self.run, "alpha", traced=False)
        (row,) = prompts.build(self.run, prompt_set=make_prompt_set(Path(self.tmp.name) / "set", ["alpha"]))["prompts"]
        self.assertIsNone(row["trace"])
        self.assertIsNone(row["checks"]["traced_vs_untraced_logits_bit_identical"])

    def test_optional_cuda_timing_and_numerical_check(self):
        directory = make_prompt_dir(self.run, "alpha", traced=False)
        (self.run / "run-manifest.json").write_text(json.dumps({
            "config": {"cuda_controls": True}, "build": {"requested": {"GGML_CUDA": "OFF"}}, "datasets": [
            {"name": "cpu", "role": "cpu_reference", "build": "baseline"},
            {"name": "cuda-cpu", "role": "cuda_build_cpu_control", "build": "cuda"},
            {"name": "cuda", "role": "accelerated_execution", "build": "cuda"},
        ]}), encoding="utf-8")
        timings(directory, "cuda-cpu-baseline", 2)
        logits(directory, "cuda-cpu-baseline")
        timings(directory, "cuda-baseline", 2)
        logits(directory, "cuda-baseline")
        (row,) = prompts.build(self.run, prompt_set=make_prompt_set(Path(self.tmp.name) / "set", ["alpha"]))["prompts"]
        self.assertAlmostEqual(row["timing"]["cuda_build_cpu"]["prefill_seconds"], 2.1)
        self.assertAlmostEqual(row["timing"]["cuda"]["prefill_seconds"], 2.1)
        self.assertEqual(len(row["checks"]["cpu_vs_cuda"]), 2)
        self.assertTrue(all(check["max_abs_difference"] == 0 for check in row["checks"]["cpu_vs_cuda"]))
        # Old control files remain readable/labeled, but no control-specific prompt reports
        # appear without a recorded diagnostic request.
        manifest_path = self.run / "run-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["config"]["cuda_controls"] = False
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        row, = prompts.build(self.run, prompt_set=make_prompt_set(Path(self.tmp.name) / "set", ["alpha"]))["prompts"]
        self.assertNotIn("cuda_build_cpu", row["timing"])
        self.assertEqual(row["checks"]["cpu_vs_cuda_build_cpu"], [])

    def test_saved_prompt_fallback_is_not_labeled_as_cuda_execution(self):
        directory = make_prompt_dir(self.run, "alpha", traced=False)
        timings(directory, "cuda-baseline", 2)
        logits(directory, "cuda-baseline")
        (self.run / "run-manifest.json").write_text(json.dumps({
            "config": {"backends": ["cpu", "cuda"]}, "build": {"requested": {"GGML_CUDA": "OFF"}},
            "prompt_backend_placements": [{"file": "prompts/alpha/cuda-baseline.jsonl",
                                           "actual": {"cpu_fallback": True}}]}), encoding="utf-8")
        row, = prompts.build(self.run, prompt_set=make_prompt_set(Path(self.tmp.name) / "set", ["alpha"]))["prompts"]
        self.assertEqual(row["timing"]["cuda"]["dataset"]["role"], "cpu_fallback")
        self.assertEqual(row["timing"]["cuda"]["dataset_label"], "CPU fallback (CUDA requested)")
        self.assertTrue(all(check["cpu_fallback"] for check in row["checks"]["cpu_vs_cuda"]))

    def test_one_row_per_prompt_in_prompt_set_order_skipping_missing(self):
        make_prompt_dir(self.run, "beta")
        make_prompt_dir(self.run, "alpha")
        listing = make_prompt_set(Path(self.tmp.name) / "set", ["alpha", "beta", "gamma"])   # gamma was never run
        rows = prompts.build(self.run, prompt_set=listing)["prompts"]
        self.assertEqual([r["prompt_id"] for r in rows], ["alpha", "beta"])

    def test_write_creates_json_only_when_there_are_rows(self):
        listing = make_prompt_set(Path(self.tmp.name) / "set", ["alpha"])
        prompts.write(self.run, prompt_set=listing)
        self.assertFalse((self.run / "prompt-profiles.json").exists())
        make_prompt_dir(self.run, "alpha")
        prompts.write(self.run, prompt_set=listing)
        self.assertEqual(len(json.loads((self.run / "prompt-profiles.json").read_text())["prompts"]), 1)


class Continuation(unittest.TestCase):
    def test_continuation_is_the_paragraph_profile_cpp_uses(self):
        text = harness.paragraph(harness.CURRENT.read_text(encoding="utf-8"))
        self.assertEqual(prompts.CONTINUATION.read_text(encoding="utf-8"), text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
