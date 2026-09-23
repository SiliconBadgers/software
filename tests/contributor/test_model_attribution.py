import importlib.util
import json
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("model_attribution", Path(__file__).resolve().parents[2] / "scripts/model-attribution.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class ModelAttributionTests(unittest.TestCase):
    def note(self, models, schema="authorship/3.0.0"):
        key = "sessions" if schema.startswith("authorship/3") else "prompts"
        return "file.py\n  id 1-3\n---\n" + json.dumps({"schema_version": schema, key: {
            str(i): {"agent_id": {"tool": t, "model": m}, "messages": "PRIVATE PROMPT CONTENT"}
            for i, (t, m) in enumerate(models)}})

    def test_multiple_exact_model_ids(self):
        pairs, status = report.recorded_models(self.note([("codex", "gpt-6-astra"), ("codex", "gpt-5.6-sol")]))
        self.assertEqual(pairs, [("codex", "gpt-5.6-sol"), ("codex", "gpt-6-astra")])
        self.assertEqual(status, "Captured metadata")

    def test_v2_metadata(self):
        self.assertEqual(report.recorded_models(self.note([("codex", "reported-id")], "authorship/2.0.0"))[0], [("codex", "reported-id")])

    def test_unknown_model_is_visible(self):
        pairs, status = report.recorded_models(self.note([("codex", "")]))
        self.assertEqual(pairs, [("codex", "unknown-model")])
        self.assertIn("missing", status)

    def test_missing_and_malformed_notes(self):
        for note in (None, "bad", "---\n{}", self.note([], "future/9")):
            self.assertFalse(report.recorded_models(note)[0])

    def test_declarations_are_real_trailers(self):
        message = "Update\n\nMention AI-Model: fake in prose.\n\nAI-Model: codex=gpt-6-astra\nAI-Model: codex=second-id\n"
        self.assertEqual(report.declared_models(message), ["codex=gpt-6-astra", "codex=second-id"])

    def test_report_omits_private_content_and_flags_conflict(self):
        pairs, status = report.recorded_models(self.note([("codex", "gpt-6-astra")]))
        text = report.render([{"commit": "a" * 40, "models": pairs, "status": status, "declared": ["none"], "merge": False}])
        self.assertIn("codex=gpt-6-astra", text)
        self.assertIn("CONFLICT", text)
        self.assertNotIn("PRIVATE PROMPT CONTENT", text)

    def test_untrusted_model_cannot_inject_markup(self):
        value = report.cell('<img src=x> | [click](https://example.org)\n`model`')
        self.assertNotIn('<img', value)
        self.assertNotIn('|', value)
        self.assertNotIn('[click]', value)
        self.assertNotIn('\n', value)

    def test_merge_missing_note_is_not_assumed_human(self):
        text = report.render([{"commit": "a" * 40, "models": [], "status": "No Git AI note", "declared": [], "merge": True}])
        self.assertIn("inspect contributing commits", text)
        self.assertNotIn("declares human-only", text)


if __name__ == "__main__":
    unittest.main()
