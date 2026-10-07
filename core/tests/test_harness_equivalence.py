"""Preserve the recorded measurement semantics while allowing backend metadata/CLI extensions.

Compare the logit writer, tokenization/output, context configuration and timed loop with the
immutable harness, including every brace. A historical formatting bug wrote logits once per
vocabulary element; mutations below confirm this gate still detects that class of regression.
"""
import ast
from pathlib import Path
import re
import unittest

REPO = Path(__file__).resolve().parents[2]
ORIGINAL = REPO / "experiments" / "llama-cpp" / "2026-09-22" / "profile.cpp"
CURRENT = REPO / "core" / "harness" / "profile.cpp"
STRING = r'"(?:[^"\\]|\\.)*"'


def significant(source):
    source = re.sub(r"//[^\n]*", "", source)       # comments
    source = re.sub(r'"\s*"', "", source)          # adjacent string literals are one literal
    return re.sub(r"\s+", "", source)


def measurement_regions(source):
    text = significant(source)
    def region(start, end=None):
        first = text.index(start)
        return text[first:text.index(end, first) + len(end)] if end else text[first:]
    return {
        "logit_writer": region("staticvoidlogits(", "intmain("),
        "tokenization_and_outputs": region("intnt=-llama_tokenize(", "std::cout<<std::setprecision(10);"),
        "context_configuration": region("for(intn:lengths){", "if(!c)return2;"),
        "warmup_and_measurement": region('phase("warmup");'),
    }


def paragraph(source):
    match = re.search(r"std::string paragraph\s*=\s*((?:\s*" + STRING + r")+)\s*;", source)
    pieces = re.findall(r'"((?:[^"\\]|\\.)*)"', match.group(1))
    return ast.literal_eval('"' + "".join(pieces) + '"')


@unittest.skipUnless(ORIGINAL.is_file(), "recorded 2026-09-22 harness not present")
class HarnessEquivalence(unittest.TestCase):
    def setUp(self):
        self.original = ORIGINAL.read_text(encoding="utf-8")
        self.current = CURRENT.read_text(encoding="utf-8")

    def test_prompt_paragraph_is_unchanged(self):
        self.assertEqual(paragraph(self.current), paragraph(self.original))

    def test_measurement_regions_match_recorded_program(self):
        self.assertEqual(measurement_regions(self.current), measurement_regions(self.original))

    def test_gate_catches_per_element_logit_write_regression(self):
        mutated = self.current.replace('if(!path.empty()){', 'for(int i=0;i<nv;i++)if(!path.empty()){')
        self.assertNotEqual(measurement_regions(mutated), measurement_regions(self.original))

    def test_gate_catches_context_and_timing_changes(self):
        for old, new in (("cp.n_ubatch=512;", "cp.n_ubatch=128;"),
                         ("llama_synchronize(c);", "/* synchronization removed */")):
            self.assertNotEqual(measurement_regions(self.current.replace(old, new)),
                                measurement_regions(self.original))

    def test_lf_line_endings(self):
        self.assertNotIn(b"\r", CURRENT.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
