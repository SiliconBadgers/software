"""core/harness/profile.cpp may be reformatted and commented, but must stay the same program as the
harness that produced the recorded 2026-09-22 baseline.

Whitespace, comments and how string literals are split are ignored; every other character,
including each brace, must match. (A formatter once moved a single brace in logits() so that the
logits file was rewritten 248,320 times per checkpoint; this test exists to catch that class of edit.)
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

    def test_program_text_is_unchanged_apart_from_layout(self):
        self.assertEqual(significant(self.current), significant(self.original))

    def test_lf_line_endings(self):
        self.assertNotIn(b"\r", CURRENT.read_bytes())


if __name__ == "__main__":
    unittest.main(verbosity=2)
