"""Shared representation has the old public identities, not a second graph copy."""
import gzip
from pathlib import Path
import tempfile
import unittest

from helpers import real_run
from cdfg import Graph, Node, Tensor
from sbengine import graph as compatibility
from characterization import graph_record


class SharedGraph(unittest.TestCase):
    def test_compatibility_exports_are_identical_objects(self):
        for name, value in (("Graph", Graph), ("Node", Node), ("Tensor", Tensor)):
            self.assertIs(getattr(compatibility, name), value)

    def test_plain_compressed_parity_for_every_phase(self):
        with tempfile.TemporaryDirectory() as temp:
            plain = Path(temp) / "graph.json"
            paths = sorted((real_run() / "graphs").glob("*/*.json.gz"))
            self.assertEqual(len(paths), 44)
            for path in paths:
                with self.subTest(capture=path.parent.name, phase=path.stem):
                    phase = path.name.split(".")[0]
                    plain.write_bytes(gzip.decompress(path.read_bytes()))
                    self.assertEqual(graph_record(Graph(path, phase)), graph_record(Graph(plain, phase)))


if __name__ == "__main__":
    unittest.main()
