"""Shared representation has the old public identities, not a second graph copy."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from helpers import real_run
from cdfg import Graph, Node, Tensor
from sbengine import graph as compatibility
from characterization import FIXTURE, digest, graph_record


class SharedGraph(unittest.TestCase):
    def test_compatibility_exports_are_identical_objects(self):
        for name, value in (("Graph", Graph), ("Node", Node), ("Tensor", Tensor)):
            self.assertIs(getattr(compatibility, name), value)

    def test_plain_compressed_parity_for_every_phase(self):
        recorded = json.loads(FIXTURE.read_text())["graphs"]
        with tempfile.TemporaryDirectory() as temp:
            plain = Path(temp) / "graph.json"
            paths = sorted((real_run() / "graphs").glob("*/*.json.gz"))
            self.assertEqual(len(paths), 44)
            for path in paths:
                with self.subTest(capture=path.parent.name, phase=path.stem):
                    phase = path.name.split(".")[0]
                    plain.write_bytes(gzip.decompress(path.read_bytes()))
                    compressed_graph, plain_graph = Graph(path, phase), Graph(plain, phase)
                    self.assertEqual(graph_record(compressed_graph), graph_record(plain_graph))
                    for graph in (compressed_graph, plain_graph):
                        self.assertEqual(digest(graph_record(graph)), recorded[f"{path.parent.name}/{phase}"][1])
                        self.assertEqual(graph.capture, json.loads(gzip.decompress(path.read_bytes())))


if __name__ == "__main__":
    unittest.main()
