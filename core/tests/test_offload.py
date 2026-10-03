"""offload.py's structural accounting on a hand-built graph: boundary bytes count only edges between
scheduled ops, leaf inputs (weights, caches, graph inputs) are counted separately and once per group."""
from pathlib import Path
import sys
import unittest

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "analysis"))
import offload  # noqa: E402


def tensor(tid, name, op, shape, nbytes, sources=()):
    return {"id": tid, "name": name, "op": op, "shape": list(shape), "elements": shape[0] * shape[1],
            "logical_bytes": nbytes, "sources": [{"slot": i, "tensor": s} for i, s in enumerate(sources)]}


# x (input) -> norm -> ffn_up matmul (weight W) -> output-head matmul (tied embedding E)
TENSORS = {t["id"]: t for t in [
    tensor(1, "inp", "NONE", (8, 1), 32),
    tensor(2, "blk.0.ffn_up.weight", "NONE", (8, 16), 1000),
    tensor(3, "token_embd.weight", "NONE", (16, 100), 5000),
    tensor(10, "norm", "RMS_NORM", (8, 1), 32, (1,)),
    tensor(11, "up", "MUL_MAT", (16, 1), 64, (2, 10)),
    tensor(12, "logits", "MUL_MAT", (100, 1), 400, (3, 11)),
]}
ORDER = [10, 11, 12]


class StructuralBoundaries(unittest.TestCase):
    def setUp(self):
        self.b = offload.structural_boundaries(TENSORS, ORDER)

    def test_boundary_bytes_are_scheduled_edges_only(self):
        mlp = self.b["MLP projections"]
        self.assertEqual(mlp["boundary_in_bytes"], 32)    # norm -> up
        self.assertEqual(mlp["boundary_out_bytes"], 64)   # up -> logits
        head = self.b["Vocabulary output head"]
        self.assertEqual(head["boundary_in_bytes"], 64)
        self.assertEqual(head["boundary_out_bytes"], 0)

    def test_weights_and_inputs_are_leaf_bytes_not_boundary(self):
        self.assertEqual(self.b["MLP projections"]["leaf_input_bytes"], 1000)
        self.assertEqual(self.b["Vocabulary output head"]["leaf_input_bytes"], 5000)
        self.assertEqual(self.b["Vector/norm/activation/other"]["leaf_input_bytes"], 32)

    def test_leaf_counted_once_per_group(self):
        tensors = dict(TENSORS)
        tensors[13] = tensor(13, "up2", "MUL_MAT", (16, 1), 64, (2, 10))  # reuses the same weight
        b = offload.structural_boundaries(tensors, [10, 11, 13, 12])
        self.assertEqual(b["MLP projections"]["leaf_input_bytes"], 1000)

    def test_macs(self):
        self.assertEqual(self.b["MLP projections"]["matrix_macs"], 16 * 8)
        self.assertEqual(self.b["Vocabulary output head"]["matrix_macs"], 100 * 16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
