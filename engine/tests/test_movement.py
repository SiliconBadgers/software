"""Data-movement plans pass the independent checker, and broken plans do not."""
import dataclasses
from pathlib import Path
import sys
import unittest

ENGINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ENGINE))

from sbengine.codegen import lower_matmul
from sbengine.codegen.movement import (MovementError, MovementPlan, StubTiledMatmul,
                                       TargetMemory, memory_done)
from sbengine.codegen.movement_check import check
from sbengine.graph import Graph

FIXTURE = ENGINE / "tests/fixtures/compiler-matmul.json"


def mutate(movement, transfers=None, matrix=None):
    """Apply a deliberate defect to a valid plan's commands."""
    if transfers:
        movement.transfers = transfers(list(movement.transfers))
    if matrix:
        original = movement.matrix_commands
        movement.matrix_commands = lambda: (matrix(c) for c in original())
    return movement


class MovementPlans(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plan = lower_matmul(Graph(str(FIXTURE), "prefill"), 30)  # 5 x 7 times 7 x 3

    def movement(self, tiles=(2, 2, 2), **target):
        return MovementPlan(StubTiledMatmul(self.plan, *tiles), TargetMemory(**target))

    def kinds(self, movement):
        result = check(movement)
        self.assertFalse(result["passed"])
        return {v["kind"] for v in result["violations"]}

    def test_valid_plans_pass_for_edge_and_whole_tiles(self):
        for tiles in [(2, 2, 2), (2, 2, 4), (1, 1, 1), (5, 3, 7), (4, 3, 3)]:
            for reuse in ("complete", "inputs_released"):
                result = check(self.movement(tiles, reuse_after=reuse))
                self.assertTrue(result["passed"], (tiles, reuse, result["violations"]))
                self.assertEqual(result["observed"]["matrix"],
                                 len(list(StubTiledMatmul(self.plan, *tiles).tiles())))

    def test_layout_rows_and_waits(self):
        movement = self.movement()
        self.assertEqual([x.operand for x in movement.transfers], ["b", "a", "a", "a", "c"])
        for x in movement.transfers:
            self.assertEqual(x.sram_address % 512, 0)
            self.assertEqual(x.length_bytes % 512, 0)
        # The third A panel reuses slot 0 once panel 0's last reader (matrix 7) has
        # finished its SRAM reads (3j+1).
        self.assertEqual(movement.transfers[3].wait_matrix, 3 * 7 + 1)
        conservative = self.movement(reuse_after="complete")
        self.assertEqual(conservative.transfers[3].wait_matrix, 3 * 7 + 3)
        self.assertEqual(movement.transfers[4].wait_matrix, 3 * (movement.matrix_count - 1) + 3)

    def test_same_input_gives_same_commands(self):
        self.assertEqual(self.movement().digest(), self.movement().digest())
        self.assertNotEqual(self.movement().digest(), self.movement((2, 2, 4)).digest())

    def test_stale_input_is_caught(self):
        # Panel 2's tiles wait only for panel 1's load, so slot 0 still holds panel 0.
        movement = mutate(self.movement(), matrix=lambda c: c._replace(
            wait_memory=memory_done(2)) if c.wait_memory == memory_done(3) else c)
        self.assertIn("data", self.kinds(movement))

    def test_read_before_load_is_caught(self):
        movement = mutate(self.movement(), matrix=lambda c: c._replace(wait_memory=0)
                          if c.index == 0 else c)
        self.assertIn("raw", self.kinds(movement))

    def test_early_reuse_is_caught(self):
        movement = mutate(self.movement(), transfers=lambda t: [
            x._replace(wait_matrix=0) if x.operand == "a" and x.wait_matrix else x for x in t])
        self.assertEqual(self.kinds(movement), {"war"})

    def test_partial_row_transfer_is_caught(self):
        movement = mutate(self.movement(), transfers=lambda t: [
            x._replace(length_bytes=100) if x.index == 1 else x for x in t])
        self.assertIn("alignment", self.kinds(movement))

    def test_early_store_is_caught(self):
        movement = mutate(self.movement(), transfers=lambda t: [
            x._replace(wait_matrix=3) if x.direction == "sram_to_hbm" else x for x in t])
        self.assertEqual(self.kinds(movement), {"store"})

    def test_wait_cycle_is_caught(self):
        movement = mutate(self.movement(), transfers=lambda t: [
            x._replace(wait_matrix=10**6) if x.index == 3 else x for x in t])
        self.assertIn("deadlock", self.kinds(movement))

    def test_sram_overflow_is_caught(self):
        movement = mutate(self.movement(), transfers=lambda t: [
            x._replace(sram_address=8 * 1024 * 1024) if x.index == 1 else x for x in t])
        self.assertIn("bounds", self.kinds(movement))

    def test_reuse_policy_matters(self):
        movement = self.movement(reuse_after="inputs_released")
        self.assertTrue(check(movement)["passed"])
        movement.target = dataclasses.replace(movement.target, reuse_after="complete")
        self.assertEqual(self.kinds(movement), {"war"})

    def test_unsupported_targets_are_rejected(self):
        with self.assertRaises(MovementError):
            self.movement(sram_bytes=1024)  # B, C and two A panels need 2 KiB
        with self.assertRaises(MovementError):
            self.movement(layout="tiled")
        with self.assertRaises(MovementError):
            self.movement(reuse_after="matrix_done")

    def test_tiles_must_visit_panels_in_order(self):
        tiles = list(StubTiledMatmul(self.plan, 2, 2, 2).tiles())

        class Reversed(StubTiledMatmul):
            def tiles(self):
                return iter(tiles[::-1])
        with self.assertRaises(MovementError):
            MovementPlan(Reversed(self.plan, 2, 2, 2))


if __name__ == "__main__":
    unittest.main()
