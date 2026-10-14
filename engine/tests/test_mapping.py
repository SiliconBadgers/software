"""Mapping is explicit, resource-dependent and consumed once by costing."""
from dataclasses import asdict
import unittest
from unittest import mock

from helpers import Builder
from sbengine import config, costs, mapping


class MappingContract(unittest.TestCase):
    def graph(self):
        b = Builder()
        w = b.add("blk.0.w", [64, 64], weight=True)
        x = b.add("x", [64, 64], op="ROPE")
        b.add("y", [64, 64], op="MUL_MAT", src=(w, x))
        return b.graph()

    def test_costing_consumes_mapping_without_selecting_again(self):
        g = self.graph()
        for cfg in (config.make_config(), config.make_config({"matrix": {"count": 0}}),
                    config.make_config({"matrix": {"count": 0}, "scalar": {"count": 0}})):
            ctx = costs.Ctx(cfg)
            for n in g.nodes:
                selected = mapping.map_node(ctx, g, n)
                expected = costs.node_cost(ctx, g, n)
                with mock.patch.object(costs, "map_node", side_effect=AssertionError("selection repeated")):
                    actual = costs.node_cost(ctx, g, n, selected)
                self.assertEqual(asdict(actual), asdict(expected))
                self.assertEqual((selected.node, selected.tensor), (n.idx, n.tensor.id))

    def test_tile_feasibility_depends_on_local_memory(self):
        g = self.graph()
        n = g.nodes[-1]
        usable = mapping.map_node(costs.Ctx(config.make_config()), g, n)
        tiny = mapping.map_node(costs.Ctx(config.make_config({"l1": {"mib": 0.000001}})), g, n)
        self.assertTrue(usable.tile["fits"])
        self.assertFalse(tiny.tile["fits"])
        self.assertIn("tile budget", tiny.error)

    def test_mapping_for_another_operation_is_rejected(self):
        g = self.graph()
        ctx = costs.Ctx(config.make_config())
        with self.assertRaisesRegex(ValueError, "captured operation"):
            costs.node_cost(ctx, g, g.nodes[-1], mapping.map_node(ctx, g, g.nodes[0]))


if __name__ == "__main__":
    unittest.main()
