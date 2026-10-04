"""Storage formats agree with the capture; configs validate; profiles load; hashes are stable."""
import unittest

from helpers import ENGINE, real_run  # noqa: F401  (sets sys.path)
from sbengine import config, formats
from sbengine.graph import Graph, capture_path


class Formats(unittest.TestCase):
    def test_block_sizes(self):
        self.assertEqual(formats.storage_bytes("q6_K", 2048 * 248320), 417177600)   # token_embd.weight in the capture
        self.assertEqual(formats.storage_bytes("f16", 10), 20)
        self.assertEqual(formats.storage_bytes("q4_K", 256), 144)
        self.assertEqual(formats.storage_bytes("q8_0", 33), 68)                      # partial block rounds up

    def test_rate_classes(self):
        self.assertEqual({d: formats.rate_class(d) for d in ("q4_K", "q5_K", "q6_K", "q8_0", "f16", "bf16", "f32")},
                         {"q4_K": "w4", "q5_K": "w5", "q6_K": "w6", "q8_0": "w8", "f16": "w16", "bf16": "w16", "f32": "w32"})

    def test_every_captured_weight_matches_its_recorded_bytes(self):
        run = real_run()
        if run is None:
            self.skipTest("no captured run")
        g = Graph(capture_path(run / "graphs" / "pp128-fa-on", "decode"), "decode")
        weights = [g.tensors[r] for r in g.roots_of_kind("weight")]
        self.assertGreater(len(weights), 100)
        for t in weights:
            self.assertEqual(formats.storage_bytes(t.dtype, t.elements), t.nbytes, t.name)


class Config(unittest.TestCase):
    def test_defaults_are_valid(self):
        self.assertEqual(config.validate(config.DEFAULTS), [])

    def test_profiles_load(self):
        for name in ("accel-balanced", "accel-efficient", "legacy-equations"):
            cfg, doc = config.load_profile(name)
            self.assertEqual(doc["name"], name)
            self.assertEqual(config.validate(cfg), [])

    def test_bad_values_rejected(self):
        for over in ({"matrix": {"efficiency": 1.5}}, {"clock_mhz": 0}, {"batch": 1.5}, {"l1": {"state_reserve": 0.6, "tile_fraction": 0.5}},
                     {"memory": {"fusion": "sometimes"}}, {"precision": {"act_bits": 12}}, {"matrix": {"caps": "gemm"}},
                     {"hbm": {"gbs": float("nan")}}):
            with self.assertRaises(ValueError, msg=str(over)):
                config.make_config(over)

    def test_dotted_paths_and_sets(self):
        cfg = config.make_config()
        self.assertEqual(config.get_path(cfg, "matrix.rates.w4"), 2.0)
        cfg2 = config.apply_sets(cfg, ["matrix.count=2", "memory.fusion=\"none\"", "vector.caps=[\"elementwise\"]"])
        self.assertEqual((cfg2["matrix"]["count"], cfg2["memory"]["fusion"], cfg2["vector"]["caps"]), (2, "none", ["elementwise"]))
        self.assertEqual(cfg["matrix"]["count"], 4)                          # input untouched
        with self.assertRaises(KeyError):
            config.apply_sets(cfg, ["matrix.nope=1"])

    def test_hash_is_stable_and_sensitive(self):
        a, b = config.make_config(), config.make_config()
        self.assertEqual(config.config_hash(a), config.config_hash(b))
        b["hbm"]["gbs"] += 1
        self.assertNotEqual(config.config_hash(a), config.config_hash(b))


if __name__ == "__main__":
    unittest.main()
