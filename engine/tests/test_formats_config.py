"""Storage formats agree with the capture; configs validate; profiles load; hashes are stable."""
import copy
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
                     {"hbm": {"gbs": float("nan")}}, {"recurrent": {"lowering": "fast"}}, {"recurrent": {"matrix_penalty": 0}}):
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

    def test_switch_register_covers_real_paths_and_valid_values(self):
        """Every registered switch exists, and its explorer and pessimistic settings are accepted values."""
        base = config.make_config()
        for s in config.SWITCHES:
            config.get_path(base, s["path"])                                    # KeyError if the path is wrong
            for key in ("explorer", "conservative", "alternative"):
                if s.get(key) is not None:
                    cfg = copy.deepcopy(base)
                    config.set_path(cfg, s["path"], s[key])
                    self.assertEqual(config.validate(cfg), [], s["path"])
        self.assertEqual(len({s["path"] for s in config.SWITCHES}), len(config.SWITCHES))

    def test_register_covers_kv_padding_and_reports_the_unbracketed_overlap(self):
        by_path = {s["path"]: s for s in config.SWITCHES}
        self.assertEqual(by_path["attention.kv_padding"]["conservative"], "captured")
        base = config.make_config()
        exact = config.make_config({"attention": {"kv_padding": "exact"}})
        self.assertNotIn("attention.kv_padding", [r["path"] for r in config.switch_report(base) if r["optimistic"]])
        self.assertIn("attention.kv_padding", [r["path"] for r in config.switch_report(exact) if r["optimistic"]])
        self.assertEqual(config.conservative(exact)["attention"]["kv_padding"], "captured")
        # within-op overlap is the explorer's node-duration rule: not bracketed, but its alternative is on record
        overlap = by_path["schedule.within_op_overlap"]
        self.assertIsNone(overlap["conservative"])
        self.assertIs(overlap["alternative"], False)
        self.assertIn("not bracketed", overlap["assumes"])
        row = {r["path"]: r for r in config.switch_report(base)}["schedule.within_op_overlap"]
        self.assertEqual((row["optimistic"], row["informational"]), (False, True))
        self.assertTrue(config.conservative(base)["schedule"]["within_op_overlap"])
        self.assertFalse(config.conservative(base, upper_bound=True)["schedule"]["within_op_overlap"])

    def test_conservative_config_has_no_optimistic_switch(self):
        base = config.make_config()
        optimistic = {r["path"] for r in config.switch_report(base) if r["optimistic"]}
        self.assertIn("recurrent.lowering", optimistic)                         # the default is "auto"
        self.assertIn("schedule.mode", optimistic)
        cautious = config.conservative(base)
        self.assertEqual([r["path"] for r in config.switch_report(cautious) if r["optimistic"]], [])
        self.assertEqual(config.validate(cautious), [])
        self.assertEqual(base["recurrent"]["lowering"], "auto")                 # input untouched
        restored = copy.deepcopy(cautious)                                     # only the switches moved
        for s in config.SWITCHES:
            config.set_path(restored, s["path"], config.get_path(base, s["path"]))
        self.assertEqual(restored, base)
        # the explorer-equations profile is pessimistic on every switch the explorer had
        legacy = config.load_profile("legacy-equations")[0]
        for r in config.switch_report(legacy):
            if r["explorer"] is not None:
                self.assertEqual(r["value"], r["explorer"], r["path"])

    def test_hash_is_stable_and_sensitive(self):
        a, b = config.make_config(), config.make_config()
        self.assertEqual(config.config_hash(a), config.config_hash(b))
        b["hbm"]["gbs"] += 1
        self.assertNotEqual(config.config_hash(a), config.config_hash(b))


if __name__ == "__main__":
    unittest.main()
