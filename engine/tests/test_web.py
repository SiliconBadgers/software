"""The local web API: routes, validation, limits. Uses a real server on an ephemeral port."""
import json
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

from helpers import real_run
from sbengine import web

RUN = real_run()


def call(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


class Sanitize(unittest.TestCase):
    def test_non_finite_numbers_become_null(self):
        self.assertEqual(web._clean({"a": float("inf"), "b": [float("nan"), 1.5], "c": "x"}), {"a": None, "b": [None, 1.5], "c": "x"})


@unittest.skipIf(RUN is None, "no captured run under results/")
class Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), web.make_handler(web.App(RUN)))
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        status, raw = call(self.base, path, body)
        return status, json.loads(raw)

    def test_page_and_meta(self):
        status, raw = call(self.base, "/")
        self.assertEqual(status, 200)
        self.assertIn(b"Accelerator model playground", raw)
        status, raw = call(self.base, "/api/meta")
        meta = json.loads(raw)
        self.assertEqual(status, 200)
        self.assertGreater(len(meta["controls"]), 50)
        self.assertIn("pp512-fa-on", [w["name"] for w in meta["workloads"]])
        self.assertEqual({p["name"] for p in meta["profiles"]}, {"accel-balanced", "accel-efficient", "legacy-equations"})
        for c in meta["controls"]:                              # every control maps onto a real config value
            cur = meta["defaults"]
            for part in c["path"].split("."):
                cur = cur[part]

    def test_eval_matches_the_engine(self):
        from sbengine.config import load_profile
        from sbengine.graph import load_workload
        from sbengine.model import evaluate
        status, r = self.post("/api/eval", {"graph": "pp128-fa-on", "config": {}, "top_nodes": 5})
        direct = evaluate(load_workload(RUN / "graphs" / "pp128-fa-on"), load_profile("accel-balanced")[0])
        self.assertEqual(status, 200)
        self.assertTrue(r["valid"])
        self.assertAlmostEqual(r["phases"]["decode"]["seconds"], direct["phases"]["decode"]["seconds"], places=12)
        self.assertEqual(len(r["phases"]["prefill"]["top_nodes"]), 5)
        self.assertNotIn("rows", r["phases"]["prefill"])
        self.assertIn("decode_binding", r["summary"])

    def test_config_changes_take_effect_and_bad_ones_are_rejected(self):
        _, base = self.post("/api/eval", {"graph": "pp128-fa-on", "config": {}})
        _, direct = self.post("/api/eval", {"graph": "pp128-fa-on", "config": {"memory": {"weight_path": "direct"}}})
        self.assertLess(direct["phases"]["decode"]["seconds"], base["phases"]["decode"]["seconds"])
        status, err = self.post("/api/eval", {"graph": "pp128-fa-on", "config": {"matrix": {"efficiency": 2}}})
        self.assertEqual(status, 400)
        self.assertIn("efficiency", err["error"])
        self.assertEqual(self.post("/api/eval", {"graph": "../../etc/passwd", "config": {}})[0], 400)
        self.assertEqual(self.post("/api/eval", {"config": {}})[0], 400)

    def test_infeasible_design_is_a_result_not_an_error(self):
        status, r = self.post("/api/eval", {"graph": "pp128-fa-on", "config": {"matrix": {"count": 64}}})
        self.assertEqual(status, 200)
        self.assertFalse(r["valid"])
        self.assertIn("DSP", " ".join(r["errors"]))

    def test_sweep_limits_and_paths(self):
        status, r = self.post("/api/sweep", {"graph": "pp128-fa-on", "config": {}, "axes": {"matrix.count": [1, 2], "recurrent.count": [0, 2]}})
        self.assertEqual(status, 200)
        self.assertEqual(len(r["records"]), 4)
        self.assertTrue(r["frontier"])
        self.assertEqual(self.post("/api/sweep", {"graph": "pp128-fa-on", "axes": {"hbm.nope": [1]}})[0], 400)
        self.assertEqual(self.post("/api/sweep", {"graph": "pp128-fa-on", "axes": {"matrix.caps": [[]]}})[0], 400)   # not a sweepable control
        many = {"matrix.count": list(range(1, 10)), "vector.count": list(range(1, 10))}
        self.assertEqual(self.post("/api/sweep", {"graph": "pp128-fa-on", "axes": many})[0], 400)
        self.assertEqual(self.post("/api/sweep", {"graph": "pp128-fa-on", "axes": {}})[0], 400)

    def test_ablation_and_sensitivity(self):
        status, r = self.post("/api/ablation", {"graph": "pp128-fa-off", "config": {}})
        self.assertEqual(status, 200)
        self.assertGreater(len(r["cumulative"]), 5)
        status, r = self.post("/api/sensitivity", {"graph": "pp128-fa-on", "config": {}, "delta": 0.2})
        self.assertEqual(status, 200)
        self.assertTrue(r["rows"])
        self.assertEqual(self.post("/api/sensitivity", {"graph": "pp128-fa-on", "delta": 5})[0], 400)

    def test_unknown_routes_and_bad_bodies(self):
        self.assertEqual(call(self.base, "/nope")[0], 404)
        self.assertEqual(call(self.base, "/api/nope", {})[0], 404)
        req = urllib.request.Request(self.base + "/api/eval", data=b"[1,2]", headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 400)
        req = urllib.request.Request(self.base + "/api/eval", data=b"{not json", headers={"Content-Type": "application/json"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 400)


if __name__ == "__main__":
    unittest.main()
