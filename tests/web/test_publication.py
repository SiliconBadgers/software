import importlib.util
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "web" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load("build")
publisher = load("publish")


class PublicationTests(unittest.TestCase):
    def test_registry_requires_attribution_and_contained_assets(self):
        registry = json.loads((ROOT / "web/registry.json").read_text())
        builder.validate_registry(registry)
        registry["components"][0]["assets"][0]["target"] = "../outside.js"
        with self.assertRaises(ValueError):
            builder.validate_registry(registry)

    def test_unsafe_paths_and_model_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            builder.safe_path(ROOT, "../outside")
        registry = json.loads((ROOT / "web/registry.json").read_text())
        registry["components"][0]["assets"][0]["source"] = "../private.gguf"
        with self.assertRaises(ValueError):
            builder.validate_registry(registry)

    def test_main_and_previews_do_not_replace_each_other(self):
        with tempfile.TemporaryDirectory() as temp:
            site = Path(temp) / "site"
            build = Path(temp) / "build"
            site.mkdir(); build.mkdir()
            (build / "index.html").write_text("main one")
            publisher.overlay(site, build, "main")
            (build / "index.html").write_text("preview one")
            preview = publisher.overlay(site, build, "feature/new-model")
            self.assertEqual((site / "index.html").read_text(), "main one")
            (build / "index.html").write_text("main two")
            publisher.overlay(site, build, "main")
            self.assertEqual((preview / "index.html").read_text(), "preview one")
            (build / "index.html").write_text("preview two")
            publisher.overlay(site, build, "feature/new-model")
            self.assertEqual((site / "index.html").read_text(), "main two")
            self.assertEqual((preview / "index.html").read_text(), "preview two")

    def test_distinct_branch_names_have_distinct_preview_paths(self):
        a, b = publisher.preview_name("feature/a-b"), publisher.preview_name("feature/a/b")
        self.assertNotEqual(a, b)
        self.assertNotIn("/", a)
        self.assertLessEqual(len(a), 64)

    def test_checkout_authentication_is_forwarded_once(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.headers.get_all("Authorization"))
                self.send_response(401)
                self.end_headers()

            def log_message(self, *_):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                url = f"http://127.0.0.1:{server.server_port}/"
                header = "Authorization: Basic test-value"
                subprocess.run(["git", "init", temp], check=True, capture_output=True)
                subprocess.run(["git", "config", f"http.{url}.extraheader", header],
                               cwd=temp, check=True)
                env = {**os.environ, **publisher.authentication_environment(header, url),
                       "GIT_TERMINAL_PROMPT": "0"}
                # The server deliberately denies access; the request headers are the assertion.
                subprocess.run(["git", "ls-remote", url + "repository"], cwd=temp,
                               env=env, capture_output=True, timeout=10)
                self.assertEqual(requests, [["Basic test-value"]])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
