"""Stdlib-only contracts for the uv project and Unix launcher."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[1]


class UvProjectTests(unittest.TestCase):
    def test_mac_has_no_default_dependencies(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(project["project"]["requires-python"], ">=3.11")
        self.assertEqual(project["project"]["dependencies"], [])
        self.assertEqual(project["tool"]["uv"]["default-groups"], [])
        self.assertFalse(project["tool"]["uv"]["package"])

    def test_dependency_groups_and_python_floor(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        groups = project["dependency-groups"]
        self.assertIn({"include-group": "engine"}, groups["analysis"])
        self.assertIn({"include-group": "analysis"}, groups["profiling"])
        lock = tomllib.loads((ROOT / "uv.lock").read_text())
        versions = {package["name"]: package["version"] for package in lock["package"]}
        for group, dependencies in groups.items():
            self.assertEqual(project["tool"]["uv"]["dependency-groups"][group],
                             {"requires-python": ">=3.12"})
            for dependency in dependencies:
                if isinstance(dependency, str):
                    name, pinned_version = dependency.split("==")
                    self.assertEqual(versions[name], pinned_version)


@unittest.skipUnless(os.name != "nt" and shutil.which("bash"), "Unix launcher requires Bash on Unix")
class UnixLauncherTests(unittest.TestCase):
    def test_arguments_working_directory_and_exit_status(self):
        with tempfile.TemporaryDirectory(prefix="sb uv launcher ") as temp:
            cwd = Path(temp)
            record = cwd / "called.json"
            uv = cwd / "fake uv"
            uv.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['UV_TEST_RECORD']).write_text(json.dumps({"
                "'args': sys.argv[1:], 'cwd': os.getcwd()}))\n"
                "sys.exit(23)\n"
            )
            uv.chmod(0o755)
            env = {**os.environ, "UV": str(uv), "UV_TEST_RECORD": str(record),
                   "PYTHON": "a python path with spaces"}
            result = subprocess.run(
                ["bash", str(ROOT / "unix/run.sh"), "analyze", "relative run", "--flag=value"],
                cwd=cwd, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 23, result.stderr)
            called = json.loads(record.read_text())
            self.assertEqual(called["cwd"], str(cwd))
            self.assertEqual(called["args"], [
                "run", "--locked", "--project", str(ROOT), "--group", "profiling",
                "--python", "a python path with spaces", "python",
                str(ROOT / "core/runner/cli.py"), "--platform-dir", str(ROOT / "unix"),
                "analyze", "relative run", "--flag=value",
            ])

    def test_missing_uv_is_actionable(self):
        env = {**os.environ, "UV": "/missing/sb-test-uv"}
        result = subprocess.run(["bash", str(ROOT / "unix/run.sh"), "doctor"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn("uv is required", result.stderr)
        self.assertIn("https://docs.astral.sh/uv/getting-started/installation/", result.stderr)


@unittest.skipUnless(os.name == "nt" and (shutil.which("pwsh") or shutil.which("powershell")),
                     "Windows launcher requires PowerShell on Windows")
class WindowsLauncherTests(unittest.TestCase):
    def test_arguments_working_directory_and_exit_status(self):
        with tempfile.TemporaryDirectory(prefix="sb uv launcher ") as temp:
            cwd = Path(temp)
            record = cwd / "called.json"
            script = cwd / "fake uv.py"
            script.write_text(
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "Path(os.environ['UV_TEST_RECORD']).write_text(json.dumps({"
                "'args': sys.argv[1:], 'cwd': os.getcwd()}))\n"
                "sys.exit(23)\n"
            )
            uv = cwd / "fake uv.cmd"
            uv.write_text(f'@"{sys.executable}" "{script}" %*\n')
            env = {**os.environ, "UV": str(uv), "UV_TEST_RECORD": str(record),
                   "PYTHON": "a python path with spaces"}
            powershell = shutil.which("pwsh") or shutil.which("powershell")
            result = subprocess.run(
                [powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                 str(ROOT / "windows/run.ps1"), "analyze", "relative run", "--flag=value"],
                cwd=cwd, env=env, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 23, result.stderr)
            called = json.loads(record.read_text())
            self.assertEqual(Path(called["cwd"]), cwd)
            self.assertEqual(called["args"], [
                "run", "--locked", "--project", str(ROOT), "--group", "profiling",
                "--python", "a python path with spaces", "python",
                str(ROOT / "core/runner/cli.py"), "--platform-dir", str(ROOT / "windows"),
                "analyze", "relative run", "--flag=value",
            ])


if __name__ == "__main__":
    unittest.main()
