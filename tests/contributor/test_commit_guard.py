"""Exercise the real hook in disposable repos; no network or real user config."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class CommitGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.repo = self.base / "repo with spaces"
        self.repo.mkdir()
        self.home = self.base / "home"
        self.home.mkdir()
        self.bin = self.base / "bin"
        self.bin.mkdir()
        # Avoid touching any installed Git AI wrapper or user configuration.
        git = next((p for p in (Path('/usr/bin/git'), Path('/bin/git')) if p.exists()), Path(shutil.which('git')))
        (self.bin / "git").symlink_to(git)
        (self.bin / "python3").symlink_to(sys.executable)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "CODEX_"))}
        self.env.update(HOME=str(self.home), CODEX_HOME=str(self.home / ".codex"),
                        PATH=str(self.bin) + os.pathsep + "/usr/bin:/bin",
                        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        for file in (".githooks/pre-commit", "scripts/install-repo-hooks.py", "scripts/check-git-ai.py"):
            dst = self.repo / file
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / file, dst)
        self.call("git", "init", "-q")
        self.call("git", "config", "user.name", "Guard Test")
        self.call("git", "config", "user.email", "guard@example.invalid")
        self.write_config()
        self.fake_ai()
        (self.repo / "change.txt").write_text("A human test fixture.\n")
        self.call("git", "add", ".")

    def call(self, *args, ok=True, cwd=None):
        p = subprocess.run(args, cwd=cwd or self.repo, env=self.env, text=True, capture_output=True)
        if ok:
            self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        return p

    def write_config(self, disabled=False, privacy="local", matcher=""):
        codex = self.home / ".codex/config.toml"
        codex.parent.mkdir(exist_ok=True)
        text = "[features]\nhooks = true\n"
        for event in ("PreToolUse", "PostToolUse", "Stop"):
            text += f'[[hooks.{event}]]\n'
            if matcher:
                text += f'matcher = "{matcher}"\n'
            text += f'[[hooks.{event}.hooks]]\ntype = "command"\ncommand = "git-ai checkpoint codex --hook-input stdin"\n'
        if disabled:
            key = str(codex) + ":post_tool_use:0:0"
            text += f'[hooks.state.{json.dumps(key)}]\nenabled = false\n'
        codex.write_text(text)
        ai = self.home / ".git-ai/config.json"
        ai.parent.mkdir(exist_ok=True)
        ai.write_text(json.dumps({"prompt_storage": privacy, "telemetry_oss": "off"}))

    def fake_ai(self, ready=True, retained_error=False, bad_status=False):
        status = {"ok": ready, "daemon_running": ready, "daemon": {"sequencer_stalled": False}}
        if retained_error:
            status["data"] = {"last_error": "old rejected notes push"}
        script = self.bin / "git-ai"
        script.write_text("#!/usr/bin/env python3\nimport sys\n"
                          "if sys.argv[1:] == ['--version']: print('1.7.5')\n"
                          f"elif sys.argv[1:3] == ['bg', 'status']: print({json.dumps(json.dumps(status))})\n"
                          f"elif sys.argv[1] == 'status': print({json.dumps('invalid' if bad_status else json.dumps({'stats': {}}))})\n"
                          "else: sys.exit(1)\n")
        script.chmod(0o755)

    def install(self):
        return self.call("python3", "scripts/install-repo-hooks.py")

    def commit(self):
        return self.call("git", "commit", "-qm", "Test human contribution", ok=False)

    def test_installer_is_idempotent_and_good_setup_allows_commit(self):
        self.install()
        self.install()
        self.assertEqual(self.commit().returncode, 0)
        message = self.call("git", "log", "-1", "--format=%B").stdout
        self.assertNotIn("Co-authored-by", message)

    def test_missing_binary_blocks_commit(self):
        self.install()
        (self.bin / "git-ai").unlink()
        result = self.commit()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not on PATH", result.stdout + result.stderr)

    def test_disabled_codex_hook_blocks_commit(self):
        self.install()
        self.write_config(disabled=True)
        self.assertNotEqual(self.commit().returncode, 0)

    def test_restricted_matcher_blocks_commit(self):
        self.install()
        self.write_config(matcher="SomeTool")
        self.assertNotEqual(self.commit().returncode, 0)

    def test_missing_codex_config_blocks_commit(self):
        self.install()
        (self.home / ".codex/config.toml").unlink()
        self.assertNotEqual(self.commit().returncode, 0)

    def test_prompt_sharing_blocks_commit(self):
        self.install()
        self.write_config(privacy="notes")
        self.assertNotEqual(self.commit().returncode, 0)

    def test_unavailable_daemon_blocks_commit(self):
        self.install()
        self.fake_ai(ready=False)
        self.assertNotEqual(self.commit().returncode, 0)

    def test_unreadable_capture_state_blocks_commit(self):
        self.install()
        self.fake_ai(bad_status=True)
        self.assertNotEqual(self.commit().returncode, 0)

    def test_retained_sync_error_is_reported_not_hidden(self):
        self.install()
        self.fake_ai(retained_error=True)
        result = self.commit()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("repository sync error", result.stdout + result.stderr)

    def test_existing_hook_path_is_preserved(self):
        self.call("git", "config", "core.hooksPath", "custom-hooks")
        result = self.call("python3", "scripts/install-repo-hooks.py", ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.call("git", "config", "core.hooksPath").stdout.strip(), "custom-hooks")

    def test_existing_executable_hook_is_preserved(self):
        hook = self.repo / ".git/hooks/pre-commit"
        hook.write_text("#!/bin/sh\nexit 0\n")
        hook.chmod(0o755)
        self.assertNotEqual(self.call("python3", "scripts/install-repo-hooks.py", ok=False).returncode, 0)
        self.assertEqual(hook.read_text(), "#!/bin/sh\nexit 0\n")

    def test_guard_not_activated_is_reported(self):
        self.assertNotEqual(self.call("python3", "scripts/check-git-ai.py", ok=False).returncode, 0)

    def test_install_from_subdirectory(self):
        self.call("python3", "install-repo-hooks.py", cwd=self.repo / "scripts")
        self.assertEqual(self.commit().returncode, 0)


if __name__ == "__main__":
    unittest.main()
