"""Run-folder naming and protection rules. Needs only Python."""
from datetime import datetime
from pathlib import Path
import sys
import tempfile
import unittest

CORE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CORE / "runner"))
import runpaths  # noqa: E402


class RunPaths(unittest.TestCase):
    def test_name_has_date_time_and_os(self):
        self.assertEqual(runpaths.run_name("macos", datetime(2026, 9, 25, 14, 30)), "2026-09-25_1430_macos")
        self.assertEqual(runpaths.run_name("windows", datetime(2026, 1, 2, 3, 4)), "2026-01-02_0304_windows")

    def test_same_minute_never_reuses_a_folder(self):
        with tempfile.TemporaryDirectory() as root:
            now = datetime(2026, 9, 25, 14, 30)
            names = [runpaths.create_run_dir(root, "linux", now).name for _ in range(3)]
            self.assertEqual(names, ["2026-09-25_1430_linux", "2026-09-25_1430_linux-2", "2026-09-25_1430_linux-3"])

    def test_dry_run_creates_nothing(self):
        with tempfile.TemporaryDirectory() as root:
            path = runpaths.create_run_dir(root, "linux", datetime(2026, 9, 25, 14, 30), create=False)
            self.assertFalse(path.exists())

    def test_experiments_directory_is_protected(self):
        with self.assertRaises(ValueError):
            runpaths.create_run_dir(runpaths.PROTECTED / "llama-cpp", "linux")
        with self.assertRaises(ValueError):
            runpaths.assert_writable(runpaths.PROTECTED)

    def test_folder_names_are_valid_on_windows(self):
        name = runpaths.run_name("windows", datetime(2026, 9, 25, 14, 30))
        self.assertFalse(set(name) & set('<>:"/\|?*'))


if __name__ == "__main__":
    unittest.main(verbosity=2)
