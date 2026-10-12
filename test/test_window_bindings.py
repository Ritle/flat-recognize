import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class WindowBindingsTests(unittest.TestCase):
    def run_adapter(self, bindings):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            folder = Path(folder)
            project = {"levels": [{"elements": [{"type": "doorItem", "uuid": "door-1"}],
                                    "segments": []}]}
            (folder / "project.json").write_text(json.dumps(project), encoding="utf-8")
            (folder / "bindings.json").write_text(json.dumps(bindings), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/add_windows.py"), str(folder / "project.json"),
                 str(folder / "bindings.json"), str(ROOT / "test/template_window.json"),
                 "--output", str(folder / "result.json")], capture_output=True, text=True)
            output = (json.loads((folder / "result.json").read_text(encoding="utf-8"))
                      if (folder / "result.json").exists() else None)
            return result, project, output

    def test_empty_windows_array_preserves_project(self):
        result, project, output = self.run_adapter({"windows": []})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(output, project)
        self.assertIn("WARNING", result.stdout)

    def test_missing_windows_array_is_still_an_error(self):
        result, _, output = self.run_adapter({})
        self.assertNotEqual(result.returncode, 0)
        self.assertIsNone(output)

    def test_window_uuid_is_recorded_in_diagnostics(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "output") as folder:
            folder = Path(folder)
            project = {"levels": [{"elements": [], "segments": [{"uuid": "wall-1"}]}]}
            binding = {"window": 1, "opening_id": 17, "wall_uuid": "wall-1",
                       "position_percent": 50, "width_cm": 80}
            (folder / "project.json").write_text(json.dumps(project), encoding="utf-8")
            (folder / "windows.json").write_text(json.dumps({"windows": [binding]}), encoding="utf-8")
            diagnostics = folder / "diagnostics.json"
            diagnostics.write_text(json.dumps({"doors": [], "windows": [binding]}), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/add_windows.py"), str(folder / "project.json"),
                 str(folder / "windows.json"), str(ROOT / "test/template_window.json"),
                 "--output", str(folder / "result.json"), "--bindings-output", str(diagnostics)],
                capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            window = json.loads((folder / "result.json").read_text(encoding="utf-8"))["levels"][0]["elements"][0]
            recorded = json.loads(diagnostics.read_text(encoding="utf-8"))["windows"][0]
            self.assertEqual(recorded["element_uuid"], window["uuid"])
            self.assertEqual(recorded["opening_id"], 17)


if __name__ == "__main__":
    unittest.main()
