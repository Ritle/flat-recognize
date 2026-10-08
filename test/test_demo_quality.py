import io
import json
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, UploadFile
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import demo.app as api
from validate_recognition import validate_recognition
from test.test_validate_recognition import fixture


class DemoQualityTest(unittest.IsolatedAsyncioTestCase):
    async def test_new_diagnostic_images_are_served(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(api, "JOBS_DIR", Path(temp)):
            job_id = str(uuid.uuid4())
            job = Path(temp) / job_id
            job.mkdir()
            for kind in ("topology_windows", "interior_windows"):
                Image.new("RGB", (10, 10), "white").save(job / f"{kind}.png")
                response = await api.job_image(job_id, kind)
                self.assertEqual(Path(response.path), job / f"{kind}.png")

    async def test_ambiguous_window_diagnostic_is_exposed(self):
        with tempfile.TemporaryDirectory() as temp:
            jobs, output = Path(temp) / "jobs", Path(temp) / "output"
            output.mkdir()
            rooms, _, _ = fixture()
            rooms["rooms"] = []
            rooms.setdefault("diagnostics", {})["interior_window_filter"] = {
                "topology_stable": True, "ambiguous_windows": 1}
            quality = validate_recognition(rooms)

            def pipeline(image_path, project_path, pipeline_v2):
                stem = image_path.stem
                (output / f"{stem}_rooms.json").write_text(json.dumps(rooms), encoding="utf-8")
                (output / f"{stem}_rooms_filtered.json").write_text(json.dumps(rooms), encoding="utf-8")
                (output / f"{stem}_interior_windows.json").write_text(
                    json.dumps({"rejected_windows": 0, "ambiguous_windows": 1}), encoding="utf-8")
                for suffix in ("_preprocess.json", "_inference.json"):
                    (output / f"{stem}{suffix}").write_text("{}", encoding="utf-8")
                Image.new("RGB", (10, 10), "white").save(
                    output / f"{stem}_interior_windows_overlay.png")
                project_path.with_suffix(".quality.json").write_text(
                    json.dumps(quality), encoding="utf-8")
                return subprocess.CompletedProcess([], 2, stdout="quality check", stderr="")

            with patch.object(api, "JOBS_DIR", jobs), patch.object(api, "OUTPUT_DIR", output), \
                    patch.object(api, "run_pipeline", side_effect=pipeline):
                response = await api.recognize(UploadFile(filename="test.png", file=io.BytesIO(b"source")), True)
                self.assertEqual(response["quality"]["status"], "invalid")
                self.assertIn("interior_windows", response["images"])
                image = await api.job_image(response["job_id"], "interior_windows")
                self.assertTrue(Path(image.path).exists())

    async def test_result_and_export_routes_for_each_status(self):
        for status in ("good", "review", "invalid", "no_rooms", "geometry_invalid"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as temp:
                jobs = Path(temp) / "jobs"
                output = Path(temp) / "output"
                output.mkdir()
                rooms, project, bindings = fixture()
                if status == "review":
                    rooms["rooms"][0]["uncertain"] = True
                elif status == "invalid":
                    project["levels"][0]["elements"][0]["wall"]["uuid"] = "missing"
                elif status == "no_rooms":
                    rooms["rooms"] = []
                elif status == "geometry_invalid":
                    project["levels"][0]["elements"] = []
                    project["levels"][0]["segments"][0]["start"] = "missing"
                quality = validate_recognition(rooms, project if status != "no_rooms" else None, bindings)

                def pipeline(image_path, project_path, pipeline_v2):
                    stem = image_path.stem
                    (output / f"{stem}_rooms.json").write_text(json.dumps(rooms), encoding="utf-8")
                    project_path.with_suffix(".quality.json").write_text(json.dumps(quality), encoding="utf-8")
                    if status not in ("no_rooms", "geometry_invalid"):
                        project_path.write_text(json.dumps(project), encoding="utf-8")
                    Image.new("RGB", (20, 20), "white").save(output / f"{stem}_rooms_overlay.png")
                    return subprocess.CompletedProcess([], 0 if quality["export_allowed"] else 2,
                                                       stdout="quality check", stderr="")

                with patch.object(api, "JOBS_DIR", jobs), patch.object(api, "OUTPUT_DIR", output), \
                        patch.object(api, "run_pipeline", side_effect=pipeline):
                    response = await api.recognize(UploadFile(filename="test.png", file=io.BytesIO(b"source")), False)
                    self.assertEqual(response["quality"]["status"], "invalid" if status in ("no_rooms", "geometry_invalid") else status)
                    if status == "geometry_invalid":
                        self.assertEqual(response["stats"]["doors"], len(rooms["doors"]))
                    self.assertEqual(await api.job_quality(response["job_id"]), quality)
                    image = await api.job_image(response["job_id"], "rooms")
                    self.assertTrue(Path(image.path).exists())
                    if quality["export_allowed"]:
                        self.assertTrue(response["ok"])
                        self.assertEqual(await api.project_json(response["job_id"]), project)
                        download = await api.download_project(response["job_id"])
                        self.assertEqual(json.loads(Path(download.path).read_text(encoding="utf-8")), project)
                    else:
                        self.assertFalse(response["ok"])
                        self.assertIsNone(response["download_url"])
                        self.assertIsNone(response["json_url"])
                        for route in (api.project_json, api.download_project):
                            with self.assertRaises(HTTPException) as caught:
                                await route(response["job_id"])
                            self.assertEqual(caught.exception.status_code, 409)

    async def test_legacy_job_without_report_cannot_bypass_gate(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(api, "JOBS_DIR", Path(temp)):
            job_id = str(uuid.uuid4())
            (Path(temp) / job_id).mkdir()
            (Path(temp) / job_id / "project.json").write_text("{}")
            with self.assertRaises(HTTPException) as caught:
                await api.project_json(job_id)
            self.assertEqual(caught.exception.status_code, 409)


if __name__ == "__main__":
    unittest.main()
