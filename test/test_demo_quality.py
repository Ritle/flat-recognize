import io
import hashlib
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
from review_decisions import apply_review_decisions
from review_rooms import apply_room_decisions
from test.test_validate_recognition import fixture


class DemoQualityTest(unittest.IsolatedAsyncioTestCase):
    async def test_v2_shows_final_partition_candidates_and_reconciled_mask(self):
        source = io.BytesIO()
        Image.new("RGB", (40, 40), "white").save(source, format="PNG")
        rooms = {"meta": {"width": 40, "height": 40}, "rooms": [], "doors": [],
                 "windows": [], "diagnostics": {"possible_room_partitions": [
                     {"start": [10, 20], "end": [30, 20], "rooms_if_closed": 2}]}}
        quality = {"status": "invalid", "export_allowed": False, "warnings": [],
                   "errors": ["Needs review"], "metrics": {}, "issues": []}
        with tempfile.TemporaryDirectory() as temp:
            jobs, output = Path(temp) / "jobs", Path(temp) / "output"
            output.mkdir()

            def pipeline(image_path, project_path, pipeline_v2, review_decisions=None):
                stem = image_path.stem
                for suffix in ("_rooms.json", "_rooms_filtered.json",
                               "_rooms_topology_reconciled.json"):
                    (output / f"{stem}{suffix}").write_text(json.dumps(rooms), encoding="utf-8")
                (output / f"{stem}_topology_reconciled_classified.json").write_text(
                    json.dumps({"meta": rooms["meta"], "doors": [], "windows": []}),
                    encoding="utf-8")
                (output / f"{stem}_topology_reconciliation.json").write_text(
                    json.dumps({"removed_from_room_mask": [2]}), encoding="utf-8")
                for suffix in ("_preprocess.json", "_inference.json"):
                    (output / f"{stem}{suffix}").write_text("{}", encoding="utf-8")
                for suffix in ("_topology_reconciled_barrier.png",
                               "_topology_reconciled_closures.png"):
                    Image.new("RGB", (40, 40), "black").save(output / f"{stem}{suffix}")
                if review_decisions:
                    answers = json.loads(review_decisions.read_text(encoding="utf-8"))["answers"]
                    selected = {**rooms, "reviewed_partitions": [
                        {"partition_id": answer["partition_id"], "choice": answer["choice"]}
                        for answer in answers]}
                    reviewed_dir = output / f"{stem}_reviewed"
                    reviewed_dir.mkdir()
                    (reviewed_dir / f"{stem}_rooms_selected.json").write_text(
                        json.dumps(selected), encoding="utf-8")
                project_path.with_suffix(".quality.json").write_text(
                    json.dumps(quality), encoding="utf-8")
                return subprocess.CompletedProcess([], 2, stdout="quality check", stderr="")

            with patch.object(api, "JOBS_DIR", jobs), patch.object(api, "OUTPUT_DIR", output), \
                    patch.object(api, "run_pipeline", side_effect=pipeline):
                response = await api.recognize(
                    UploadFile(filename="plan.png", file=io.BytesIO(source.getvalue())), True)
                self.assertEqual(len(response["possible_room_partitions"]), 1)
                question = next(item for item in response["review_questions"]
                                if item["kind"] == "partition")
                job = jobs / response["job_id"]
                self.assertTrue((job / "topology_reconciliation.json").exists())
                with Image.open(job / "rooms.png") as overlay:
                    self.assertEqual(overlay.convert("RGB").getpixel((20, 20)), (255, 165, 0))
                with Image.open(job / "barrier.png") as barrier:
                    self.assertEqual(barrier.convert("RGB").getpixel((20, 20)), (0, 0, 0))
                answer = {"partition_id": question["partition_id"],
                          "fingerprint": question["fingerprint"], "choice": "open"}
                reviewed = await api.recognize(
                    UploadFile(filename="plan.png", file=io.BytesIO(source.getvalue())), True,
                    json.dumps({"version": 1, "image_sha256": response["image_sha256"],
                                "answers": [answer]}))
                self.assertEqual(reviewed["applied_decisions"], [answer])
                self.assertEqual(reviewed["quality"]["status"], "invalid")

    async def test_room_only_choice_uses_original_openings_and_keeps_invalid_gate(self):
        source = b"source image"
        source_hash = hashlib.sha256(source).hexdigest()
        large = {"id": 0, "bbox": [10, 10, 70, 70], "area_px": 3600,
                 "polygon": [[10, 10], [70, 10], [70, 70], [10, 70]]}
        small = {"id": 1, "bbox": [80, 80, 85, 85], "area_px": 25,
                 "polygon": [[80, 80], [85, 80], [85, 85], [80, 85]]}
        rooms = {"meta": {"width": 100, "height": 100}, "rooms": [large, small],
                 "walls": [], "doors": [], "windows": []}
        quality = {"status": "invalid", "export_allowed": False, "warnings": [],
                   "metrics": {}, "issues": [{"code": "unclosed_area", "severity": "error"}]}
        with tempfile.TemporaryDirectory() as temp:
            jobs, output = Path(temp) / "jobs", Path(temp) / "output"
            output.mkdir()

            def pipeline(image_path, project_path, pipeline_v2, review_decisions=None):
                stem = image_path.stem
                (output / f"{stem}_rooms.json").write_text(json.dumps(rooms), encoding="utf-8")
                (output / f"{stem}_classified.json").write_text(json.dumps({
                    "meta": rooms["meta"], "doors": [], "windows": []}), encoding="utf-8")
                (output / f"{stem}_geometry.json").write_text(json.dumps({
                    "unsupported_closure_groups": [{"closure_index": 2,
                        "center": [35, 40], "unsupported_edge_count": 3}]}), encoding="utf-8")
                if review_decisions:
                    selected = apply_room_decisions(
                        rooms, json.loads(review_decisions.read_text(encoding="utf-8")), source_hash)
                    (output / f"{stem}_rooms_selected.json").write_text(
                        json.dumps(selected), encoding="utf-8")
                project_path.with_suffix(".quality.json").write_text(
                    json.dumps(quality), encoding="utf-8")
                return subprocess.CompletedProcess([], 2, stdout="quality check", stderr="")

            with patch.object(api, "JOBS_DIR", jobs), patch.object(api, "OUTPUT_DIR", output), \
                    patch.object(api, "run_pipeline", side_effect=pipeline):
                first = await api.recognize(
                    UploadFile(filename="test.png", file=io.BytesIO(source)), False, None)
                question = next(item for item in first["review_questions"] if item.get("kind") == "room")
                self.assertEqual(first["geometry_review_candidates"][0]["image_size"], [100, 100])
                self.assertEqual(first["geometry_review_candidates"][0]["unsupported_edge_count"], 3)
                self.assertEqual(question["room_id"], 1)
                answer = {"room_id": 1, "fingerprint": question["fingerprint"],
                          "choice": "ignore"}
                second = await api.recognize(
                    UploadFile(filename="test.png", file=io.BytesIO(source)), False,
                    json.dumps({"version": 1, "image_sha256": source_hash, "answers": [answer]}))
                self.assertFalse(second["ok"])
                self.assertEqual(second["applied_decisions"], [answer])
                selected = json.loads((jobs / second["job_id"] / "rooms.json").read_text(encoding="utf-8"))
                self.assertEqual([room["id"] for room in selected["rooms"]], [0])
                self.assertEqual([room["id"] for room in selected["ignored_rooms"]], [1])
                self.assertTrue((jobs / second["job_id"] / "classified.json").exists())

    async def test_opening_preflight_is_available_for_invalid_job(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(api, "JOBS_DIR", Path(temp)):
            job_id = str(uuid.uuid4())
            job = Path(temp) / job_id
            job.mkdir()
            report = {"status": "audited", "summary": {"examined": 5, "suspect": 2},
                      "wall_geometry": {"sampled": 12, "suspect": 3},
                      "openings": [{"opening_id": 9, "reasons": ["exceeds_segment"]}]}
            (job / "opening_preflight.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(await api.job_opening_preflight(job_id), report)

    async def test_review_question_can_be_submitted_for_same_image(self):
        source = b"source image"
        classified = {"meta": {"width": 100, "height": 80}, "walls": [],
                      "doors": [{"id": 3, "type": "door",
                                 "outer": [[10, 10], [20, 10], [20, 50], [10, 50]],
                                 "classification": {"requires_review": True}}],
                      "windows": [], "rejected_openings": []}
        rooms = {"rooms": [], "doors": classified["doors"], "windows": [], "diagnostics": {}}
        quality = validate_recognition(rooms)
        with tempfile.TemporaryDirectory() as temp:
            jobs, output = Path(temp) / "jobs", Path(temp) / "output"
            output.mkdir()

            def pipeline(image_path, project_path, pipeline_v2, review_decisions=None):
                stem = image_path.stem
                result = (apply_review_decisions(
                    classified, json.loads(review_decisions.read_text(encoding="utf-8")),
                    hashlib.sha256(source).hexdigest()) if review_decisions else classified)
                suffix = "_reviewed_classified.json" if review_decisions else "_classified.json"
                (output / f"{stem}{suffix}").write_text(json.dumps(result), encoding="utf-8")
                (output / f"{stem}_rooms.json").write_text(json.dumps(rooms), encoding="utf-8")
                if review_decisions:
                    (output / f"{stem}_rooms_selected.json").write_text(
                        json.dumps(rooms), encoding="utf-8")
                (output / f"{stem}_opening_preflight.json").write_text(json.dumps({
                    "status": "audited", "summary": {"examined": 1, "suspect": 1},
                    "wall_geometry": {"sampled": 6, "suspect": 2},
                    "openings": [{"opening_id": 3, "reasons": ["far_from_wall"]}]}),
                    encoding="utf-8")
                project_path.with_suffix(".quality.json").write_text(json.dumps(quality), encoding="utf-8")
                return subprocess.CompletedProcess([], 2, stdout="quality check", stderr="")

            with patch.object(api, "JOBS_DIR", jobs), patch.object(api, "OUTPUT_DIR", output), \
                    patch.object(api, "run_pipeline", side_effect=pipeline):
                first = await api.recognize(UploadFile(filename="test.png", file=io.BytesIO(source)), False, None)
                question = first["review_questions"][0]
                self.assertEqual(question["opening_id"], 3)
                self.assertEqual(first["opening_preflight"]["summary"]["suspect"], 1)
                self.assertEqual(first["opening_preflight"]["wall_geometry"]["suspect"], 2)
                self.assertEqual(first["opening_preflight_url"],
                                 f"/api/jobs/{first['job_id']}/opening-preflight")
                payload = {"version": 1, "image_sha256": first["image_sha256"],
                           "answers": [{"opening_id": 3, "fingerprint": question["fingerprint"],
                                        "choice": "window"}]}
                with self.assertRaises(HTTPException) as stale:
                    await api.recognize(UploadFile(filename="test.png", file=io.BytesIO(source)),
                                        False, json.dumps({**payload, "image_sha256": "wrong"}))
                self.assertEqual(stale.exception.status_code, 400)
                self.assertEqual(len(list(jobs.iterdir())), 1)
                second = await api.recognize(UploadFile(filename="test.png", file=io.BytesIO(source)),
                                             False, json.dumps(payload))
                self.assertEqual(second["applied_decisions"], payload["answers"])
                self.assertEqual(second["quality"]["status"], "invalid")
                reviewed = json.loads((jobs / second["job_id"] / "classified.json").read_text(encoding="utf-8"))
                self.assertEqual([item["id"] for item in reviewed["windows"]], [3])
                self.assertEqual(reviewed["doors"], [])

    async def test_new_diagnostic_images_are_served(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(api, "JOBS_DIR", Path(temp)):
            job_id = str(uuid.uuid4())
            job = Path(temp) / job_id
            job.mkdir()
            for kind in ("topology_windows", "interior_windows", "wall_geometry"):
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

            def pipeline(image_path, project_path, pipeline_v2, review_decisions=None):
                stem = image_path.stem
                (output / f"{stem}_rooms.json").write_text(json.dumps(rooms), encoding="utf-8")
                (output / f"{stem}_rooms_filtered.json").write_text(json.dumps(rooms), encoding="utf-8")
                (output / f"{stem}_interior_filtered_classified.json").write_text(
                    json.dumps({"meta": {"width": 10, "height": 10}, "doors": [], "windows": []}),
                    encoding="utf-8")
                (output / f"{stem}_interior_windows.json").write_text(
                    json.dumps({"rejected_windows": 0, "ambiguous_windows": 1}), encoding="utf-8")
                for suffix in ("_preprocess.json", "_inference.json"):
                    (output / f"{stem}{suffix}").write_text("{}", encoding="utf-8")
                Image.new("RGB", (10, 10), "white").save(
                    output / f"{stem}_interior_windows_overlay.png")
                Image.new("RGB", (10, 10), "white").save(
                    output / f"{stem}_wall_geometry_overlay.png")
                project_path.with_suffix(".quality.json").write_text(
                    json.dumps(quality), encoding="utf-8")
                return subprocess.CompletedProcess([], 2, stdout="quality check", stderr="")

            with patch.object(api, "JOBS_DIR", jobs), patch.object(api, "OUTPUT_DIR", output), \
                    patch.object(api, "run_pipeline", side_effect=pipeline):
                response = await api.recognize(UploadFile(filename="test.png", file=io.BytesIO(b"source")), True)
                self.assertEqual(response["quality"]["status"], "invalid")
                self.assertIn("interior_windows", response["images"])
                self.assertIn("wall_geometry", response["images"])
                image = await api.job_image(response["job_id"], "interior_windows")
                self.assertTrue(Path(image.path).exists())
                image = await api.job_image(response["job_id"], "wall_geometry")
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

                def pipeline(image_path, project_path, pipeline_v2, review_decisions=None):
                    stem = image_path.stem
                    (output / f"{stem}_rooms.json").write_text(json.dumps(rooms), encoding="utf-8")
                    (output / f"{stem}_classified.json").write_text(
                        json.dumps({"meta": {"width": 20, "height": 20}, "doors": [], "windows": []}),
                        encoding="utf-8")
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
