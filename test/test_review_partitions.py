import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from extract_rooms_v2 import extract_rooms
from polygon_geometry import build_polygon_project
from review_partitions import apply_partition_decisions, partition_questions
from validate_recognition import validate_recognition


def rectangle(x1, y1, x2, y2):
    return {"outer": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "holes": []}


def source(walls):
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    for wall in walls:
        draw.polygon([tuple(point) for point in wall["outer"]], fill="black")
    return {"meta": {"width": 400, "height": 300}, "walls": walls,
            "doors": [], "windows": []}, np.asarray(image)


class PartitionReviewTest(unittest.TestCase):
    def setUp(self):
        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 380, 280),
                 rectangle(195, 20, 205, 130), rectangle(195, 170, 205, 280)]
        self.classified, self.rgb = source(walls)
        self.rooms, _, _ = extract_rooms(self.classified, self.rgb)
        self.sha = "a" * 64
        self.question = partition_questions(self.rooms, self.sha)[0]

    def answer(self, choice, fingerprint=None):
        return {"version": 1, "image_sha256": self.sha, "answers": [{
            "partition_id": 0, "choice": choice,
            "fingerprint": fingerprint or self.question["fingerprint"]}]}

    def test_separate_changes_only_reviewed_room_mask_and_stays_invalid(self):
        original = copy.deepcopy(self.rooms)
        result, barrier, inferred, reviewed = apply_partition_decisions(
            self.classified, self.rooms, self.rgb, self.answer("separate"), self.sha)
        self.assertEqual(len(original["rooms"]), 1)
        self.assertEqual(self.rooms, original)
        self.assertEqual(len(result["rooms"]), 2)
        self.assertIsNotNone(barrier)
        self.assertIsNotNone(inferred)
        self.assertEqual(len(result["diagnostics"]["reviewed_partition_closures"]), 1)
        self.assertEqual(reviewed, self.classified)
        self.assertIn("reviewed_partition_geometry_unconfirmed",
                      {issue["code"] for issue in validate_recognition(result)["issues"]})

    def test_open_keeps_room_topology(self):
        result, barrier, inferred, reviewed = apply_partition_decisions(
            self.classified, self.rooms, self.rgb, self.answer("open"), self.sha)
        self.assertEqual(len(result["rooms"]), 1)
        self.assertIsNone(barrier)
        self.assertIsNone(inferred)
        self.assertEqual(reviewed, self.classified)
        self.assertEqual(result["reviewed_partitions"],
                         [{"partition_id": 0, "choice": "open"}])

    def test_nearly_closed_leaf_hint_is_exposed_without_changing_choices(self):
        rooms = copy.deepcopy(self.rooms)
        rooms["diagnostics"]["possible_room_partitions"][0]["unconfirmed_leaf_hint"] = {
            "hinge": [200, 170], "tip": [188, 132], "swing_angle_degrees": 18}
        question = partition_questions(rooms, self.sha)[0]
        self.assertIn("near_closed_leaf_unconfirmed", question["reasons"])
        self.assertEqual(question["choices"], ["door", "separate", "open"])
        self.assertEqual(question["unconfirmed_leaf_hint"]["hinge"], [200, 170])

    def test_confirmed_leaf_creates_reviewed_door_and_splits_room(self):
        image = Image.fromarray(self.rgb)
        ImageDraw.Draw(image).line((200, 170, 188, 132), fill="black", width=2)
        rgb = np.asarray(image)
        rooms, _, _ = extract_rooms(self.classified, rgb)
        question = partition_questions(rooms, self.sha)[0]
        self.assertIn("door", question["choices"])
        answer = {"version": 1, "image_sha256": self.sha, "answers": [{
            "partition_id": 0, "choice": "door", "fingerprint": question["fingerprint"]}]}
        result, barrier, inferred, reviewed = apply_partition_decisions(
            self.classified, rooms, rgb, answer, self.sha)
        self.assertEqual(len(result["rooms"]), 2)
        self.assertEqual(len(result["doors"]), 1)
        self.assertEqual(result["doors"], reviewed["doors"])
        self.assertTrue(result["doors"][0]["classification"]["user_confirmed"])
        self.assertIsNotNone(barrier)
        self.assertIsNotNone(inferred)
        self.assertEqual(result["diagnostics"]["reviewed_partition_closures"], [])
        self.assertEqual(len(result["diagnostics"]["reviewed_door_closures"]), 1)
        issues = {item["code"] for item in validate_recognition(result)["issues"]}
        self.assertIn("reviewed_door_binding_pending", issues)
        self.assertNotIn("reviewed_partition_geometry_unconfirmed", issues)
        self.assertEqual(self.classified["doors"], [])
        root = Path(__file__).resolve().parent.parent
        template = json.loads((root / "test/template.json").read_text(encoding="utf-8"))
        project, geometry = build_polygon_project(result, template, thickness=10)
        self.assertEqual(geometry["unsupported_segments"], [])
        quality = validate_recognition(result, project, geometry=geometry)
        self.assertEqual(quality["status"], "review")
        self.assertTrue(quality["export_allowed"])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name, value in (("source.json", reviewed), ("rooms.json", result),
                                ("project.json", project), ("geometry.json", geometry)):
                (directory / name).write_text(json.dumps(value), encoding="utf-8")
            command = [sys.executable, str(root / "scripts/add_openings.py"),
                       str(directory / "project.json"), str(directory / "source.json"),
                       str(directory / "rooms.json"), str(root / "test/template.json"),
                       "--output", str(directory / "doors.json"),
                       "--windows-output", str(directory / "windows.json"),
                       "--bindings-output", str(directory / "bindings.json"),
                       "--geometry-report", str(directory / "geometry.json")]
            completed = subprocess.run(command, capture_output=True, text=True,
                                       encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            exported = json.loads((directory / "doors.json").read_text(encoding="utf-8"))
        elements = [item for item in exported["levels"][0]["elements"]
                    if item["type"] == "doorItem"]
        self.assertEqual(len(elements), 1)
        self.assertIn(elements[0]["wall"]["uuid"],
                      {segment["uuid"] for segment in exported["levels"][0]["segments"]})

    def test_door_choice_requires_leaf_hint(self):
        with self.assertRaisesRegex(ValueError, "visible leaf"):
            apply_partition_decisions(self.classified, self.rooms, self.rgb,
                                      self.answer("door"), self.sha)
        stale = copy.deepcopy(self.rooms)
        stale["diagnostics"]["possible_room_partitions"][0]["unconfirmed_leaf_hint"] = {
            "hinge": [200, 170], "tip": [188, 132]}
        with self.assertRaisesRegex(ValueError, "no longer supported"):
            apply_partition_decisions(self.classified, stale, self.rgb,
                                      self.answer("door"), self.sha)

    def test_stale_geometry_and_duplicate_answers_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_partition_decisions(self.classified, self.rooms, self.rgb,
                                      self.answer("separate", "stale"), self.sha)
        doubled = self.answer("separate")
        doubled["answers"].append(doubled["answers"][0])
        with self.assertRaises(ValueError):
            apply_partition_decisions(self.classified, self.rooms, self.rgb,
                                      doubled, self.sha)

    def test_old_exterior_candidate_cannot_be_confirmed_as_partition(self):
        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 150, 280),
                 rectangle(190, 270, 380, 280), rectangle(100, 20, 110, 110),
                 rectangle(20, 100, 110, 110)]
        classified, rgb = source(walls)
        rooms, _, _ = extract_rooms(classified, rgb)
        candidate = {"start": [144.0, 275.0], "end": [195.0, 275.0],
                     "width": 14.0, "length_px": 51.0}
        rooms["diagnostics"]["possible_room_partitions"] = [candidate]
        question = partition_questions(rooms, self.sha)[0]
        payload = {"version": 1, "image_sha256": self.sha, "answers": [{
            "partition_id": 0, "fingerprint": question["fingerprint"],
            "choice": "separate"}]}
        with self.assertRaisesRegex(ValueError, "inside one room"):
            apply_partition_decisions(classified, rooms, rgb, payload, self.sha)


if __name__ == "__main__":
    unittest.main()
