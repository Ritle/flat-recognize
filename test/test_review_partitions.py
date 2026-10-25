import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from extract_rooms_v2 import extract_rooms
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
        result, barrier, inferred = apply_partition_decisions(
            self.classified, self.rooms, self.rgb, self.answer("separate"), self.sha)
        self.assertEqual(len(original["rooms"]), 1)
        self.assertEqual(self.rooms, original)
        self.assertEqual(len(result["rooms"]), 2)
        self.assertIsNotNone(barrier)
        self.assertIsNotNone(inferred)
        self.assertEqual(len(result["diagnostics"]["reviewed_partition_closures"]), 1)
        self.assertIn("reviewed_partition_geometry_unconfirmed",
                      {issue["code"] for issue in validate_recognition(result)["issues"]})

    def test_open_keeps_room_topology(self):
        result, barrier, inferred = apply_partition_decisions(
            self.classified, self.rooms, self.rgb, self.answer("open"), self.sha)
        self.assertEqual(len(result["rooms"]), 1)
        self.assertIsNone(barrier)
        self.assertIsNone(inferred)
        self.assertEqual(result["reviewed_partitions"],
                         [{"partition_id": 0, "choice": "open"}])

    def test_stale_geometry_and_duplicate_answers_are_rejected(self):
        with self.assertRaises(ValueError):
            apply_partition_decisions(self.classified, self.rooms, self.rgb,
                                      self.answer("separate", "stale"), self.sha)
        doubled = self.answer("separate")
        doubled["answers"].append(doubled["answers"][0])
        with self.assertRaises(ValueError):
            apply_partition_decisions(self.classified, self.rooms, self.rgb,
                                      doubled, self.sha)


if __name__ == "__main__":
    unittest.main()
