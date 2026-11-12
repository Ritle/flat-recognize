import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from shapely.geometry import box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from detect_exterior_enclosures_v2 import detect_exterior_enclosures
from extract_rooms_v2 import extract_rooms
from review_exterior_enclosures import (apply_exterior_decisions, exterior_fingerprint,
                                        exterior_questions)
from test.test_detect_exterior_enclosures_v2 import fixture


class ExteriorReviewTests(unittest.TestCase):
    def setUp(self):
        self.classified, _, self.rgb = fixture()
        ring = box(10, 150, 390, 390).difference(box(20, 160, 380, 380))
        self.classified["walls"] = [{"outer": list(ring.exterior.coords[:-1]),
                                     "holes": [list(inner.coords[:-1]) for inner in ring.interiors]}]
        self.rooms, _, _ = extract_rooms(self.classified, self.rgb)
        report = detect_exterior_enclosures(self.classified, self.rooms, self.rgb)
        self.assertEqual(len(report["candidates"]), 1)
        self.hash = "a" * 64
        self.payload = {"version": 1, "image_sha256": self.hash, "answers": [{
            "exterior_id": 0, "choice": "confirm",
            "fingerprint": exterior_fingerprint(report["candidates"][0])}]}

    def test_confirmation_adds_one_trial_room_but_does_not_mutate_project_recognition(self):
        original = copy.deepcopy((self.classified, self.rooms))
        report, barrier = apply_exterior_decisions(
            self.classified, self.rooms, self.rgb, self.payload, self.hash)
        trial = report["candidates"][0]["trial"]
        self.assertEqual(trial["status"], "accepted")
        self.assertEqual(trial["rooms_after"], trial["rooms_before"] + 1)
        self.assertTrue(trial["preserved_rooms"])
        self.assertIsNotNone(barrier)
        self.assertEqual((self.classified, self.rooms), original)
        question = exterior_questions(report, self.hash, [400, 400])[0]
        self.assertEqual(question["current"], "confirm")
        self.assertEqual(question["trial"], trial)

    def test_ignore_is_reversible_without_running_a_trial(self):
        self.payload["answers"][0]["choice"] = "ignore"
        report, barrier = apply_exterior_decisions(
            self.classified, self.rooms, self.rgb, self.payload, self.hash)
        self.assertIsNone(barrier)
        question = exterior_questions(report, self.hash, [400, 400])[0]
        self.assertEqual(question["current"], "ignore")
        self.assertIsNone(question["trial"])
        self.assertEqual(question["fingerprint"], self.payload["answers"][0]["fingerprint"])

    def test_stale_image_geometry_missing_rails_and_duplicate_answers_are_rejected(self):
        for defect in ("image", "geometry", "rails", "duplicate", "boolean_id"):
            payload, rgb = copy.deepcopy(self.payload), self.rgb.copy()
            if defect == "image":
                payload["image_sha256"] = "b" * 64
            elif defect == "geometry":
                payload["answers"][0]["fingerprint"] = "changed"
            elif defect == "rails":
                rgb[90:115, 95:110] = 255
            elif defect == "duplicate":
                payload["answers"].append(copy.deepcopy(payload["answers"][0]))
            else:
                payload["answers"][0]["exterior_id"] = False
            with self.subTest(defect=defect), self.assertRaises(ValueError):
                apply_exterior_decisions(self.classified, self.rooms, rgb, payload, self.hash)

    def test_confirmation_does_not_override_an_unsafe_baseline(self):
        with patch("review_exterior_enclosures.preserved_rooms", return_value=False):
            report, _ = apply_exterior_decisions(
                self.classified, self.rooms, self.rgb, self.payload, self.hash)
        self.assertEqual(report["candidates"][0]["reviewed_choice"], "confirm")
        self.assertEqual(report["candidates"][0]["trial"]["status"], "rejected")
        self.assertNotIn("new_room", report["candidates"][0]["trial"])


if __name__ == "__main__":
    unittest.main()
