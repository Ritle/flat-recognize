import copy
import sys
import unittest
from pathlib import Path

from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from polygon_geometry import build_polygon_project
from review_rooms import apply_room_decisions, room_questions
from validate_recognition import validate_recognition


def room(uid, bounds):
    x0, y0, x1, y1 = bounds
    return {"id": uid, "bbox": bounds, "area_px": (x1 - x0) * (y1 - y0),
            "polygon": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


class RoomReviewTests(unittest.TestCase):
    def setUp(self):
        self.data = {"meta": {"width": 200, "height": 200},
                     "rooms": [room(0, [10, 10, 110, 110]),
                               room(1, [140, 10, 150, 20])],
                     "walls": [], "doors": [], "windows": []}
        self.hash = "a" * 64

    def test_only_tiny_room_is_questioned_and_choice_is_reversible(self):
        questions = room_questions(self.data, self.hash)
        self.assertEqual([item["room_id"] for item in questions], [1])
        answer = {"room_id": 1, "fingerprint": questions[0]["fingerprint"],
                  "choice": "ignore"}
        original = copy.deepcopy(self.data)
        reviewed = apply_room_decisions(
            self.data, {"version": 1, "image_sha256": self.hash,
                        "answers": [{"opening_id": 3, "choice": "window"}, answer]}, self.hash)
        self.assertEqual(self.data, original)
        self.assertEqual([item["id"] for item in reviewed["rooms"]], [0])
        self.assertEqual([item["id"] for item in reviewed["ignored_rooms"]], [1])
        self.assertEqual(room_questions(reviewed, self.hash)[0]["current"], "ignore")
        self.assertTrue(any(issue["code"] == "user_room_review_applied"
                            for issue in validate_recognition(reviewed)["issues"]))

        answer["choice"] = "room"
        restored = apply_room_decisions(
            self.data, {"version": 1, "image_sha256": self.hash,
                        "answers": [answer]}, self.hash)
        self.assertEqual(len(restored["rooms"]), 2)
        self.assertEqual(restored["ignored_rooms"], [])

    def test_stale_or_all_room_removal_is_rejected(self):
        question = room_questions(self.data, self.hash)[0]
        answer = {"room_id": 1, "fingerprint": question["fingerprint"],
                  "choice": "ignore"}
        payload = {"version": 1, "image_sha256": self.hash, "answers": [answer]}
        with self.assertRaisesRegex(ValueError, "another image"):
            apply_room_decisions(self.data, payload, "b" * 64)
        with self.assertRaisesRegex(ValueError, "changed"):
            apply_room_decisions(self.data, {**payload, "answers": [
                {**answer, "fingerprint": "stale"}]}, self.hash)
        with self.assertRaisesRegex(ValueError, "repeated"):
            apply_room_decisions(self.data, {**payload, "answers": [answer, answer]}, self.hash)
        large = self.data["rooms"][0]
        from review_rooms import room_fingerprint
        with self.assertRaisesRegex(ValueError, "changed or missing"):
            apply_room_decisions(self.data, {**payload, "answers": [{
                "room_id": 0, "fingerprint": room_fingerprint(large),
                "choice": "ignore"}]}, self.hash)
        single = dict(self.data, rooms=[self.data["rooms"][1]])
        with self.assertRaisesRegex(ValueError, "At least one room"):
            apply_room_decisions(single, payload, self.hash)

    def test_excluding_small_area_preserves_other_exported_area(self):
        original = self.data["rooms"]
        self.data["walls"] = [{"outer": [list(point) for point in
                                 box(*bounds).buffer(20, join_style=2).difference(box(*bounds))
                                 .exterior.coords[:-1]]}
                                for bounds in ([10, 10, 110, 110], [140, 10, 150, 20])]
        template = {"levels": [{"connectors": [], "segments": [{}],
                                "areas": [{}], "elements": []}]}
        before, old_report = build_polygon_project(self.data, template)
        question = room_questions(self.data, self.hash)[0]
        chosen = apply_room_decisions(self.data, {"version": 1,
            "image_sha256": self.hash, "answers": [{"room_id": 1,
                "fingerprint": question["fingerprint"], "choice": "ignore"}]}, self.hash)
        after, new_report = build_polygon_project(chosen, template)
        self.assertEqual(len(after["levels"][0]["areas"]), 1)
        self.assertEqual(new_report["unsupported_segments"], [])
        self.assertLess(Polygon(old_report["rooms"][0]["polygon"]).symmetric_difference(
            Polygon(new_report["rooms"][0]["polygon"])).area, 1e-4)
        self.assertEqual(self.data["rooms"], original)


if __name__ == "__main__":
    unittest.main()
