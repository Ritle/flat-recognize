import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from extract_rooms_v2 import axis_pairs, extract_rooms, gap_candidates


def rectangle(x1, y1, x2, y2):
    return {"outer": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "holes": []}


def source(walls, leaf=False):
    image = Image.new("RGB", (400, 300), "white")
    draw = ImageDraw.Draw(image)
    for wall in walls:
        draw.polygon([tuple(p) for p in wall["outer"]], fill="black")
        for hole in wall.get("holes", []):
            draw.polygon([tuple(p) for p in hole], fill="white")
    if leaf:
        draw.line((150, 270, 150, 230), fill="black", width=2)
    return {"meta": {"width": 400, "height": 300}, "walls": walls,
            "doors": [], "windows": []}, np.asarray(image)


class RoomsTests(unittest.TestCase):
    def test_axis_sampling_rows_do_not_shift_with_barrier_thickness(self):
        barrier = np.zeros((300, 400), np.uint8)
        cv2.rectangle(barrier, (20, 120), (140, 160), 255, -1)
        cv2.rectangle(barrier, (220, 120), (380, 160), 255, -1)
        first = axis_pairs(barrier, 27, sampling_thickness=26)
        second = axis_pairs(barrier, 28, sampling_thickness=26)
        first_rows = [int(a["point"][1]) for a, b in first if 70 < b["point"][0] - a["point"][0] < 90]
        second_rows = [int(a["point"][1]) for a, b in second if 70 < b["point"][0] - a["point"][0] < 90]
        self.assertIn(144, first_rows)
        self.assertIn(144, second_rows)

    def test_missed_exterior_door_is_closed_only_for_topology(self):
        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 150, 280),
                 rectangle(190, 270, 380, 280)]
        data, rgb = source(walls, leaf=True)
        result, _, inferred = extract_rooms(data, rgb)
        self.assertEqual(len(result["rooms"]), 1)
        self.assertTrue(np.any(inferred))
        self.assertTrue(result["rooms"][0]["uncertain"])
        self.assertEqual(result["walls"], walls)
        self.assertTrue(result["diagnostics"]["warnings"])

    def test_unmarked_wide_passage_is_not_closed(self):
        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 380, 280),
                 rectangle(195, 20, 205, 120), rectangle(195, 180, 205, 280)]
        data, rgb = source(walls)
        result, _, inferred = extract_rooms(data, rgb)
        self.assertEqual(len(result["rooms"]), 1)
        self.assertFalse(np.any(inferred))
        self.assertFalse(result["rooms"][0]["uncertain"])

    def test_rejected_window_candidate_is_reported_for_review(self):
        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 380, 280)]
        data, rgb = source(walls)
        data["rejected_openings"] = [{"classification": {
            "method": "rejected_detached_thin_wall_window"}}]
        result, _, _ = extract_rooms(data, rgb)
        self.assertEqual(len(result["rooms"]), 1)
        self.assertEqual(result["diagnostics"]["rejected_openings"], 1)
        data["rejected_openings"][0]["classification"]["method"] = "user_ignored"
        data["review_decisions"] = [{"opening_id": 3, "choice": "ignore"}]
        reviewed, _, _ = extract_rooms(data, rgb)
        self.assertEqual(reviewed["review_decisions"], data["review_decisions"])
        self.assertFalse(any("изолированных тонких" in warning
                             for warning in reviewed["diagnostics"]["warnings"]))
        self.assertTrue(any("Оконные кандидаты" in warning
                            for warning in result["diagnostics"]["warnings"]))

    def test_isolated_furniture_enclosure_is_not_a_room(self):
        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 380, 280),
                 rectangle(150, 100, 186, 103), rectangle(150, 133, 186, 136),
                 rectangle(150, 100, 153, 136), rectangle(183, 100, 186, 136)]
        data, rgb = source(walls)
        result, _, _ = extract_rooms(data, rgb)
        self.assertEqual(len(result["rooms"]), 1)
        rejected = [item for pass_result in result["diagnostics"]["passes"]
                    for item in pass_result["rejected_components"]]
        self.assertTrue(any(item["reason"] == "isolated_wall_enclosure"
                            for item in rejected))
        self.assertEqual(result["diagnostics"]["isolated_wall_enclosures"], 1)
        self.assertTrue(result["diagnostics"]["warnings"])

    def test_small_room_joined_to_main_walls_is_preserved(self):
        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 380, 280),
                 rectangle(90, 20, 100, 110), rectangle(20, 100, 100, 110)]
        data, rgb = source(walls)
        result, _, _ = extract_rooms(data, rgb)
        self.assertEqual(len(result["rooms"]), 2)
        self.assertEqual(result["diagnostics"]["isolated_wall_enclosures"], 0)

    def test_furniture_line_away_from_jamb_does_not_close_passage(self):
        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 150, 280),
                 rectangle(190, 270, 380, 280)]
        data, rgb = source(walls)
        image = Image.fromarray(rgb)
        draw = ImageDraw.Draw(image)
        draw.line((170, 250, 170, 210), fill="black", width=2)
        result, _, inferred = extract_rooms(data, np.asarray(image))
        self.assertEqual(result["rooms"], [])
        self.assertFalse(np.any(inferred))
        rejected = [candidate for pass_result in result["diagnostics"]["passes"]
                    for candidate in pass_result.get("rejected_leaf_gaps", [])]
        self.assertTrue(rejected)
        self.assertTrue(all(candidate["reason"] == "no_jamb_anchored_leaf" for candidate in rejected))

    def test_large_exterior_gap_without_leaf_remains_uncertain(self):
        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 150, 280),
                 rectangle(190, 270, 380, 280)]
        data, rgb = source(walls)
        result, _, inferred = extract_rooms(data, rgb)
        self.assertEqual(result["rooms"], [])
        self.assertFalse(np.any(inferred))
        self.assertFalse(result["diagnostics"]["exterior_envelope"]["accepted"])
        self.assertTrue(result["diagnostics"]["warnings"])

    def test_fragmented_exterior_envelope_is_closed_only_for_topology(self):
        walls = [rectangle(20, 20, 145, 30), rectangle(255, 20, 380, 30),
                 rectangle(20, 270, 145, 280), rectangle(255, 270, 380, 280),
                 rectangle(20, 20, 30, 115), rectangle(20, 185, 30, 280),
                 rectangle(370, 20, 380, 115), rectangle(370, 185, 380, 280)]
        data, rgb = source(walls)
        result, _, inferred = extract_rooms(data, rgb)
        self.assertEqual(len(result["rooms"]), 1)
        self.assertTrue(np.any(inferred))
        self.assertEqual(result["walls"], walls)
        envelope = result["diagnostics"]["exterior_envelope"]
        self.assertTrue(envelope["accepted"])
        self.assertEqual(envelope["reason"], "fragmented_exterior_envelope")
        self.assertTrue(result["rooms"][0]["uncertain"])

    def test_extended_leaf_requires_free_space_at_its_tip(self):
        barrier = np.zeros((300, 300), dtype=np.uint8)
        cv2.rectangle(barrier, (95, 20), (105, 120), 255, -1)
        cv2.rectangle(barrier, (95, 170), (105, 280), 255, -1)
        gray = np.full_like(barrier, 255)
        gray[barrier > 0] = 0
        cv2.line(gray, (100, 170), (150, 170), 0, 2)

        closures, _ = gap_candidates(barrier, gray, relaxed=True, extended=True)
        self.assertEqual(len(closures), 1)
        self.assertEqual(closures[0]["evidence"], "door_leaf")

        cv2.rectangle(barrier, (146, 166), (154, 174), 255, -1)
        closures, diagnostics = gap_candidates(barrier, gray, relaxed=True, extended=True)
        self.assertEqual(closures, [])
        self.assertTrue(diagnostics["rejected_leaf_gaps"])
        self.assertTrue(all(candidate["reason"] == "leaf_tip_touches_barrier"
                            for candidate in diagnostics["rejected_leaf_gaps"]))

    def test_arc_less_door_marks_room_result_for_review(self):
        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 380, 280)]
        data, rgb = source(walls)
        data["doors"] = [{"outer": [[150, 270], [190, 270], [190, 280], [150, 280]],
                          "classification": {"requires_review": True}}]
        result, _, _ = extract_rooms(data, rgb)
        self.assertEqual(result["diagnostics"]["review_doors"], 1)
        self.assertTrue(any("без подтверждённой дуги" in warning
                            for warning in result["diagnostics"]["warnings"]))

    def test_topology_recovered_window_marks_room_result_for_review(self):
        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 380, 280)]
        data, rgb = source(walls)
        data["windows"] = [{"outer": [[150, 20], [190, 20], [190, 30], [150, 30]],
                            "classification": {"requires_review": True}}]
        result, _, _ = extract_rooms(data, rgb)
        self.assertEqual(result["diagnostics"]["review_windows"], 1)
        self.assertTrue(any("Окна, восстановленные" in warning
                            for warning in result["diagnostics"]["warnings"]))

    def test_l_shaped_polygon_and_missing_walls(self):
        wall = {"outer": [[20, 20], [380, 20], [380, 170], [220, 170], [220, 280], [20, 280]],
                "holes": [[[30, 30], [370, 30], [370, 160], [210, 160], [210, 270], [30, 270]]]}
        data, rgb = source([wall])
        result, _, _ = extract_rooms(data, rgb)
        self.assertEqual(len(result["rooms"]), 1)
        self.assertGreater(len(result["rooms"][0]["polygon"]), 4)
        data, rgb = source([])
        result, _, _ = extract_rooms(data, rgb)
        self.assertEqual(result["rooms"], [])
        self.assertTrue(result["diagnostics"]["warnings"])


if __name__ == "__main__":
    unittest.main()
