import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from recover_windows_topology_v2 import eligible_candidates, preserve_rooms, recover_windows


def candidate(uid, x1, x2, thickness=24):
    return {"id": uid, "outer": [[x1, 20], [x2, 20], [x2, 20 + thickness],
                                 [x1, 20 + thickness]],
            "bbox": [x1, 20, x2, 20 + thickness], "source_class": "door",
            "reject_reason": ["too_thick"], "thickness": thickness,
            "length": x2 - x1, "aspect_ratio": (x2 - x1) / thickness,
            "wall_distance": 2}


def room(x1, y1, x2, y2):
    return {"polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}


def topology_result(rooms, outside, area):
    return {"rooms": rooms, "diagnostics": {
        "outside_fraction_in_structure": outside, "interior_area_px": area}}


class TopologyWindowRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((400, 600, 3), 255, dtype=np.uint8)
        self.prediction = {"meta": {"width": 600, "height": 400},
                           "diagnostics": {"wall_tolerance": 6, "rejected": [
                               candidate(11, 80, 160), candidate(12, 300, 380)]}}
        self.classified = {"meta": {"width": 600, "height": 400},
                           "openings": [], "doors": [], "windows": []}

    def test_two_windows_required_to_recover_room(self):
        small = room(50, 50, 150, 150)
        large = room(200, 50, 500, 350)

        def extract(data, _):
            count = len(data["windows"])
            result = (topology_result([small, large], .15, 100000) if count == 2
                      else topology_result([small], .45 if count == 1 else .55, 10000))
            return result, None, None

        original = copy.deepcopy(self.classified)
        with patch("recover_windows_topology_v2.extract_rooms", side_effect=extract):
            result, report = recover_windows(self.prediction, self.classified, self.rgb)
        self.assertEqual(report["recovered_windows"], 2)
        self.assertEqual(report["selected_indices"], [0, 1])
        self.assertTrue(report["requires_review"])
        self.assertEqual(len(result["windows"]), 2)
        self.assertEqual(result["windows"][0]["type"], "window")
        self.assertTrue(result["windows"][0]["classification"]["requires_review"])
        self.assertNotIn("reject_reason", result["windows"][0])
        self.assertEqual(self.classified, original)

    def test_candidate_filters_exclude_furniture_and_known_opening_overlap(self):
        self.prediction["diagnostics"]["rejected"].append(candidate(13, 420, 530, 50))
        self.classified["openings"] = [{"outer": candidate(0, 80, 160)["outer"]}]
        eligible = eligible_candidates(self.prediction, self.classified, 400, 600)
        self.assertEqual([item["id"] for item in eligible], [12])

    def test_new_room_cannot_replace_existing_room(self):
        self.assertFalse(preserve_rooms([room(10, 10, 90, 90)],
                                        [room(50, 50, 150, 150)]))

    def test_room_count_alone_is_not_enough(self):
        small = room(50, 50, 150, 150)

        def extract(data, _):
            if data["windows"]:
                return topology_result([small, room(200, 50, 500, 350)], .38, 100000), None, None
            return topology_result([small], .55, 10000), None, None

        with patch("recover_windows_topology_v2.extract_rooms", side_effect=extract):
            result, report = recover_windows(self.prediction, self.classified, self.rgb)
        self.assertEqual(result, self.classified)
        self.assertEqual(report["recovered_windows"], 0)
        self.assertEqual(report["reason"], "no_topology_preserving_improvement")

    def test_rejects_coordinate_mismatch(self):
        self.prediction["meta"]["width"] = 599
        with self.assertRaisesRegex(ValueError, "coordinates"):
            recover_windows(self.prediction, self.classified, self.rgb)


if __name__ == "__main__":
    unittest.main()
