import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from filter_interior_windows_v2 import filter_interior_windows
from validate_recognition import validate_recognition


def window(uid, x1, y1, x2, y2):
    return {"id": uid, "outer": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "bbox": [x1, y1, x2, y2], "center": [(x1+x2)/2, (y1+y2)/2],
            "type": "window"}


class InteriorWindowFilterTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((200, 200, 3), 255, np.uint8)
        interior = window(1, 70, 95, 110, 101)
        boundary = window(2, 70, 20, 110, 26)
        self.classified = {"meta": {"width": 200, "height": 200}, "walls": [],
                           "doors": [], "windows": [interior, boundary],
                           "openings": copy.deepcopy([interior, boundary])}
        self.rooms = {"meta": {"width": 200, "height": 200}, "windows": [interior, boundary],
                      "doors": [], "rooms": [{"id": 0, "polygon": [[20, 20], [180, 20],
                                                               [180, 180], [20, 180]]}],
                      "diagnostics": {"warnings": []}}
    def test_rejects_only_deep_interior_and_rebuilds_rooms(self):
        original = copy.deepcopy((self.classified, self.rooms))
        rebuilt = copy.deepcopy(self.rooms)
        rebuilt["windows"] = rebuilt["windows"][1:]
        with patch("filter_interior_windows_v2.extract_rooms", return_value=(rebuilt, None, None)):
            classified, rooms, report = filter_interior_windows(
                self.classified, self.rooms, self.rgb)
        self.assertEqual(report["rejected_windows"], 1)
        self.assertEqual([w["id"] for w in classified["windows"]], [2])
        self.assertEqual([w["id"] for w in rooms["windows"]], [2])
        self.assertEqual(len(rooms["rooms"]), 1)
        self.assertEqual(classified["rejected_openings"][0]["id"], 1)
        self.assertTrue(report["topology_stable"])
        self.assertEqual((self.classified, self.rooms), original)

    def test_no_candidate_needs_no_recheck(self):
        self.classified["windows"] = self.classified["windows"][1:]
        self.classified["openings"] = self.classified["openings"][1:]
        self.rooms["windows"] = self.rooms["windows"][1:]
        with patch("filter_interior_windows_v2.extract_rooms") as extract:
            _, _, report = filter_interior_windows(self.classified, self.rooms, self.rgb)
        self.assertEqual(report["rejected_windows"], 0)
        extract.assert_not_called()

    def test_changed_topology_keeps_provisional_rooms_and_blocks_export(self):
        rebuilt = copy.deepcopy(self.rooms)
        rebuilt["rooms"] = []
        with patch("filter_interior_windows_v2.extract_rooms", return_value=(rebuilt, None, None)):
            _, rooms, report = filter_interior_windows(self.classified, self.rooms, self.rgb)
        self.assertFalse(report["topology_stable"])
        self.assertEqual(len(rooms["rooms"]), 1)
        self.assertEqual([w["id"] for w in rooms["windows"]], [2])
        quality = validate_recognition(rooms)
        self.assertEqual(quality["status"], "invalid")
        self.assertTrue(any(issue["code"] == "unstable_room_topology"
                            for issue in quality["issues"]))

    def test_coordinate_mismatch(self):
        self.rooms["meta"]["width"] = 199
        with self.assertRaisesRegex(ValueError, "coordinates"):
            filter_interior_windows(self.classified, self.rooms, self.rgb)


if __name__ == "__main__":
    unittest.main()
