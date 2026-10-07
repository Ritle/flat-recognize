import copy
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from filter_interior_windows_v2 import filter_interior_windows


def window(uid, x1, y1, x2, y2):
    return {"id": uid, "outer": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "bbox": [x1, y1, x2, y2], "center": [(x1+x2)/2, (y1+y2)/2],
            "type": "window"}


class InteriorWindowFilterTests(unittest.TestCase):
    def setUp(self):
        self.gray = np.full((200, 200), 255, np.uint8)
        interior = window(1, 70, 95, 110, 101)
        boundary = window(2, 70, 20, 110, 26)
        self.classified = {"meta": {"width": 200, "height": 200}, "walls": [],
                           "doors": [], "windows": [interior, boundary],
                           "openings": copy.deepcopy([interior, boundary])}
        self.rooms = {"meta": {"width": 200, "height": 200}, "windows": [interior, boundary],
                      "doors": [], "rooms": [{"id": 0, "polygon": [[20, 20], [180, 20],
                                                               [180, 180], [20, 180]]}],
                      "diagnostics": {"warnings": []}}
        self.geometry = {"adapter": "polygon", "errors": [], "warnings": []}

    def test_rejects_only_deep_interior_and_blocks_provisional_geometry(self):
        original = copy.deepcopy((self.classified, self.rooms, self.geometry))
        classified, rooms, geometry, report = filter_interior_windows(
            self.classified, self.rooms, self.geometry, self.gray)
        self.assertEqual(report["rejected_windows"], 1)
        self.assertEqual([w["id"] for w in classified["windows"]], [2])
        self.assertEqual([w["id"] for w in rooms["windows"]], [2])
        self.assertEqual(len(rooms["rooms"]), 1)
        self.assertEqual(classified["rejected_openings"][0]["id"], 1)
        self.assertTrue(geometry["errors"])
        self.assertEqual((self.classified, self.rooms, self.geometry), original)

    def test_no_candidate_does_not_block_geometry(self):
        self.classified["windows"] = self.classified["windows"][1:]
        self.classified["openings"] = self.classified["openings"][1:]
        self.rooms["windows"] = self.rooms["windows"][1:]
        _, _, geometry, report = filter_interior_windows(
            self.classified, self.rooms, self.geometry, self.gray)
        self.assertEqual(report["rejected_windows"], 0)
        self.assertFalse(geometry["errors"])

    def test_coordinate_mismatch(self):
        self.rooms["meta"]["width"] = 199
        with self.assertRaisesRegex(ValueError, "coordinates"):
            filter_interior_windows(self.classified, self.rooms, self.geometry, self.gray)


if __name__ == "__main__":
    unittest.main()
