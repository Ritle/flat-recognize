import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from classify_openings import classify_opening, classify_openings


def fixture(arc=True, leaf=True):
    image = Image.new("RGB", (240, 240), "white")
    draw = ImageDraw.Draw(image)
    walls = [
        {"outer": [[40, 96], [100, 96], [100, 104], [40, 104]]},
        {"outer": [[160, 96], [220, 96], [220, 104], [160, 104]]},
    ]
    for wall in walls:
        draw.polygon([tuple(point) for point in wall["outer"]], fill="black")
    if leaf:
        draw.line((100, 96, 100, 40), fill="black", width=2)
    if arc:
        draw.arc((44, 40, 156, 152), 270, 360, fill="black", width=2)
    opening = {"outer": [[100, 96], [160, 96], [160, 104], [100, 104]],
               "source_class": "door"}
    data = {"meta": {"width": 240, "height": 240}, "walls": walls,
            "openings": [opening]}
    return data, np.asarray(image)


class ClassifyOpeningsTests(unittest.TestCase):
    def test_leaf_and_arc_confirm_door_without_mutating_source(self):
        data, rgb = fixture()
        before = copy.deepcopy(data)
        result = classify_openings(data, rgb)
        self.assertEqual(len(result["doors"]), 1)
        self.assertEqual(result["windows"], [])
        self.assertEqual(result["doors"][0]["classification"]["method"], "jamb_leaf_and_arc")
        self.assertGreaterEqual(result["doors"][0]["classification"]["evidence"]["arc_hits"], 7)
        self.assertEqual(data, before)

    def test_leaf_or_arc_alone_remains_window(self):
        for arc, leaf in ((False, True), (True, False), (False, False)):
            with self.subTest(arc=arc, leaf=leaf):
                data, rgb = fixture(arc=arc, leaf=leaf)
                result = classify_openings(data, rgb)
                self.assertEqual(result["doors"], [])
                self.assertEqual(len(result["windows"]), 1)
                self.assertFalse(result["windows"][0]["classification"]["confirmed"])

    def test_legacy_gap_classifier_keeps_perpendicular_line_signal(self):
        data, rgb = fixture(arc=False, leaf=True)
        gray = np.asarray(Image.fromarray(rgb).convert("L"))
        opening_type, evidence = classify_opening(gray, data["openings"][0])
        self.assertEqual(opening_type, "door")
        self.assertTrue(evidence)

    def test_coordinate_mismatch_is_rejected(self):
        data, rgb = fixture()
        data["meta"]["width"] = 100
        with self.assertRaises(ValueError):
            classify_openings(data, rgb)


if __name__ == "__main__":
    unittest.main()
