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

        data["walls"] = data["walls"][:1]
        result = classify_openings(data, rgb)
        self.assertEqual(result["doors"], [])

    def test_leaf_or_arc_alone_remains_window(self):
        for arc, leaf in ((False, True), (True, False), (False, False)):
            with self.subTest(arc=arc, leaf=leaf):
                data, rgb = fixture(arc=arc, leaf=leaf)
                result = classify_openings(data, rgb)
                self.assertEqual(result["doors"], [])
                self.assertEqual(len(result["windows"]), 1)
                self.assertFalse(result["windows"][0]["classification"]["confirmed"])

    def test_source_door_with_two_jambs_accepts_diagonal_leaf_without_arc(self):
        data, rgb = fixture(arc=False, leaf=False)
        image = Image.fromarray(rgb)
        ImageDraw.Draw(image).line((100, 96, 142, 54), fill="black", width=2)
        result = classify_openings(data, np.asarray(image))
        self.assertEqual(len(result["doors"]), 1)
        classification = result["doors"][0]["classification"]
        self.assertEqual(classification["method"], "source_door_two_jamb_diagonal_leaf")
        self.assertTrue(classification["requires_review"])
        self.assertEqual(classification["evidence"]["jamb_support"], [3, 3])

        data["walls"] = data["walls"][:1]
        result = classify_openings(data, np.asarray(image))
        self.assertEqual(result["doors"], [])

    def test_source_window_needs_two_wall_jambs_to_override_semantics(self):
        data, rgb = fixture()
        data["openings"][0]["source_class"] = "window"
        result = classify_openings(data, rgb)
        self.assertEqual(len(result["doors"]), 1)
        self.assertEqual(result["doors"][0]["classification"]["evidence"]["jamb_support"],
                         [3, 3])

        data["walls"] = data["walls"][:1]
        result = classify_openings(data, rgb)
        self.assertEqual(result["doors"], [])

    def test_window_on_detached_thin_furniture_line_is_rejected(self):
        def rectangle(x1, y1, x2, y2):
            return {"outer": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]}

        walls = [rectangle(20, 20, 380, 30), rectangle(20, 20, 30, 280),
                 rectangle(370, 20, 380, 280), rectangle(20, 270, 380, 280),
                 rectangle(250, 98, 340, 99)]
        opening = {"outer": [[325, 100], [374, 100], [374, 114], [325, 114]],
                   "source_class": "window"}
        data = {"meta": {"width": 400, "height": 300}, "walls": walls,
                "openings": [opening]}
        image = Image.new("RGB", (400, 300), "white")
        draw = ImageDraw.Draw(image)
        for wall in walls:
            draw.polygon([tuple(point) for point in wall["outer"]], fill="black")
        result = classify_openings(data, np.asarray(image))
        self.assertEqual(result["doors"], [])
        self.assertEqual(result["windows"], [])
        self.assertEqual(result["openings"], [])
        self.assertEqual(len(result["rejected_openings"]), 1)
        classification = result["rejected_openings"][0]["classification"]
        self.assertEqual(classification["method"], "rejected_detached_thin_wall_window")
        self.assertTrue(classification["rejection"]["detached_linework"])

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
