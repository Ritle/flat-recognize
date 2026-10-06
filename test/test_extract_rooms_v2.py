import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from extract_rooms_v2 import extract_rooms


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
