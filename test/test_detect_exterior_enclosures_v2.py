import copy
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from detect_exterior_enclosures_v2 import detect_exterior_enclosures


def fixture():
    rgb = np.full((400, 400, 3), 255, np.uint8)
    for y in (58, 64):
        rgb[y:y + 2, 98:246] = 0
    for x in (98, 104, 238, 244):
        rgb[58:165, x:x + 2] = 0
    def item(shape, uid=None):
        return {"id": uid, "outer": list(shape.exterior.coords[:-1])}
    classified = {"meta": {"width": 400, "height": 400},
        "walls": [item(box(90, 150, 115, 170)), item(box(230, 150, 255, 170))],
        "windows": [item(box(120, 57, 160, 67), 1), item(box(237, 85, 247, 135), 2)]}
    rooms = {"meta": classified["meta"], "rooms": [
        {"id": 0, "polygon": list(box(20, 160, 380, 380).exterior.coords[:-1])}]}
    return classified, rooms, rgb


class ExteriorEnclosureTests(unittest.TestCase):
    def test_two_seed_frame_outside_room_is_proposed_without_mutation(self):
        classified, rooms, rgb = fixture()
        original = copy.deepcopy((classified, rooms))
        report = detect_exterior_enclosures(classified, rooms, rgb)
        self.assertEqual(len(report["candidates"]), 1)
        candidate = report["candidates"][0]
        self.assertEqual(candidate["seed_ids"], [1, 2])
        self.assertTrue(candidate["requires_review"])
        self.assertGreater(candidate["outside_fraction"], .98)
        self.assertEqual(len(candidate["rail_spans"]), 3)
        self.assertGreater(Polygon(candidate["polygon"]).area, 13000)
        self.assertEqual((classified, rooms), original)

    def test_rotation_does_not_depend_on_a_top_balcony(self):
        classified, rooms, rgb = fixture()
        for turn in range(4):
            with self.subTest(turn=turn):
                self.assertEqual(len(detect_exterior_enclosures(classified, rooms, rgb)["candidates"]), 1)
            rgb = np.rot90(rgb).copy()
            for key in ("walls", "windows"):
                for item in classified[key]:
                    shape = affine_transform(Polygon(item["outer"]), [0, 1, -1, 0, 0, 399])
                    item["outer"] = list(shape.exterior.coords[:-1])
            for room in rooms["rooms"]:
                shape = affine_transform(Polygon(room["polygon"]), [0, 1, -1, 0, 0, 399])
                room["polygon"] = list(shape.exterior.coords[:-1])

    def test_furniture_broken_rails_missing_anchor_or_single_seed_are_not_proposed(self):
        for failure in ("inside_room", "broken_side", "missing_anchor", "single_seed", "ambiguous_rails"):
            classified, rooms, rgb = fixture()
            if failure == "inside_room":
                rooms["rooms"][0]["polygon"] = list(box(20, 20, 380, 380).exterior.coords[:-1])
            elif failure == "broken_side":
                rgb[90:115, 95:110] = 255
            elif failure == "missing_anchor":
                classified["walls"] = classified["walls"][:1]
            elif failure == "single_seed":
                classified["windows"] = classified["windows"][:1]
            else:
                rgb[70:72, 98:246] = 0
            with self.subTest(failure=failure):
                self.assertEqual(detect_exterior_enclosures(classified, rooms, rgb)["candidates"], [])


if __name__ == "__main__":
    unittest.main()
