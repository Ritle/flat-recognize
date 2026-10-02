import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from recover_openings_v2 import recover_openings
from validate_recognition import validate_recognition
from polygon_geometry import build_polygon_project
from shapely.geometry import box


def fixture(arc=True, leaf=True, vertical=False):
    image = Image.new("RGB", (240, 240), "white")
    draw = ImageDraw.Draw(image)
    walls = [{"outer": [[40, 96], [100, 96], [100, 104], [40, 104]]},
             {"outer": [[160, 96], [220, 96], [220, 104], [160, 104]]}]
    for wall in walls:
        draw.polygon([tuple(p) for p in wall["outer"]], fill="black")
    if leaf:
        draw.line((100, 96, 100, 40), fill="black", width=2)
    if arc:
        draw.arc((44, 40, 156, 152), 270, 360, fill="black", width=2)
    classified = {"meta": {"width": 240, "height": 240}, "walls": walls,
                  "doors": [], "windows": [], "openings": []}
    rooms = {**copy.deepcopy(classified), "rooms": [], "diagnostics": {
        "warnings": [], "temporary_closures": [{"start": [100, 100], "end": [160, 100],
                                                 "width": 8, "evidence": "door_leaf"}]}}
    if vertical:
        image = image.transpose(Image.Transpose.TRANSPOSE)
        for data in (classified, rooms):
            for wall in data["walls"]:
                wall["outer"] = [p[::-1] for p in wall["outer"]]
        closure = rooms["diagnostics"]["temporary_closures"][0]
        closure["start"], closure["end"] = closure["start"][::-1], closure["end"][::-1]
    return classified, rooms, np.asarray(image)


class RecoveryTest(unittest.TestCase):
    def test_leaf_arc_jambs_recover_both_orientations_without_mutating_sources(self):
        for vertical in (False, True):
            with self.subTest(vertical=vertical):
                classified, rooms, rgb = fixture(vertical=vertical)
                before = copy.deepcopy((classified, rooms))
                output, final_rooms, report = recover_openings(classified, rooms, rgb)
                self.assertEqual(report["recovered_doors"], 1)
                self.assertEqual(output["doors"], final_rooms["doors"])
                self.assertEqual(output["doors"][0]["orientation"], "vertical" if vertical else "horizontal")
                self.assertTrue(output["doors"][0]["recovery"]["requires_review"])
                self.assertGreaterEqual(report["candidates"][0]["evidence"]["arc_hits"], 7)
                self.assertEqual((classified, rooms), before)
                self.assertEqual(final_rooms["walls"], rooms["walls"])
                self.assertEqual(final_rooms["rooms"], rooms["rooms"])
                self.assertIn("room_diagnostic", [i["code"] for i in validate_recognition(final_rooms)["issues"]])

    def test_plain_line_or_arc_alone_cannot_invent_a_door(self):
        for arc, leaf in ((False, True), (True, False), (False, False)):
            with self.subTest(arc=arc, leaf=leaf):
                classified, rooms, rgb = fixture(arc=arc, leaf=leaf)
                self.assertEqual(recover_openings(classified, rooms, rgb)[2]["recovered_doors"], 0)

    def test_missing_jamb_and_unmarked_passage_stay_rejected(self):
        for mode in ("missing_jamb", "small_gap", "raster_line"):
            classified, rooms, rgb = fixture()
            if mode == "missing_jamb":
                classified["walls"].pop()
            else:
                rooms["diagnostics"]["temporary_closures"][0]["evidence"] = mode
            self.assertEqual(recover_openings(classified, rooms, rgb)[2]["recovered_doors"], 0)

    def test_existing_window_and_repeated_gap_are_not_duplicated(self):
        classified, rooms, rgb = fixture()
        closure = rooms["diagnostics"]["temporary_closures"][0]
        rooms["diagnostics"]["temporary_closures"].append(copy.deepcopy(closure))
        output, _, report = recover_openings(classified, rooms, rgb)
        self.assertEqual(report["recovered_doors"], 1)
        self.assertEqual(report["candidates"][1]["reason"], "existing_opening")
        window = copy.deepcopy(output["doors"][0])
        window.update(type="window", id=10)
        classified["windows"] = [window]
        self.assertEqual(recover_openings(classified, rooms, rgb)[2]["recovered_doors"], 0)

    def test_empty_baseline_is_unchanged_and_coordinates_are_checked(self):
        classified, rooms, rgb = fixture()
        rooms["diagnostics"]["temporary_closures"] = []
        output, final_rooms, report = recover_openings(classified, rooms, rgb)
        self.assertEqual(output, classified)
        self.assertEqual(final_rooms["doors"], rooms["doors"])
        self.assertEqual(report["recovered_doors"], 0)
        classified["meta"]["width"] = 123
        with self.assertRaises(ValueError):
            recover_openings(classified, rooms, rgb)

    def test_confirmed_door_is_exported_with_leaf_and_existing_wall(self):
        classified, rooms, rgb = fixture()
        shape = box(40, 104, 220, 220)
        walls = shape.buffer(8, join_style=2).difference(shape).difference(box(100, 95, 160, 105))
        classified["walls"] = [{"outer": [list(p) for p in walls.exterior.coords[:-1]],
                                 "holes": [[list(p) for p in r.coords[:-1]] for r in walls.interiors]}]
        rooms["walls"] = copy.deepcopy(classified["walls"])
        rooms["rooms"] = [{"polygon": [list(p) for p in shape.exterior.coords[:-1]],
                            "bbox": list(shape.bounds)}]
        root = Path(__file__).resolve().parent.parent
        template = json.loads((root / "test/template.json").read_text(encoding="utf8"))
        candidate, report = build_polygon_project(rooms, template, thickness=8)
        self.assertFalse(validate_recognition(rooms, candidate, geometry=report)["export_allowed"])
        classified, rooms, recovery = recover_openings(classified, rooms, rgb)
        self.assertEqual(recovery["recovered_doors"], 1)
        project, report = build_polygon_project(rooms, template, thickness=8)
        self.assertEqual(report["unsupported_segments"], [])
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name, value in (("source.json", classified), ("rooms.json", rooms),
                                ("project.json", project), ("geometry.json", report)):
                (directory / name).write_text(json.dumps(value), encoding="utf8")
            result = subprocess.run([sys.executable, str(root / "scripts/add_openings.py"),
                str(directory / "project.json"), str(directory / "source.json"),
                str(directory / "rooms.json"), str(root / "test/template.json"),
                "--output", str(directory / "doors.json"), "--windows-output", str(directory / "windows.json"),
                "--bindings-output", str(directory / "bindings.json"),
                "--geometry-report", str(directory / "geometry.json")],
                capture_output=True, encoding="utf8", env={**os.environ, "PYTHONUTF8": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            final = json.loads((directory / "doors.json").read_text(encoding="utf8"))
            bindings = json.loads((directory / "bindings.json").read_text(encoding="utf8"))
        doors = [e for e in final["levels"][0]["elements"] if e["type"] == "doorItem"]
        self.assertEqual(len(doors), 1)
        self.assertTrue(doors[0]["model"])
        self.assertIn(doors[0]["wall"]["uuid"], {s["uuid"] for s in final["levels"][0]["segments"]})
        self.assertTrue(all(m["itemUuid"] == doors[0]["uuid"] for m in doors[0]["materials"]))
        quality = validate_recognition(rooms, final, bindings, geometry=report)
        self.assertEqual(quality["status"], "review")
        self.assertTrue(quality["export_allowed"])


if __name__ == "__main__":
    unittest.main()
