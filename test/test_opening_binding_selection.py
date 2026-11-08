import sys
import unittest
import json
import os
import subprocess
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from add_openings import (binding_rejection_reasons, choose_segment_binding,
                          door_width_cm, opening_export_geometry)
from test.test_validate_recognition import fixture
from validate_recognition import validate_recognition


def segment(uid, ax, ay, bx, by):
    return {"segment": {"uuid": uid, "thickness": 20},
            "ax": ax, "ay": ay, "bx": bx, "by": by}


class OpeningBindingSelectionTests(unittest.TestCase):
    def test_polygon_export_keeps_unplaced_openings_visible_and_invalid(self):
        rooms, project, _ = fixture()
        rooms["doors"] = []
        rooms["windows"] = [
            {"id": 1, "center": [50, 0], "bbox": [40, -3, 60, 3]},
            {"id": 2, "center": [50, 200], "bbox": [40, 197, 60, 203]},
            {"id": 3, "center": [50, 0], "bbox": [47, -10, 53, 10]}]
        project["levels"][0]["elements"] = []
        geometry = {"adapter": "polygon", "pixel_to_world": {
            "scale_x": 1, "scale_y": 1, "origin_x": 0, "origin_y": 0}}
        root = Path(__file__).resolve().parent.parent
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            for name, data in (("source", rooms), ("project", project), ("geometry", geometry)):
                (folder / f"{name}.json").write_text(json.dumps(data), encoding="utf8")
            commands = [
                [sys.executable, str(root / "scripts/add_openings.py"),
                 str(folder / "project.json"), str(folder / "source.json"),
                 str(folder / "source.json"), str(root / "test/template.json"),
                 "--output", str(folder / "doors.json"),
                 "--windows-output", str(folder / "windows.json"),
                 "--geometry-report", str(folder / "geometry.json"),
                 "--bindings-output", str(folder / "bindings.json")],
                [sys.executable, str(root / "scripts/add_windows.py"),
                 str(folder / "doors.json"), str(folder / "windows.json"),
                 str(root / "test/template_window.json"), "--output", str(folder / "final.json"),
                 "--bindings-output", str(folder / "bindings.json")]]
            for command in commands:
                run = subprocess.run(command, capture_output=True, encoding="utf8",
                    env={**os.environ, "PYTHONUTF8": "1"}, check=False)
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            final = json.loads((folder / "final.json").read_text(encoding="utf8"))
            bindings = json.loads((folder / "bindings.json").read_text(encoding="utf8"))
        self.assertEqual(len(final["levels"][0]["elements"]), 1)
        self.assertEqual([item["opening_id"] for item in bindings["unplaced"]], [2, 3])
        self.assertIn("far_from_wall", bindings["unplaced"][0]["reasons"])
        self.assertIn("perpendicular_wall", bindings["unplaced"][1]["reasons"])
        quality = validate_recognition(rooms, final, bindings, geometry=geometry)
        self.assertEqual(quality["status"], "invalid")
        self.assertEqual([item["opening_id"] for item in quality["issues"]
                          if item["code"] == "opening_unplaced"], [2, 3])

    def test_close_diagonal_or_thick_wall_is_not_rejected_as_perpendicular(self):
        diagonal = choose_segment_binding([segment("diagonal", 0, 0, 100, 100)],
                                           50, 50, 40, "horizontal")
        self.assertEqual(binding_rejection_reasons(diagonal, 40), [])
        thick = segment("thick", 0, 0, 100, 0)
        thick["segment"]["thickness"] = 66
        binding = choose_segment_binding([thick], 50, 70, 40, "horizontal")
        self.assertEqual(binding_rejection_reasons(binding, 40), [])

    def test_recovered_window_uses_bounded_raster_bay_for_export(self):
        opening = {"id": 19, "center": [60, 42], "bbox": [0, 0, 120, 80],
                   "length": 120, "orientation": "horizontal",
                   "recovery": {"method": "topology_recovered_thick_window",
                                "raster_axis_evidence": {
                                    "span": [[10, 38], [90, 38]],
                                    "extent_basis": "raster_clear_bay",
                                    "minimum_support": .95,
                                    "bay_evidence": {"start": 10, "end": 90,
                                                     "jamb_support": [.9, .9]}}}}
        geometry = opening_export_geometry(opening)
        self.assertEqual(geometry, {"center": (50, 38), "orientation": "horizontal",
                                    "length_px": 80, "source": "raster_clear_bay"})
        self.assertEqual(opening_export_geometry(opening, door=True)["length_px"], 120)

        opening["recovery"]["raster_axis_evidence"]["bay_evidence"]["jamb_support"] = [.9, .5]
        self.assertEqual(opening_export_geometry(opening)["source"], "segmentation")
        opening["recovery"]["raster_axis_evidence"]["bay_evidence"] = {}
        self.assertEqual(opening_export_geometry(opening)["source"], "segmentation")
        opening["recovery"]["raster_axis_evidence"]["bay_evidence"] = {
            "start": 30, "end": 90, "jamb_support": [.9, .9]}
        self.assertEqual(opening_export_geometry(opening)["source"], "segmentation")

    def test_door_minimum_width_is_used_for_selection_and_export(self):
        self.assertEqual(door_width_cm(20), 40)
        self.assertAlmostEqual(door_width_cm(80), 89.6)

    def test_nearby_full_span_beats_closer_short_segment(self):
        segments = [segment("short", 0, 0, 70, 0),
                    segment("full", -20, 2, 100, 2)]
        result = choose_segment_binding(segments, 35, 0, 90, "horizontal")
        self.assertEqual(result["segment"]["uuid"], "full")
        self.assertEqual(result["selection"], "span_aware")
        self.assertEqual(result["overhang"], 0)
        self.assertEqual(result["nearest_distance"], 0)

    def test_fitting_nearest_segment_is_unchanged(self):
        segments = [segment("nearest", 0, 0, 100, 0),
                    segment("other", -20, 2, 120, 2)]
        result = choose_segment_binding(segments, 50, 0, 80, "horizontal")
        self.assertEqual(result["segment"]["uuid"], "nearest")
        self.assertEqual(result["selection"], "nearest")

    def test_nearby_aligned_wall_beats_perpendicular_crossing(self):
        segments = [segment("crossing", 50, -60, 50, 60),
                    segment("wall", 0, 3, 100, 3)]
        result = choose_segment_binding(segments, 50, 0, 80, "horizontal")
        self.assertEqual(result["segment"]["uuid"], "wall")
        self.assertEqual(result["selection"], "span_aware")
        self.assertEqual(result["nearest_distance"], 0)

    def test_distant_aligned_wall_does_not_replace_crossing(self):
        segments = [segment("crossing", 50, -60, 50, 60),
                    segment("wall", 0, 25, 100, 25)]
        result = choose_segment_binding(segments, 50, 0, 80, "horizontal")
        self.assertEqual(result["segment"]["uuid"], "crossing")

    def test_distant_or_perpendicular_alternative_is_not_used(self):
        segments = [segment("short", 0, 0, 70, 0),
                    segment("far", -20, 12, 100, 12),
                    segment("perpendicular", 35, -60, 35, 60)]
        result = choose_segment_binding(segments, 35, 0, 90, "horizontal")
        self.assertEqual(result["segment"]["uuid"], "short")
        self.assertEqual(result["selection"], "nearest")
        self.assertGreater(result["overhang"], 5)

    def test_equally_plausible_alternatives_are_left_for_review(self):
        segments = [segment("short", 0, 0, 70, 0),
                    segment("upper", -20, 2, 100, 2),
                    segment("lower", -20, -2, 100, -2)]
        result = choose_segment_binding(segments, 35, 0, 90, "horizontal")
        self.assertEqual(result["segment"]["uuid"], "short")
        self.assertEqual(result["selection"], "ambiguous")
        self.assertEqual(result["alternative_wall_uuids"], ["lower", "upper"])
        self.assertGreater(result["overhang"], 5)

    def test_alternative_must_contain_exported_width(self):
        segments = [segment("short", 0, 0, 70, 0),
                    segment("almost", -7, 2, 77, 2)]
        result = choose_segment_binding(segments, 35, 0, 90, "horizontal")
        self.assertEqual(result["segment"]["uuid"], "short")
        self.assertEqual(result["selection"], "nearest")


if __name__ == "__main__":
    unittest.main()
