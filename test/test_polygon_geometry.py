import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from polygon_geometry import build_polygon_project, partition_rooms, requires_polygons
from validate_recognition import validate_recognition


def source(shapes):
    walls = unary_union([p.buffer(20, join_style=2).difference(p) for p in shapes])
    polygons = [walls] if walls.geom_type == "Polygon" else list(walls.geoms)
    return {"rooms": [{"polygon": [list(p) for p in shape.exterior.coords[:-1]],
                       "bbox": list(shape.bounds)} for shape in shapes],
            "walls": [{"outer": [list(p) for p in shape.exterior.coords[:-1]],
                       "holes": [[list(p) for p in r.coords[:-1]] for r in shape.interiors]} for shape in polygons],
            "doors": [], "windows": [], "diagnostics": {}}


def template():
    return {"levels": [{"connectors": [], "segments": [{}], "areas": [{}], "elements": []}]}


class PolygonGeometryTest(unittest.TestCase):
    def test_l_shape_does_not_contain_neighbor(self):
        a = Polygon([(0, 0), (200, 0), (200, 80), (80, 80), (80, 200), (0, 200)])
        b = box(100, 100, 200, 200)
        data = source([a, b])
        before = copy.deepcopy(data)
        project, report = build_polygon_project(data, template())
        quality = validate_recognition(data, project, geometry=report)
        self.assertEqual(quality["errors"], [])
        self.assertEqual(quality["metrics"]["level_0_max_overlap_ratio"], 0)
        self.assertGreater(len(project["levels"][0]["areas"][0]["connectors"]), 4)
        # Match the legacy area winding in project x/y coordinates.
        points = {c["uuid"]: (c["x"], c["y"]) for c in project["levels"][0]["connectors"]}
        cycle = [points[c] for c in project["levels"][0]["areas"][0]["connectors"]]
        signed = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(cycle, cycle[1:] + cycle[:1]))
        self.assertGreater(signed, 0)
        self.assertEqual(data, before)
        self.assertTrue(requires_polygons(data["rooms"]))

    def test_shared_wall_and_t_junction_have_common_nodes(self):
        data = source([box(0, 0, 100, 100), box(120, 0, 220, 40), box(120, 60, 220, 100)])
        project, report = build_polygon_project(data, template())
        level = project["levels"][0]
        self.assertEqual(validate_recognition(data, project, geometry=report)["errors"], [])
        self.assertTrue(any(len(c["adjacency"]) == 3 for c in level["connectors"]))
        edges = [frozenset((s["start"], s["end"])) for s in level["segments"]]
        self.assertEqual(len(edges), len(set(edges)))
        self.assertTrue(set(level["areas"][0]["connectors"]) & set(level["areas"][1]["connectors"]))

    def test_diagonal_boundary_survives(self):
        data = source([Polygon([(0, 0), (120, 0), (120, 80), (80, 120), (0, 120)])])
        project, report = build_polygon_project(data, template())
        level = project["levels"][0]
        coords = {c["uuid"]: c for c in level["connectors"]}
        self.assertTrue(any(coords[s["start"]]["x"] != coords[s["end"]]["x"]
                            and coords[s["start"]]["y"] != coords[s["end"]]["y"] for s in level["segments"]))
        self.assertEqual(validate_recognition(data, project, geometry=report)["errors"], [])

    def test_temporary_closure_never_becomes_wall(self):
        data = source([box(0, 0, 100, 100)])
        gap = box(-25, 30, 1, 70)
        wall = Polygon(data["walls"][0]["outer"], data["walls"][0]["holes"]).difference(gap)
        data["walls"] = [{"outer": [list(p) for p in wall.exterior.coords[:-1]],
                          "holes": [[list(p) for p in r.coords[:-1]] for r in wall.interiors]}]
        data["diagnostics"]["temporary_closures"] = [{"start": [-10, 30], "end": [-10, 70], "width": 20}]
        project, report = build_polygon_project(data, template())
        self.assertTrue(report["unsupported_segments"])
        self.assertFalse(validate_recognition(data, project, geometry=report)["export_allowed"])
        # A real detected opening supplies an opening carrier, not an inferred wall.
        data["doors"] = [{"outer": [[-25, 30], [1, 30], [1, 70], [-25, 70]]}]
        _, known = build_polygon_project(data, template())
        self.assertEqual(known["unsupported_segments"], [])

    def test_opening_coordinate_tolerance_covers_small_closure_cap_mismatch(self):
        data = source([box(0, 0, 100, 100)])
        gap = box(-25, 30, 1, 70)
        wall = Polygon(data["walls"][0]["outer"], data["walls"][0]["holes"]).difference(gap)
        data["walls"] = [{"outer": [list(p) for p in wall.exterior.coords[:-1]],
                          "holes": [[list(p) for p in r.coords[:-1]] for r in wall.interiors]}]
        data["diagnostics"]["temporary_closures"] = [{"start": [-10, 29.5], "end": [-10, 70.5], "width": 20}]
        data["doors"] = [{"outer": [[-25, 30.5], [1, 30.5], [1, 69.5], [-25, 69.5]]}]
        _, report = build_polygon_project(data, template())
        self.assertEqual(report["unsupported_segments"], [])

    def test_source_overlap_is_rejected(self):
        with self.assertRaises(ValueError):
            partition_rooms([box(0, 0, 100, 100), box(50, 50, 150, 150)], 10)

    def test_source_hole_cannot_be_flattened(self):
        shape = Polygon(box(0, 0, 100, 100).exterior, [box(40, 40, 60, 60).exterior])
        with self.assertRaises(ValueError):
            partition_rooms([shape], 10)

    def test_large_hole_created_by_offset_is_not_filled(self):
        shape = Polygon([(0, 0), (100, 0), (100, 45), (80, 45), (80, 20), (20, 20),
                         (20, 80), (80, 80), (80, 55), (100, 55), (100, 100), (0, 100)])
        with self.assertRaises(ValueError):
            partition_rooms([shape], 10)

    def test_bbox_mode_selection_keeps_rectangular_baseline(self):
        self.assertFalse(requires_polygons(source([box(0, 0, 100, 100), box(120, 0, 200, 100)])["rooms"]))

    def test_opening_binding_uses_exact_transform_at_non_default_scale(self):
        original_template = json.loads((ROOT / "test/template.json").read_text(encoding="utf-8"))
        data = source([box(0, 0, 100, 100)])
        data["windows"] = [{"center": [50, -5], "outer": [[40, -10], [60, -10], [60, 0], [40, 0]],
                            "orientation": "horizontal", "length": 20}]
        data["doors"] = [{"center": [50, 105], "outer": [[40, 100], [60, 100], [60, 110], [40, 110]],
                          "orientation": "horizontal", "length": 20}]
        project, report = build_polygon_project(data, original_template, scale=2)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name, value in (("source.json", data), ("project.json", project), ("geometry.json", report)):
                (directory / name).write_text(json.dumps(value), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/add_openings.py"), str(directory / "project.json"),
                       str(directory / "source.json"), str(directory / "source.json"), str(ROOT / "test/template.json"),
                       "--output", str(directory / "doors.json"), "--windows-output", str(directory / "windows.json"),
                       "--bindings-output", str(directory / "bindings.json"), "--geometry-report", str(directory / "geometry.json")]
            result = subprocess.run(command, capture_output=True, encoding="utf-8", env={**os.environ, "PYTHONUTF8": "1"})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            binding = json.loads((directory / "windows.json").read_text(encoding="utf-8"))["windows"][0]
            self.assertEqual(binding["distance_to_wall_cm"], 0)
            self.assertAlmostEqual(binding["position_percent"], 50)
            transform = report["pixel_to_world"]
            self.assertEqual(binding["world_center"], [round(transform["origin_x"] + 100, 3),
                                                       round(transform["origin_y"] - 10, 3)])


if __name__ == "__main__":
    unittest.main()
