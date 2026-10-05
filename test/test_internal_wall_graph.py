import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from shapely.geometry import LineString, box
from shapely.ops import unary_union

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from internal_wall_graph import extend_project_walls
from polygon_geometry import build_polygon_project, parts
from validate_recognition import validate_recognition

from test.test_polygon_geometry import source, template


def fixture(partitions):
    data = source([box(0, 0, 200, 200)])
    for shape in partitions:
        for p in parts(shape, "Polygon"):
            data["walls"].append({"outer": [list(p) for p in p.exterior.coords[:-1]],
                                  "holes": [[list(p) for p in r.coords[:-1]] for r in p.interiors]})
    return data


def graph_lines(project):
    level = project["levels"][0]
    coords = {c["uuid"]: (c["x"], c["y"]) for c in level["connectors"]}
    return [(s, LineString([coords[s["start"]], coords[s["end"]]])) for s in level["segments"]]


class InternalWallsTest(unittest.TestCase):
    def build(self, data, scale=1):
        project, report = build_polygon_project(data, template(), scale=scale)
        original = copy.deepcopy(data)
        project, report = extend_project_walls(project, data, report, thickness=20)
        self.assertEqual(data, original)
        quality = validate_recognition(data, project, geometry=report, scale=scale)
        self.assertEqual(quality["errors"], [], quality)
        self.assertEqual(len(project["levels"][0]["areas"]), len(data["rooms"]))
        # Every geometric intersection must share a connector, including T/X nodes.
        edges = graph_lines(project)
        for i, (a, line_a) in enumerate(edges):
            for b, line_b in edges[i+1:]:
                intersection = line_a.intersection(line_b)
                if not intersection.is_empty:
                    self.assertEqual(intersection.geom_type, "Point")
                    self.assertTrue({a["start"], a["end"]} & {b["start"], b["end"]})
        return project, report, quality

    def test_existing_boundaries_are_unchanged_when_no_partition_exists(self):
        data = fixture([])
        project, report = build_polygon_project(data, template())
        original = copy.deepcopy(project)
        result, report = extend_project_walls(project, data, report)
        self.assertEqual(result, original)
        self.assertEqual(report["internal_wall_graph"]["added_segments"], [])

    def test_attached_partition_splits_boundary_without_creating_a_room(self):
        data = fixture([box(90, -10, 110, 120)])
        project, report, quality = self.build(data)
        self.assertTrue(report["internal_wall_graph"]["added_segments"])
        self.assertEqual(report["internal_wall_graph"]["split_existing_segments"], 1)
        self.assertEqual(quality["status"], "review")
        self.assertTrue(any(len(c["adjacency"]) == 3 for c in project["levels"][0]["connectors"]))

    def test_isolated_orthogonal_furniture_edge_is_not_exported_as_wall(self):
        data = fixture([box(40, 90, 160, 110)])
        _, report, _ = self.build(data)
        graph = report["internal_wall_graph"]
        self.assertEqual(graph["added_segments"], [])
        rejected = [item for item in graph["rejected"]
                    if item.get("reason") == "isolated_wall_component"]
        self.assertTrue(rejected)
        self.assertGreater(rejected[0]["boundary_distance_px"], 20)

    def test_t_and_x_junctions_are_noded_without_duplicate_segments(self):
        for shape, degree in ((unary_union([box(90, -10, 110, 105), box(20, 90, 180, 110)]), 3),
                              (unary_union([box(90, 20, 110, 180), box(20, 90, 180, 110)]), 4)):
            with self.subTest(degree=degree):
                project, _, _ = self.build(fixture([shape]))
                self.assertTrue(any(len(c["adjacency"]) == degree for c in project["levels"][0]["connectors"]))
                pairs = [frozenset((s["start"], s["end"])) for s in project["levels"][0]["segments"]]
                self.assertEqual(len(pairs), len(set(pairs)))

    def test_diagonal_centerline_survives_and_scale_is_preserved(self):
        data = fixture([LineString([(40, 40), (160, 160)]).buffer(8, cap_style=2)])
        for scale in (1, 2):
            project, report, _ = self.build(data, scale=scale)
            internal = {s["uuid"] for s in report["internal_wall_graph"]["added_segments"]}
            self.assertTrue(any(abs(line.coords[-1][0] - line.coords[0][0]) > 20
                                and abs(line.coords[-1][1] - line.coords[0][1]) > 20
                                for s, line in graph_lines(project) if s["uuid"] in internal))
            self.assertEqual(report["pixel_to_world"]["scale_x"], scale)

    def test_thin_leaf_near_a_door_never_becomes_a_wall(self):
        data = fixture([box(0, 96, 45, 100)])
        data["doors"] = [{"outer": [[-20, 100], [0, 100], [0, 145], [-20, 145]]}]
        _project, report, _ = self.build(data)
        self.assertEqual(report["internal_wall_graph"]["added_segments"], [])
        self.assertTrue(report["internal_wall_graph"]["rejected"])

    def test_temporary_closure_is_not_a_wall_or_a_bridge(self):
        data = fixture([box(20, 90, 85, 110), box(115, 90, 180, 110)])
        data["diagnostics"]["temporary_closures"] = [{"start": [85, 100], "end": [115, 100], "width": 20}]
        project, report, _ = self.build(data)
        transform = report["pixel_to_world"]
        gap = box(transform["origin_x"] + 86, transform["origin_y"] + 91,
                  transform["origin_x"] + 114, transform["origin_y"] + 109)
        self.assertTrue(report["internal_wall_graph"]["added_segments"])
        self.assertTrue(all(line.intersection(gap).length == 0 for _, line in graph_lines(project)))

    def test_internal_door_binds_to_new_carrier_with_leaf_and_valid_materials(self):
        data = fixture([box(20, 90, 85, 110), box(115, 90, 180, 110)])
        data["doors"] = [{"id": 0, "type": "door", "orientation": "horizontal", "length": 30,
                          "outer": [[85, 90], [115, 90], [115, 110], [85, 110]]}]
        real_template = json.loads((ROOT / "test/template.json").read_text(encoding="utf8"))
        project, report = build_polygon_project(data, real_template)
        project, report = extend_project_walls(project, data, report)
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name, value in (("source.json", data), ("project.json", project), ("geometry.json", report)):
                (directory / name).write_text(json.dumps(value), encoding="utf8")
            result = subprocess.run([sys.executable, str(ROOT / "scripts/add_openings.py"),
                str(directory / "project.json"), str(directory / "source.json"), str(directory / "source.json"),
                str(ROOT / "test/template.json"), "--output", str(directory / "final.json"),
                "--windows-output", str(directory / "windows.json"), "--bindings-output", str(directory / "bindings.json"),
                "--geometry-report", str(directory / "geometry.json")], capture_output=True, encoding="utf8",
                env={**os.environ, "PYTHONUTF8": "1"}, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            final = json.loads((directory / "final.json").read_text(encoding="utf8"))
            bindings = json.loads((directory / "bindings.json").read_text(encoding="utf8"))
        door = next(e for e in final["levels"][0]["elements"] if e["type"] == "doorItem")
        self.assertIn(door["wall"]["uuid"], {s["uuid"] for s in report["internal_wall_graph"]["added_segments"]})
        self.assertTrue(door["model"])
        self.assertTrue(all(m["itemUuid"] == door["uuid"] for m in door["materials"]))
        self.assertEqual(validate_recognition(data, final, bindings, geometry=report)["errors"], [])


if __name__ == "__main__":
    unittest.main()
