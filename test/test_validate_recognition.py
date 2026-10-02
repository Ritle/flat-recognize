import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from validate_recognition import validate_recognition


def fixture():
    rooms = {"rooms": [{"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}],
             "doors": [{}], "windows": [{}], "diagnostics": {}}
    connectors = [dict(uuid=str(i), x=x, y=y, adjacency=[str((i-1) % 4), str((i+1) % 4)])
                  for i, (x, y) in enumerate(rooms["rooms"][0]["polygon"])]
    segments = [dict(uuid=f"s{i}", start=str(i), end=str((i+1) % 4), thickness=20, height=270)
                for i in range(4)]
    elements = [dict(uuid=kind, type=kind, wall={"uuid": "s0", "position": 50},
                     size={"x": 40, "y": 200, "z": 20},
                     materials=[{"itemUuid": kind}], model={"name": "template-model"})
                for kind in ("doorItem", "windowItem")]
    project = {"levels": [{"connectors": connectors, "segments": segments,
                           "areas": [{"uuid": "a", "connectors": ["0", "1", "2", "3"]}],
                           "elements": elements}]}
    bindings = {key: [{"wall_uuid": "s0", "distance_to_wall_cm": 10}] for key in ("doors", "windows")}
    return rooms, project, bindings


class QualityGateTest(unittest.TestCase):
    def test_clean_good_and_scale_warning(self):
        r = validate_recognition(*fixture())
        self.assertEqual(r["status"], "good")
        self.assertTrue(r["export_allowed"])
        self.assertEqual(r["quality"], 1)
        self.assertIn("approximate_scale", [i["code"] for i in r["issues"]])

    def test_invalid_source(self):
        for polygons, code in (([], "no_rooms"),
                               ([[[0, 0], [100, 100], [100, 0], [0, 100]]], "invalid_polygon"),
                               ([[[0, 0], [1, 0], [float('nan'), 1]]], "invalid_polygon"),
                               ([[[0, 0], [100, 0], [100, 100], [0, 100]],
                                 [[10, 10], [20, 10], [20, 20], [10, 20]]], "massive_overlap")):
            with self.subTest(code=code):
                rooms, _, _ = fixture()
                rooms["rooms"] = [{"polygon": p} for p in polygons]
                r = validate_recognition(rooms)
                self.assertFalse(r["export_allowed"])
                self.assertIn(code, [i["code"] for i in r["issues"]])

    def test_review_keeps_export(self):
        rooms, project, bindings = fixture()
        rooms["rooms"][0]["uncertain"] = True
        rooms["diagnostics"]["outside_fraction_in_structure"] = 0.4
        bindings["doors"][0]["distance_to_wall_cm"] = 35
        r = validate_recognition(rooms, project, bindings)
        self.assertEqual(r["status"], "review")
        self.assertTrue(r["export_allowed"])
        self.assertIn("opening_far_from_wall", [i["code"] for i in r["issues"]])

    def test_broken_project_is_blocked(self):
        mutations = {
            "missing_connector": lambda l: l["segments"][0].update(start="absent"),
            "zero_segment": lambda l: l["connectors"][1].update(x=0, y=0),
            "invalid_uuid": lambda l: l["connectors"].append(copy.deepcopy(l["connectors"][0])),
            "unclosed_area": lambda l: l["segments"].pop(),
            "missing_wall": lambda l: l["elements"][0]["wall"].update(uuid="absent"),
            "opening_position": lambda l: l["elements"][0]["wall"].update(position=101),
            "material_reference": lambda l: l["elements"][0]["materials"][0].update(itemUuid="template"),
            "no_door_leaf": lambda l: l["elements"][0].pop("model"),
            "invalid_coordinate": lambda l: l["connectors"][0].update(x=float('inf')),
            "invalid_wall_size": lambda l: l["segments"][0].update(thickness=-20),
        }
        for code, mutate in mutations.items():
            with self.subTest(code=code):
                rooms, project, bindings = fixture()
                mutate(project["levels"][0])
                r = validate_recognition(rooms, project, bindings)
                self.assertEqual(r["status"], "invalid")
                self.assertFalse(r["export_allowed"])
                self.assertIn(code, [i["code"] for i in r["issues"]])

    def test_massive_export_overlap_even_when_source_disjoint(self):
        rooms, project, bindings = fixture()
        rooms["rooms"].append({"polygon": [[200, 0], [300, 0], [300, 100], [200, 100]]})
        project["levels"][0]["areas"].append({"uuid": "b", "connectors": ["0", "1", "2", "3"]})
        r = validate_recognition(rooms, project, bindings)
        self.assertFalse(r["export_allowed"])
        self.assertEqual(r["metrics"]["level_0_max_overlap_ratio"], 1)

    def test_unbound_opening_requires_review(self):
        rooms, project, bindings = fixture()
        project["levels"][0]["elements"].pop()
        bindings["windows"] = []
        self.assertEqual(validate_recognition(rooms, project, bindings)["status"], "review")

    def test_bad_payloads_and_scale_do_not_crash(self):
        for rooms, project, bindings in ((None, [], []), ({"rooms": [None]}, {}, None)):
            self.assertFalse(validate_recognition(rooms, project, bindings, float('nan'))["export_allowed"])

    def test_explicit_closed_cycle(self):
        rooms, project, bindings = fixture()
        project["levels"][0]["areas"][0]["connectors"].append("0")
        self.assertEqual(validate_recognition(rooms, project, bindings)["status"], "good")


if __name__ == "__main__":
    unittest.main()
