import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from validate_recognition import collinear_wall_span, validate_recognition


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
    def test_collinear_span_is_diagnostic_and_stops_at_junctions(self):
        coords = {"a": (0, 0), "b": (50, 0), "c": (100, 1), "d": (50, 40)}
        segments = {"s0": {"uuid": "s0", "start": "a", "end": "b"},
                    "s1": {"uuid": "s1", "start": "b", "end": "c"}}
        span = collinear_wall_span(segments, coords, "s0", 90, 90)
        self.assertEqual(span["wall_chain_segments"], ["s0", "s1"])
        self.assertEqual(span["wall_chain_overhang_cm"], 0)
        coords["e"] = (150, 9)
        segments["s2"] = {"uuid": "s2", "start": "c", "end": "e"}
        self.assertEqual(collinear_wall_span(segments, coords, "s0", 90, 90)
                         ["wall_chain_segments"], ["s0", "s1"])
        segments["s2"] = {"uuid": "s2", "start": "b", "end": "d"}
        span = collinear_wall_span(segments, coords, "s0", 90, 90)
        self.assertEqual(span["wall_chain_segments"], ["s0"])
        self.assertEqual(span["wall_chain_overhang_cm"], 40)

    def test_clean_good_and_scale_warning(self):
        r = validate_recognition(*fixture())
        self.assertEqual(r["status"], "good")
        self.assertTrue(r["export_allowed"])
        self.assertEqual(r["quality"], 1)
        self.assertIn("approximate_scale", [i["code"] for i in r["issues"]])

    def test_automatic_topology_mask_change_cannot_clear_invalid_quality(self):
        rooms, project, bindings = fixture()
        rooms["diagnostics"]["topology_window_reconciliation"] = {
            "removed_from_room_mask": [2]}
        rooms["topology_only_openings"] = [{"id": 2, "topology_mask_use": False}]
        result = validate_recognition(rooms, project, bindings)
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["export_allowed"])
        self.assertIn("topology_mask_change_unconfirmed",
                      [item["code"] for item in result["issues"]])
        self.assertNotIn("topology_only_openings",
                         [item["code"] for item in result["issues"]])

    def test_only_verified_duplicate_pair_can_clear_mask_change_error(self):
        rooms, project, bindings = fixture()
        entries = [{"id": 7, "classification": {"method": "duplicate_parallel_window_layer", "paired_with": 1}},
                   {"id": 1, "classification": {"method": "one_sided_unconfirmed_window"}}]
        rooms["topology_only_openings"] = [dict(e, topology_mask_use=False) for e in entries]
        change = dict(reason="duplicate_layers_at_verified_wall", status="accepted", opening_ids=[7, 1],
                      unsupported_before=4, unsupported_after=0,
                      raster_wall={"min_wall_radius_px": 2, "max_removed_ink_radius_px": 1})
        report = {"removed_from_room_mask": [7, 1], "candidates": [change]}
        rooms["diagnostics"]["topology_window_reconciliation"] = report
        self.assertEqual(validate_recognition(rooms, project, bindings)["status"], "review")
        report["removed_from_room_mask"].append(9)
        result = validate_recognition(rooms, project, bindings)
        issue = next(i for i in result["issues"] if i["code"] == "topology_mask_change_unconfirmed")
        self.assertEqual(issue["opening_ids"], [9])
        self.assertFalse(result["export_allowed"])
        change["raster_wall"]["min_wall_radius_px"] = 1
        result = validate_recognition(rooms, project, bindings)
        issue = next(i for i in result["issues"] if i["code"] == "topology_mask_change_unconfirmed")
        self.assertEqual(issue["opening_ids"], [7, 1, 9])

    def test_verified_single_strip_must_refer_to_a_released_window_and_an_existing_room(self):
        rooms, project, bindings = fixture()
        rooms["rooms"][0]["id"] = 3
        rooms["topology_only_openings"] = [{"id": 7, "topology_mask_use": False,
            "classification": {"method": "one_sided_unconfirmed_window"}}]
        change = dict(reason="one_sided_strip_at_verified_wall", status="accepted", opening_id=7,
                      room_id=3, mask_pixels_removed=30, unsupported_before=2, unsupported_after=0,
                      raster_wall={"min_wall_radius_px": 2, "max_removed_ink_radius_px": 0})
        rooms["diagnostics"]["topology_window_reconciliation"] = {
            "removed_from_room_mask": [7], "candidates": [change]}
        self.assertEqual(validate_recognition(rooms, project, bindings)["status"], "review")
        for key, value in (("room_id", 0), ("opening_id", 2), ("mask_pixels_removed", 0),
                           ("unsupported_after", 1)):
            with self.subTest(key=key), patch.dict(change, {key: value}):
                self.assertFalse(validate_recognition(rooms, project, bindings)["export_allowed"])

    def test_manual_opening_choice_remains_review(self):
        rooms, project, bindings = fixture()
        rooms["review_decisions"] = [{"opening_id": 1, "choice": "window"}]
        result = validate_recognition(rooms, project, bindings)
        self.assertEqual(result["status"], "review")
        self.assertTrue(result["export_allowed"])
        self.assertEqual(result["metrics"]["user_review_choices"], 1)

    def test_preserved_room_contour_with_material_endcap_shift_requires_review(self):
        rooms, project, bindings = fixture()
        rooms["diagnostics"]["interior_window_filter"] = {
            "topology_stable": True,
            "endcap_room_geometry_changes": [{
                "opening_id": 15, "parent_window_id": 7,
                "added_room_area_px2": 1499.15,
                "removed_room_area_px2": 0.0, "material": True}]}
        result = validate_recognition(rooms, project, bindings)
        self.assertEqual(result["status"], "review")
        change = next(issue for issue in result["issues"]
                      if issue["code"] == "endcap_room_geometry_change")
        self.assertEqual(change["parent_window_id"], 7)
        self.assertEqual(change["added_room_area_px2"], 1499.15)

        rooms["diagnostics"]["interior_window_filter"][
            "endcap_room_geometry_changes"][0]["material"] = False
        self.assertEqual(validate_recognition(rooms, project, bindings)["status"], "good")

    def test_raster_supported_window_corner_adjustment_stays_review(self):
        rooms, project, bindings = fixture()
        geometry = {"adapter": "polygon", "extended_window_endcaps": [{
            "window_id": 7, "room_id": 0, "corner_shift_px": 35,
            "added_room_area_px2": 1802.5}]}
        result = validate_recognition(rooms, project, bindings, geometry=geometry)
        self.assertEqual(result["status"], "review")
        self.assertTrue(result["export_allowed"])
        adjustment = next(issue for issue in result["issues"]
                          if issue["code"] == "window_endcap_corner_adjusted")
        self.assertEqual(adjustment["opening_id"], 7)
        self.assertEqual(adjustment["corner_shift_px"], 35)

    def test_measured_facade_thickness_stays_review(self):
        rooms, project, bindings = fixture()
        geometry = {"adapter": "polygon", "calibrated_wall_thickness": [{
            "wall_uuid": "wall-1", "opening_ids": [19],
            "measured_thickness_cm": 66.07}]}
        result = validate_recognition(rooms, project, bindings, geometry=geometry)
        self.assertEqual(result["status"], "review")
        issue = next(item for item in result["issues"]
                     if item["code"] == "facade_wall_thickness_calibrated")
        self.assertEqual(issue["wall_uuid"], "wall-1")
        self.assertEqual(issue["opening_ids"], [19])

    def test_realigning_exterior_axis_requires_review(self):
        rooms, project, bindings = fixture()
        geometry = {"adapter": "polygon", "realigned_exterior_wall_axes": [{
            "wall_uuid": "wall-1", "room_index": 0,
            "axis_shift_px": 7.75, "source_support": .997}]}
        result = validate_recognition(rooms, project, bindings, geometry=geometry)
        self.assertEqual(result["status"], "review")
        self.assertTrue(result["export_allowed"])
        adjustment = next(issue for issue in result["issues"]
                          if issue["code"] == "exterior_wall_axis_realigned")
        self.assertEqual(adjustment["axis_shift_px"], 7.75)

    def test_thick_topology_window_stays_invalid_after_type_confirmation(self):
        rooms, project, bindings = fixture()
        recovered = {"method": "topology_recovered_thick_window", "requires_review": True}
        rooms["windows"][0].update(id=20, length=161, thickness=89,
                                   classification=recovered)
        room_check = validate_recognition(rooms)
        self.assertEqual(room_check["status"], "review")
        self.assertTrue(any(i["code"] == "thick_window_geometry_unresolved"
                            and i["severity"] == "warning" for i in room_check["issues"]))

        final = validate_recognition(rooms, project, bindings)
        self.assertEqual(final["status"], "invalid")
        self.assertFalse(final["export_allowed"])
        issue = next(i for i in final["issues"]
                     if i["code"] == "thick_window_geometry_unresolved")
        self.assertEqual((issue["opening_id"], issue["thickness_px"]), (20, 89))

        rooms["windows"][0]["classification"] = {
            "method": "user_selected", "user_confirmed": True, "previous": recovered}
        confirmed = validate_recognition(rooms, project, bindings)
        self.assertEqual(confirmed["status"], "invalid")
        self.assertIn("thick_window_geometry_unresolved",
                      [i["code"] for i in confirmed["issues"]])

    def test_exported_window_cut_must_match_raster_and_calibrated_wall(self):
        rooms, project, bindings = fixture()
        level = project["levels"][0]
        rooms["doors"] = []
        bindings["doors"] = []
        item = level["elements"].pop()
        level["elements"] = [item]
        wall = item["wall"]
        wall["holeShape"] = [dict(x=x, y=y) for x, y in
                             ((-.2, 0), (.2, 0), (.2, 1), (-.2, 1))]
        method = "topology_recovered_thick_window"
        evidence = dict(method="paired_raster_rails", span=[[5, 20], [5, 40]],
                        extent_basis="raster_clear_bay", minimum_support=1, rails_px=[3, 7],
                        bay_evidence=dict(start=20, end=40, jamb_support=[1, 1]))
        rooms["windows"][0].update(id=5, bbox=[0, 18, 10, 42], classification=dict(method=method),
                                   recovery=dict(method=method, raster_axis_evidence=evidence))
        # The left edge is downward in image coordinates and reversed in the project cycle.
        wall["uuid"] = "s3"
        bindings["windows"][0].update(wall_uuid="s3", element_uuid=item["uuid"],
                                      opening_id=5, geometry_source="raster_clear_bay")
        calibration = dict(wall_uuid="original", final_wall_uuids=["s3"],
                           opening_ids=[5], measured_thickness_cm=20)
        geometry = dict(adapter="polygon", calibrated_wall_thickness=[calibration],
                        pixel_to_world=dict(scale_x=2, scale_y=2, origin_x=-10, origin_y=-10))
        result = validate_recognition(rooms, project, bindings, geometry=geometry)
        self.assertTrue(result["export_allowed"], result["errors"])
        self.assertIn("recovered_window_cut_verified", [i["code"] for i in result["issues"]])
        for target, key, replacement in (
                (wall, "position", 65), (wall, "holeShape", []),
                (evidence["bay_evidence"], "jamb_support", [1, .3]),
                (geometry["pixel_to_world"], "origin_x", 10),
                (calibration, "final_wall_uuids", ["s1"]),
                (calibration, "measured_thickness_cm", 42)):
            with self.subTest(key=key):
                old = target[key]
                target[key] = replacement
                blocked = validate_recognition(rooms, project, bindings, geometry=geometry)
                self.assertFalse(blocked["export_allowed"])
                self.assertIn("thick_window_geometry_unresolved", [i["code"] for i in blocked["issues"]])
                target[key] = old

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

    def test_severe_binding_errors_block_export(self):
        rooms, project, bindings = fixture()
        bindings["windows"][0]["distance_to_wall_cm"] = 65
        result = validate_recognition(rooms, project, bindings)
        self.assertEqual(result["status"], "invalid")
        self.assertFalse(result["export_allowed"])
        self.assertTrue(any(i["code"] == "opening_far_from_wall" and i["severity"] == "error"
                            for i in result["issues"]))

        rooms, project, bindings = fixture()
        project["levels"][0]["elements"][0]["wall"]["position"] = 14
        self.assertEqual(validate_recognition(rooms, project, bindings)["status"], "review")
        project["levels"][0]["elements"][0]["wall"]["position"] = 1
        project["levels"][0]["elements"][0]["size"]["x"] = 60
        bindings["doors"][0].update(door=1, opening_id=42, element_uuid="doorItem")
        result = validate_recognition(rooms, project, bindings)
        self.assertEqual(result["status"], "invalid")
        overhang = next(i for i in result["issues"] if i["code"] == "opening_exceeds_segment")
        self.assertEqual(overhang["severity"], "error")
        self.assertEqual(overhang["opening_id"], 42)
        self.assertEqual(overhang["element_uuid"], "doorItem")
        self.assertEqual(overhang["wall_length_cm"], 100)
        self.assertEqual(overhang["opening_width_cm"], 60)
        self.assertEqual(overhang["wall_chain_segments"], ["s0"])

    def test_perpendicular_binding_blocks_export_with_opening_location(self):
        rooms, project, bindings = fixture()
        project["levels"][0]["elements"][1]["wall"]["uuid"] = "s1"
        bindings["windows"][0].update(wall_uuid="s1", orientation="horizontal",
                                      window=1, pixel_center=[50, 20])
        result = validate_recognition(rooms, project, bindings)
        self.assertEqual(result["status"], "invalid")
        issue = next(i for i in result["issues"]
                     if i["code"] == "opening_perpendicular_wall")
        self.assertEqual(issue["opening_index"], 1)
        self.assertEqual(issue["wall_uuid"], "s1")
        self.assertEqual(issue["pixel_center"], [50, 20])

        bindings["windows"][0]["orientation"] = "vertical"
        self.assertEqual(validate_recognition(rooms, project, bindings)["status"], "good")

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

    def test_geometry_failure_cannot_pass_with_valid_bbox(self):
        rooms, project, bindings = fixture()
        report = {"adapter": "bbox", "warnings": [], "errors": ["Polygon export failed"]}
        self.assertFalse(validate_recognition(rooms, project, bindings, geometry=report)["export_allowed"])


if __name__ == "__main__":
    unittest.main()
