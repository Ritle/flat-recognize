import sys
import unittest
from pathlib import Path
from unittest.mock import patch
import subprocess
import io
from contextlib import redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from audit_opening_bindings import audit_opening_bindings
from shapely.geometry import Polygon
from process_floorplan import run_command


def project():
    return {"levels": [{"connectors": [
        {"uuid": "a", "x": 0, "y": 0}, {"uuid": "b", "x": 100, "y": 0}],
        "segments": [{"uuid": "wall", "start": "a", "end": "b", "thickness": 20}]}]}


def opening(uid, center, bbox):
    return {"id": uid, "center": center, "bbox": bbox}


class OpeningBindingAuditTests(unittest.TestCase):
    def test_reports_specific_unsupported_openings_without_changing_project(self):
        classified = {"doors": [opening(1, [50, 0], [35, -2, 65, 2])],
                      "windows": [opening(2, [50, 60], [40, 57, 60, 63]),
                                  opening(3, [95, 0], [80, -2, 110, 2]),
                                  opening(4, [50, 0], [47, -10, 53, 10])]}
        geometry = {"adapter": "polygon", "pixel_to_world": {
            "scale_x": 1, "scale_y": 1, "origin_x": 0, "origin_y": 0}}
        original = project()
        report = audit_opening_bindings(classified, original, geometry)
        self.assertEqual(report["status"], "audited")
        self.assertEqual(report["summary"]["examined"], 4)
        self.assertEqual(report["summary"]["suspect"], 3)
        self.assertEqual(report["openings"][0]["reasons"], [])
        self.assertEqual(report["openings"][1]["reasons"], ["far_from_wall"])
        self.assertEqual(report["openings"][2]["reasons"], ["exceeds_segment"])
        self.assertEqual(report["openings"][3]["reasons"], ["misaligned_wall"])
        self.assertEqual(len(original["levels"][0]["segments"]), 1)

    def test_skips_bbox_adapter_without_exact_transform(self):
        report = audit_opening_bindings({"doors": [], "windows": []}, project(),
                                        {"adapter": "bbox"})
        self.assertEqual(report["status"], "skipped")
        self.assertEqual(report["reason"], "exact_pixel_to_world_transform_unavailable")

    def test_raster_span_is_counterfactual_and_keeps_primary_quality_result(self):
        window = opening(19, [50, 60], [20, 50, 80, 70])
        window["recovery"] = {"raster_axis_evidence": {
            "extent_basis": "raster_clear_bay", "span": [[25, 40], [75, 40]]}}
        classified = {"doors": [], "windows": [window]}
        geometry = {"adapter": "polygon", "pixel_to_world": {
            "scale_x": 1, "scale_y": 1, "origin_x": 0, "origin_y": 0}}
        report = audit_opening_bindings(classified, project(), geometry)
        row = report["openings"][0]
        self.assertEqual(row["distance_to_wall_cm"], 60)
        self.assertEqual(row["reasons"], ["far_from_wall"])
        alternative = row["raster_span_on_current_walls"]
        self.assertEqual(alternative["distance_to_wall_cm"], 40)
        self.assertEqual(alternative["width_cm"], 50)
        self.assertEqual(alternative["reasons"], ["far_from_wall"])
        self.assertEqual(report["summary"]["suspect"], 1)
        self.assertEqual(window["center"], [50, 60])

        window["recovery"]["raster_axis_evidence"]["extent_basis"] = "segmentation_bbox"
        row = audit_opening_bindings(classified, project(), geometry)["openings"][0]
        self.assertNotIn("raster_span_on_current_walls", row)

    def test_thick_flanking_wall_is_reported_without_changing_primary_binding(self):
        window = opening(19, [50, 60], [20, 50, 80, 70])
        window["recovery"] = {"raster_axis_evidence": {
            "extent_basis": "raster_clear_bay", "span": [[25, 0], [75, 0]]}}
        classified = {"doors": [], "windows": [window], "walls": [
            {"outer": [[0, -40], [20, -40], [20, 40], [0, 40]]},
            {"outer": [[80, -40], [100, -40], [100, 40], [80, 40]]}]}
        geometry = {"adapter": "polygon", "pixel_to_world": {
            "scale_x": 1, "scale_y": 1, "origin_x": 0, "origin_y": 0}}
        report = audit_opening_bindings(classified, project(), geometry)
        row = report["openings"][0]
        self.assertEqual(row["reasons"], ["far_from_wall"])
        alternative = row["raster_span_on_current_walls"]
        self.assertEqual(alternative["source_wall_band"]["median_thickness_px"], 80)
        self.assertEqual(alternative["source_wall_band"]["bound_wall_thickness_cm"], 20)
        self.assertEqual(alternative["reasons"], ["wall_thickness_mismatch"])
        self.assertEqual(report["summary"]["suspect"], 1)

    def test_normal_flanking_wall_does_not_flag_thickness_mismatch(self):
        window = opening(19, [50, 0], [25, -3, 75, 3])
        window["recovery"] = {"raster_axis_evidence": {
            "extent_basis": "raster_clear_bay", "span": [[25, 0], [75, 0]]}}
        classified = {"doors": [], "windows": [window], "walls": [
            {"outer": [[0, -10], [20, -10], [20, 10], [0, 10]]},
            {"outer": [[80, -10], [100, -10], [100, 10], [80, 10]]}]}
        geometry = {"adapter": "polygon", "pixel_to_world": {
            "scale_x": 1, "scale_y": 1, "origin_x": 0, "origin_y": 0}}
        alternative = audit_opening_bindings(classified, project(), geometry)[
            "openings"][0]["raster_span_on_current_walls"]
        self.assertEqual(alternative["source_wall_band"]["median_thickness_px"], 20)
        self.assertEqual(alternative["reasons"], [])

    def test_one_flank_does_not_claim_wall_thickness_mismatch(self):
        window = opening(19, [50, 0], [25, -3, 75, 3])
        window["recovery"] = {"raster_axis_evidence": {
            "extent_basis": "raster_clear_bay", "span": [[25, 0], [75, 0]]}}
        classified = {"doors": [], "windows": [window], "walls": [
            {"outer": [[0, -40], [20, -40], [20, 40], [0, 40]]}]}
        geometry = {"adapter": "polygon", "pixel_to_world": {
            "scale_x": 1, "scale_y": 1, "origin_x": 0, "origin_y": 0}}
        alternative = audit_opening_bindings(classified, project(), geometry)[
            "openings"][0]["raster_span_on_current_walls"]
        self.assertFalse(alternative["source_wall_band"]["consistent_flanks"])
        self.assertNotIn("wall_thickness_mismatch", alternative["reasons"])

    def test_flanking_wall_band_works_for_vertical_bay(self):
        from audit_opening_bindings import flanking_wall_band
        walls = Polygon([[-35, 0], [35, 0], [35, 20], [-35, 20]]).union(
            Polygon([[-35, 80], [35, 80], [35, 100], [-35, 100]]))
        band = flanking_wall_band(walls, ((0, 25), (0, 75)))
        self.assertEqual(band["median_thickness_px"], 70)
        self.assertTrue(band["consistent_flanks"])

    def test_invalid_candidate_is_reported_without_stopping_audit(self):
        data = {"doors": [{"id": 1}],
                "windows": [opening(2, [50, 0], [40, -2, 60, 2])]}
        geometry = {"adapter": "polygon", "pixel_to_world": {
            "scale_x": 1, "scale_y": 1, "origin_x": 0, "origin_y": 0}}
        report = audit_opening_bindings(data, project(), geometry)
        self.assertEqual(report["summary"]["examined"], 2)
        self.assertEqual(report["openings"][0]["reasons"], ["invalid_opening_geometry"])
        self.assertEqual(report["openings"][1]["reasons"], [])

    def test_advisory_failure_does_not_block_pipeline(self):
        with patch("process_floorplan.subprocess.run",
                   return_value=subprocess.CompletedProcess([], 2)):
            with redirect_stdout(io.StringIO()):
                self.assertFalse(run_command("diagnostic", ["diagnostic"], required=False))


if __name__ == "__main__":
    unittest.main()
