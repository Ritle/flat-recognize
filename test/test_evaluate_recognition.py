import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_recognition import (evaluate, evaluate_files, exported_openings,
                                 prediction_span, validate_ground_truth)


def rectangle(x1, y1, x2, y2):
    return {"outer": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "holes": []}


class EvaluateRecognitionTests(unittest.TestCase):
    def test_thick_opening_is_matched_by_centerline_not_one_rectangle_edge(self):
        truth = {"version": 1, "image": {"width": 100, "height": 80},
                 "coverage": {"windows": "geometry"},
                 "windows": [{"id": "w", "span": [[40, 15], [60, 15]]}],
                 "tolerance": {"opening_center_px": 2}}
        found = {"meta": {"width": 100, "height": 80},
                 "windows": [rectangle(40, 10, 60, 20)]}
        result = evaluate(truth, found)["metrics"]["windows"]
        self.assertEqual((result["true_positive"], result["false_positive"],
                          result["false_negative"]), (1, 0, 0))
        self.assertEqual(result["matches"][0]["predicted_thickness_px"], 10)
        span = prediction_span(found["windows"][0])
        self.assertAlmostEqual(sum(point[1] for point in span) / 2, 15)

    def test_rotated_opening_uses_middle_of_both_long_edges(self):
        truth = {"version": 1, "image": {"width": 100, "height": 100},
                 "coverage": {"doors": "geometry"},
                 "doors": [{"id": "d", "span": [[37, 43], [57, 63]]}],
                 "tolerance": {"opening_center_px": 2}}
        found = {"meta": {"width": 100, "height": 100},
                 "doors": [{"outer": [[40, 40], [60, 60], [54, 66], [34, 46]],
                            "holes": []}]}
        result = evaluate(truth, found)["metrics"]["doors"]
        self.assertEqual((result["true_positive"], result["false_positive"],
                          result["false_negative"]), (1, 0, 0))
        self.assertAlmostEqual(result["matches"][0]["predicted_thickness_px"], 8.49, places=2)

    def test_count_only_does_not_treat_missing_geometry_as_empty(self):
        truth = {
            "version": 1,
            "image": {"width": 100, "height": 80},
            "coverage": {"rooms": "count", "doors": "count"},
            "expected_counts": {"rooms": 1, "doors": 1},
        }
        recognition = {
            "meta": {"width": 100, "height": 80},
            "rooms": [rectangle(10, 10, 90, 70)],
            "doors": [rectangle(45, 65, 55, 70)],
        }
        result = evaluate(truth, recognition)
        self.assertTrue(result["all_annotated_counts_exact"])
        self.assertEqual(result["unannotated"], ["walls", "windows", "junctions"])

    def test_exact_geometry_scores_one(self):
        truth = {
            "version": 1,
            "image": {"width": 100, "height": 80},
            "coverage": {category: "geometry" for category in
                         ("rooms", "walls", "doors", "windows", "junctions")},
            "rooms": [{"id": "r1", "polygon": [[10, 10], [90, 10], [90, 70], [10, 70]]}],
            "walls": [
                {"id": "top", "centerline": [[10, 10], [90, 10]], "thickness_px": 4},
                {"id": "middle", "centerline": [[50, 10], [50, 70]], "thickness_px": 4},
            ],
            "doors": [{"id": "d1", "span": [[46, 70], [54, 70]], "wall_id": "bottom"}],
            "windows": [{"id": "w1", "span": [[10, 30], [10, 50]], "wall_id": "left"}],
            "junctions": [{"id": "j1", "type": "T", "point": [50, 10],
                           "wall_ids": ["top", "middle"]}],
        }
        recognition = {
            "meta": {"width": 100, "height": 80},
            "rooms": [rectangle(10, 10, 90, 70)],
            "walls": [rectangle(8, 8, 92, 12), rectangle(48, 8, 52, 72)],
            "doors": [rectangle(46, 68, 54, 72)],
            "windows": [rectangle(8, 30, 12, 50)],
            "junctions": [{"type": "T", "point": [50, 10]}],
        }
        result = evaluate(truth, recognition)
        self.assertEqual(result["metrics"]["rooms"]["f1"], 1)
        self.assertEqual(result["metrics"]["walls"]["area_recall"], 1)
        self.assertEqual(result["metrics"]["doors"]["f1"], 1)
        self.assertEqual(result["metrics"]["windows"]["f1"], 1)
        self.assertEqual(result["metrics"]["junctions"]["f1"], 1)

    def test_wrong_opening_type_is_not_a_match(self):
        truth = {
            "version": 1,
            "image": {"width": 100, "height": 80},
            "coverage": {"doors": "geometry", "windows": "geometry"},
            "doors": [{"id": "d1", "span": [[40, 40], [60, 40]]}],
            "windows": [],
        }
        recognition = {
            "meta": {"width": 100, "height": 80},
            "doors": [],
            "windows": [rectangle(40, 38, 60, 42)],
        }
        result = evaluate(truth, recognition)
        self.assertEqual(result["metrics"]["doors"]["false_negative"], 1)
        self.assertEqual(result["metrics"]["windows"]["false_positive"], 1)
        self.assertEqual(result["metrics"]["doors"]["unmatched_truth"], ["d1"])
        self.assertEqual(result["metrics"]["windows"]["unmatched_predictions"], [0])

    def test_invalid_ground_truth_geometry_is_rejected_before_scoring(self):
        truth = {"version": 1, "image": {"width": 100, "height": 80},
                 "coverage": {"rooms": "geometry", "doors": "geometry"},
                 "rooms": [{"id": "r1", "polygon": [[10, 10], [90, 70],
                                                      [10, 70], [90, 10]]}],
                 "doors": [{"id": "d1", "span": [[40, 40], [60, 40]]}]}
        with self.assertRaisesRegex(ValueError, "invalid or empty polygon"):
            validate_ground_truth(truth)
        truth["rooms"][0]["polygon"] = [[10, 10], [90, 10], [90, 70], [10, 70]]
        truth["doors"][0]["span"] = [[40, 40], [140, 40]]
        with self.assertRaisesRegex(ValueError, "inside the source image"):
            validate_ground_truth(truth)
        truth["doors"][0]["span"] = [[40, 40], [40, 40]]
        with self.assertRaisesRegex(ValueError, "zero length"):
            validate_ground_truth(truth)

    def test_duplicate_ids_and_invalid_count_are_rejected(self):
        truth = {"version": 1, "image": {"width": 100, "height": 80},
                 "coverage": {"rooms": "geometry"},
                 "rooms": [{"id": "r1", "polygon": [[10, 10], [90, 10],
                                                      [90, 70], [10, 70]]}]}
        duplicate = copy.deepcopy(truth["rooms"][0])
        truth["rooms"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "Duplicate rooms id"):
            validate_ground_truth(truth)
        truth["rooms"].pop()
        truth["coverage"] = {"rooms": "count"}
        truth["expected_counts"] = {"rooms": -1}
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            validate_ground_truth(truth)

    def test_geometry_count_mismatch_is_not_reported_exact(self):
        truth = {"version": 1, "image": {"width": 100, "height": 80},
                 "coverage": {"rooms": "geometry"},
                 "rooms": [{"id": "r1", "polygon": [[10, 10], [90, 10],
                                                      [90, 70], [10, 70]]}]}
        recognition = {"meta": {"width": 100, "height": 80}, "rooms": []}
        result = evaluate(truth, recognition)
        self.assertFalse(result["all_annotated_counts_exact"])

    def test_no_coverage_is_not_reported_exact(self):
        truth = {"version": 1, "image": {"width": 100, "height": 80},
                 "coverage": {}}
        recognition = {"meta": {"width": 100, "height": 80}}
        result = evaluate(truth, recognition)
        self.assertFalse(result["all_annotated_counts_exact"])

    def test_scoped_opening_geometry_excludes_unannotated_area(self):
        truth = {"version": 1, "image": {"width": 100, "height": 80},
                 "coverage": {"windows": "geometry"},
                 "annotation_regions": {"windows": [[0, 0, 100, 30]]},
                 "windows": [{"id": "w1", "span": [[20, 10], [40, 10]]}]}
        recognition = {"meta": {"width": 100, "height": 80}, "windows": [
            rectangle(18, 48, 42, 52),  # Outside annotated band.
            rectangle(19, 8, 41, 12),   # Match.
            rectangle(65, 8, 85, 12),   # False positive inside annotated band.
        ]}
        result = evaluate(truth, recognition)["metrics"]["windows"]
        self.assertEqual((result["true_positive"], result["false_positive"]), (1, 1))
        self.assertEqual(result["scoped_prediction_indices"], [1, 2])
        self.assertEqual(result["unmatched_predictions"], [2])
        self.assertEqual(result["matches"][0]["prediction"], 1)
        self.assertEqual(result["excluded_prediction_count"], 1)

    def test_scoped_truth_must_be_inside_annotated_area(self):
        truth = {"version": 1, "image": {"width": 100, "height": 80},
                 "coverage": {"doors": "geometry"},
                 "annotation_regions": {"doors": [[0, 0, 100, 30]]},
                 "doors": [{"id": "d1", "span": [[20, 50], [40, 50]]}]}
        with self.assertRaisesRegex(ValueError, "outside annotation_regions"):
            validate_ground_truth(truth)

    def test_partial_room_negative_regions_count_only_enclosed_false_rooms(self):
        truth = {"version": 1, "image": {"width": 200, "height": 100},
                 "coverage": {"rooms": "geometry"}, "rooms": [],
                 "annotation_regions": {"rooms": [[100, 0, 160, 100]]}}
        recognition = {"meta": {"width": 200, "height": 100}, "rooms": [
            rectangle(0, 0, 80, 100),    # Outside the reviewed region.
            rectangle(110, 20, 130, 40),  # Furniture enclosure inside it.
            rectangle(80, 50, 120, 90),   # Only half lies inside.
        ]}
        metrics = evaluate(truth, recognition)["metrics"]["rooms"]
        self.assertEqual((metrics["predicted"], metrics["false_positive"]), (2, 2))
        self.assertEqual(metrics["scoped_prediction_indices"], [1, 2])
        self.assertEqual(metrics["unmatched_predictions"], [1, 2])
        self.assertEqual(metrics["excluded_prediction_count"], 1)

    def test_partial_room_truth_must_be_inside_reviewed_region(self):
        truth = {"version": 1, "image": {"width": 200, "height": 100},
                 "coverage": {"rooms": "geometry"},
                 "annotation_regions": {"rooms": [[100, 0, 160, 100]]},
                 "rooms": [{"id": "r1", "polygon": [[90, 10], [130, 10],
                                                       [130, 50], [90, 50]]}]}
        with self.assertRaisesRegex(ValueError, "outside annotation_regions"):
            validate_ground_truth(truth)

    def test_partial_room_matches_keep_original_prediction_index(self):
        truth = {"version": 1, "image": {"width": 200, "height": 100},
                 "coverage": {"rooms": "geometry"},
                 "annotation_regions": {"rooms": [[100, 0, 160, 100]]},
                 "rooms": [{"id": "r1", "polygon": [[110, 10], [140, 10],
                                                       [140, 60], [110, 60]]}]}
        recognition = {"meta": {"width": 200, "height": 100}, "rooms": [
            rectangle(0, 0, 80, 100), rectangle(110, 10, 140, 60)]}
        metrics = evaluate(truth, recognition)["metrics"]["rooms"]
        self.assertEqual((metrics["true_positive"], metrics["false_positive"]), (1, 0))
        self.assertEqual(metrics["matches"][0]["prediction"], 1)

    def test_partial_wall_region_clips_false_geometry_outside_it(self):
        truth = {"version": 1, "image": {"width": 200, "height": 100},
                 "coverage": {"walls": "geometry"},
                 "annotation_regions": {"walls": [[40, 10, 160, 40]]},
                 "walls": [{"id": "top", "centerline": [[70, 25], [130, 25]],
                            "thickness_px": 10}]}
        found = {"meta": {"width": 200, "height": 100}, "walls": [
            rectangle(65, 20, 135, 30), rectangle(60, 60, 140, 70)]}
        metrics = evaluate(truth, found)["metrics"]["walls"]
        self.assertEqual(metrics["scope"], "annotation_regions")
        self.assertEqual(metrics["predicted"], 1)
        self.assertEqual(metrics["area_iou"], 1)
        self.assertEqual(metrics["per_region"][0]["area_iou"], 1)
        truth["walls"][0]["centerline"] = [[10, 25], [130, 25]]
        with self.assertRaisesRegex(ValueError, "outside annotation_regions"):
            validate_ground_truth(truth)

    def test_exported_wall_scoring_uses_geometry_transform(self):
        truth = {"version": 1, "image": {"width": 200, "height": 100},
                 "coverage": {"walls": "geometry"},
                 "annotation_regions": {"walls": [[40, 10, 160, 40]]},
                 "walls": [{"id": "top", "centerline": [[70, 25], [130, 25]],
                            "thickness_px": 10}]}
        found = {"meta": {"width": 200, "height": 100}, "walls": []}
        # Pixel coordinates are mapped to world by x*2+100, y*2-40.
        project = {"levels": [{"connectors": [
            {"uuid": "a", "x": 240, "y": 10},
            {"uuid": "b", "x": 360, "y": 10}],
            "segments": [{"start": "a", "end": "b", "thickness": 20}]}]}
        geometry = {"adapter": "polygon", "pixel_to_world": {
            "scale_x": 2, "scale_y": 2, "origin_x": 100, "origin_y": -40}}
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{name}.json" for name in
                     ("truth", "found", "project", "geometry")]
            for path, data in zip(paths, (truth, found, project, geometry)):
                path.write_text(json.dumps(data), encoding="utf-8")
            result = evaluate_files(paths[0], paths[1], project_path=paths[2],
                                    geometry_path=paths[3])
        self.assertEqual(result["metrics"]["walls"]["area_iou"], 0)
        self.assertEqual(result["export_geometry"]["walls"]["area_iou"], 1)
        self.assertEqual(result["export_geometry"]["walls"]["scope"],
                         "annotation_regions")

    def test_equal_counts_with_false_positive_and_miss_are_not_exact(self):
        truth = {"version": 1, "image": {"width": 100, "height": 80},
                 "coverage": {"windows": "geometry"},
                 "windows": [{"id": "w1", "span": [[20, 10], [40, 10]]}]}
        recognition = {"meta": {"width": 100, "height": 80},
                       "windows": [rectangle(65, 8, 85, 12)]}
        result = evaluate(truth, recognition)
        self.assertEqual(result["metrics"]["windows"]["expected"], 1)
        self.assertEqual(result["metrics"]["windows"]["predicted"], 1)
        self.assertFalse(result["all_annotated_counts_exact"])

    def test_exported_windows_are_scored_at_actual_carrier_positions(self):
        truth = {"version": 1, "image": {"width": 100, "height": 80},
                 "coverage": {"windows": "geometry"},
                 "annotation_regions": {"windows": [[10, 0, 50, 35]]},
                 "tolerance": {"opening_center_px": 2},
                 "windows": [{"id": "w", "span": [[20, 10], [40, 10]]}]}
        found = {"meta": truth["image"], "windows": [rectangle(20, 8, 40, 12)]}
        # Reversed carrier and unequal raster scales must preserve position/width.
        project = {"levels": [{"connectors": [
            {"uuid": "a", "x": 300, "y": -10}, {"uuid": "b", "x": 100, "y": -10}],
            "segments": [{"uuid": "wall", "start": "a", "end": "b", "thickness": 30}],
            "elements": [{"uuid": uid, "type": "windowItem", "size": {"x": 40},
                          "wall": {"uuid": "wall", "position": position}}
                         for uid, position in (("target", 70), ("outside-scope", 10))]}]}
        geometry = {"adapter": "polygon", "pixel_to_world": {
            "scale_x": 2, "scale_y": 3, "origin_x": 100, "origin_y": -40}}
        with tempfile.TemporaryDirectory() as temp:
            paths = [Path(temp) / f"{name}.json" for name in
                     ("truth", "found", "project", "geometry")]
            for path, data in zip(paths, (truth, found, project, geometry)):
                path.write_text(json.dumps(data), encoding="utf8")
            result = evaluate_files(paths[0], paths[1], project_path=paths[2], geometry_path=paths[3])
            exported = result["export_geometry"]["windows"]
            self.assertEqual(exported["true_positive"], 1)
            self.assertEqual(exported["excluded_prediction_count"], 1)
            self.assertEqual(exported["matches"][0]["prediction_id"], "target")
            self.assertEqual(exported["matches"][0]["center_error_px"], 0)
            self.assertEqual(exported["matches"][0]["length_error_px"], 0)
            self.assertEqual(exported["matches"][0]["predicted_thickness_px"], 10)
            for node in project["levels"][0]["connectors"]:
                node["y"] += 60
            paths[2].write_text(json.dumps(project), encoding="utf8")
            shifted = evaluate_files(paths[0], paths[1], project_path=paths[2], geometry_path=paths[3])
        self.assertEqual(shifted["metrics"]["windows"]["true_positive"], 1)
        self.assertEqual(shifted["export_geometry"]["windows"]["false_negative"], 1)
        self.assertEqual(shifted["export_geometry"]["windows"]["false_positive"], 1)

    def test_exported_door_on_diagonal_wall_and_missing_carrier(self):
        project = {"levels": [{"connectors": [
            {"uuid": "a", "x": 0, "y": 0}, {"uuid": "b", "x": 60, "y": 80}],
            "segments": [{"uuid": "wall", "start": "a", "end": "b", "thickness": 20}],
            "elements": [{"uuid": "door", "type": "doorItem", "size": {"x": 50},
                          "wall": {"uuid": "wall", "position": 50}}]}]}
        geometry = {"adapter": "polygon", "pixel_to_world": {
            "scale_x": 1, "scale_y": 1, "origin_x": 0, "origin_y": 0}}
        self.assertEqual(exported_openings(project, geometry)["doors"][0]["span"],
                         [[15, 20], [45, 60]])
        geometry["pixel_to_world"].update(scale_x=2, scale_y=4)
        stretched = exported_openings(project, geometry)["doors"][0]
        self.assertEqual(stretched["span"], [[7.5, 5], [22.5, 15]])
        self.assertAlmostEqual(stretched["thickness_px"], 6.93375245)
        project["levels"][0]["elements"][0]["wall"]["uuid"] = "missing"
        with self.assertRaisesRegex(ValueError, "Invalid exported opening geometry"):
            exported_openings(project, geometry)


if __name__ == "__main__":
    unittest.main()
