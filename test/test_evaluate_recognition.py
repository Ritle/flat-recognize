import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_recognition import evaluate, validate_ground_truth


def rectangle(x1, y1, x2, y2):
    return {"outer": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]], "holes": []}


class EvaluateRecognitionTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
