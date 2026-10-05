import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_recognition import evaluate


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


if __name__ == "__main__":
    unittest.main()
