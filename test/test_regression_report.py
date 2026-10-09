import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from run_recognition_regression import write_report


class RegressionReportTest(unittest.TestCase):
    def test_count_only_ground_truth_is_not_reported_as_geometry_accuracy(self):
        report = {"pipeline": "v2", "cases": [
            {"id": "labeled", "status": "pipeline_ok", "seconds": 1,
             "accuracy": {"coverage": {"rooms": "count", "doors": "count"},
                          "metrics": {"rooms": {"exact": True}, "doors": {"exact": False}}}},
            {"id": "unlabeled", "status": "quality_invalid", "seconds": 2},
        ]}
        with tempfile.TemporaryDirectory() as temp:
            write_report(report, Path(temp))
            markdown = (Path(temp) / "report.md").read_text(encoding="utf-8")
        self.assertIn("1/2 cases with count-only annotations; 0/2 with geometric annotations", markdown)
        self.assertIn("counts 1/2; geometry 0", markdown)
        self.assertIn("[counts 1/2; geometry 0](labeled/accuracy.json)", markdown)

    def test_geometry_coverage_has_no_count_accuracy_claim(self):
        report = {"pipeline": "v2", "cases": [
            {"id": "geometry", "status": "pipeline_ok", "seconds": 1,
             "accuracy": {"coverage": {"rooms": "geometry", "doors": "geometry"},
                          "metrics": {"rooms": {"f1": 1}, "doors": {"f1": 0.5}}}},
        ]}
        with tempfile.TemporaryDirectory() as temp:
            write_report(report, Path(temp))
            markdown = (Path(temp) / "report.md").read_text(encoding="utf-8")
        self.assertIn("0/1 cases with count-only annotations; 1/1 with geometric annotations", markdown)
        self.assertIn("[geometry 2 categories](geometry/accuracy.json)", markdown)
        self.assertNotIn("counts 0/0", markdown)

    def test_one_geometry_category_uses_singular_label(self):
        report = {"pipeline": "v2", "cases": [
            {"id": "windows", "status": "quality_invalid", "seconds": 1,
             "accuracy": {"coverage": {"windows": "geometry"}, "metrics": {}}},
        ]}
        with tempfile.TemporaryDirectory() as temp:
            write_report(report, Path(temp))
            markdown = (Path(temp) / "report.md").read_text(encoding="utf-8")
        self.assertIn("[geometry 1 category](windows/accuracy.json)", markdown)

    def test_style_diagnostics_keep_unannotated_metrics_unknown(self):
        report = {"pipeline": "v2", "cases": [
            {"id": "colored", "status": "quality_invalid", "seconds": 1,
             "tags": ["colored", "furniture"],
             "inference": {"passes": {"original": {"wall": 19},
                                      "normalized": {"wall": 27654}},
                           "disagreement_fraction": 0.147833},
             "accuracy": {"coverage": {"windows": "geometry"},
                          "metrics": {"windows": {"coverage": "geometry",
                                                  "true_positive": 3,
                                                  "false_positive": 4,
                                                  "false_negative": 2}}}},
            {"id": "unknown", "status": "quality_invalid", "seconds": 2,
             "tags": ["hatching"]},
        ]}
        with tempfile.TemporaryDirectory() as temp:
            write_report(report, Path(temp))
            markdown = (Path(temp) / "report.md").read_text(encoding="utf-8")
        self.assertIn("| colored | colored, furniture | 19/27654 | 14.8% | 3/4/2 | — |", markdown)
        self.assertIn("| unknown | hatching | — | — | — | — |", markdown)

    def test_scoped_geometry_is_labeled_in_report(self):
        report = {"pipeline": "v2", "cases": [
            {"id": "partial", "status": "quality_invalid", "seconds": 1,
             "accuracy": {"coverage": {"windows": "geometry"},
                          "metrics": {"windows": {"coverage": "geometry",
                                                  "scope": "annotation_regions",
                                                  "true_positive": 2,
                                                  "false_positive": 1,
                                                  "false_negative": 1}}}},
        ]}
        with tempfile.TemporaryDirectory() as temp:
            write_report(report, Path(temp))
            markdown = (Path(temp) / "report.md").read_text(encoding="utf-8")
        self.assertIn("geometry 1 category; scoped 1", markdown)
        self.assertIn("2/1/1 (region)", markdown)


if __name__ == "__main__":
    unittest.main()
