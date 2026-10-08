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
        self.assertIn("1/2 cases with confirmed counts; 0/2 with geometric annotations", markdown)
        self.assertIn("counts 1/2; geometry 0", markdown)
        self.assertIn("[counts 1/2; geometry 0](labeled/accuracy.json)", markdown)


if __name__ == "__main__":
    unittest.main()
