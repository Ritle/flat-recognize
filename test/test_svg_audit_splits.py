import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_svg_audit_splits import prepare
from buildingcv.train import StrictSvgDataset


class SvgAuditSplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        self.rows, self.hashes = [], {}
        for split, names in (("train", ["good", "bad"]), ("val", ["val"]), ("test", ["test"])):
            path = self.data / f"{split}.txt"
            path.write_text("".join(f"/{name}/\n" for name in names))
            self.hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
            for name in names:
                source = self.data / name / "model.svg"
                source.parent.mkdir()
                source.write_text("<svg/>")
                self.rows.append({"split": split, "sample": name,
                                  "status": "error" if name in ("bad", "test") else "ok",
                                  "svg_sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
        self.report = self.root / "audit.json"
        self.write_report()

    def write_report(self):
        self.report.write_text(json.dumps({"settings": {"dataset": str(self.data)},
                                         "samples": self.rows, "total": len(self.rows),
                                         "substituted_samples": 0, "source_split_sha256": self.hashes}))

    def test_only_training_failures_are_removed_and_test_problem_stays_visible(self):
        before = {split: (self.data / f"{split}.txt").read_bytes() for split in self.hashes}
        out = self.root / "selected"
        result = prepare(self.report, self.data, out)
        self.assertEqual(result["excluded_train_samples"], ["bad"])
        self.assertEqual(result["known_test_failures_retained"], ["test"])
        self.assertEqual(result["input_mode"], "structural")
        self.assertEqual(result["counts"], {"train": 1, "val": 1, "test": 1})
        self.assertEqual((out / "val.txt").read_bytes(), before["val"])
        self.assertEqual((out / "test.txt").read_bytes(), before["test"])
        self.assertEqual((self.data / "train.txt").read_bytes(), before["train"])

    def test_failed_validation_or_changed_svg_cannot_be_hidden_by_preparation(self):
        self.rows[2]["status"] = "error"
        self.write_report()
        out = self.root / "selected"
        with self.assertRaisesRegex(ValueError, "Validation contains"):
            prepare(self.report, self.data, out)
        self.assertFalse(out.exists())
        self.rows[2]["status"] = "ok"
        self.write_report()
        (self.data / "good/model.svg").write_text("<svg changed='yes'/>")
        with self.assertRaisesRegex(ValueError, "SVG changed"):
            prepare(self.report, self.data, out)

    def test_strict_training_view_never_calls_substitution_loader(self):
        class Source:
            def __len__(self):
                return 1

            def __getitem__(self, index):
                raise AssertionError("Fallback loader must not be called")

            def load_strict(self, index):
                raise ValueError("Original bad SVG")

        strict = StrictSvgDataset(Source())
        self.assertEqual(len(strict), 1)
        with self.assertRaisesRegex(ValueError, "Original bad SVG"):
            strict[0]


if __name__ == "__main__":
    unittest.main()
