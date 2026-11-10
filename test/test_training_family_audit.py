import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from audit_training_families import audit, distance, signatures


class FamilyAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.image = Image.new("RGB", (230, 170), "white")
        draw = ImageDraw.Draw(self.image)
        draw.rectangle((20, 15, 200, 145), outline="black", width=7)
        draw.line((90, 15, 90, 110), fill="black", width=5)
        draw.line((90, 80, 200, 80), fill="black", width=5)
        draw.rectangle((128, 109, 149, 127), outline="black", width=2)
        self.image.save(self.root / "original.png")
        self.training = self.root / "training.json"
        self.regression = self.root / "regression.json"

    def sample(self, sid, file, split, group=None):
        return {"id": sid, "image": file, "split": split, "group_id": group or sid,
                "sha256": hashlib.sha256((self.root / file).read_bytes()).hexdigest()}

    def run_audit(self, samples, cases=None):
        self.training.write_text(json.dumps({"samples": samples}), encoding="utf-8")
        self.regression.write_text(json.dumps({"cases": cases or []}), encoding="utf-8")
        before = self.training.read_bytes()
        result = audit(self.training, self.regression, self.root, self.root / "audit")
        self.assertEqual(self.training.read_bytes(), before)
        return result

    def test_recompressed_resized_rotated_and_margin_variants_are_candidates(self):
        rotated = self.image.rotate(90, expand=True)
        canvas = Image.new("RGB", (rotated.width + 60, rotated.height + 40), "white")
        canvas.paste(rotated, (30, 20))
        canvas.resize((canvas.width * 2, canvas.height * 2)).save(self.root / "variant.jpg", quality=90)
        result = self.run_audit([self.sample("first", "original.png", "train"),
                                 self.sample("second", "variant.jpg", "val")])
        pair = result["candidate_pairs"][0]
        self.assertFalse(pair["exact"])
        self.assertTrue(pair["requires_manual_review"])
        self.assertEqual(pair["second_rotation_deg"], 270)
        self.assertEqual(result["conflicts"], [])
        self.assertFalse(result["automatic_group_approval"])
        self.assertFalse(result["base_training_dataset_checked"])
        self.assertTrue((self.root / "audit/contact-sheet.jpg").exists())

    def test_exact_copies_across_train_and_test_are_conflicts(self):
        (self.root / "copy.png").write_bytes((self.root / "original.png").read_bytes())
        result = self.run_audit([self.sample("first", "original.png", "train"),
                                 self.sample("second", "copy.png", "test")])
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertTrue(result["candidate_pairs"][0]["exact"])

    def test_known_family_cannot_cross_splits_even_when_rasters_differ(self):
        Image.new("RGB", (100, 100), "black").save(self.root / "different.png")
        result = self.run_audit([self.sample("first", "original.png", "train", "same"),
                                 self.sample("second", "different.png", "val", "same")])
        self.assertEqual(len(result["conflicts"]), 1)
        self.assertTrue(result["candidate_pairs"][0]["same_group"])

    def test_validation_source_already_in_development_is_flagged(self):
        result = self.run_audit([self.sample("first", "original.png", "val")],
                                [{"id": "old-case", "image": "original.png"}])
        self.assertEqual(len(result["images"]), 1)
        self.assertEqual(result["conflicts"][0]["kind"], "evaluation_image_used_in_development")

    def test_same_split_copies_are_diagnostic_and_outputs_cannot_be_overwritten(self):
        (self.root / "copy.png").write_bytes((self.root / "original.png").read_bytes())
        result = self.run_audit([self.sample("first", "original.png", "train"),
                                 self.sample("second", "copy.png", "train")])
        self.assertEqual(result["conflicts"], [])
        before = (self.root / "audit/report.json").read_bytes()
        with self.assertRaises(FileExistsError):
            audit(self.training, self.regression, self.root, self.root / "audit")
        self.assertEqual((self.root / "audit/report.json").read_bytes(), before)

    def test_changed_source_fails_before_creating_a_report(self):
        sample = self.sample("first", "original.png", "train")
        sample["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "source changed"):
            self.run_audit([sample])
        self.assertFalse((self.root / "audit").exists())

    def test_signature_is_stable_for_uniform_added_margins(self):
        padded = Image.new("RGB", (400, 350), "white")
        padded.paste(self.image, (40, 50))
        self.assertEqual(distance(signatures(self.image), signatures(padded))[:2], (0, 0))


if __name__ == "__main__":
    unittest.main()
