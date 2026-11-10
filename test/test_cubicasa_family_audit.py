import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from audit_cubicasa_families import audit, prepare_deduplicated_splits


class CubiCasaFamilyAuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.image = Image.new("RGB", (160, 110), "white")
        draw = ImageDraw.Draw(self.image)
        draw.rectangle((12, 10, 145, 98), outline="black", width=5)
        draw.line((60, 10, 60, 70), fill="black", width=4)
        draw.rectangle((91, 30, 118, 43), outline="black", width=2)
        for index, split in enumerate(("train", "val", "test")):
            path = self.base / f"plan-{index}"
            path.mkdir()
            image = self.image.copy()
            ImageDraw.Draw(image).line((20, 75 + index * 5, 45, 75 + index * 5), fill="black", width=3)
            image.save(path / "F1_original.png")
            (self.base / f"{split}.txt").write_text(f"/plan-{index}/\n")
        self.local = self.root / "local.png"
        self.local.write_bytes((self.base / "plan-2/F1_original.png").read_bytes())
        self.manifest = self.root / "local.json"
        self.manifest.write_text(json.dumps({"samples": [
            {"id": "new-plan", "image": "local.png", "split": "train", "group_review": "pending",
             "sha256": hashlib.sha256(self.local.read_bytes()).hexdigest()}]}))

    def test_checks_all_splits_and_records_exact_match_without_approving(self):
        before = self.manifest.read_bytes()
        report = audit(self.manifest, self.base, self.root / "out", root=self.root,
                       workers=1, nearest=2, cache=self.root / "cache.json")
        self.assertEqual(report["base_count"], 3)
        self.assertEqual(report["exact_local_matches"][0]["base_split"], "test")
        self.assertEqual(report["local_plans"][0]["nearest"][0]["base_sample"], "plan-2")
        self.assertFalse(report["automatic_group_approval"])
        self.assertEqual(self.manifest.read_bytes(), before)
        self.assertTrue((self.root / "out/new-plan.jpg").is_file())
        cached = audit(self.manifest, self.base, self.root / "again", root=self.root,
                       workers=1, nearest=2, cache=self.root / "cache.json")
        self.assertEqual(cached["local_plans"], report["local_plans"])

    def test_rotated_recompressed_image_is_a_visual_candidate(self):
        image = Image.open(self.local).rotate(90, expand=True)
        image.save(self.root / "rotated.jpg", quality=90)
        data = json.loads(self.manifest.read_text())
        data["samples"][0].update(image="rotated.jpg", sha256=hashlib.sha256((self.root / "rotated.jpg").read_bytes()).hexdigest())
        self.manifest.write_text(json.dumps(data))
        report = audit(self.manifest, self.base, self.root / "out", root=self.root, workers=1)
        self.assertEqual(report["exact_local_matches"], [])
        pair = next(r for r in report["local_plans"][0]["suggested_pairs"] if r["base_sample"] == "plan-2")
        self.assertIn(pair["rotation_deg"], (90, 270))

    def test_duplicate_base_files_across_splits_are_reported(self):
        (self.base / "plan-1/F1_original.png").write_bytes((self.base / "plan-0/F1_original.png").read_bytes())
        report = audit(self.manifest, self.base, self.root / "out", root=self.root, workers=1)
        self.assertEqual(len(report["base_cross_split_exact_duplicates"]), 1)
        before = (self.base / "train.txt").read_bytes()
        result = prepare_deduplicated_splits(report, self.root / "splits")
        self.assertEqual(result["excluded_train_samples"], ["plan-0"])
        self.assertEqual((self.base / "train.txt").read_bytes(), before)
        self.assertEqual((self.root / "splits/val.txt").read_bytes(), (self.base / "val.txt").read_bytes())
        # An empty train list here is a property of this tiny test fixture.
        self.assertEqual(result["counts"]["train"], 0)

    def test_changed_base_duplicates_and_val_test_duplicates_are_not_silently_removed(self):
        (self.base / "plan-1/F1_original.png").write_bytes((self.base / "plan-0/F1_original.png").read_bytes())
        report = audit(self.manifest, self.base, self.root / "out", root=self.root, workers=1)
        (self.base / "plan-1/F1_original.png").write_bytes(self.local.read_bytes())
        with self.assertRaisesRegex(ValueError, "changed after"):
            prepare_deduplicated_splits(report, self.root / "splits")
        pair = report["base_cross_split_exact_duplicates"][0]
        pair["splits"] = ["val", "test"]
        with self.assertRaisesRegex(ValueError, "Evaluation splits"):
            prepare_deduplicated_splits(report, self.root / "splits")

    def test_missing_or_repeated_paths_and_changed_local_source_fail(self):
        for split in ("train", "val", "test"):
            original = (self.base / f"{split}.txt").read_text()
            (self.base / f"{split}.txt").write_text("/missing/\n")
            with self.assertRaisesRegex(ValueError, "Missing"):
                audit(self.manifest, self.base, self.root / "out", root=self.root)
            (self.base / f"{split}.txt").write_text(original)
        self.local.write_bytes(self.local.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "Local image changed"):
            audit(self.manifest, self.base, self.root / "out", root=self.root)
        self.assertFalse((self.root / "out").exists())

    def test_bad_local_ids_and_existing_output_are_rejected(self):
        (self.root / "out").mkdir()
        with self.assertRaises(FileExistsError):
            audit(self.manifest, self.base, self.root / "out", root=self.root)
        data = json.loads(self.manifest.read_text())
        data["samples"][0]["id"] = "../escape"
        self.manifest.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "sample ID"):
            audit(self.manifest, self.base, self.root / "other", root=self.root)


if __name__ == "__main__":
    unittest.main()
