import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from prepare_training_annotations import (
    build_masks, checked_polygon, draft_annotation, load_samples, prepare_drafts,
    preview_annotation, read_json, render_annotation, write_json,
)
import prepare_training_annotations as annotation_cli


class TrainingAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        Image.new("RGB", (40, 30), "white").save(self.root / "source.png")
        self.sample = {
            "id": "sample", "image": "source.png", "width": 40, "height": 30,
            "sha256": hashlib.sha256((self.root / "source.png").read_bytes()).hexdigest(),
            "split": "train", "group_id": "family", "group_review": "pending",
        }
        self.manifest = self.root / "samples.json"
        write_json(self.manifest, {"version": 1, "samples": [self.sample]})

    def reviewed(self):
        annotation = draft_annotation(self.sample)
        annotation["review"] = {"status": "reviewed", "coverage": "full", "reviewer": "tester"}
        # Deliberately put the door first; it must overwrite walls anyway.
        annotation["objects"] = [
            {"id": "d1", "class": "door", "polygon": [[10, 2], [14, 2], [14, 5], [10, 5]]},
            {"id": "w1", "class": "wall", "polygon": [[2, 2], [35, 2], [35, 5], [2, 5]]},
            {"id": "v1", "class": "window", "polygon": [[22, 2], [28, 2], [28, 5], [22, 5]]},
        ]
        return annotation

    def test_drafts_never_produce_a_mask_and_cannot_overwrite_edits(self):
        output = self.root / "drafts"
        self.assertEqual(prepare_drafts(self.manifest, output, self.root), 1)
        draft = read_json(output / "sample.annotation.json")
        self.assertEqual(draft["review"]["coverage"], "none")
        self.assertEqual(list(output.glob("*.png")), [])
        with self.assertRaisesRegex(ValueError, "full-image"):
            render_annotation(draft, self.sample, self.root)
        before = (output / "sample.annotation.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_drafts(self.manifest, output, self.root)
        self.assertEqual((output / "sample.annotation.json").read_bytes(), before)

    def test_mask_keeps_source_coordinates_and_fixed_class_priority(self):
        mask, overlay = render_annotation(self.reviewed(), self.sample, self.root)
        self.assertEqual(mask.mode, "L")
        self.assertEqual(mask.size, (40, 30))
        self.assertEqual(mask.getpixel((4, 3)), 1)
        self.assertEqual(mask.getpixel((12, 3)), 2)
        self.assertEqual(mask.getpixel((25, 3)), 3)
        self.assertEqual(mask.getpixel((12, 15)), 0)
        self.assertEqual(set(np.unique(mask)), {0, 1, 2, 3})
        self.assertEqual(overlay.getpixel((12, 15)), (255, 255, 255))
        self.assertNotEqual(overlay.getpixel((25, 3)), (255, 255, 255))

    def test_draft_preview_marks_questions_without_enabling_training(self):
        annotation = self.reviewed()
        annotation["review"] = {"status": "draft", "coverage": "partial", "reviewer": ""}
        annotation["unresolved"] = [{"bbox": [20, 17, 35, 27], "question": "wall or furniture?"}]
        original = copy.deepcopy(annotation)
        overlay = preview_annotation(annotation, self.sample, self.root)
        self.assertEqual(overlay.size, (40, 30))
        self.assertEqual(overlay.getpixel((20, 24)), (220, 0, 0))
        self.assertEqual(annotation, original)
        with self.assertRaisesRegex(ValueError, "full-image"):
            render_annotation(annotation, self.sample, self.root)
        annotation["unresolved"][0]["bbox"] = [0, 0, 40, 20]
        with self.assertRaisesRegex(ValueError, "within"):
            preview_annotation(annotation, self.sample, self.root)

    def test_preview_cli_writes_only_diagnostic_image_and_preserves_existing_output(self):
        annotation = self.reviewed()
        annotation["review"] = {"status": "draft", "coverage": "partial", "reviewer": ""}
        path = self.root / "draft.annotation.json"
        write_json(path, annotation)
        output = self.root / "preview.png"
        argv = ["prepare_training_annotations", "--manifest", str(self.manifest),
                "preview", str(path), "--output", str(output)]
        with patch.object(annotation_cli, "ROOT", self.root), patch.object(sys, "argv", argv):
            annotation_cli.main()
            before = output.read_bytes()
            with self.assertRaises(SystemExit) as error:
                annotation_cli.main()
            self.assertEqual(error.exception.code, 2)
            self.assertEqual(output.read_bytes(), before)
        self.assertFalse(list(self.root.glob("*.mask.png")))
        self.assertEqual(read_json(path)["review"]["status"], "draft")

    def test_incomplete_or_ambiguous_annotations_are_rejected(self):
        for patch in (
            {"unresolved": [{"reason": "shaft or furniture"}]},
            {"objects": []},
            {"review": {"status": "reviewed", "coverage": "partial", "reviewer": "tester"}},
            {"review": {"status": "reviewed", "coverage": "full", "reviewer": " "}},
        ):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                annotation = self.reviewed()
                annotation.update(patch)
                render_annotation(annotation, self.sample, self.root)

    def test_wrong_source_or_annotation_fingerprint_is_rejected(self):
        annotation = self.reviewed()
        annotation["image"]["width"] = 39
        with self.assertRaisesRegex(ValueError, "coordinate system"):
            render_annotation(annotation, self.sample, self.root)
        Image.new("RGB", (40, 30), "black").save(self.root / "source.png")
        with self.assertRaisesRegex(ValueError, "changed"):
            render_annotation(self.reviewed(), self.sample, self.root)

    def test_invalid_geometry_and_unsupported_holes_fail_explicitly(self):
        for polygon in (
            [[-1, 0], [10, 0], [10, 8]],
            [[0, 0], [40, 0], [10, 8]],
            [[0, 0], [1, 1], [2, 2]],
            [[1, 1], [20, 20], [20, 1], [1, 15]],
            [[True, 0], [10, 0], [10, 8]],
            [[0, 0], [float("nan"), 0], [10, 8]],
        ):
            with self.subTest(polygon=polygon), self.assertRaises(ValueError):
                checked_polygon(polygon, 40, 30)
        annotation = self.reviewed()
        annotation["objects"][0]["holes"] = []
        with self.assertRaisesRegex(ValueError, "holes"):
            render_annotation(annotation, self.sample, self.root)

    def test_duplicate_ids_and_unknown_classes_are_rejected(self):
        for change in ("duplicate", "class"):
            annotation = self.reviewed()
            if change == "duplicate":
                annotation["objects"][1]["id"] = "d1"
            else:
                annotation["objects"][0]["class"] = "furniture"
            with self.subTest(change=change), self.assertRaises(ValueError):
                render_annotation(annotation, self.sample, self.root)

    def test_identical_images_and_families_cannot_cross_splits(self):
        for same_family in (True, False):
            extra = copy.deepcopy(self.sample)
            extra.update(id="other", image="other.png", split="val",
                         group_id="family" if same_family else "other-family")
            (self.root / "other.png").write_bytes((self.root / "source.png").read_bytes())
            write_json(self.manifest, {"version": 1, "samples": [self.sample, extra]})
            with self.subTest(same_family=same_family), self.assertRaisesRegex(ValueError, "cross"):
                load_samples(self.manifest, self.root)

    def test_build_validates_all_inputs_before_writing_and_keeps_readiness_separate(self):
        reviewed = self.root / "reviewed.json"
        write_json(reviewed, self.reviewed())
        draft = self.root / "draft.json"
        write_json(draft, draft_annotation(self.sample))
        output = self.root / "masks"
        with self.assertRaises(ValueError):
            build_masks(self.manifest, [draft], output, self.root)
        self.assertFalse(output.exists())
        self.assertEqual(build_masks(self.manifest, [reviewed], output, self.root), 1)
        report = read_json(output / "masks.json")
        self.assertFalse(report["samples"][0]["training_ready"])
        self.assertEqual(report["samples"][0]["split"], "train")
        with Image.open(output / "sample.mask.png") as mask:
            self.assertEqual(mask.getpixel((25, 3)), 3)
        with self.assertRaises(FileExistsError):
            build_masks(self.manifest, [reviewed], output, self.root)


if __name__ == "__main__":
    unittest.main()
