import argparse
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from prepare_training_annotations import build_masks, read_json, render_annotation, write_json
from buildingcv.labels import IGNORE_ID
from buildingcv.raster_data import RasterMaskDataset
from buildingcv.train import build_datasets, check_data, evaluate, train
from test import test_raster_training as fixtures


class PartialRasterTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RasterTrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.annotation_paths = [self.root / f"sample-{split}.json" for split in ("train", "val", "test")]
        for path in self.annotation_paths[:2]:
            annotation = read_json(path)
            annotation["review"]["coverage"] = "outside_unresolved"
            annotation["unresolved"] = [{"bbox": [20, 1, 30, 8], "question": "uncertain structure"}]
            write_json(path, annotation)

    def build(self):
        output = self.root / "partial"
        build_masks(self.root / "inventory.json", self.annotation_paths, output,
                    self.root, allow_partial=True)
        return output / "masks.json"

    def test_partial_build_requires_opt_in_and_named_review_of_remaining_image(self):
        annotation = read_json(self.annotation_paths[0])
        with self.assertRaisesRegex(ValueError, "allow_partial"):
            render_annotation(annotation, self.fixture.samples[0], self.root)
        with self.assertRaisesRegex(ValueError, "allow_partial"):
            render_annotation(annotation, self.fixture.samples[0], self.root, allow_partial="false")
        mask, overlay = render_annotation(annotation, self.fixture.samples[0], self.root, allow_partial=True)
        self.assertEqual(mask.getpixel((25, 3)), IGNORE_ID)
        self.assertEqual(mask.getpixel((4, 3)), 1)
        self.assertNotEqual(overlay.getpixel((25, 3)), (255, 255, 255))
        annotation["review"]["status"] = "draft"
        with self.assertRaisesRegex(ValueError, "full-image"):
            render_annotation(annotation, self.fixture.samples[0], self.root, allow_partial=True)
        annotation["review"]["status"] = "reviewed"
        annotation["unresolved"] = []
        with self.assertRaisesRegex(ValueError, "explicit unresolved"):
            render_annotation(annotation, self.fixture.samples[0], self.root, allow_partial=True)

    def test_empty_or_unbounded_known_area_is_rejected(self):
        annotation = read_json(self.annotation_paths[0])
        for bbox in ([0, 0, 39, 19], [-1, 0, 10, 10], [0, 0, 40, 20]):
            annotation["unresolved"] = [{"bbox": bbox}]
            with self.subTest(bbox=bbox), self.assertRaises(ValueError):
                render_annotation(annotation, self.fixture.samples[0], self.root, allow_partial=True)

    def test_loader_requires_own_opt_in_and_preserves_ignore_through_resize_and_padding(self):
        manifest = self.build()
        with self.assertRaisesRegex(ValueError, "allow_partial"):
            RasterMaskDataset(manifest, "train", root=self.root)
        dataset = RasterMaskDataset(manifest, "train", root=self.root,
                                    size=(40, 40), allow_partial=True)
        _, mask = dataset[0]
        self.assertEqual(mask[13, 25].item(), IGNORE_ID)
        self.assertEqual(mask[13, 4].item(), 1)
        self.assertTrue(torch.all(mask[:10] == IGNORE_ID))
        self.assertEqual(set(mask.unique().tolist()), {0, 1, 2, IGNORE_ID})

    def test_loader_detects_unmasked_unknown_pixels_even_if_file_hash_is_updated(self):
        manifest = self.build()
        row = read_json(manifest)["samples"][0]
        path = manifest.parent / row["mask"]
        with Image.open(path) as image:
            values = np.array(image)
        values[3, 25] = 0
        Image.fromarray(values).save(path)
        payload = read_json(manifest)
        payload["samples"][0]["mask_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(manifest, payload)
        with self.assertRaisesRegex(ValueError, "do not match unresolved"):
            RasterMaskDataset(manifest, "train", root=self.root, allow_partial=True)

    def test_loss_and_gradient_do_not_depend_on_unknown_predictions(self):
        logits = torch.zeros(1, 4, 1, 2, requires_grad=True)
        truth = torch.tensor([[[1, IGNORE_ID]]])
        loss = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_ID)(logits, truth)
        loss.backward()
        self.assertTrue(torch.all(logits.grad[:, :, :, 1] == 0))
        changed = logits.detach().clone()
        changed[0, 3, 0, 1] = 100
        self.assertEqual(loss.item(), torch.nn.CrossEntropyLoss(ignore_index=IGNORE_ID)(changed, truth).item())

    def test_evaluation_excludes_ignored_pixels_from_loss_accuracy_and_false_positives(self):
        truth = torch.tensor([[[0, 1, 2, 3, IGNORE_ID, IGNORE_ID]]])
        logits = torch.full((1, 4, 1, 6), -10.0)
        for index, cls in enumerate((0, 1, 2, 3, 1, 3)):
            logits[0, cls, 0, index] = 10
        loader = DataLoader(TensorDataset(logits, truth), batch_size=1)
        result = evaluate(torch.nn.Identity(), loader,
                          torch.nn.CrossEntropyLoss(ignore_index=IGNORE_ID), torch.device("cpu"))
        self.assertEqual(result["pixel_acc"], 1)
        self.assertEqual(result["iou"], [1, 1, 1, 1])
        self.assertEqual(result["evaluated_pixels"], 4)
        self.assertEqual(result["ignored_pixels"], 2)
        empty = DataLoader(TensorDataset(logits, torch.full_like(truth, IGNORE_ID)))
        with self.assertRaisesRegex(ValueError, "no reviewed pixels"):
            evaluate(torch.nn.Identity(), empty, torch.nn.CrossEntropyLoss(ignore_index=IGNORE_ID),
                     torch.device("cpu"))

    def test_partial_cpu_training_and_check_data_use_explicit_setting(self):
        manifest = self.build()
        cfg = self.fixture.cfg()
        cfg["data"]["raster_manifest"] = str(manifest)
        with self.assertRaisesRegex(ValueError, "allow_partial"):
            build_datasets(cfg)
        cfg["data"]["allow_partial_annotations"] = True
        args = argparse.Namespace(limit_train=None, limit_val=None, resume=None, epochs=1,
                                  config=self.root / "partial-config.yaml")
        args.config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        stream = io.StringIO()
        with redirect_stdout(stream):
            check_data(cfg, args)
        self.assertIn("ignored=", stream.getvalue())
        tiny_model = torch.nn.Conv2d(3, 4, 1)
        before = tiny_model.weight.detach().clone()
        with patch("buildingcv.model.build_model", return_value=tiny_model):
            with redirect_stdout(io.StringIO()):
                train(cfg, args)
        self.assertFalse(torch.equal(before, tiny_model.weight.detach()))
        self.assertTrue(list((self.root / "runs").glob("*/last.pt")))

    def test_validation_loss_is_weighted_by_reviewed_pixels_across_batches(self):
        truth = torch.tensor([[[0, 1, 2, 3, IGNORE_ID, IGNORE_ID]],
                              [[0, IGNORE_ID, IGNORE_ID, IGNORE_ID, IGNORE_ID, IGNORE_ID]]])
        logits = torch.zeros(2, 4, 1, 6)
        logits[0, 0, 0, 0] = 3
        logits[1, 1, 0, 0] = 5
        dataset = TensorDataset(logits, truth)
        results = [evaluate(torch.nn.Identity(), DataLoader(dataset, batch_size=batch_size),
                            torch.nn.CrossEntropyLoss(ignore_index=IGNORE_ID), torch.device("cpu"))
                   for batch_size in (1, 2)]
        self.assertAlmostEqual(results[0]["loss"], results[1]["loss"], places=6)
        self.assertEqual(results[0]["evaluated_pixels"], 5)
        self.assertEqual(results[0]["ignored_pixels"], 7)
        self.assertEqual(results[0]["pixel_acc"], results[1]["pixel_acc"])


if __name__ == "__main__":
    unittest.main()
