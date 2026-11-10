import argparse
import copy
import hashlib
import io
import subprocess
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw
import torch
from torch.utils.data import ConcatDataset, TensorDataset
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from prepare_training_annotations import build_masks, draft_annotation, read_json, write_json
from buildingcv.raster_data import RasterMaskDataset
from buildingcv.train import build_datasets, check_data, train


class RasterTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.samples, annotations = [], []
        for index, split in enumerate(("train", "val", "test")):
            sid = f"sample-{split}"
            source = Image.new("RGB", (40, 20), (220 + index, 230, 240))
            ImageDraw.Draw(source).rectangle((2, 2, 35, 5), fill="black")
            path = self.root / f"{sid}.png"
            source.save(path)
            sample = {"id": sid, "image": path.name, "width": 40, "height": 20,
                      "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "split": split, "group_id": sid, "group_review": "reviewed"}
            self.samples.append(sample)
            annotation = draft_annotation(sample)
            annotation["review"] = {"status": "reviewed", "coverage": "full", "reviewer": "test"}
            annotation["objects"] = [
                {"id": "wall", "class": "wall", "polygon": [[2, 2], [35, 2], [35, 5], [2, 5]]},
                {"id": "door", "class": "door", "polygon": [[10, 2], [14, 2], [14, 5], [10, 5]]},
                {"id": "window", "class": "window", "polygon": [[22, 2], [28, 2], [28, 5], [22, 5]]},
            ]
            annotation_path = self.root / f"{sid}.json"
            write_json(annotation_path, annotation)
            annotations.append(annotation_path)
        inventory = self.root / "inventory.json"
        write_json(inventory, {"version": 1, "samples": self.samples})
        self.output = self.root / "masks"
        build_masks(inventory, annotations, self.output, self.root)
        self.manifest = self.output / "masks.json"
        self.original = read_json(self.manifest)

    def dataset(self, split="train", **kwargs):
        return RasterMaskDataset(self.manifest, split, root=self.root, size=(40, 40), **kwargs)

    def cfg(self):
        return {
            "data": {"include_cubicasa": False, "raster_manifest": str(self.manifest),
                     "raster_root": str(self.root), "image_size": [32, 32],
                     "normalize": True, "letterbox": True, "num_workers": 0},
            "train": {"device": "cpu", "epochs": 1, "log_every": 1},
            "optim": {"lr": 0.001, "weight_decay": 0, "batch_size": 1},
            "model": {"encoder_name": "unused", "encoder_weights": None},
            "output": {"run_dir": str(self.root / "runs")},
        }

    def rewrite_mask(self, image):
        path = self.output / "sample-train.mask.png"
        image.save(path)
        manifest = copy.deepcopy(self.original)
        manifest["samples"][0]["mask_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(self.manifest, manifest)

    def test_letterbox_alignment_normalization_and_nearest_labels(self):
        dataset = self.dataset(normalize=True)
        image, mask = dataset[0]
        self.assertEqual(tuple(image.shape), (3, 40, 40))
        self.assertEqual(mask.dtype, torch.int64)
        self.assertTrue(torch.all(image[:, :10] == 0))
        self.assertTrue(torch.all(mask[:10] == 0))
        self.assertEqual(mask[13, 12].item(), 2)
        self.assertEqual(mask[13, 25].item(), 3)
        self.assertEqual(mask[13, 4].item(), 1)
        raw, _ = self.dataset(normalize=False)[0]
        self.assertTrue(torch.allclose(raw[:, 0, 0], torch.tensor([0.485, 0.456, 0.406])))
        self.assertTrue(torch.allclose(image[:, 13, 4],
                                      -torch.tensor([0.485, 0.456, 0.406]) /
                                      torch.tensor([0.229, 0.224, 0.225])))
        resized = RasterMaskDataset(self.manifest, "train", root=self.root,
                                    size=(31, 37), letterbox=False)[0][1]
        self.assertEqual(set(resized.unique().tolist()), {0, 1, 2, 3})

    def test_only_selected_split_is_loaded_and_pending_data_is_not_silently_skipped(self):
        self.assertEqual(self.dataset().samples[0]["sample_id"], "sample-train")
        changed = copy.deepcopy(self.original)
        changed["samples"][2]["training_ready"] = False
        write_json(self.manifest, changed)
        self.assertEqual(len(self.dataset()), 1)
        with self.assertRaisesRegex(ValueError, "training-ready"):
            self.dataset("test")
        changed["samples"][0]["training_ready"] = False
        write_json(self.manifest, changed)
        with self.assertRaisesRegex(ValueError, "training-ready"):
            self.dataset()

    def test_leaked_family_or_identical_source_is_rejected_across_any_splits(self):
        for kind in ("family", "hash"):
            changed = copy.deepcopy(self.original)
            if kind == "family":
                changed["samples"][2]["group_id"] = changed["samples"][1]["group_id"]
            else:
                changed["samples"][2]["image"]["sha256"] = changed["samples"][1]["image"]["sha256"]
            write_json(self.manifest, changed)
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "crosses splits"):
                self.dataset()

    def test_changed_image_mask_and_annotation_are_rejected(self):
        dataset = self.dataset()
        for name in ("sample-train.mask.png", "sample-train.annotation.json"):
            path = self.output / name
            old = path.read_bytes()
            path.write_bytes(old + b" ")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "changed"):
                dataset[0]
            path.write_bytes(old)
        source = self.root / "sample-train.png"
        source.write_bytes(source.read_bytes() + b" ")
        with self.assertRaisesRegex(ValueError, "changed"):
            dataset[0]

    def test_wrong_mask_dimensions_rgb_unknown_labels_and_empty_walls_are_rejected(self):
        for mask in (Image.new("L", (40, 19), 1), Image.new("RGB", (40, 20), "black"),
                     Image.new("L", (40, 20), 255), Image.new("L", (40, 20), 0)):
            self.rewrite_mask(mask)
            with self.subTest(mode=mask.mode, size=mask.size), self.assertRaises(ValueError):
                self.dataset()

    def test_draft_cannot_be_enabled_by_only_flipping_manifest_readiness(self):
        path = self.output / "sample-train.annotation.json"
        annotation = read_json(path)
        annotation["review"]["status"] = "draft"
        write_json(path, annotation)
        changed = copy.deepcopy(self.original)
        changed["samples"][0]["annotation_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(self.manifest, changed)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.dataset()

    def test_mixed_datasets_keep_svg_default_and_never_train_on_test_rows(self):
        cfg = self.cfg()
        cfg["data"].update(include_cubicasa=True, data_dir="unused")
        old = TensorDataset(torch.zeros(2, 3, 32, 32), torch.zeros(2, 32, 32).long())
        with patch("buildingcv.train.CubiCasaDataset", return_value=old) as factory:
            training, validation = build_datasets(cfg)
        self.assertIsInstance(training, ConcatDataset)
        self.assertEqual((len(training), len(validation)), (3, 3))
        self.assertEqual([c.args[1] for c in factory.call_args_list], ["train", "val"])
        self.assertEqual(training.datasets[1].samples[0]["split"], "train")
        with patch("buildingcv.train.CubiCasaDataset", return_value=old):
            selected, _ = build_datasets(cfg, limit_train=1, limit_val=1, limits_per_source=True)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(selected.datasets[0]), 1)
        self.assertEqual(len(selected.datasets[1]), 1)
        self.assertEqual(selected.datasets[1].samples[0]["split"], "train")
        cfg["data"].pop("raster_manifest")
        cfg["data"].pop("include_cubicasa")
        with patch("buildingcv.train.CubiCasaDataset", return_value=old):
            self.assertIs(build_datasets(cfg)[0], old)

    def test_check_data_cli_works_without_model_or_run_settings(self):
        config = self.root / "check-only.yaml"
        config.write_text(yaml.safe_dump({"data": self.cfg()["data"]}), encoding="utf-8")
        result = subprocess.run(
            [sys.executable, "-m", "buildingcv.train", "--config", str(config), "--check-data"],
            capture_output=True, text=True, timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("train: 1 samples", result.stdout)
        self.assertIn("val: 1 samples", result.stdout)
        self.assertNotIn("test:", result.stdout)
        self.assertFalse((self.root / "runs").exists())

    def test_check_data_creates_no_run_and_cpu_training_saves_snapshot(self):
        cfg = self.cfg()
        args = argparse.Namespace(limit_train=None, limit_val=None, resume=None, epochs=1,
                                  config=self.root / "config.yaml")
        args.config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        captured = io.StringIO()
        with redirect_stdout(captured):
            check_data(cfg, args)
        self.assertIn("train: 1 samples", captured.getvalue())
        self.assertIn("val: 1 samples", captured.getvalue())
        self.assertFalse((self.root / "runs").exists())
        tiny_model = torch.nn.Conv2d(3, 4, 1)
        before = tiny_model.weight.detach().clone()
        with patch("buildingcv.model.build_model", return_value=tiny_model):
            with redirect_stdout(io.StringIO()):
                train(cfg, args)
        self.assertFalse(torch.equal(before, tiny_model.weight.detach()))
        run = next((self.root / "runs").iterdir())
        self.assertTrue((run / "last.pt").is_file())
        self.assertEqual((run / "raster_masks.json").read_bytes(), self.manifest.read_bytes())
        args.resume = run / "last.pt"
        args.epochs = 2
        with patch("buildingcv.model.build_model", return_value=tiny_model):
            with redirect_stdout(io.StringIO()):
                train(cfg, args)
        self.assertEqual(len((run / "metrics.csv").read_text().splitlines()), 3)
        self.manifest.write_bytes(self.manifest.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "dataset changed"):
            train(cfg, args)


if __name__ == "__main__":
    unittest.main()
