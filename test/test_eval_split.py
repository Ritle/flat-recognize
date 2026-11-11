import io
import json
from contextlib import redirect_stdout
import unittest
from unittest.mock import patch

from safetensors.torch import save_file
import torch
import yaml

from buildingcv.eval_split import evaluation_dataset, evaluate_checkpoint
from test import test_raster_training as fixtures


class SplitEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RasterTrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root

    def test_raster_source_selects_only_requested_split_and_records_reviewed_files(self):
        dataset, sources = evaluation_dataset(self.fixture.cfg(), "val", "raster")
        self.assertEqual(len(dataset), 1)
        self.assertEqual(sources[0]["samples"][0]["sample_id"], "sample-val")
        self.assertEqual(sources[0]["count"], 1)
        with self.assertRaisesRegex(ValueError, "No samples"):
            evaluation_dataset(self.fixture.cfg(), "val", "svg")
        for limit in (0, -1, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                evaluation_dataset(self.fixture.cfg(), "val", limit=limit)

    def test_svg_source_honors_derived_lists_and_never_substitutes_failed_sample(self):
        data = self.root / "svg"
        splits = self.root / "splits"
        (data / "one").mkdir(parents=True)
        splits.mkdir()
        (data / "one/model.svg").write_text("<svg/>")
        (splits / "val.txt").write_text("/one/\n")
        cfg = self.fixture.cfg()
        cfg["data"].update(include_cubicasa=True, data_dir=str(data), split_dir=str(splits))
        with patch("buildingcv.data.CubiCasaDataset.load_strict", side_effect=ValueError("broken SVG")):
            dataset, sources = evaluation_dataset(cfg, "val", "svg")
            with self.assertRaisesRegex(ValueError, "broken SVG"):
                dataset[0]
        self.assertTrue(sources[0]["strict"])
        self.assertEqual(sources[0]["samples"][0]["sample"], "one")
        combined, sources = evaluation_dataset(cfg, "val", "all", limit=1)
        self.assertEqual(len(combined), 2)
        self.assertEqual([row["source"] for row in sources], ["svg", "raster"])
        self.assertEqual([row["count"] for row in sources], [1, 1])

    def test_evaluation_ignores_unknown_pixels_preserves_weights_and_refuses_overwrite(self):
        from scripts.prepare_training_annotations import build_masks, read_json, write_json
        annotation_paths = [self.root / f"sample-{split}.json" for split in ("train", "val", "test")]
        annotation = read_json(annotation_paths[1])
        annotation["review"]["coverage"] = "outside_unresolved"
        annotation["unresolved"] = [{"bbox": [20, 1, 30, 8], "question": "uncertain"}]
        write_json(annotation_paths[1], annotation)
        partial = self.root / "partial"
        build_masks(self.root / "inventory.json", annotation_paths, partial,
                    self.root, allow_partial=True)
        cfg = self.fixture.cfg()
        cfg["data"].update(raster_manifest=str(partial / "masks.json"), allow_partial_annotations=True)
        config = self.root / "config.yaml"
        config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        model = torch.nn.Conv2d(3, 4, 1)
        weights = self.root / "weights.safetensors"
        save_file(model.state_dict(), weights)
        before = weights.read_bytes()
        output = self.root / "metrics.json"
        with patch("buildingcv.eval_split.build_model", return_value=model) as factory:
            with redirect_stdout(io.StringIO()):
                result = evaluate_checkpoint(config, weights, output, split="val", source="raster")
        self.assertGreater(result["ignored_pixels"], 0)
        self.assertEqual(result["ignored_pixels"] + result["evaluated_pixels"], 32 * 32)
        self.assertFalse(result["model_training_started"])
        self.assertEqual(json.loads(output.read_text())["checkpoint_sha256"], result["checkpoint_sha256"])
        self.assertEqual(before, weights.read_bytes())
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertIsNone(factory.call_args.kwargs["encoder_weights"])
        with self.assertRaises(FileExistsError):
            evaluate_checkpoint(config, weights, output)


if __name__ == "__main__":
    unittest.main()
