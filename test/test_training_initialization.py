import argparse
import hashlib
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from safetensors.torch import load_file, save_file
import torch
import yaml

from buildingcv.checkpoint import initialize_training_weights
from buildingcv.train import train
from test import test_raster_training as fixtures


class WeightInitializationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = torch.nn.Conv2d(3, 4, 1)
        with torch.no_grad():
            self.source.weight.fill_(0.25)
            self.source.bias.fill_(-0.1)
        self.path = self.root / "start.safetensors"
        save_file(self.source.state_dict(), self.path, metadata={"epoch": "99", "best_miou": "0.9"})

    def test_exact_tensor_loading_is_read_only_and_does_not_restore_training_state(self):
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        target = torch.nn.Conv2d(3, 4, 1)
        receipt = initialize_training_weights(target, self.path)
        self.assertTrue(torch.equal(target.weight, self.source.weight))
        self.assertTrue(torch.equal(target.bias, self.source.bias))
        self.assertEqual(receipt["sha256"], digest)
        self.assertEqual(receipt["source_metadata"]["epoch"], "99")
        self.assertFalse(receipt["optimizer_restored"])
        self.assertFalse(receipt["epoch_restored"])
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), digest)

    def test_missing_extra_wrong_shape_dtype_and_nonfinite_values_fail_before_loading(self):
        for kind in ("missing", "extra", "shape", "dtype", "nan"):
            state = {k: v.clone() for k, v in self.source.state_dict().items()}
            if kind == "missing":
                del state["bias"]
            elif kind == "extra":
                state["unknown"] = torch.zeros(1)
            elif kind == "shape":
                state["weight"] = torch.zeros(5, 3, 1, 1)
            elif kind == "dtype":
                state["weight"] = state["weight"].double()
            else:
                state["weight"][0, 0, 0, 0] = float("nan")
            save_file(state, self.path)
            target = torch.nn.Conv2d(3, 4, 1)
            before = target.weight.detach().clone()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                initialize_training_weights(target, self.path)
            self.assertTrue(torch.equal(target.weight, before))
        with self.assertRaisesRegex(ValueError, "safetensors"):
            initialize_training_weights(self.source, self.root / "untrusted.pt")

    def test_new_experiment_starts_epoch_one_and_disables_encoder_download(self):
        fixture = fixtures.RasterTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        cfg = fixture.cfg()
        cfg["train"]["seed"] = 42
        cfg["model"]["encoder_weights"] = "imagenet"
        config = fixture.root / "config.yaml"
        config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
        args = argparse.Namespace(limit_train=None, limit_val=None, resume=None,
                                  init_weights=self.path, epochs=1, config=config)
        target = torch.nn.Conv2d(3, 4, 1)
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        with patch("buildingcv.model.build_model", return_value=target) as factory:
            with redirect_stdout(io.StringIO()):
                train(cfg, args)
        self.assertIsNone(factory.call_args.kwargs["encoder_weights"])
        run = next((fixture.root / "runs").iterdir())
        receipt = json.loads((run / "initialization.json").read_text())
        checkpoint = torch.load(run / "last.pt", map_location="cpu", weights_only=False)
        self.assertEqual(checkpoint["epoch"], 1)
        self.assertIn("configured seed: 42", (run / "train.log").read_text())
        self.assertEqual(receipt["sha256"], digest)
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(), digest)
        self.assertFalse(torch.equal(target.weight, load_file(self.path)["weight"]))
        self.assertTrue(all(int(row["step"]) == 1 for row in checkpoint["optimizer"]["state"].values()))

    def test_resume_and_initialization_conflict_before_any_run_is_created(self):
        args = argparse.Namespace(resume=self.root / "last.pt", init_weights=self.path)
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            train({}, args)


if __name__ == "__main__":
    unittest.main()
