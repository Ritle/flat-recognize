import json
import unittest

import yaml

from scripts.prepare_training_pilot import prepare
from buildingcv.train import build_datasets
from test import test_raster_training as fixtures


class TrainingPilotTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RasterTrainingTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.root
        self.source = self.root / "svg"
        self.source.mkdir()
        names = [f"{bucket}/{index}" for bucket in ("colorful", "high_quality", "architectural")
                 for index in range(4)]
        for name in names + ["val/one", "test/one"]:
            svg = self.source / name / "model.svg"
            svg.parent.mkdir(parents=True)
            svg.write_text("<svg/>")
        for split, rows in (("train", names), ("val", ["val/one"]), ("test", ["test/one"])):
            (self.source / f"{split}.txt").write_text("".join(f"/{name}/\n" for name in rows))
        cfg = self.fixture.cfg()
        cfg["data"].update(include_cubicasa=True, data_dir=str(self.source), strict_svg=True)
        self.config = self.root / "config.yaml"
        self.config.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    def test_seeded_selection_keeps_all_sources_and_preserves_held_out_files(self):
        one, two = self.root / "pilot-one", self.root / "pilot-two"
        first = prepare(self.config, one, count=6, seed=42)
        second = prepare(self.config, two, count=6, seed=42)
        self.assertEqual(first["selected_base_train"], second["selected_base_train"])
        self.assertEqual(set(first["selected_buckets"].values()), {2})
        self.assertEqual(first["mixed_train"], 7)
        self.assertEqual(first["mixed_val"], 2)
        for split in ("val", "test"):
            self.assertEqual((one / "splits" / f"{split}.txt").read_bytes(),
                             (self.source / f"{split}.txt").read_bytes())
        cfg = yaml.safe_load((one / "config.yaml").read_text())
        train, val = build_datasets(cfg)
        self.assertEqual((len(train), len(val)), (7, 2))
        self.assertEqual(cfg["train"]["seed"], 42)
        self.assertFalse(json.loads((one / "selection.json").read_text())["training_started"])
        with self.assertRaises(FileExistsError):
            prepare(self.config, one, count=6)

    def test_invalid_size_seed_or_split_overlap_fails_before_creating_output(self):
        out = self.root / "bad-pilot"
        for kwargs in ({"count": 0}, {"count": 99}, {"seed": True}, {"seed": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                prepare(self.config, out, **kwargs)
            self.assertFalse(out.exists())
        (self.source / "val.txt").write_text("/colorful/0/\n")
        with self.assertRaisesRegex(ValueError, "crosses"):
            prepare(self.config, out, count=6)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
