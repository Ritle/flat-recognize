from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image
import torch

from buildingcv.data import CubiCasaDataset
from buildingcv.eval_split import evaluation_dataset
from buildingcv.svg_render import render_input_png
from buildingcv.train import build_datasets

SCENE = '''<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50" viewBox="0 0 100 50">
<rect width="100" height="50" fill="white"/>
<g class="Wall"><rect x="10" y="5" width="10" height="10" fill="black"/></g>
<g class="Door"><rect x="20" y="5" width="4" height="10" fill="brown"/></g>
<g class="Window"><rect x="24" y="5" width="6" height="10" fill="blue"/></g>
<g class="FixedFurniture"><rect x="50" y="10" width="10" height="10" fill="black"/></g>
<g class="TextLabel"><text x="75" y="30" font-size="12">A7</text>
<path d="M80 5 L90 15" stroke="black" stroke-width="2"/></g>
<g class="Wall" style="display:none"><rect x="70" y="35" width="10" height="5" fill="black"/></g>
</svg>'''


class CubiCasaFullSceneTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "scene").mkdir()
        self.svg = self.root / "scene/model.svg"
        self.svg.write_text(SCENE, encoding="utf-8")
        for split in ("train", "val", "test"):
            (self.root / f"{split}.txt").write_text("/scene/\n")

    def test_default_is_unchanged_and_full_view_keeps_furniture_and_annotations(self):
        before = self.svg.read_bytes()
        default = render_input_png(self.svg, (100, 50))
        clean = render_input_png(self.svg, (100, 50), input_mode="structural")
        full = render_input_png(self.svg, (100, 50), input_mode="full")
        self.assertEqual(default, clean)
        with Image.open(BytesIO(clean)) as a, Image.open(BytesIO(full)) as b:
            self.assertEqual(a.getpixel((55, 15)), (255, 255, 255))
            self.assertEqual(b.getpixel((55, 15)), (0, 0, 0))
            self.assertNotEqual(a.getpixel((85, 10)), b.getpixel((85, 10)))
            self.assertEqual(a.getpixel((15, 10)), b.getpixel((15, 10)))
            self.assertEqual(b.getpixel((75, 37)), (255, 255, 255))
        self.assertEqual(before, self.svg.read_bytes())

    def test_furniture_changes_input_but_not_labels_or_letterbox_coordinates(self):
        kwargs = dict(size=(100, 100), normalize=False, letterbox=True)
        clean_image, clean_mask = CubiCasaDataset(self.root, "train", **kwargs).load_strict(0)
        full_image, full_mask = CubiCasaDataset(self.root, "train", input_mode="full", **kwargs)[0]
        self.assertTrue(torch.equal(clean_mask, full_mask))
        self.assertFalse(torch.equal(clean_image, full_image))
        self.assertEqual(full_mask[35, 15].item(), 1)
        self.assertEqual(full_mask[35, 22].item(), 2)
        self.assertEqual(full_mask[35, 27].item(), 3)
        self.assertEqual(full_mask[40, 55].item(), 0)
        self.assertTrue(torch.all(full_image[:, 40, 55] == 0))
        self.assertTrue(torch.equal(full_image[:, :25], clean_image[:, :25]))
        self.assertEqual(full_mask[62, 75].item(), 0)

    def test_full_view_never_substitutes_failed_sample_and_rejects_unknown_modes(self):
        dataset = CubiCasaDataset(self.root, "train", input_mode="full")
        with patch.object(dataset, "_load", side_effect=ValueError("bad requested SVG")) as load:
            with self.assertRaisesRegex(ValueError, "bad requested SVG"):
                dataset[0]
        load.assert_called_once_with(0)
        with self.assertRaisesRegex(ValueError, "input mode"):
            CubiCasaDataset(self.root, "train", input_mode="typo")
        with self.assertRaisesRegex(ValueError, "input mode"):
            render_input_png(self.svg, (100, 50), input_mode="typo")

    def test_training_and_evaluation_select_and_record_the_explicit_mode(self):
        cfg = {"data": dict(data_dir=str(self.root), image_size=[100, 100], normalize=True,
                            letterbox=True, strict_svg=True, svg_input_mode="full")}
        train, val = build_datasets(cfg)
        self.assertEqual((train.dataset.input_mode, val.dataset.input_mode), ("full", "full"))
        dataset, receipts = evaluation_dataset(cfg, "val", "svg")
        self.assertEqual(dataset.dataset.input_mode, "full")
        self.assertEqual(receipts[0]["input_mode"], "full")
        del cfg["data"]["svg_input_mode"]
        default, receipts = evaluation_dataset(cfg, "val", "svg")
        self.assertEqual(default.dataset.input_mode, "structural")
        self.assertNotIn("input_mode", receipts[0])


if __name__ == "__main__":
    unittest.main()
