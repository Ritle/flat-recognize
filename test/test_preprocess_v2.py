import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from preprocess_v2 import model_point_to_source, preprocess


class PreprocessTests(unittest.TestCase):
    def test_faded_plan_keeps_thin_detail_and_source_coordinates(self):
        source = Image.new("RGB", (500, 400), (252, 252, 252))
        draw = ImageDraw.Draw(source)
        draw.rectangle((60, 60, 440, 340), outline=(213, 213, 213), width=8)
        draw.line((100, 150, 160, 150), fill=(225, 225, 225), width=1)
        normalized, _, meta = preprocess(source, light_fill_cleanup=True)
        x, y, _, _ = meta["roi"]
        self.assertEqual(meta["diagnostics"]["normalization"], "faint_scan_contrast")
        self.assertFalse(meta["diagnostics"]["light_fill_applied"])
        self.assertLess(normalized.getpixel((62-x, 120-y))[0], 40)
        self.assertLess(normalized.getpixel((130-x, 150-y))[0], 125)
        self.assertGreater(normalized.getpixel((130-x, 153-y))[0], 235)

    def test_faint_noise_paper_shading_and_single_rule_stay_unchanged(self):
        gradient = np.tile(np.linspace(208, 255, 360, dtype=np.uint8), (300, 1))
        noise = np.random.default_rng(1).integers(208, 256, (300,360), dtype=np.uint8)
        ruled = np.full((300,360), 252, np.uint8)
        ruled[100:107] = 212
        for gray in (gradient, noise, ruled):
            normalized, _, meta = preprocess(Image.fromarray(gray))
            self.assertEqual(meta["diagnostics"]["normalization"], "original")
            x, y, right, bottom = meta["roi"]
            np.testing.assert_array_equal(np.asarray(normalized)[:,:,0], gray[y:bottom,x:right])

    def test_light_floor_fill_does_not_become_wall(self):
        image = Image.new("RGB", (500, 400), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((70, 60, 430, 320), fill=(245, 220, 155), outline="black", width=7)
        draw.line((250, 60, 250, 320), fill="black", width=5)
        draw.line((100, 100, 150, 100), fill="black", width=1)
        draw.line((70, 365, 430, 365), fill="black", width=1)
        normalized, _, meta = preprocess(image, light_fill_cleanup=True)
        self.assertEqual(meta["diagnostics"]["normalization"], "light_fill_neutralization")
        x, y, _, bottom = meta["roi"]
        self.assertLess(bottom, 365)
        self.assertEqual(normalized.getpixel((170-x, 150-y)), (255, 255, 255))
        self.assertEqual(normalized.getpixel((120-x, 100-y)), (0, 0, 0))

    def test_letterbox_coordinates_restore_crop_offset_and_actual_size(self):
        meta = {"roi": [180, 220, 850, 850], "source_size": [1000, 1200]}
        rect = (17, 25, 511, 320)
        self.assertEqual(model_point_to_source((17, 25), rect, meta), [180, 220])
        self.assertEqual(model_point_to_source((528, 345), rect, meta), [850, 850])
        self.assertEqual(model_point_to_source((272.5, 185), rect, meta), [515, 535])

    def test_clean_tightly_framed_image_is_preserved(self):
        image = Image.new("RGB", (400, 300), "white")
        ImageDraw.Draw(image).rectangle((12, 12, 387, 287), outline="black", width=10)
        normalized, _, meta = preprocess(image)
        self.assertEqual(meta["roi"], [0, 0, 400, 300])
        self.assertTrue(np.array_equal(np.asarray(normalized), np.asarray(image)))

    def test_crop_keeps_plan_and_ignores_separate_footer_and_logo(self):
        image = Image.new("RGB", (1000, 1200), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((180, 220, 850, 850), outline="black", width=15)
        draw.line((510, 220, 510, 850), fill="black", width=10)
        draw.rectangle((30, 30, 120, 100), outline="black", width=7)
        draw.text((240, 1120), "VectorStock footer www.example.test", fill="black")
        _, _, meta = preprocess(image)
        x1, y1, x2, y2 = meta["roi"]
        self.assertLessEqual(x1, 180)
        self.assertLessEqual(y1, 220)
        self.assertGreater(x2, 850)
        self.assertGreater(y2, 850)
        self.assertGreater(y1, 100)
        self.assertLess(y2, 1120)
        self.assertEqual(meta["processed_size"], [x2 - x1, y2 - y1])

    def test_colored_wall_becomes_dark_without_erasing_thin_detail(self):
        image = Image.new("RGB", (400, 300), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((12, 12, 387, 287), outline=(180, 90, 20), width=10)
        draw.line((100, 100, 150, 100), fill=(40, 40, 40), width=1)
        normalized, _, meta = preprocess(image)
        self.assertEqual(meta["diagnostics"]["normalization"], "min_channel_contrast")
        self.assertLess(normalized.getpixel((15, 100))[0], 40)
        self.assertLess(normalized.getpixel((120, 100))[0], 80)
        self.assertEqual(normalized.getpixel((200, 200)), (255, 255, 255))

    def test_blank_and_ambiguous_images_keep_full_frame(self):
        blank = Image.new("RGB", (800, 500), "white")
        _, _, meta = preprocess(blank)
        self.assertEqual(meta["roi"], [0, 0, 800, 500])
        draw = ImageDraw.Draw(blank)
        draw.rectangle((50, 150, 350, 400), outline="black", width=10)
        draw.rectangle((450, 150, 750, 400), outline="black", width=10)
        _, _, meta = preprocess(blank)
        self.assertEqual(meta["roi"], [0, 0, 800, 500])
        self.assertEqual(meta["diagnostics"]["fallback"], "ambiguous_structural_regions")


if __name__ == "__main__":
    unittest.main()
