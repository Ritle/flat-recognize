import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from outlined_wall_layout import framed_bays, outlined_wall_layout


class OutlinedWallLayoutTest(unittest.TestCase):
    def test_recover_two_closed_rooms_only_when_model_leaks(self):
        gray = np.full((180, 240), 255, np.uint8)
        network = np.zeros_like(gray)
        for start, end in [
            ((12, 12), (225, 19)), ((12, 160), (225, 167)),
            ((12, 19), (19, 160)), ((218, 19), (225, 160)),
            ((115, 19), (122, 160)),
        ]:
            cv2.rectangle(gray, start, end, 235, -1)
            cv2.rectangle(gray, start, end, 60, 1)
            cv2.rectangle(network, start, end, 1, -1)
        for x in (42, 153):
            cv2.rectangle(gray, (x, 12), (x + 36, 19), 255, -1)
            cv2.rectangle(gray, (x, 12), (x + 36, 19), 140, 1)
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        leaky = network.copy()
        leaky[:20, 42:78] = leaky[:20, 153:189] = 0
        walls, frames, info = outlined_wall_layout(rgb, leaky)
        self.assertTrue(walls)
        self.assertEqual(len(frames), 2)
        self.assertEqual(info["enclosed_regions"], 2)
        self.assertIsNone(outlined_wall_layout(rgb, network))
        fragment = np.zeros_like(network)
        fragment[15, 20:35] = 1
        self.assertIsNone(outlined_wall_layout(rgb, fragment))
        rgb[gray == 60] = 255
        self.assertIsNone(outlined_wall_layout(rgb, leaky))

    def test_compound_bay_preserves_mullion_between_door_and_glazing(self):
        mask = np.zeros((160, 190), np.uint8)
        mask[30:38, 5:40] = mask[30:38, 105:150] = 1
        gray = np.full_like(mask, 255)
        gray[30, 39:106] = gray[37, 39:106] = 130
        gray[30:38, 71:73] = gray[34, 73:106] = 130
        for vertical in (False, True):
            frames = framed_bays(mask.T if vertical else mask, gray.T if vertical else gray, vertical)
            self.assertEqual([frame["glazing_rail"] for frame in frames], [False, True])
            dimension = int(vertical)
            ends = [max(p[dimension] for p in frames[0]["outer"]),
                    min(p[dimension] for p in frames[1]["outer"])]
            self.assertEqual(ends, [70.5, 72.5])
        gray[34, 39:71] = 130
        self.assertEqual(len(framed_bays(mask, gray, False)), 1)
