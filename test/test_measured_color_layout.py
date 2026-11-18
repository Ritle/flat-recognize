import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from measured_color_layout import glazed_corners, rectangle
from outlined_wall_layout import outlined_wall_layout


class MeasuredColorLayoutTest(unittest.TestCase):
    def drawing(self, arc):
        rgb = np.full((190, 260, 3), 255, np.uint8)
        mask = np.zeros(rgb.shape[:2], np.uint8)
        for a, b in [((12, 12), (245, 19)), ((12, 167), (245, 174)),
                     ((12, 12), (19, 174)), ((238, 12), (245, 174)),
                     ((120, 12), (127, 174))]:
            cv2.rectangle(mask, a, b, 1, -1)
        mask[12:20, 46:98] = mask[82:126, 238:246] = 0
        rgb[mask != 0] = 150
        pen = (32, 164, 180)
        for y in (13, 16, 19):
            cv2.line(rgb, (45, y), (98, y), pen)
        cv2.line(rgb, (197, 82), (245, 82), pen)
        cv2.line(rgb, (238, 126), (245, 126), pen)
        if arc:
            cv2.ellipse(rgb, (241, 82), (44, 44), 0, 90, 180, pen, 1, cv2.LINE_AA)
        # Dimension lines cross both spaces.
        cv2.line(rgb, (4, 95), (253, 95), (10, 10, 10))
        return rgb, mask

    def test_recover_colored_symbols_without_exporting_dimension_lines(self):
        rgb, mask = self.drawing(arc=True)
        walls, bays, info = outlined_wall_layout(rgb, mask)
        self.assertEqual(info["enclosed_regions"], 2)
        self.assertTrue(info["colored_symbols"])
        self.assertCountEqual([b["source_class"] for b in bays],
                              ["measured_glazing", "measured_swing"])
        recovered = np.zeros_like(mask)
        for wall in walls:
            cv2.fillPoly(recovered, [np.int32(wall["outer"])], 1)
            for hole in wall["holes"]:
                cv2.fillPoly(recovered, [np.int32(hole)], 0)
        self.assertFalse(recovered[95, 60])
        rgb, mask = self.drawing(arc=False)
        self.assertIsNone(outlined_wall_layout(rgb, mask))

    def test_glazing_turn_leaves_the_corner_post_solid(self):
        walls = np.zeros((80, 100), np.uint8)
        along = rectangle(10, 80, 60, 69, False)
        across = rectangle(12, 69, 10, 18, True)
        glazed_corners(walls, [along], [across])
        self.assertEqual(along["outer"][0], [18, 60])
        self.assertEqual(across["outer"][2], [18, 60])
        self.assertEqual(walls.sum(), 90)
