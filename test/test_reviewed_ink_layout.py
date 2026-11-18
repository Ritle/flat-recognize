import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from extract_rooms_v2 import wall_mask
from reviewed_ink_layout import measured_frame_at, reviewed_ink_layout


class ReviewedInkTest(unittest.TestCase):
    def test_open_stair_edge_keeps_the_adjoining_door_carrier(self):
        gray = np.full((90, 120), 255, np.uint8)
        gray[12:19, 20:106] = gray[70:78, 20:106] = 0
        gray[18:71, 20:24] = 0
        gray[12:19, 50:76] = 255
        review = {"openings": [{"box": [49, 12, 76, 18], "swing": True}],
                  "empty_regions": [[19, 19, 24, 69]]}
        walls, openings, _ = reviewed_ink_layout(np.repeat(gray[:, :, None], 3, axis=2),
                                                  gray == 0, review)
        mask = wall_mask({"walls": walls}, *gray.shape)
        self.assertFalse(mask[19:70, 19:25].any())
        self.assertTrue(mask[15, 40] and mask[15, 84] and mask[72, 22])
        self.assertEqual(len(openings), 1)
        self.assertEqual(openings[0]["source_class"], "surveyed_door")

    def test_measured_glazing_evidence_serializes_in_prediction_json(self):
        gray = np.full((64, 96), 255, np.uint8)
        gray[20:31, 8:32] = gray[20:31, 62:87] = 0
        gray[[20, 23, 27, 30], 32:62] = 100
        outer = [[31, 20], [62, 20], [62, 30], [31, 30]]
        self.assertEqual(json.loads(json.dumps(measured_frame_at(gray, outer)))["bbox"],
                         [31, 20, 62, 30])
        gray[23, 32:62] = 255
        self.assertIsNone(measured_frame_at(gray, outer))
