import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from dual_pass_inference import fuse_probabilities, inference_diagnostics
from predict_raster import source_mask


class DualPassInferenceTest(unittest.TestCase):
    def test_normalized_baseline_and_supported_original_rescue(self):
        original = torch.zeros(4, 1, 4)
        normalized = torch.zeros(4, 1, 4)
        original[:, 0, 0] = torch.tensor([.1, .8, .1, 0])
        normalized[:, 0, 0] = torch.tensor([.65, .25, .1, 0])
        original[:, 0, 1] = torch.tensor([.05, .1, .8, .05])
        normalized[:, 0, 1] = torch.tensor([.6, .15, .2, .05])
        original[:, 0, 2] = torch.tensor([.1, .1, .75, .05])
        normalized[:, 0, 2] = torch.tensor([.1, .7, .15, .05])
        original[:, 0, 3] = torch.tensor([.1, .8, .1, 0])
        normalized[:, 0, 3] = torch.tensor([.1, .75, .15, 0])
        fused = fuse_probabilities(original, normalized)
        self.assertEqual(fused.argmax(dim=0).tolist(), [[1, 2, 1, 1]])

    def test_original_pass_without_wall_support_is_ignored(self):
        original = torch.zeros(4, 1, 4)
        normalized = torch.zeros(4, 1, 4)
        original[0] = 1
        original[:, 0, 0] = torch.tensor([.05, .05, .85, .05])
        normalized[1] = 1
        self.assertTrue(torch.equal(fuse_probabilities(original, normalized), normalized))

    def test_shapes_and_diagnostics_are_checked(self):
        probabilities = torch.zeros(4, 3, 4)
        probabilities[0] = 1
        with self.assertRaises(ValueError):
            fuse_probabilities(probabilities, torch.zeros(4, 2, 4))
        diagnostics = inference_diagnostics(probabilities, probabilities, probabilities, (1, 1, 2, 2))
        self.assertEqual(diagnostics["disagreement_fraction"], 0)
        self.assertEqual(diagnostics["passes"]["fused"]["floor"], 4)

    def test_model_mask_is_restored_into_source_roi(self):
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        restored = source_mask(mask, (1, 1, 2, 2),
                               {"roi": [1, 1, 3, 3]}, (4, 4))
        expected = np.zeros((4, 4), dtype=np.uint8)
        expected[1:3, 1:3] = 1
        np.testing.assert_array_equal(restored, expected)


if __name__ == "__main__":
    unittest.main()
