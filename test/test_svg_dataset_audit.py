import unittest
from unittest.mock import patch

import torch

from scripts import audit_svg_dataset as svg_audit


class StrictSvgAuditTests(unittest.TestCase):
    def test_records_requested_sample_and_checks_loader_without_fallback(self):
        class Sample:
            size = (4, 4)

            def load_strict(self, index):
                if index == 1:
                    raise ValueError("bad source SVG")
                return torch.zeros(3, 4, 4), torch.ones(4, 4, dtype=torch.int64)

        with patch.dict(svg_audit._DATASETS, {"train": Sample()}, clear=True):
            good = svg_audit.inspect_sample(("train", 0, "plan-a", "abc"))
            bad = svg_audit.inspect_sample(("train", 1, "plan-b", "def"))
        self.assertEqual(good["status"], "ok")
        self.assertEqual(bad["sample"], "plan-b")
        self.assertEqual(bad["status"], "error")
        self.assertEqual(bad["error_type"], "ValueError")
        self.assertIn("bad source SVG", bad["error"])

    def test_nonfinite_bad_labels_shapes_and_empty_wall_masks_are_not_accepted(self):
        variants = [
            (torch.full((3, 4, 4), float("nan")), torch.ones(4, 4).long()),
            (torch.zeros(3, 4, 4), torch.full((4, 4), 255).long()),
            (torch.zeros(3, 4, 4), torch.zeros(4, 4).long()),
            (torch.zeros(3, 4, 4), torch.ones(4, 4)),
            (torch.zeros(3, 2, 2), torch.ones(4, 4).long()),
        ]
        for tensors in variants:
            class Sample:
                size = (4, 4)

                def load_strict(self, index):
                    return tensors

            with self.subTest(), patch.dict(svg_audit._DATASETS, {"train": Sample()}, clear=True):
                self.assertEqual(svg_audit.inspect_sample(("train", 0, "plan", "abc"))["status"], "error")


if __name__ == "__main__":
    unittest.main()
