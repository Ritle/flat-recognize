from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import ConcatDataset, Subset, TensorDataset

from buildingcv.data import CubiCasaDataset
from buildingcv.train import checked_item


class CubiCasaSplitSelectionTests(unittest.TestCase):
    def test_derived_split_keeps_data_root_and_original_default(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "dataset"
            root.mkdir()
            (root / "train.txt").write_text("/original/\n/duplicate/\n")
            derived = Path(folder) / "derived"
            derived.mkdir()
            (derived / "train.txt").write_text("/original/\n")
            original = CubiCasaDataset(root, "train")
            selected = CubiCasaDataset(root, "train", split_dir=derived)
            self.assertEqual(original.samples, ["original", "duplicate"])
            self.assertEqual(selected.samples, ["original"])
            self.assertEqual(selected.data_dir, root)

    def test_preflight_preserves_selected_index_and_never_substitutes_a_bad_svg(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "train.txt").write_text("/first/\n/bad/\n")
            ds = CubiCasaDataset(root, "train")
            prefix = TensorDataset(torch.zeros(1, 3, 2, 2), torch.zeros(1, 2, 2))
            mixed = ConcatDataset([prefix, Subset(ds, [1])])
            with patch.object(ds, "_load", side_effect=ValueError("bad selected SVG")) as strict:
                with self.assertRaisesRegex(ValueError, "bad selected SVG"):
                    checked_item(mixed, 1)
                strict.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()
