"""PyTorch Dataset for CubiCasa5K. Image is the SVG rasterized via cairosvg, not F1_*.png (those don't share the SVG's coordinate space)."""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .labels import FLOOR_ID
from .svg_render import render_input_png
from .svg_to_mask import svg_to_mask

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _read_split(data_dir: Path, split: str, split_dir: str | Path | None = None) -> list[str]:
    """Return cleaned relative sample paths from train.txt / val.txt / test.txt."""
    p = (Path(split_dir) if split_dir is not None else data_dir) / f"{split}.txt"
    return [line.strip().strip("/") for line in p.read_text().splitlines() if line.strip()]


def _read_svg_dims(svg_path: Path) -> tuple[float, float]:
    """Return the SVG's native (width, height) in user units."""
    root = ET.parse(svg_path).getroot()
    viewbox = (root.get("viewBox") or "").split()
    w = float(root.get("width") or (viewbox[2] if len(viewbox) == 4 else 0))
    h = float(root.get("height") or (viewbox[3] if len(viewbox) == 4 else 0))
    if w <= 0 or h <= 0:
        raise ValueError(f"could not determine SVG size for {svg_path}")
    return w, h


class CubiCasaDataset(Dataset):
    """Floor-plan segmentation dataset. Returns (image_tensor, mask_tensor) per item.

    Args:
        data_dir: path to the unzipped `cubicasa5k/` folder.
        split: one of "train", "val", "test".
        size: target (H, W) in pixels. Both image and mask are returned at this size.
        normalize: if True, image is normalized with ImageNet stats (matches what
            ImageNet-pretrained backbones expect).
        letterbox: if True, render the SVG at its native aspect ratio scaled to
            fit inside (H, W), then center-pad to (H, W) — image padded with the
            ImageNet mean (zero in normalized space), mask padded with `floor`.
            If False (default), the SVG is rendered directly at (H, W), which
            stretches non-square plans.
        split_dir: optional directory containing derived train/val/test lists;
            images and SVG annotations still come from data_dir.

    Each item:
        image: float32 tensor of shape (3, H, W). Channel-first, in [0, 1] or
            ImageNet-normalized depending on `normalize`.
        mask:  int64 tensor of shape (H, W). Class indices in [0, NUM_CLASSES).
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str,
        size: tuple[int, int] = (384, 384),
        normalize: bool = True,
        letterbox: bool = False,
        split_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.size = size  # (H, W)
        self.normalize = normalize
        self.letterbox = letterbox
        self.samples = _read_split(self.data_dir, split, split_dir)

        if normalize:
            self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

        # Per-worker memo of indices whose SVG cairosvg can't render. CubiCasa5K
        # has a small number of these (rotate(NaN) etc.); we skip past them in
        # __getitem__ and log each path the first time we hit it.
        self._bad_seen: set[int] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def load_strict(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Read the requested sample without substituting a later valid SVG."""
        return self._load(idx)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        n = len(self.samples)
        for offset in range(n):
            i = (idx + offset) % n
            try:
                return self._load(i)
            except Exception as e:
                if i not in self._bad_seen:
                    self._bad_seen.add(i)
                    print(
                        f"[CubiCasaDataset] skipping bad sample {self.samples[i]}: {e}",
                        file=sys.stderr,
                    )
        raise RuntimeError("CubiCasaDataset: every sample failed to load")

    def _load(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rel = self.samples[idx]
        svg_path = self.data_dir / rel / "model.svg"
        H, W = self.size

        if self.letterbox:
            svg_w, svg_h = _read_svg_dims(svg_path)
            scale = min(W / svg_w, H / svg_h)
            inner_w = max(1, int(round(svg_w * scale)))
            inner_h = max(1, int(round(svg_h * scale)))
        else:
            inner_w, inner_h = W, H

        png_bytes = render_input_png(svg_path, out_size=(inner_w, inner_h))
        img = np.array(Image.open(BytesIO(png_bytes)).convert("RGB"))
        mask = svg_to_mask(svg_path, size=(inner_w, inner_h))

        image_t = torch.from_numpy(img).permute(2, 0, 1).contiguous().float().div_(255.0)
        if self.normalize:
            image_t = (image_t - self._mean) / self._std
        mask_t = torch.from_numpy(mask).long()

        if self.letterbox and (inner_h, inner_w) != (H, W):
            top = (H - inner_h) // 2
            left = (W - inner_w) // 2
            if self.normalize:
                canvas_img = torch.zeros(3, H, W)
            else:
                canvas_img = torch.tensor(IMAGENET_MEAN).view(3, 1, 1).expand(3, H, W).clone()
            canvas_mask = torch.full((H, W), FLOOR_ID, dtype=torch.long)
            canvas_img[:, top:top + inner_h, left:left + inner_w] = image_t
            canvas_mask[top:top + inner_h, left:left + inner_w] = mask_t
            return canvas_img, canvas_mask

        return image_t, mask_t
