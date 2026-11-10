"""Strict loader for reviewed source-coordinate PNG segmentation masks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .data import IMAGENET_MEAN, IMAGENET_STD
from .labels import CLASS_TO_ID, FLOOR_ID, IGNORE_ID, NUM_CLASSES
from .raster_annotations import reviewed_ignore_mask


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verified_bytes(path, expected):
    if (not isinstance(expected, str) or len(expected) != 64 or
            any(c not in "0123456789abcdef" for c in expected)):
        raise ValueError(f"Missing or malformed SHA-256: {path}")
    data = Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"File changed since annotation review: {path}")
    return data


class RasterMaskDataset(Dataset):
    """Read masks.json produced by prepare_training_annotations.py.

    Image paths are relative to root; masks and annotations are relative to
    the mask manifest. Only the requested split is loaded. Metadata across
    all splits is inspected for leaked families or duplicate source hashes.
    Unreviewed rows in the requested split fail, rather than being skipped.
    """

    def __init__(self, manifest, split, *, root=".", size=(512, 512),
                 normalize=True, letterbox=True, allow_partial=False):
        if split not in ("train", "val", "test"):
            raise ValueError("Raster split must be train, val or test")
        if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError("Raster image_size must contain two positive integers")
        self.manifest = Path(manifest).resolve()
        self.manifest_sha256 = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
        self.root = Path(root).resolve()
        self.size = tuple(size)
        self.normalize = normalize
        self.letterbox = letterbox
        self.split = split
        self.allow_partial = allow_partial
        self._mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        payload = _read_json(self.manifest)
        if (not isinstance(payload, dict) or payload.get("version") != 1 or
                payload.get("classes") != CLASS_TO_ID or
                not isinstance(payload.get("samples"), list)):
            raise ValueError("Expected reviewed masks.json with the four-class taxonomy")
        self.samples = []
        ids, image_splits, groups = set(), {}, {}
        for row in payload["samples"]:
            if not isinstance(row, dict):
                raise ValueError("Raster sample must be an object")
            sid = row.get("sample_id")
            if (not isinstance(sid, str) or not sid or sid in ids or
                    any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in sid)):
                raise ValueError("Invalid or duplicate raster sample ID")
            ids.add(sid)
            group, part, image = row.get("group_id"), row.get("split"), row.get("image")
            if (not isinstance(group, str) or not group or part not in ("train", "val", "test") or
                    not isinstance(image, dict) or not isinstance(image.get("sha256"), str)):
                raise ValueError(f"Missing image, family or split metadata: {sid}")
            for seen, key in ((groups, group), (image_splits, image["sha256"])):
                if key in seen and seen[key] != part:
                    raise ValueError(f"Raster family or identical image crosses splits: {sid}")
                seen[key] = part
            if part == split:
                if row.get("training_ready") is not True or row.get("group_review") != "reviewed":
                    raise ValueError(f"Raster sample is not reviewed and training-ready: {sid}")
                self.samples.append(row)
        if not self.samples:
            raise ValueError(f"No reviewed raster samples in split {split}")
        # Fail before a model/run is created, including for malformed masks.
        for row in self.samples:
            self._load_pair(row)

    def __len__(self):
        return len(self.samples)

    def _load_pair(self, row):
        from io import BytesIO

        sid, meta = row["sample_id"], row["image"]
        image_path = self.root / meta["image"]
        mask_path = self.manifest.parent / row["mask"]
        annotation_path = self.manifest.parent / f"{sid}.annotation.json"
        annotation = json.loads(_verified_bytes(annotation_path, row.get("annotation_sha256")))
        if not isinstance(annotation, dict):
            raise ValueError(f"Raster annotation must be an object: {sid}")
        review = annotation.get("review", {})
        if (annotation.get("version") != 1 or annotation.get("sample_id") != sid or
                annotation.get("image") != meta or not isinstance(review, dict) or
                review.get("status") != "reviewed" or
                not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip() or
                not isinstance(annotation.get("objects"), list) or not annotation["objects"]):
            raise ValueError(f"Raster annotation is incomplete or does not match source: {sid}")
        with Image.open(BytesIO(_verified_bytes(image_path, meta.get("sha256")))) as image:
            if (image.size != (meta.get("width"), meta.get("height")) or
                    image.getexif().get(274, 1) != 1):
                raise ValueError(f"Raster source dimensions or orientation changed: {sid}")
            source = image.convert("RGB")
        ignored = reviewed_ignore_mask(annotation, source.size, allow_partial=self.allow_partial)
        if review["coverage"] == "outside_unresolved":
            if row.get("coverage") != "outside_unresolved" or row.get("ignore_index") != IGNORE_ID:
                raise ValueError(f"Partial raster metadata does not match annotation: {sid}")
        elif row.get("coverage", "full") != "full" or row.get("ignore_index") is not None:
            raise ValueError(f"Full raster metadata does not match annotation: {sid}")
        with Image.open(BytesIO(_verified_bytes(mask_path, row.get("mask_sha256")))) as mask:
            if mask.format != "PNG" or mask.mode not in ("L", "P") or mask.size != source.size:
                raise ValueError(f"Raster mask must be an indexed PNG in source dimensions: {sid}")
            values = np.array(mask)
        if (np.any((values >= NUM_CLASSES) & (values != IGNORE_ID)) or
                not np.any(values == CLASS_TO_ID["wall"])):
            raise ValueError(f"Raster mask contains unknown labels or no walls: {sid}")
        if not np.array_equal(values == IGNORE_ID, ignored):
            raise ValueError(f"Ignored mask pixels do not match unresolved regions: {sid}")
        return source, Image.fromarray(values)

    def __getitem__(self, index):
        image, mask = self._load_pair(self.samples[index])
        height, width = self.size
        if self.letterbox:
            scale = min(width / image.width, height / image.height)
            inner_w = max(1, min(width, round(image.width * scale)))
            inner_h = max(1, min(height, round(image.height * scale)))
        else:
            inner_w, inner_h = width, height
        image = image.resize((inner_w, inner_h), Image.Resampling.BILINEAR)
        mask = mask.resize((inner_w, inner_h), Image.Resampling.NEAREST)
        image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float().div_(255)
        if self.normalize:
            image_t = (image_t - self._mean) / self._std
        mask_t = torch.from_numpy(np.array(mask)).long()
        if not torch.any(mask_t != IGNORE_ID):
            raise ValueError("No reviewed pixels remain after raster resizing")
        if (inner_h, inner_w) != (height, width):
            top, left = (height - inner_h) // 2, (width - inner_w) // 2
            canvas = (torch.zeros(3, height, width) if self.normalize else
                      self._mean.expand(3, height, width).clone())
            padding = IGNORE_ID if self.samples[index].get("coverage") == "outside_unresolved" else FLOOR_ID
            labels = torch.full((height, width), padding, dtype=torch.long)
            canvas[:, top:top + inner_h, left:left + inner_w] = image_t
            labels[top:top + inner_h, left:left + inner_w] = mask_t
            return canvas, labels
        return image_t, mask_t
