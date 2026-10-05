"""Conservative fusion for aligned original and normalized inference passes."""

from __future__ import annotations

import numpy as np
import torch

from buildingcv.labels import CLASS_NAMES, CLASS_TO_ID


def fuse_probabilities(original: torch.Tensor, normalized: torch.Tensor) -> torch.Tensor:
    """Keep the normalized pass and rescue only strongly supported original pixels."""
    if original.shape != normalized.shape or original.ndim != 3:
        raise ValueError("Dual-pass probability tensors must have the same CxHxW shape")
    if original.shape[0] != len(CLASS_NAMES):
        raise ValueError("Dual-pass probability tensors use an unexpected class count")
    fused = normalized.clone()
    original_mask = original.argmax(dim=0)
    normalized_mask = normalized.argmax(dim=0)
    wall_id = CLASS_TO_ID["wall"]
    floor_id = CLASS_TO_ID["floor"]
    original_wall_pixels = torch.count_nonzero(original_mask == wall_id)
    normalized_wall_pixels = torch.count_nonzero(normalized_mask == wall_id)
    # A pass which sees almost no wall cannot safely contribute furniture or
    # text-shaped fragments (the characteristic colored-plan failure mode).
    wall_reliable = bool(original_wall_pixels * 20 >= torch.clamp(normalized_wall_pixels, min=1) * 9)
    if not wall_reliable:
        return fused

    wall_rescue = ((original_mask == wall_id) & (normalized_mask == floor_id)
                   & (original[wall_id] >= .72) & (normalized[wall_id] >= .18)
                   & (normalized[floor_id] < .80))
    fused[wall_id, wall_rescue] = torch.maximum(
        fused[wall_id, wall_rescue], original[wall_id, wall_rescue] * .92)

    structural_support = normalized[wall_id] + normalized[CLASS_TO_ID["door"]] + normalized[CLASS_TO_ID["window"]]
    for name in ("door", "window"):
        class_id = CLASS_TO_ID[name]
        opening_rescue = ((original_mask == class_id) & (normalized_mask == floor_id)
                          & (original[class_id] >= .75) & (structural_support >= .25))
        fused[class_id, opening_rescue] = torch.maximum(
            fused[class_id, opening_rescue], original[class_id, opening_rescue] * .90)
    return fused


def inference_diagnostics(original: torch.Tensor, normalized: torch.Tensor,
                          fused: torch.Tensor, content_rect) -> dict:
    """Return compact deterministic metrics for the non-letterboxed content."""
    left, top, width, height = content_rect
    slices = (..., slice(top, top + height), slice(left, left + width))
    masks = {
        "original": original[slices].argmax(dim=0).cpu().numpy(),
        "normalized": normalized[slices].argmax(dim=0).cpu().numpy(),
        "fused": fused[slices].argmax(dim=0).cpu().numpy(),
    }
    result = {
        "version": 1,
        "method": "normalized_with_confidence_gated_original_rescue",
        "disagreement_fraction": round(float(np.mean(masks["original"] != masks["normalized"])), 6),
        "changed_from_normalized_pixels": int(np.count_nonzero(masks["fused"] != masks["normalized"])),
        "passes": {},
    }
    for name, mask in masks.items():
        result["passes"][name] = {
            class_name: int(np.count_nonzero(mask == CLASS_TO_ID[class_name]))
            for class_name in CLASS_NAMES
        }
    return result
