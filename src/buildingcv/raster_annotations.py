"""Shared review and exclusion rules for manual raster annotations."""

import math

import numpy as np
from PIL import Image, ImageDraw


def unresolved_mask(regions, size):
    if not isinstance(regions, list):
        raise ValueError("Unresolved regions must be a list")
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for region in regions:
        bbox = region.get("bbox") if isinstance(region, dict) else None
        if (not isinstance(bbox, list) or len(bbox) != 4 or
                any(type(v) not in (int, float) or not math.isfinite(v) for v in bbox)):
            raise ValueError("Unresolved regions need finite bbox coordinates")
        x1, y1, x2, y2 = bbox
        if not (0 <= x1 < x2 < width and 0 <= y1 < y2 < height):
            raise ValueError("Unresolved region bbox must lie within the source image")
        draw.rectangle(bbox, fill=1)
    return np.array(mask, dtype=bool)


def reviewed_ignore_mask(annotation, size, *, allow_partial=False):
    review = annotation.get("review", {})
    if (not isinstance(review, dict) or review.get("status") != "reviewed" or
            not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip()):
        raise ValueError("A named reviewer must explicitly confirm full-image coverage")
    coverage = review.get("coverage")
    if coverage == "full":
        if annotation.get("unresolved") != []:
            raise ValueError("Resolve all ambiguous regions before building a training mask")
        return unresolved_mask([], size)
    if coverage != "outside_unresolved":
        raise ValueError("A named reviewer must explicitly confirm full-image coverage")
    if allow_partial is not True:
        raise ValueError("Partial annotations require explicit allow_partial opt-in")
    if not annotation.get("unresolved"):
        raise ValueError("Partial coverage needs explicit unresolved regions")
    return unresolved_mask(annotation["unresolved"], size)
