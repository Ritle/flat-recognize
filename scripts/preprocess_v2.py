"""Conservative structural ROI and input normalization, with explicit transforms."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw


def structural_roi(rgb, *, separated_graphics=False):
    """Group long structural lines; use the whole image if evidence is weak."""
    height, width = rgb.shape[:2]
    ratio = min(1.0, 1024 / max(width, height))
    sample = cv2.resize(rgb, (max(1, round(width * ratio)), max(1, round(height * ratio))))
    h, w = sample.shape[:2]
    minimum = min(h, w)
    gray = sample.min(axis=2)
    ink = (gray < 200).astype(np.uint8) * 255
    length = max(12, round(minimum * 0.045))
    horizontal = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    vertical = cv2.morphologyEx(ink, cv2.MORPH_OPEN, np.ones((length, 1), np.uint8))
    lines = cv2.bitwise_or(horizontal, vertical)
    # Include long diagonal contours without relying on orthogonal walls alone.
    segments = cv2.HoughLinesP(ink, 1, np.pi / 180, threshold=length,
                               minLineLength=length * 2, maxLineGap=3)
    if segments is not None and not separated_graphics:
        for x1, y1, x2, y2 in segments.reshape(-1, 4):
            cv2.line(lines, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1)
    radius = max(3, round(minimum * (0.009 if separated_graphics else 0.025)))
    groups = cv2.dilate(lines, cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(groups, connectivity=8)
    candidates = []
    for label in range(1, count):
        x, y, bw, bh, _ = map(int, stats[label])
        support = int(np.count_nonzero(lines[labels == label]))
        if bw * bh >= w * h * 0.025 and bw >= length * 2 and bh >= length * 2:
            candidates.append((support * math.sqrt(bw * bh), label, support))
    full = [0, 0, width, height]
    diagnostics = {"structural_candidates": len(candidates), "method": "long_line_regions"}
    if not candidates:
        return full, {**diagnostics, "fallback": "no_reliable_structural_region"}
    candidates.sort(reverse=True)
    _, selected, support = candidates[0]
    if len(candidates) > 1 and candidates[1][0] > candidates[0][0] * 0.65:
        return full, {**diagnostics, "fallback": "ambiguous_structural_regions"}
    ys, xs = np.where((labels == selected) & (lines > 0))
    if support < length * 4:
        return full, {**diagnostics, "fallback": "insufficient_structural_support"}
    pad = max(6, round(minimum * 0.035))
    x1, y1 = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x2, y2 = min(w, int(xs.max()) + pad + 1), min(h, int(ys.max()) + pad + 1)
    # Retain nearby structural islands (e.g. balcony contours and window blocks).
    for _, label, _ in candidates[1:]:
        x, y, bw, bh, _ = map(int, stats[label])
        if x < x2 + pad and x + bw > x1 - pad and y < y2 + pad and y + bh > y1 - pad:
            x1, y1 = min(x1, max(0, x - pad)), min(y1, max(0, y - pad))
            x2, y2 = max(x2, min(w, x + bw + pad)), max(y2, min(h, y + bh + pad))
    # Avoid perturbing already tightly framed clean plans.
    if (x2 - x1) * (y2 - y1) >= w * h * 0.85:
        return full, {**diagnostics, "fallback": "already_tightly_framed"}
    roi = [math.floor(x1 * width / w), math.floor(y1 * height / h),
           math.ceil(x2 * width / w), math.ceil(y2 * height / h)]
    return roi, {**diagnostics, "fallback": None}


def normalize_crop(rgb, *, light_fill_cleanup=False):
    """Preserve high-contrast neutral inputs; darken structural colors otherwise.

    No thin-line erasure: furniture and architectural details both remain visible.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    projected = rgb.min(axis=2)
    maximum = rgb.max(axis=2)
    colorful_light = (maximum > 180) & (maximum.astype(np.int16) - projected > 25)
    # Broad light fills are floor/background, unlike narrow colored wall bands.
    # Use the brightest channel to avoid turning yellow/blue floors into walls.
    # Require plentiful dark ink; this branch must not erase pale scan linework.
    fill_fraction = float(np.mean(colorful_light))
    dark_fraction = float(np.mean(maximum < 65))
    if light_fill_cleanup and fill_fraction > 0.12 and dark_fraction > 0.01:
        neutral = np.clip((maximum.astype(np.float32) - 50) * (255 / 130), 0, 255).astype(np.uint8)
        return np.repeat(neutral[:, :, None], 3, axis=2), gray, {
            "normalization": "light_fill_neutralization",
            "light_color_fraction": fill_fraction, "dark_ink_fraction": dark_fraction,
        }
    dark = (projected < 200).astype(np.uint8)
    distance = cv2.distanceTransform(dark, cv2.DIST_L2, 5)
    thick = distance >= max(2.0, min(rgb.shape[:2]) * 0.003)
    if np.count_nonzero(thick) < 20:
        return rgb.copy(), gray, {"normalization": "original", "reason": "weak_wall_color_evidence"}
    wall_level = float(np.median(projected[thick]))
    colors = rgb[thick].astype(np.int16)
    saturation = float(np.mean(colors.max(axis=1) - colors.min(axis=1) > 35))
    background = float(np.percentile(projected, 95))
    if wall_level <= 65 and saturation < 0.05 and background >= 220:
        return rgb.copy(), gray, {"normalization": "original", "wall_level": wall_level}
    if background - wall_level < 30:
        return rgb.copy(), gray, {"normalization": "original", "reason": "low_dynamic_range"}
    black = max(0, wall_level - 15)
    normalized = np.clip((projected.astype(np.float32) - black)
                         * 255 / max(30, background - black), 0, 255).astype(np.uint8)
    return np.repeat(normalized[:, :, None], 3, axis=2), gray, {
        "normalization": "min_channel_contrast", "wall_level": wall_level,
        "background_level": background, "structural_color_fraction": saturation,
    }


def faint_scan_view(rgb):
    """Recover pale neutral ink only when a bright page contains structural linework."""
    gray = rgb.min(axis=2)
    ink, mid, paper = np.percentile(gray, [1, 60, 95])
    if (ink < 190 or paper < 230 or paper - ink < 15 or paper - mid > .15 * (paper - ink)
            or np.percentile(rgb.max(axis=2) - gray, 99) > 10):
        return None
    lut = np.clip(24 + (np.arange(256) - ink) * 226 / (paper - ink), 0, 255).astype(np.uint8)
    enhanced = lut[gray]
    height, width = gray.shape
    ratio = min(1., 896 / max(height, width))
    sample = cv2.resize(enhanced, (max(1, round(width * ratio)), max(1, round(height * ratio))),
                        interpolation=cv2.INTER_AREA)
    # Check actual stroke edges. A broad shadow alone can form an ink region.
    edges = cv2.Canny(sample, 50, 150)
    minimum = max(12, round(min(sample.shape) * .07))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, minimum, minLineLength=minimum, maxLineGap=3)
    if lines is None:
        return None
    segments = lines.reshape(-1, 4)
    vectors = segments[:, 2:] - segments[:, :2]
    lengths = np.linalg.norm(vectors, axis=1)
    unit = vectors / lengths[:, None]
    if np.all(np.abs(unit @ unit[np.argmax(lengths)]) > math.cos(math.radians(25))):
        return None
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
    roi, diagnostics = structural_roi(enhanced_rgb)
    if not diagnostics["structural_candidates"] or diagnostics["fallback"] not in (None, "already_tightly_framed"):
        return None
    levels = {"normalization": "faint_scan_contrast", "ink_level": float(ink), "paper_level": float(paper)}
    return enhanced_rgb, roi, diagnostics, levels


def preprocess(image, *, light_fill_cleanup=False):
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    neutral, _, preliminary = normalize_crop(rgb, light_fill_cleanup=light_fill_cleanup)
    separated = preliminary.get("normalization") == "light_fill_neutralization"
    faint = faint_scan_view(rgb) if preliminary.get("reason") == "weak_wall_color_evidence" else None
    if faint is None:
        roi, diagnostics = structural_roi(neutral if separated else rgb, separated_graphics=separated)
        x1, y1, x2, y2 = roi
        crop = rgb[y1:y2, x1:x2]
        normalized, gray, normalization = normalize_crop(crop, light_fill_cleanup=light_fill_cleanup)
    else:
        enhanced, roi, diagnostics, normalization = faint
        x1, y1, x2, y2 = roi
        normalized = enhanced[y1:y2, x1:x2]
        gray = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2GRAY)
    meta = {
        "version": 2, "source_size": [width, height], "roi": roi,
        "processed_size": [x2 - x1, y2 - y1], "coordinate_space": "source_raster",
        "source_to_crop": [[1, 0, -x1], [0, 1, -y1], [0, 0, 1]],
        "crop_to_source": [[1, 0, x1], [0, 1, y1], [0, 0, 1]],
        "diagnostics": {**diagnostics, **normalization,
                        "light_fill_applied": separated or normalization["normalization"] == "light_fill_neutralization"},
    }
    return Image.fromarray(normalized), Image.fromarray(gray), meta


def model_point_to_source(point, content_rect, meta):
    left, top, inner_w, inner_h = content_rect
    x1, y1, x2, y2 = meta["roi"]
    width, height = meta["source_size"]
    if inner_w <= 0 or inner_h <= 0:
        raise ValueError("Invalid letterbox content dimensions")
    x = (point[0] - left) * (x2 - x1) / inner_w + x1
    y = (point[1] - top) * (y2 - y1) / inner_h + y1
    return [round(max(0, min(width, x)), 2), round(max(0, min(height, y)), 2)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--light-fill-cleanup", action="store_true", help="Experimental light floor fill cleanup")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as source:
        source = source.convert("RGB")
        normalized, gray, meta = preprocess(source, light_fill_cleanup=args.light_fill_cleanup)
        preview = source.copy()
    stem = args.image.stem
    normalized_path = args.output_dir / f"{stem}_preprocessed.png"
    gray_path = args.output_dir / f"{stem}_grayscale.png"
    preview_path = args.output_dir / f"{stem}_preprocess_preview.png"
    meta_path = args.output_dir / f"{stem}_preprocess.json"
    meta.update({"source": str(args.image.resolve()),
                 "normalized_image": str(normalized_path.resolve()),
                 "grayscale_image": str(gray_path.resolve())})
    normalized.save(normalized_path)
    gray.save(gray_path)
    ImageDraw.Draw(preview).rectangle(
        (meta["roi"][0], meta["roi"][1], meta["roi"][2] - 1, meta["roi"][3] - 1),
        outline="magenta", width=3)
    preview.save(preview_path)
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ROI: {meta['roi']}; {meta['diagnostics']}")
    print(f"Transform metadata: {meta_path}")


if __name__ == "__main__":
    main()
