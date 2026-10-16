"""Recover thick window candidates only when they repair room topology.

The segmentation polygon must have failed only the thickness filter. A small
set of candidates is tried against extract_rooms_v2; accepted rooms must add a
substantial interior without losing any room already found. All decisions are
diagnostic and recovered windows require review.
"""

import argparse
import copy
import itertools
import json
import math
from pathlib import Path

import numpy as np
from extract_rooms_v2 import extract_rooms
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import unary_union


def raster_frame_axis(gray, candidate):
    """Suggest an axis only for a unique pair of long, close raster rails.

    The broad segmentation mask still closes rooms. This evidence is never
    substituted for the source polygon or used to approve export.
    """
    bounds = candidate.get("bbox")
    if not isinstance(bounds, list) or len(bounds) != 4:
        return None
    x0, y0, x1, y1 = (float(value) for value in bounds)
    if not all(math.isfinite(value) for value in (x0, y0, x1, y1)):
        return None
    horizontal = x1 - x0 >= y1 - y0
    long_lo, long_hi = (x0, x1) if horizontal else (y0, y1)
    side_lo, side_hi = (y0, y1) if horizontal else (x0, x1)
    length, thickness = long_hi - long_lo, side_hi - side_lo
    if length < 30 or thickness < 8:
        return None
    height, width = gray.shape
    long_limit, side_limit = (width, height) if horizontal else (height, width)
    sample_lo = max(0, math.ceil(long_lo + length * .2))
    sample_hi = min(long_limit, math.floor(long_hi - length * .2) + 1)
    transverse_lo = max(0, math.floor(side_lo))
    transverse_hi = min(side_limit, math.ceil(side_hi) + 1)
    if sample_hi - sample_lo < 20 or transverse_hi - transverse_lo < 8:
        return None
    raster = gray[transverse_lo:transverse_hi, sample_lo:sample_hi] if horizontal else \
        gray[sample_lo:sample_hi, transverse_lo:transverse_hi].T
    support = np.mean(raster < 140, axis=1)
    rows = np.flatnonzero(support >= .75)
    if len(rows) < 2:
        return None
    bands = []
    for group in np.split(rows, np.flatnonzero(np.diff(rows) > 1) + 1):
        if len(group) <= max(3, thickness * .08):
            bands.append((transverse_lo + float(np.mean(group)),
                          float(np.max(support[group]))))
    pairs = []
    for (a, support_a), (b, support_b) in zip(bands, bands[1:]):
        separation = b - a
        center = (a + b) / 2
        if (3 <= separation <= min(18, thickness * .25)
                and side_lo + thickness * .2 <= center <= side_hi - thickness * .2):
            pairs.append((a, b, support_a, support_b))
    if len(pairs) != 1:
        return None
    a, b, support_a, support_b = pairs[0]
    center = (a + b) / 2
    span = ([[round(long_lo, 2), round(center, 2)],
             [round(long_hi, 2), round(center, 2)]] if horizontal else
            [[round(center, 2), round(long_lo, 2)],
             [round(center, 2), round(long_hi, 2)]])
    return {"method": "paired_raster_rails", "span": span,
            "extent_basis": "segmentation_bbox",
            "rails_px": [round(a, 2), round(b, 2)],
            "minimum_support": round(min(support_a, support_b), 3)}


def eligible_candidates(prediction, classified, height, width):
    known_shapes = []
    for item in classified.get("openings", []):
        shape = Polygon(item.get("outer", []))
        if not shape.is_valid:
            shape = shape.buffer(0)
        if not shape.is_empty:
            known_shapes.append(shape)
    known = unary_union(known_shapes)
    wall_tolerance = prediction.get("diagnostics", {}).get("wall_tolerance", 6)
    limit = min(height, width) * .08
    candidates = []
    for item in prediction.get("diagnostics", {}).get("rejected", []):
        if (item.get("reject_reason") != ["too_thick"]
                or item.get("source_class") not in ("door", "window")
                or not 0 < item.get("thickness", 0) <= limit
                or item.get("aspect_ratio", 0) < 1.8
                or item.get("length", 0) < max(30, min(height, width) * .06)
                or item.get("wall_distance", float("inf")) > wall_tolerance):
            continue
        shape = Polygon(item.get("outer", []))
        if not shape.is_valid:
            shape = shape.buffer(0)
        if shape.is_empty or shape.geom_type != "Polygon" or shape.area <= 0:
            continue
        if not known.is_empty and shape.intersection(known).area > shape.area * .2:
            continue
        candidates.append(item)
    return candidates


def preserve_rooms(before, after):
    trial = [Polygon(room["polygon"]) for room in after]
    for room in before:
        source = Polygon(room["polygon"])
        if source.area <= 0 or not trial:
            return False
        if max(source.intersection(shape).area for shape in trial) < source.area * .98:
            return False
    return True


def with_windows(classified, candidates, indices, axis_evidence=None):
    result = copy.deepcopy(classified)
    existing_ids = [item.get("id") for item in result.get("openings", [])]
    next_id = max([uid for uid in existing_ids if isinstance(uid, int)] + [-1]) + 1
    for index in indices:
        candidate = copy.deepcopy(candidates[index])
        candidate.pop("reject_reason", None)
        candidate.update(id=next_id, type="window",
                         classification={"method": "topology_recovered_thick_window",
                                         "requires_review": True,
                                         "source_class": candidate["source_class"]},
                         recovery={"method": "topology_recovered_thick_window",
                                   "requires_review": True})
        if axis_evidence and axis_evidence[index] is not None:
            candidate["recovery"]["raster_axis_evidence"] = axis_evidence[index]
        next_id += 1
        result.setdefault("windows", []).append(candidate)
        result.setdefault("openings", []).append(copy.deepcopy(candidate))
    return result


def recover_windows(prediction, classified, rgb):
    height, width = rgb.shape[:2]
    for data in (prediction, classified):
        meta = data.get("meta", {})
        if (meta.get("width"), meta.get("height")) != (width, height):
            raise ValueError("Window recovery coordinates must match the raster")
    candidates = eligible_candidates(prediction, classified, height, width)
    gray = np.asarray(Image.fromarray(rgb).convert("L")) if candidates else None
    axis_evidence = [raster_frame_axis(gray, candidate) for candidate in candidates]
    report = {"version": 1, "method": "topology_guided_thick_window",
              "eligible": len(candidates), "recovered_windows": 0,
              "candidates": [{"source_id": x.get("id"), "bbox": x.get("bbox"),
                              "raster_axis_evidence": axis_evidence[index]}
                             for index, x in enumerate(candidates)],
              "trials": [], "requires_review": False}
    if not candidates:
        return copy.deepcopy(classified), report
    if len(candidates) > 4:
        report["reason"] = "too_many_candidates_for_conservative_search"
        return copy.deepcopy(classified), report
    base, _, _ = extract_rooms(classified, rgb)
    baseline = base["diagnostics"]
    report["baseline"] = {"rooms": len(base["rooms"]),
                          "outside_fraction": baseline["outside_fraction_in_structure"]}
    if baseline["outside_fraction_in_structure"] < .4:
        report["reason"] = "no_significant_room_leak"
        return copy.deepcopy(classified), report
    for count in range(1, len(candidates) + 1):
        passing = []
        for indices in itertools.combinations(range(len(candidates)), count):
            trial_classified = with_windows(classified, candidates, indices, axis_evidence)
            trial, _, _ = extract_rooms(trial_classified, rgb)
            diagnostics = trial["diagnostics"]
            outside = diagnostics["outside_fraction_in_structure"]
            preserved = preserve_rooms(base["rooms"], trial["rooms"])
            accepted = (len(trial["rooms"]) > len(base["rooms"])
                        and outside < .3
                        and outside <= baseline["outside_fraction_in_structure"] - .2
                        and preserved
                        and diagnostics["interior_area_px"] > baseline["interior_area_px"] * 1.1)
            report["trials"].append({"indices": list(indices), "rooms": len(trial["rooms"]),
                                     "outside_fraction": round(outside, 4),
                                     "preserved_rooms": preserved, "accepted": accepted})
            if accepted:
                passing.append((outside, indices))
        if passing:
            _, selected = min(passing)
            result = with_windows(classified, candidates, selected, axis_evidence)
            report.update(recovered_windows=len(selected), selected_indices=list(selected),
                          requires_review=True)
            return result, report
    report["reason"] = "no_topology_preserving_improvement"
    return copy.deepcopy(classified), report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("prediction", type=Path)
    parser.add_argument("classified", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    classified, report = recover_windows(
        json.loads(args.prediction.read_text(encoding="utf-8")),
        json.loads(args.classified.read_text(encoding="utf-8")), np.asarray(image))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    for suffix, data in (("_topology_classified.json", classified),
                         ("_topology_windows.json", report)):
        (args.output_dir / (stem + suffix)).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    draw = ImageDraw.Draw(image)
    for item in classified.get("windows", []):
        if item.get("classification", {}).get("method") == "topology_recovered_thick_window":
            points = [tuple(p) for p in item["outer"]]
            draw.line(points + points[:1], fill="cyan", width=4)
            evidence = (item.get("recovery") or {}).get("raster_axis_evidence")
            if evidence:
                draw.line([tuple(point) for point in evidence["span"]], fill="yellow", width=3)
    image.save(args.output_dir / (stem + "_topology_windows_overlay.png"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
