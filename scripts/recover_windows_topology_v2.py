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
from pathlib import Path

import numpy as np
from extract_rooms_v2 import extract_rooms
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import unary_union


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


def with_windows(classified, candidates, indices):
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
    report = {"version": 1, "method": "topology_guided_thick_window",
              "eligible": len(candidates), "recovered_windows": 0,
              "candidates": [{"source_id": x.get("id"), "bbox": x.get("bbox")}
                             for x in candidates], "trials": [], "requires_review": False}
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
            trial_classified = with_windows(classified, candidates, indices)
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
            result = with_windows(classified, candidates, selected)
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
    image.save(args.output_dir / (stem + "_topology_windows_overlay.png"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
