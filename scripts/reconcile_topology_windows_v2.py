"""Remove unsupported window shapes from the room mask only when geometry improves.

The opening stays available for user review. No export or quality decision is
changed here; the normal project builder and quality gate still run afterward.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon

from extract_rooms_v2 import extract_rooms
from filter_interior_windows_v2 import same_room_topology
from polygon_geometry import build_polygon_project, requires_polygons


REVIEW_METHODS = {"one_sided_unconfirmed_window", "no_jamb_unconfirmed_window"}


def unsupported_length(report):
    return sum(math.dist(edge["start"], edge["end"])
               for edge in report.get("unsupported_segments", []))


def near_unsupported(candidate, report, margin):
    shape = Polygon(candidate["outer"])
    if not shape.is_valid or shape.is_empty:
        return False
    return any(shape.distance(LineString([edge["start"], edge["end"]])) <= margin
               for edge in report.get("unsupported_segments", []))


def project_geometry(rooms, template, scale, thickness, gray):
    _, report = build_polygon_project(rooms, template, scale=scale, thickness=thickness,
                                      raster_gray=gray)
    return report


def reconcile_topology_windows(classified, rooms, rgb, template, scale=1, thickness=20):
    """Try one weak topology-only window at a time, retaining only improvements."""
    result_classified, result_rooms = copy.deepcopy(classified), copy.deepcopy(rooms)
    report = {"version": 1, "method": "stable_rooms_and_supported_project_boundary",
              "candidates": [], "removed_from_room_mask": []}
    if not rooms.get("rooms") or not requires_polygons(rooms["rooms"]):
        report["reason"] = "no_polygon_rooms"
        return result_classified, result_rooms, report
    gray = rgb.min(axis=2)
    try:
        geometry = project_geometry(result_rooms, template, scale, thickness, gray)
    except ValueError as exc:
        report["reason"] = "baseline_geometry_unavailable"
        report["detail"] = str(exc)
        return result_classified, result_rooms, report
    if not geometry.get("unsupported_segments"):
        report["reason"] = "no_unsupported_boundary"
        return result_classified, result_rooms, report
    candidates = [opening for opening in result_classified.get("topology_only_openings", [])
                  if (opening.get("classification") or {}).get("method") in REVIEW_METHODS
                  and opening.get("topology_mask_use") is not False]
    for candidate in candidates:
        entry = {"opening_id": candidate.get("id"), "status": "rejected"}
        report["candidates"].append(entry)
        if not near_unsupported(candidate, geometry, max(6, thickness / scale)):
            entry["reason"] = "not_near_unsupported_boundary"
            continue
        trial_classified = copy.deepcopy(result_classified)
        trial_opening = next(item for item in trial_classified["topology_only_openings"]
                             if item.get("id") == candidate.get("id"))
        trial_opening["topology_mask_use"] = False
        trial_rooms, _, _ = extract_rooms(trial_classified, rgb)
        if not same_room_topology(result_rooms["rooms"], trial_rooms["rooms"]):
            entry["reason"] = "room_topology_changed"
            continue
        try:
            trial_geometry = project_geometry(trial_rooms, template, scale, thickness, gray)
        except ValueError:
            entry["reason"] = "trial_geometry_unavailable"
            continue
        before = geometry.get("unsupported_segments", [])
        after = trial_geometry.get("unsupported_segments", [])
        old_length, new_length = unsupported_length(geometry), unsupported_length(trial_geometry)
        entry.update(unsupported_before=len(before), unsupported_after=len(after),
                     unsupported_length_before_px=round(old_length, 2),
                     unsupported_length_after_px=round(new_length, 2))
        if len(after) >= len(before) or new_length > old_length - 5:
            entry["reason"] = "boundary_support_not_improved"
            continue
        entry.update(status="accepted", reason="stable_rooms_and_fewer_unsupported_edges")
        report["removed_from_room_mask"].append(candidate["id"])
        result_classified, result_rooms, geometry = trial_classified, trial_rooms, trial_geometry
        previous = rooms.get("diagnostics", {})
        for key in ("opening_recovery", "interior_window_filter"):
            if key in previous:
                result_rooms.setdefault("diagnostics", {})[key] = copy.deepcopy(previous[key])
        if "reclassified_wall_support" in rooms:
            result_rooms["reclassified_wall_support"] = copy.deepcopy(
                rooms["reclassified_wall_support"])
    result_rooms.setdefault("diagnostics", {})["topology_window_reconciliation"] = report
    return result_classified, result_rooms, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("classified", type=Path)
    parser.add_argument("rooms", type=Path)
    parser.add_argument("template", type=Path)
    parser.add_argument("--scale", type=float, default=1)
    parser.add_argument("--wall-thickness", type=float, default=20)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert("RGB"))
    classified = json.loads(args.classified.read_text(encoding="utf-8"))
    rooms = json.loads(args.rooms.read_text(encoding="utf-8"))
    template = json.loads(args.template.read_text(encoding="utf-8"))
    result_classified, result_rooms, report = reconcile_topology_windows(
        classified, rooms, rgb, template, args.scale, args.wall_thickness)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    for suffix, payload in (("_topology_reconciled_classified.json", result_classified),
                            ("_rooms_topology_reconciled.json", result_rooms),
                            ("_topology_reconciliation.json", report)):
        (args.output_dir / f"{stem}{suffix}").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["removed_from_room_mask"]:
        _, barrier, inferred = extract_rooms(result_classified, rgb)
        Image.fromarray(barrier).save(args.output_dir / f"{stem}_topology_reconciled_barrier.png")
        Image.fromarray(inferred).save(args.output_dir / f"{stem}_topology_reconciled_closures.png")
        overlay = Image.fromarray(rgb)
        draw = ImageDraw.Draw(overlay)
        for room in result_rooms["rooms"]:
            ring = [tuple(point) for point in room["polygon"]]
            draw.line(ring + ring[:1], fill="magenta", width=3)
            draw.text(tuple(room["center"]), f"Room {room['id'] + 1}", fill="magenta")
        for candidate in result_rooms.get("diagnostics", {}).get("possible_room_partitions", []):
            draw.line([tuple(candidate["start"]), tuple(candidate["end"])],
                      fill="orange", width=3)
        overlay.save(args.output_dir / f"{stem}_topology_reconciled_overlay.png")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
