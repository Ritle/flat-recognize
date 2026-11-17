"""Remove unsupported window shapes from the room mask only when geometry improves.

The opening stays available for user review. No export or quality decision is
changed here; the normal project builder and quality gate still run afterward.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon

from classify_openings import structural_union
from extract_rooms_v2 import (extract_rooms, known_opening_barrier, non_rectangular_count,
                              possible_room_partitions, room_candidates, wall_mask)
from filter_interior_windows_v2 import same_room_topology
from polygon_geometry import build_polygon_project, requires_polygons


REVIEW_METHODS = {"one_sided_unconfirmed_window", "no_jamb_unconfirmed_window"}


def unsupported_length(report):
    return sum(math.dist(edge["start"], edge["end"])
               for edge in report.get("unsupported_segments", []))


def near_unsupported(candidate, report, margin):
    shape = Polygon(candidate["outer"])
    if not shape.is_valid:
        shape = shape.buffer(0)
    if shape.is_empty:
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
        trial_rooms, trial_barrier, trial_inferred = extract_rooms(trial_classified, rgb)
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
        if (not after and not trial_geometry.get("errors") and trial_barrier is not None
                and candidate["classification"]["method"] == "one_sided_unconfirmed_window"):
            verification = verify_single_strip(result_classified, result_rooms, trial_rooms, rgb,
                                               trial_barrier, trial_inferred, candidate)
            if verification:
                entry.update(reason="one_sided_strip_at_verified_wall", **verification)
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


def expanded_component(before, after, barrier, trial, removed):
    """All old free pixels survive. Only one room may gain pixels, inside the edit."""
    if len(before) != len(after) or not np.any(removed):
        return None
    changed = [(old, new) for old, new in zip(before, after)
               if old["polygon"] != new["polygon"] or old["area_px"] != new["area_px"]]
    if len(changed) != 1:
        return None
    old, new = changed[0]
    if new["area_px"] <= old["area_px"]:
        return None
    x, y = old["polygon"][0]
    nx, ny = new["polygon"][0]
    _, labels = cv2.connectedComponents(cv2.bitwise_not(trial), connectivity=8)
    if labels[y, x] == 0 or labels[y, x] != labels[ny, nx]:
        return None
    component = labels == labels[y, x]
    _, previous = cv2.connectedComponents(cv2.bitwise_not(barrier), connectivity=8)
    gain = component & (previous != previous[y, x])
    left, top, width, height = cv2.boundingRect(removed)
    if np.all(component[removed > 0]) and gain.sum() == gain[top:top+height, left:left+width].sum():
        return new["id"], component
    return None


def verify_single_strip(classified, rooms, trial_rooms, rgb, trial, inferred, candidate):
    """Verify that the only barrier edit releases a single strip into one existing room."""
    original, barrier, previous_inferred = extract_rooms(classified, rgb)
    # Saved rooms may precede door recovery. Compare the two pixel masks using
    # identical physical openings, and require agreement with the saved room topology.
    if (not same_room_topology(original["rooms"], rooms["rooms"]) or not np.array_equal(previous_inferred, inferred)
            or np.any(trial & ~barrier)):
        return None
    removed = barrier & ~trial
    # All changed pixels must come from the sealing footprint of this opening.
    footprint = known_opening_barrier(np.zeros_like(barrier), [candidate])
    if np.any(removed & ~footprint):
        return None
    expansion = expanded_component(original["rooms"], trial_rooms["rooms"], barrier, trial, removed)
    if expansion is None:
        return None
    room_id, component = expansion
    walls = wall_mask(classified, *barrier.shape)
    # Only a saturated solid band may attest to this single-sided hypothesis.
    # Count dark neutral strokes as well when checking the released pixels.
    colored = (np.ptp(rgb, axis=2) >= 60) & (rgb.min(axis=2) < 140)
    if not np.any(colored) or np.any(colored[removed > 0]):
        return None
    ink = colored | (rgb.max(axis=2) < 95)
    ink_radius = cv2.distanceTransform(ink.astype(np.uint8), cv2.DIST_L2, 5)
    erased_radius = float(ink_radius[removed > 0].max())
    if erased_radius > 1.4:
        return None
    radii = cv2.distanceTransform(colored.astype(np.uint8), cv2.DIST_L2, 5)
    # At a wall cap, the room contour can turn diagonally by one raster pixel.
    nearby = cv2.dilate(component.astype(np.uint8), np.ones((3, 3), np.uint8))
    support = solid_wall_span(radii, walls, removed, nearby)
    if support:
        support["max_removed_ink_radius_px"] = erased_radius
        return dict(raster_wall=support, room_id=room_id, mask_pixels_removed=int(np.count_nonzero(removed)))
    return None


def adjacent_solid_wall(rgb, walls, removed, component):
    """Look across the released strip for a continuous segmented band of dark ink."""
    radii = cv2.distanceTransform((rgb.max(axis=2) < 95).astype(np.uint8), cv2.DIST_L2, 5)
    return solid_wall_span(radii, walls, removed, component)


def solid_wall_span(radii, walls, removed, component):
    erased_radius = float(radii[removed > 0].max())
    if erased_radius > 1.4:
        return None
    x, y, width, height = cv2.boundingRect(removed)
    vertical = height > width
    if vertical:
        x, y, width, height = y, x, height, width
        radii, walls, component = radii.T, walls.T, component.T
    if width < max(12, 2 * height):
        return None
    span = slice(x, x + width)
    for distance in range(1, min(10, height) + 1):
        for row, inward in ((y - distance, 1), (y + height - 1 + distance, -1)):
            if not 0 <= row < radii.shape[0] or not np.all(walls[row, span]):
                continue
            radius = float(radii[row, span].min())
            if radius < 2:
                continue
            limit = min(radii.shape[0] - 1, max(0, row + inward * math.ceil(2*radius + 3)))
            strip = component[min(row, limit):max(row, limit)+1, span]
            if np.all(strip.any(axis=0)):
                ends = [[x, row], [x + width - 1, row]]
                return dict(span=[p[::-1] for p in ends] if vertical else ends,
                            min_wall_radius_px=radius, max_removed_ink_radius_px=erased_radius)
    return None


def reconcile_duplicate_layers(classified, rooms, report, rgb, template, scale, thickness, barrier, inferred):
    """Test a demoted duplicate and its demoted parent together on the retained mask."""
    topology = classified.get("topology_only_openings", [])
    active = {o["id"]: o for o in topology if o.get("topology_mask_use") is not False}
    pairs = [(o["id"], o["classification"].get("paired_with")) for o in active.values()
             if o.get("classification", {}).get("method") == "duplicate_parallel_window_layer"]
    pairs = [(child, parent) for child, parent in pairs if parent in active
             and active[parent]["classification"].get("method") in REVIEW_METHODS]
    if not pairs or not rooms.get("rooms") or not requires_polygons(rooms["rooms"]):
        return barrier
    if barrier.shape != rgb.shape[:2] or inferred.shape != barrier.shape:
        return barrier
    walls = wall_mask(classified, *barrier.shape)
    baseline, _ = room_candidates(barrier, walls, inferred)
    if baseline != rooms["rooms"]:
        return barrier
    gray = rgb.min(axis=2)
    try:
        geometry = project_geometry(rooms, template, scale, thickness, gray)
    except ValueError:
        return barrier
    if not geometry["unsupported_segments"]:
        return barrier
    physical = classified.get("doors", []) + classified.get("windows", [])
    sealed = known_opening_barrier(walls, physical + list(active.values()))
    for child, parent in pairs:
        ids = (child, parent)
        if not any(near_unsupported(active[uid], geometry, thickness/scale) for uid in ids):
            continue
        remaining = physical + [o for uid, o in active.items() if uid not in ids]
        removed = barrier & sealed & ~known_opening_barrier(walls, remaining) & ~inferred
        trial = barrier & ~removed
        rebuilt, diagnostics = room_candidates(trial, walls, inferred)
        expansion = expanded_component(baseline, rebuilt, barrier, trial, removed)
        if expansion is None:
            continue
        room_id, component = expansion
        support = adjacent_solid_wall(rgb, walls, removed, component)
        if support is None:
            continue
        try:
            verified = project_geometry(dict(rooms, rooms=rebuilt), template, scale, thickness, gray)
        except ValueError:
            continue
        if verified["unsupported_segments"] or verified["errors"]:
            continue
        for obj in topology + rooms.get("topology_only_openings", []):
            if obj["id"] in ids:
                obj["topology_mask_use"] = False
        rooms["rooms"] = rebuilt
        info = rooms["diagnostics"]
        info.update(diagnostics, uncertain_rooms=sum(r["uncertain"] for r in rebuilt),
                    non_rectangular_rooms=non_rectangular_count(rebuilt),
                    possible_room_partitions=possible_room_partitions(
                        trial, walls, inferred, rebuilt, diagnostics, gray,
                        structural_union(classified.get("walls", []))))
        report["removed_from_room_mask"].extend(ids)
        report["candidates"].append(dict(opening_ids=list(ids), status="accepted",
                                        reason="duplicate_layers_at_verified_wall", room_id=room_id, raster_wall=support,
                                        mask_pixels_removed=int(np.count_nonzero(removed)),
                                        unsupported_before=len(geometry["unsupported_segments"]), unsupported_after=0))
        info["topology_window_reconciliation"] = report
        return trial
    return barrier


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
    barrier_file = args.output_dir / f"{stem}_barrier.png"
    closures_file = args.output_dir / f"{stem}_closures.png"
    if report["removed_from_room_mask"]:
        _, barrier, inferred = extract_rooms(result_classified, rgb)
    elif report.get("candidates") and barrier_file.is_file() and closures_file.is_file():
        with Image.open(barrier_file) as im, Image.open(closures_file) as closures:
            barrier, inferred = np.asarray(im.convert("L")), np.asarray(closures.convert("L"))
    else:
        barrier = None
    if barrier is not None:
        barrier = reconcile_duplicate_layers(result_classified, result_rooms, report, rgb, template,
                                             args.scale, args.wall_thickness, barrier, inferred)
    for suffix, payload in (("_topology_reconciled_classified.json", result_classified),
                            ("_rooms_topology_reconciled.json", result_rooms),
                            ("_topology_reconciliation.json", report)):
        (args.output_dir / f"{stem}{suffix}").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["removed_from_room_mask"]:
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
