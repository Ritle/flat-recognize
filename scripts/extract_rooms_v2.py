"""Room topology repair; inferred closures never become exported wall polygons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from classify_openings import classify_opening, opening_axis, structural_union
from door_leaf_evidence import leaf_and_arc, leaf_candidates, wall_jamb_support
from extract_rooms import fill_polygon, polygon_to_int, simplify_contour
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon


def skeletonize(mask):
    """Eight-connected thinning with two alternating topology-preserving passes."""
    pixels = np.pad((mask > 0).astype(np.uint8), 1)
    while True:
        removed = 0
        for step in (0, 1):
            p = [pixels[:-2, 1:-1], pixels[:-2, 2:], pixels[1:-1, 2:], pixels[2:, 2:],
                 pixels[2:, 1:-1], pixels[2:, :-2], pixels[1:-1, :-2], pixels[:-2, :-2]]
            neighbors = sum(p)
            transitions = sum((p[i] == 0) & (p[(i + 1) % 8] == 1) for i in range(8))
            if step == 0:
                condition = (p[0] * p[2] * p[4] == 0) & (p[2] * p[4] * p[6] == 0)
            else:
                condition = (p[0] * p[2] * p[6] == 0) & (p[0] * p[4] * p[6] == 0)
            delete = ((pixels[1:-1, 1:-1] == 1) & (neighbors >= 2) & (neighbors <= 6)
                      & (transitions == 1) & condition)
            removed += int(np.count_nonzero(delete))
            pixels[1:-1, 1:-1][delete] = 0
        if not removed:
            return pixels[1:-1, 1:-1]


def wall_endpoints(barrier):
    height, width = barrier.shape
    factor = min(1.0, 1024 / max(width, height))
    small = cv2.resize(barrier, (max(1, round(width * factor)), max(1, round(height * factor))),
                       interpolation=cv2.INTER_NEAREST)
    skeleton = skeletonize(small)
    neighbors = cv2.filter2D(skeleton, cv2.CV_16S, np.ones((3, 3), np.int16)) - skeleton
    distance = cv2.distanceTransform((small > 0).astype(np.uint8), cv2.DIST_L2, 5)
    radii = distance[skeleton > 0]
    thickness = max(2.0, float(np.median(radii)) * 2 / factor) if radii.size else 2.0
    points = []
    sh, sw = skeleton.shape
    sx, sy = width / sw, height / sh
    for y, x in np.argwhere((skeleton > 0) & (neighbors == 1)):
        start = (int(x), int(y))
        visited = {start}
        current = start
        for _ in range(max(6, round(thickness * factor * 1.5))):
            cx, cy = current
            choices = [(nx, ny) for ny in range(max(0, cy - 1), min(sh, cy + 2))
                       for nx in range(max(0, cx - 1), min(sw, cx + 2))
                       if skeleton[ny, nx] and (nx, ny) not in visited]
            if len(choices) != 1:
                break
            current = choices[0]
            visited.add(current)
        if len(visited) < 4:
            continue
        direction = np.array([(start[0] - current[0]) * sx,
                              (start[1] - current[1]) * sy], dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm < 2:
            continue
        local_width = max(2.0, float(distance[y, x]) * 2 / factor)
        points.append({"point": np.array([x * sx, y * sy]), "direction": direction / norm,
                       "width": max(local_width, thickness * 0.6)})
    return points, thickness


def axis_pairs(barrier, thickness, extended=False, sampling_thickness=None):
    """Recover wall caps obscured by short skeleton branches in noisy masks."""
    pairs = []
    distance = cv2.distanceTransform((barrier > 0).astype(np.uint8), cv2.DIST_L2, 5)
    length = max(7, round(thickness * (1.5 if extended else 2)))
    # Include a fixed wall-only grid as well as the original barrier grid.
    # An unrelated opening can change barrier thickness enough to shift rows
    # past a real gap; the original grid still retains nearby leaf evidence.
    stride_ratio = 0.25 if extended else 0.65
    stride = max(2, round((sampling_thickness or thickness) * stride_ratio))
    barrier_stride = max(2, round(thickness * stride_ratio))
    for vertical in (False, True):
        mask = barrier.T if vertical else barrier
        radii = distance.T if vertical else distance
        long_runs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
        rows = set(range(stride // 2, mask.shape[0], stride))
        rows.update(range(barrier_stride // 2, mask.shape[0], barrier_stride))
        for y in sorted(rows):
            changes = np.diff(np.pad((long_runs[y] > 0).astype(np.int8), 1))
            starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
            for end, start in zip(ends[:-1], starts[1:]):
                if start - end > thickness * (8 if extended else 6) or start - end < 3:
                    continue
                left = max(0, int(end) - round(thickness))
                right = min(mask.shape[1] - 1, int(start) + round(thickness))
                width = float(min(radii[y, left], radii[y, right])) * 2
                if width < thickness * (0.35 if extended else 0.6):
                    continue
                if extended:
                    width = max(width, thickness * 1.45)
                a, b = np.array([end - 1, y], dtype=float), np.array([start, y], dtype=float)
                direction = np.array([1., 0.])
                if vertical:
                    a, b, direction = a[::-1], b[::-1], direction[::-1]
                pairs.append(({"point": a, "direction": direction, "width": width},
                              {"point": b, "direction": -direction, "width": width}))
    return pairs


def endpoint_wall_arc_candidates(barrier, gray, endpoints, thickness, walls):
    """Find arc-confirmed doorways ending at a transverse wall instead of a wall cap."""
    if walls is None:
        return []
    height, width = barrier.shape
    candidates = []
    for endpoint in endpoints:
        start = endpoint["point"]
        direction = endpoint["direction"]
        if max(abs(direction)) < np.cos(np.radians(12)):
            continue
        local_width = max(thickness, endpoint["width"])
        maximum = min(round(local_width * 6), round(min(barrier.shape) * .16 + local_width))
        left_wall = False
        hit = None
        for distance in range(1, maximum + 1):
            point = np.rint(start + direction * distance).astype(int)
            x, y = point
            if not (0 <= x < width and 0 <= y < height):
                break
            occupied = bool(barrier[y, x])
            if not occupied:
                left_wall = True
            elif left_wall:
                hit = point.astype(float)
                break
        if hit is None:
            continue
        end = hit - direction * 2
        gap = float(np.linalg.norm(end - start))
        if gap < max(15, local_width * 1.5):
            continue
        normal = np.array([-direction[1], direction[0]])
        wall_probe = hit + direction * max(2, round(local_width * .25))
        offset = max(3, round(local_width * .75))
        transverse = []
        for sign in (-1, 1):
            x, y = np.rint(wall_probe + normal * offset * sign).astype(int)
            transverse.append(0 <= x < width and 0 <= y < height and bool(barrier[y, x]))
        if not all(transverse) or min(wall_jamb_support(start, end, local_width, walls)) < 2:
            continue
        strip = LineString([start, end]).buffer(local_width / 2, cap_style=2)
        if walls.intersection(strip).area > strip.area * .25:
            continue
        if leaf_and_arc(gray, start, end, local_width, walls) is None:
            continue
        candidates.append({"start": start.tolist(), "end": end.tolist(),
                           "width": local_width, "length": gap, "alignment": 1.0,
                           "evidence": "door_leaf", "method": "endpoint_to_wall_arc"})
    return candidates


def window_edge_seal(window, barrier, walls):
    """Offer a bounded topology-only seal at supported window ends."""
    if len(window.get("outer", [])) < 3:
        return None
    start, end, thickness = opening_axis(window)
    if (np.linalg.norm(end - start) < 15 or thickness <= 0
            or min(wall_jamb_support(start, end, thickness, walls)) < 2):
        return None
    horizontal = abs(end[0] - start[0]) >= abs(end[1] - start[1])
    mask = np.zeros_like(barrier)
    cv2.fillPoly(mask, [polygon_to_int(window["outer"])], 255)
    kernel = np.ones((5, 11) if horizontal else (11, 5), np.uint8)
    grown = cv2.dilate(cv2.dilate(mask, kernel), np.ones((3, 3), np.uint8))
    return grown & cv2.bitwise_not(barrier)


def preserves_existing_rooms(existing, proposed):
    proposed_shapes = [Polygon(room["polygon"]) for room in proposed]
    for room in existing:
        shape = Polygon(room["polygon"])
        if (not proposed_shapes
                or max(shape.intersection(other).area for other in proposed_shapes) < shape.area * .95):
            return False
    return True


def gap_candidates(barrier, gray, relaxed=False, extended=False, sampling_thickness=None,
                   wall_shapes=None):
    endpoints, thickness = wall_endpoints(barrier)
    candidates = []
    rejected_leaves = []
    free_distance = (cv2.distanceTransform((barrier == 0).astype(np.uint8), cv2.DIST_L2, 5)
                     if extended else None)
    pairs = [(i, j, a, endpoints[j], "skeleton") for i, a in enumerate(endpoints)
             for j in range(i + 1, len(endpoints))]
    axes = axis_pairs(barrier, thickness, extended=extended,
                      sampling_thickness=sampling_thickness)
    for index, (a, b) in enumerate(axes):
        offset = len(endpoints) + index * 2
        pairs.append((offset, offset + 1, a, b, "axis_caps"))
    for i, j, a, b, method in pairs:
        delta = b["point"] - a["point"]
        gap = float(np.linalg.norm(delta))
        local_width = (a["width"] + b["width"]) / 2
        gap_ratio = 8 if extended else 6 if relaxed else 4.5
        if gap < 3 or gap > min(local_width * gap_ratio,
                               min(barrier.shape) * 0.16 + local_width):
            continue
        direction = delta / gap
        alignment = min(float(np.dot(a["direction"], direction)),
                        float(np.dot(b["direction"], -direction)))
        if alignment < (0.85 if relaxed else 0.92):
            continue
        samples = a["point"] + np.linspace(0.2, 0.8, max(5, round(gap)))[:, None] * delta
        xx = np.clip(np.rint(samples[:, 0]).astype(int), 0, barrier.shape[1] - 1)
        yy = np.clip(np.rint(samples[:, 1]).astype(int), 0, barrier.shape[0] - 1)
        if np.mean(barrier[yy, xx] > 0) > 0.25:
            continue
        evidence = "small_gap" if gap <= local_width * 1.5 else None
        if evidence is None and float(np.mean(gray[yy, xx] < 140)) >= 0.65:
            evidence = "raster_line"
        if evidence is None:
            polygon = LineString([a["point"], b["point"]]).buffer(local_width / 2)
            opening = {"outer": [list(p) for p in polygon.exterior.coords[:-1]]}
            opening_type, lines = classify_opening(gray, opening)
            if opening_type == "door" and lines:
                usable = []
                blocked_tips = 0
                anchored_count = 0

                def inspect_leaf_axis(start, end, width=local_width, accepted=usable):
                    nonlocal blocked_tips, anchored_count
                    anchored, _, _ = leaf_candidates(gray, start, end, width)
                    anchored_count += len(anchored)
                    for leaf in anchored:
                        hinge, tip = np.array(leaf["hinge"]), np.array(leaf["tip"])
                        samples = hinge + np.linspace(.25, .9, 20)[:, None] * (tip - hinge)
                        xx = np.clip(np.rint(samples[:, 0]).astype(int), 0, barrier.shape[1] - 1)
                        yy = np.clip(np.rint(samples[:, 1]).astype(int), 0, barrier.shape[0] - 1)
                        if np.mean(barrier[yy, xx] > 0) > .55:
                            continue
                        if extended:
                            tx, ty = np.clip(np.rint(tip).astype(int), [0, 0],
                                             [barrier.shape[1] - 1, barrier.shape[0] - 1])
                            if free_distance[ty, tx] < max(3, width * .48):
                                blocked_tips += 1
                                continue
                        accepted.append(leaf)

                inspect_leaf_axis(a["point"], b["point"])
                # Hough endpoints can change at adjacent raster rows. Retry a narrow
                # perpendicular band around an existing axis-cap gap rather than
                # scanning every image row and multiplying false candidates.
                if not usable and extended and method == "axis_caps":
                    normal = np.array([-direction[1], direction[0]])
                    jitter = max(2, round(thickness * .2))
                    for offset in range(1, jitter + 1):
                        for sign in (-1, 1):
                            shift = normal * offset * sign
                            inspect_leaf_axis(a["point"] + shift, b["point"] + shift)
                            if usable:
                                break
                        if usable:
                            break
                if usable:
                    if (wall_shapes is not None
                            and max(abs(direction)) < np.cos(np.radians(12))
                            and leaf_and_arc(gray, a["point"], b["point"], local_width,
                                             wall_shapes) is None):
                        rejected_leaves.append({"start": a["point"].tolist(),
                                                "end": b["point"].tolist(),
                                                "width": local_width,
                                                "reason": "oblique_arc_not_confirmed"})
                        continue
                    evidence = "door_leaf"
                else:
                    rejected_leaves.append({"start": a["point"].tolist(), "end": b["point"].tolist(),
                                            "width": local_width,
                                            "reason": ("leaf_tip_touches_barrier" if blocked_tips else
                                                       "leaf_overlaps_barrier" if anchored_count else
                                                       "no_jamb_anchored_leaf")})
        # A broad passage without supporting evidence remains open.
        if evidence is None:
            continue
        candidates.append({"indices": (i, j), "start": a["point"].tolist(),
                           "end": b["point"].tolist(), "width": local_width,
                           "length": gap, "alignment": alignment, "evidence": evidence,
                           "method": method})
    candidates.sort(key=lambda c: (c["length"] / c["width"], -c["alignment"]))
    used = set()
    closures = []

    def same_gap(a, b):
        tolerance = max(a["width"], b["width"])
        direct = (np.linalg.norm(np.array(a["start"]) - b["start"]) < tolerance
                  and np.linalg.norm(np.array(a["end"]) - b["end"]) < tolerance)
        reversed_order = (np.linalg.norm(np.array(a["start"]) - b["end"]) < tolerance
                          and np.linalg.norm(np.array(a["end"]) - b["start"]) < tolerance)
        return direct or reversed_order

    for candidate in candidates:
        if used.intersection(candidate["indices"]):
            continue
        if any(same_gap(candidate, closure) for closure in closures):
            continue
        used.update(candidate.pop("indices"))
        closures.append(candidate)
    return closures, {"wall_endpoints": len(endpoints), "candidate_gaps": len(candidates),
                      "rejected_leaf_gaps": rejected_leaves,
                      "axis_cap_pairs": len(axes),
                      "estimated_wall_thickness": round(thickness, 2),
                      "axis_sampling_thickness": round(sampling_thickness or thickness, 2)}


def apply_closures(barrier, closures):
    result = barrier.copy()
    additions = np.zeros_like(barrier)
    for closure in closures:
        a, b = [tuple(round(v) for v in closure[k]) for k in ("start", "end")]
        cv2.line(additions, a, b, 255, max(2, round(closure["width"])))
    additions[barrier > 0] = 0
    return cv2.bitwise_or(result, additions), additions


def exterior_envelope_candidate(barrier, walls):
    """Close a heavily fragmented rectangular exterior only for topology."""
    ys, xs = np.where(walls > 0)
    if not len(xs):
        return barrier, np.zeros_like(barrier), {"accepted": False, "reason": "no_walls"}
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    distance = cv2.distanceTransform((walls > 0).astype(np.uint8), cv2.DIST_L2, 5)
    positive = distance[distance > 1]
    thickness = max(2.0, float(np.percentile(positive, 75)) * 2) if positive.size else 2.0
    band = max(4, round(thickness * 2))

    def horizontal(y):
        strip = barrier[max(0, y - band):min(barrier.shape[0], y + band + 1), x1:x2 + 1]
        return float(np.mean(np.max(strip, axis=0) > 0)) if strip.size else 0.0

    def vertical(x):
        strip = barrier[y1:y2 + 1, max(0, x - band):min(barrier.shape[1], x + band + 1)]
        return float(np.mean(np.max(strip, axis=1) > 0)) if strip.size else 0.0

    support = {"top": horizontal(y1), "right": vertical(x2),
               "bottom": horizontal(y2), "left": vertical(x1)}
    minimum, average = min(support.values()), float(np.mean(list(support.values())))
    diagnostics = {
        "bbox": [x1, y1, x2, y2],
        "estimated_wall_thickness": round(thickness, 2),
        "side_support": {key: round(value, 4) for key, value in support.items()},
        "accepted": False,
    }
    # A nearly complete envelope with one unmarked opening is intentionally
    # left open. This fallback is only for widespread segmentation loss.
    if minimum < 0.5 or minimum >= 0.8 or average < 0.7:
        diagnostics["reason"] = "insufficient_or_local_gap_evidence"
        return barrier, np.zeros_like(barrier), diagnostics
    additions = np.zeros_like(barrier)
    cv2.rectangle(additions, (x1, y1), (x2, y2), 255, max(2, round(thickness * .2)))
    additions[barrier > 0] = 0
    diagnostics.update(accepted=True, reason="fragmented_exterior_envelope")
    return cv2.bitwise_or(barrier, additions), additions, diagnostics


def room_candidates(barrier, walls, inferred):
    h, w = barrier.shape
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        cv2.bitwise_not(barrier), connectivity=8)
    ys, xs = np.where(walls > 0)
    if not len(xs):
        return [], {"outside_fraction_in_structure": 1.0, "rejected_components": [],
                    "min_room_area": 0, "interior_area_px": 0}
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    x1, y1, x2, y2 = bbox
    wall_distance = cv2.distanceTransform((walls > 0).astype(np.uint8), cv2.DIST_L2, 5)
    # Estimate from thick wall interiors, avoiding thin contour tails.
    positive = wall_distance[wall_distance > 1]
    thickness = max(2.0, float(np.percentile(positive, 75)) * 2) if positive.size else 2.0
    minimum = max(64, round((x2 - x1) * (y2 - y1) * 0.002), round(thickness ** 2 * 2))
    join_radius = max(1, round(thickness * .125))
    joined_walls = cv2.morphologyEx(
        walls, cv2.MORPH_CLOSE, np.ones((join_radius * 2 + 1,) * 2, np.uint8))
    wall_count, wall_labels, wall_stats, _ = cv2.connectedComponentsWithStats(
        (joined_walls > 0).astype(np.uint8), connectivity=8)
    largest_wall_component = max(
        [int(wall_stats[index, cv2.CC_STAT_AREA]) for index in range(1, wall_count)] + [0])
    outside = np.zeros_like(barrier, dtype=bool)
    rejected, rooms = [], []
    adjacent_inferred = cv2.dilate(inferred, np.ones((5, 5), np.uint8))
    free_distance = cv2.distanceTransform((barrier == 0).astype(np.uint8), cv2.DIST_L2, 5)
    for label in range(1, count):
        x, y, bw, bh, area = map(int, stats[label])
        if x == 0 or y == 0 or x + bw == w or y + bh == h:
            outside[labels == label] = True
            rejected.append({"label": label, "reason": "outside", "area_px": area})
            continue
        component = labels == label
        if area < minimum or float(free_distance[component].max()) < thickness * 0.6:
            rejected.append({"label": label, "reason": "too_small_or_thin", "area_px": area})
            continue
        contours, _ = cv2.findContours(component.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        points = simplify_contour(max(contours, key=cv2.contourArea)).tolist()
        polygon = Polygon(points) if len(points) >= 3 else Polygon()
        if polygon.is_empty or not polygon.is_valid:
            rejected.append({"label": label, "reason": "invalid_polygon", "area_px": area})
            continue
        support_radius = max(2, round(thickness * .1))
        nearby = cv2.dilate(component.astype(np.uint8),
                            np.ones((support_radius * 2 + 1,) * 2, np.uint8))
        support_ids, support_counts = np.unique(wall_labels[nearby > 0], return_counts=True)
        supports = [(int(index), int(support), int(wall_stats[index, cv2.CC_STAT_AREA]))
                    for index, support in zip(support_ids, support_counts)
                    if index and support >= 20]
        if (len(supports) == 1 and largest_wall_component
                and supports[0][2] < largest_wall_component * .05):
            rejected.append({"label": label, "reason": "isolated_wall_enclosure",
                             "area_px": area, "wall_component_area_px": supports[0][2],
                             "largest_wall_component_area_px": largest_wall_component})
            continue
        rooms.append({"id": len(rooms), "polygon": points,
                      "center": [round(float(v), 2) for v in centroids[label]],
                      "bbox": [x, y, x + bw, y + bh], "area_px": area,
                      "area_ratio": round(area / (w * h), 4),
                      "uncertain": bool(np.any(component & (adjacent_inferred > 0)))})
    free_in_bbox = barrier[y1:y2, x1:x2] == 0
    leak_fraction = float(np.count_nonzero(outside[y1:y2, x1:x2] & free_in_bbox)
                          / max(1, np.count_nonzero(free_in_bbox)))
    return rooms, {"outside_fraction_in_structure": round(leak_fraction, 4),
                   "connected_components": count - 1, "rejected_components": rejected,
                   "min_room_area": minimum,
                   "interior_area_px": sum(room["area_px"] for room in rooms)}


def extract_rooms(data, rgb):
    h, w = rgb.shape[:2]
    if (data.get("meta", {}).get("width"), data.get("meta", {}).get("height")) != (w, h):
        raise ValueError("Structural coordinates must refer to the original raster dimensions")
    walls = np.zeros((h, w), dtype=np.uint8)
    for polygon in data.get("walls", []):
        mask = np.zeros_like(walls)
        fill_polygon(mask, polygon)
        walls |= mask
    openings = np.zeros_like(walls)
    topology_only = data.get("topology_only_openings", [])
    for opening in data.get("doors", []) + data.get("windows", []) + topology_only:
        if len(opening.get("outer", [])) >= 3:
            cv2.fillPoly(openings, [polygon_to_int(opening["outer"])], 255)
    barrier = walls | cv2.dilate(openings, np.ones((5, 5), np.uint8))
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    barrier = cv2.dilate(barrier, np.ones((3, 3), np.uint8))
    inferred = np.zeros_like(walls)
    rooms, diagnostics = room_candidates(barrier, walls, inferred)
    passes = [{"name": "known_openings", "rooms": len(rooms), **diagnostics}]
    gray = rgb.min(axis=2)
    wall_shapes = structural_union(data.get("walls", []))
    _, wall_only_thickness = wall_endpoints(walls)
    closures = []
    envelope = None
    envelope_applied = False
    if not rooms and diagnostics["outside_fraction_in_structure"] >= 0.9:
        trial, additions, envelope = exterior_envelope_candidate(barrier, walls)
        trial_inferred = inferred | additions
        trial_rooms, trial_diagnostics = room_candidates(trial, walls, trial_inferred)
        accepted = (envelope["accepted"] and bool(trial_rooms)
                    and trial_diagnostics["outside_fraction_in_structure"] < 0.25)
        envelope["accepted"] = accepted
        passes.append({"name": "exterior_envelope", "rooms": len(trial_rooms),
                       **envelope, **trial_diagnostics})
        if accepted:
            barrier, inferred, rooms, diagnostics = trial, trial_inferred, trial_rooms, trial_diagnostics
            envelope_applied = True
    for relaxed in (False, True):
        if (relaxed and rooms and diagnostics["outside_fraction_in_structure"] < 0.25
                and not envelope_applied):
            break
        extended = envelope_applied and relaxed
        candidates, info = gap_candidates(barrier, gray, relaxed=relaxed, extended=extended,
                                          sampling_thickness=wall_only_thickness,
                                          wall_shapes=wall_shapes)
        trial, additions = apply_closures(barrier, candidates)
        trial_inferred = inferred | additions
        trial_rooms, trial_diagnostics = room_candidates(trial, walls, trial_inferred)
        base_area = diagnostics["interior_area_px"]
        lost_area = max(0, base_area - trial_diagnostics["interior_area_px"])
        closure_area = int(np.count_nonzero(additions))
        unexplained_loss = max(0, lost_area - closure_area)
        area_preserved = (
            closure_area <= base_area * .03 and unexplained_loss <= base_area * .005
            if extended else trial_diagnostics["interior_area_px"] >= base_area * .98
        )
        improved = (len(trial_rooms) > len(rooms)
                    or trial_diagnostics["outside_fraction_in_structure"]
                    < diagnostics["outside_fraction_in_structure"] - 0.02)
        accepted = (bool(candidates) and improved and len(trial_rooms) >= len(rooms)
                    and area_preserved)
        passes.append({"name": "relaxed" if relaxed else "conservative", "rooms": len(trial_rooms),
                       "temporary_closures": len(candidates), "accepted": accepted,
                       "closure_area_px": closure_area,
                       "lost_interior_area_px": lost_area,
                       "unexplained_area_loss_px": unexplained_loss,
                       **info, **trial_diagnostics})
        if accepted:
            barrier, inferred, rooms, diagnostics = trial, trial_inferred, trial_rooms, trial_diagnostics
            closures.extend(candidates)
    endpoints, thickness = wall_endpoints(barrier)
    transverse_candidates = endpoint_wall_arc_candidates(
        barrier, gray, endpoints, thickness, wall_shapes)
    transverse_candidates.sort(key=lambda candidate: candidate["length"])
    transverse_accepted = []
    for candidate in transverse_candidates:
        trial, additions = apply_closures(barrier, [candidate])
        trial_inferred = inferred | additions
        trial_rooms, trial_diagnostics = room_candidates(trial, walls, trial_inferred)
        if (len(trial_rooms) <= len(rooms)
                or trial_diagnostics["outside_fraction_in_structure"]
                >= diagnostics["outside_fraction_in_structure"] - .02
                or trial_diagnostics["interior_area_px"] < diagnostics["interior_area_px"] * .98):
            continue
        barrier, inferred, rooms, diagnostics = trial, trial_inferred, trial_rooms, trial_diagnostics
        closures.append(candidate)
        transverse_accepted.append(candidate)
    passes.append({"name": "transverse_arc", "rooms": len(rooms),
                   "candidate_gaps": len(transverse_candidates),
                   "temporary_closures": len(transverse_accepted),
                   "rejected_candidates": len(transverse_candidates) - len(transverse_accepted),
                   "accepted": bool(transverse_accepted), **diagnostics})
    sealed_windows = []
    considered_windows = 0
    for window in data.get("windows", []) + topology_only:
        additions = window_edge_seal(window, barrier, wall_shapes)
        if additions is None or not np.any(additions):
            continue
        considered_windows += 1
        trial = barrier | additions
        trial_inferred = inferred | additions
        trial_rooms, trial_diagnostics = room_candidates(trial, walls, trial_inferred)
        if (len(trial_rooms) <= len(rooms)
                or trial_diagnostics["outside_fraction_in_structure"]
                >= diagnostics["outside_fraction_in_structure"] - .02
                or not preserves_existing_rooms(rooms, trial_rooms)):
            continue
        sealed_windows.append({"window_id": window.get("id"),
                               "added_pixels": int(np.count_nonzero(additions))})
        barrier, inferred, rooms, diagnostics = trial, trial_inferred, trial_rooms, trial_diagnostics
    passes.append({"name": "window_edge_seals", "rooms": len(rooms),
                   "considered_windows": considered_windows,
                   "accepted_windows": sealed_windows,
                   "accepted": bool(sealed_windows), **diagnostics})
    warnings = []
    if envelope_applied:
        warnings.append("Наружный контур восстановлен предположительно только для topology; проверьте границы.")
    if closures:
        warnings.append("Границы некоторых помещений восстановлены предположительно; проверьте результат.")
    if sealed_windows:
        warnings.append(
            f"Границы окон продлены только в маске помещений ({len(sealed_windows)}); проверьте восстановленные комнаты.")
    if topology_only:
        warnings.append(
            f"Слои повторных окон оставлены только в маске помещений ({len(topology_only)}); "
            "проверьте проёмы и границы комнат.")
    rejected_oblique = sum(
        candidate.get("reason") == "oblique_arc_not_confirmed"
        for phase in passes for candidate in phase.get("rejected_leaf_gaps", []))
    if rejected_oblique:
        warnings.append(
            f"Наклонные разрывы без подтверждённой дуги оставлены открытыми ({rejected_oblique}); "
            "проверьте возможные пропущенные двери и границы помещений.")
    if diagnostics["outside_fraction_in_structure"] >= 0.25:
        warnings.append("Возможны незамкнутые границы и потерянные помещения.")
    if not rooms:
        warnings.append("Замкнутые помещения не найдены.")
    isolated_enclosures = sum(item["reason"] == "isolated_wall_enclosure"
                              for item in diagnostics["rejected_components"])
    if isolated_enclosures:
        warnings.append(
            f"Изолированные небольшие контуры маски стен исключены из комнат ({isolated_enclosures}); "
            "проверьте мебель и возможные пропущенные помещения.")
    rejected_openings = len(data.get("rejected_openings", []))
    automatic_rejections = sum(
        (opening.get("classification") or {}).get("method") != "user_ignored"
        and "duplicate_parallel_window_layer" not in opening.get("reject_reason", [])
        for opening in data.get("rejected_openings", []))
    if automatic_rejections:
        warnings.append(
            f"Оконные кандидаты на изолированных тонких линиях отклонены ({automatic_rejections}); "
            "проверьте окна и возможные пропущенные проёмы.")
    review_doors = sum(door.get("classification", {}).get("requires_review") is True
                       for door in data.get("doors", []))
    if review_doors:
        warnings.append(
            f"Двери без подтверждённой дуги, найденные по полотну и двум косякам "
            f"({review_doors}), требуют проверки."
        )
    review_windows = sum(window.get("classification", {}).get("requires_review") is True
                         for window in data.get("windows", []))
    if review_windows:
        warnings.append(f"Окна, восстановленные по топологии ({review_windows}), требуют проверки.")
    non_rectangular = sum(r["area_px"] / max(1, (r["bbox"][2] - r["bbox"][0])
                                            * (r["bbox"][3] - r["bbox"][1])) < 0.9 for r in rooms)
    if non_rectangular:
        warnings.append("Есть помещения сложной формы; текущий экспорт упрощает их до прямоугольников.")
    result = {"meta": data.get("meta", {}),
              "review_decisions": data.get("review_decisions", []),
              "walls": data.get("walls", []),
              "doors": data.get("doors", []), "windows": data.get("windows", []),
              "topology_only_openings": topology_only, "rooms": rooms,
              "diagnostics": {"version": 2, "passes": passes, "temporary_closures": closures,
                              "exterior_envelope": envelope,
                              "window_edge_seals": sealed_windows,
                              "rooms_found": len(rooms), "uncertain_rooms": sum(r["uncertain"] for r in rooms),
                              "isolated_wall_enclosures": isolated_enclosures,
                              "rejected_openings": rejected_openings,
                              "review_doors": review_doors,
                              "review_windows": review_windows,
                              "rejected_oblique_closures": rejected_oblique,
                              "non_rectangular_rooms": non_rectangular,
                              "warnings": warnings, **diagnostics}}
    return result, barrier, inferred


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("classified_json", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    with Image.open(args.image) as image:
        image = image.convert("RGB")
        result, barrier, inferred = extract_rooms(
            json.loads(args.classified_json.read_text(encoding="utf-8")), np.asarray(image))
        overlay = image.copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    (args.output_dir / f"{stem}_rooms.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    Image.fromarray(barrier).save(args.output_dir / f"{stem}_barrier.png")
    Image.fromarray(inferred).save(args.output_dir / f"{stem}_closures.png")
    draw = ImageDraw.Draw(overlay)
    for wall in result["walls"]:
        for ring in [wall["outer"], *wall.get("holes", [])]:
            if len(ring) >= 3:
                points = [tuple(p) for p in ring]
                draw.line(points + points[:1], fill="red", width=2)
    for closure in result["diagnostics"]["temporary_closures"]:
        draw.line([tuple(closure["start"]), tuple(closure["end"])], fill="cyan", width=4)
    for room in result["rooms"]:
        points = [tuple(p) for p in room["polygon"]]
        draw.line(points + points[:1], fill="magenta", width=3)
        draw.text(tuple(room["center"]), f"Room {room['id'] + 1}" + (" ?" if room["uncertain"] else ""),
                  fill="magenta")
    overlay.save(args.output_dir / f"{stem}_rooms_overlay.png")
    print(json.dumps(result["diagnostics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
