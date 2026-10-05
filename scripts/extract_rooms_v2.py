"""Room topology repair; inferred closures never become exported wall polygons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Polygon

from classify_openings import classify_opening
from door_leaf_evidence import leaf_candidates
from extract_rooms import fill_polygon, polygon_to_int, simplify_contour


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


def axis_pairs(barrier, thickness):
    """Recover wall caps obscured by short skeleton branches in noisy masks."""
    pairs = []
    distance = cv2.distanceTransform((barrier > 0).astype(np.uint8), cv2.DIST_L2, 5)
    length = max(7, round(thickness * 2))
    stride = max(2, round(thickness * 0.65))
    for vertical in (False, True):
        mask = barrier.T if vertical else barrier
        radii = distance.T if vertical else distance
        long_runs = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
        for y in range(stride // 2, mask.shape[0], stride):
            changes = np.diff(np.pad((long_runs[y] > 0).astype(np.int8), 1))
            starts, ends = np.where(changes == 1)[0], np.where(changes == -1)[0]
            for end, start in zip(ends[:-1], starts[1:]):
                if start - end > thickness * 6 or start - end < 3:
                    continue
                left = max(0, int(end) - round(thickness))
                right = min(mask.shape[1] - 1, int(start) + round(thickness))
                width = float(min(radii[y, left], radii[y, right])) * 2
                if width < thickness * 0.6:
                    continue
                a, b = np.array([end - 1, y], dtype=float), np.array([start, y], dtype=float)
                direction = np.array([1., 0.])
                if vertical:
                    a, b, direction = a[::-1], b[::-1], direction[::-1]
                pairs.append(({"point": a, "direction": direction, "width": width},
                              {"point": b, "direction": -direction, "width": width}))
    return pairs


def gap_candidates(barrier, gray, relaxed=False):
    endpoints, thickness = wall_endpoints(barrier)
    candidates = []
    rejected_leaves = []
    pairs = [(i, j, a, endpoints[j], "skeleton") for i, a in enumerate(endpoints)
             for j in range(i + 1, len(endpoints))]
    axes = axis_pairs(barrier, thickness)
    for index, (a, b) in enumerate(axes):
        offset = len(endpoints) + index * 2
        pairs.append((offset, offset + 1, a, b, "axis_caps"))
    for i, j, a, b, method in pairs:
        delta = b["point"] - a["point"]
        gap = float(np.linalg.norm(delta))
        local_width = (a["width"] + b["width"]) / 2
        if gap < 3 or gap > min(local_width * (6 if relaxed else 4.5),
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
                anchored, _, _ = leaf_candidates(gray, a["point"], b["point"], local_width)
                usable = []
                for leaf in anchored:
                    hinge, tip = np.array(leaf["hinge"]), np.array(leaf["tip"])
                    samples = hinge + np.linspace(.25, .9, 20)[:, None] * (tip - hinge)
                    xx = np.clip(np.rint(samples[:, 0]).astype(int), 0, barrier.shape[1] - 1)
                    yy = np.clip(np.rint(samples[:, 1]).astype(int), 0, barrier.shape[0] - 1)
                    if np.mean(barrier[yy, xx] > 0) <= .55:
                        usable.append(leaf)
                if usable:
                    evidence = "door_leaf"
                else:
                    rejected_leaves.append({"start": a["point"].tolist(), "end": b["point"].tolist(),
                                            "width": local_width,
                                            "reason": "leaf_overlaps_barrier" if anchored else "no_jamb_anchored_leaf"})
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
    for candidate in candidates:
        if used.intersection(candidate["indices"]):
            continue
        if any(np.linalg.norm(np.array(candidate["start"]) - c["start"]) < max(candidate["width"], c["width"])
               and np.linalg.norm(np.array(candidate["end"]) - c["end"]) < max(candidate["width"], c["width"])
               for c in closures):
            continue
        used.update(candidate.pop("indices"))
        closures.append(candidate)
    return closures, {"wall_endpoints": len(endpoints), "candidate_gaps": len(candidates),
                      "rejected_leaf_gaps": rejected_leaves,
                      "axis_cap_pairs": len(axes),
                      "estimated_wall_thickness": round(thickness, 2)}


def apply_closures(barrier, closures):
    result = barrier.copy()
    additions = np.zeros_like(barrier)
    for closure in closures:
        a, b = [tuple(int(round(v)) for v in closure[k]) for k in ("start", "end")]
        cv2.line(additions, a, b, 255, max(2, round(closure["width"])))
    additions[barrier > 0] = 0
    return cv2.bitwise_or(result, additions), additions


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
    for opening in data.get("doors", []) + data.get("windows", []):
        if len(opening.get("outer", [])) >= 3:
            cv2.fillPoly(openings, [polygon_to_int(opening["outer"])], 255)
    barrier = walls | cv2.dilate(openings, np.ones((5, 5), np.uint8))
    barrier = cv2.morphologyEx(barrier, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    barrier = cv2.dilate(barrier, np.ones((3, 3), np.uint8))
    inferred = np.zeros_like(walls)
    rooms, diagnostics = room_candidates(barrier, walls, inferred)
    passes = [{"name": "known_openings", "rooms": len(rooms), **diagnostics}]
    gray = rgb.min(axis=2)
    closures = []
    for relaxed in (False, True):
        if relaxed and rooms and diagnostics["outside_fraction_in_structure"] < 0.25:
            break
        candidates, info = gap_candidates(barrier, gray, relaxed=relaxed)
        trial, additions = apply_closures(barrier, candidates)
        trial_inferred = inferred | additions
        trial_rooms, trial_diagnostics = room_candidates(trial, walls, trial_inferred)
        improved = (len(trial_rooms) > len(rooms)
                    or trial_diagnostics["outside_fraction_in_structure"]
                    < diagnostics["outside_fraction_in_structure"] - 0.02)
        accepted = (bool(candidates) and improved and len(trial_rooms) >= len(rooms)
                    and trial_diagnostics["interior_area_px"] >= diagnostics["interior_area_px"] * 0.98)
        passes.append({"name": "relaxed" if relaxed else "conservative", "rooms": len(trial_rooms),
                       "temporary_closures": len(candidates), "accepted": accepted,
                       **info, **trial_diagnostics})
        if accepted:
            barrier, inferred, rooms, diagnostics = trial, trial_inferred, trial_rooms, trial_diagnostics
            closures.extend(candidates)
    warnings = []
    if closures:
        warnings.append("Границы некоторых помещений восстановлены предположительно; проверьте результат.")
    if diagnostics["outside_fraction_in_structure"] >= 0.25:
        warnings.append("Возможны незамкнутые границы и потерянные помещения.")
    if not rooms:
        warnings.append("Замкнутые помещения не найдены.")
    non_rectangular = sum(r["area_px"] / max(1, (r["bbox"][2] - r["bbox"][0])
                                            * (r["bbox"][3] - r["bbox"][1])) < 0.9 for r in rooms)
    if non_rectangular:
        warnings.append("Есть помещения сложной формы; текущий экспорт упрощает их до прямоугольников.")
    result = {"meta": data.get("meta", {}), "walls": data.get("walls", []),
              "doors": data.get("doors", []), "windows": data.get("windows", []), "rooms": rooms,
              "diagnostics": {"version": 2, "passes": passes, "temporary_closures": closures,
                              "rooms_found": len(rooms), "uncertain_rooms": sum(r["uncertain"] for r in rooms),
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
