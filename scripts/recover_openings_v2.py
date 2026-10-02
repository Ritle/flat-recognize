"""Conservative recovery from accepted topology gaps; never invent unmarked passages.

Only a jamb-anchored leaf AND a matching swing arc can promote a gap to a door.
Original segmentation, classification and room extraction artifacts remain intact.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


def structural_union(items):
    shapes = []
    for item in items:
        shape = Polygon(item["outer"], item.get("holes", []))
        if not shape.is_valid:
            shape = shape.buffer(0)
        if not shape.is_empty:
            shapes.append(shape)
    return unary_union(shapes)


def gap_polygon(start, end, width):
    return LineString([start, end]).buffer(width / 2, cap_style=2)


def leaf_and_arc(gray, start, end, width, walls):
    """Return reproducible source-coordinate evidence, without altering V1 classifier."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    axis = (end - start) / length
    normal = np.array([-axis[1], axis[0]])
    corners = [(p + sign * normal * width / 2, direction)
               for p, direction in ((start, axis), (end, -axis)) for sign in (-1, 1)]
    margin = length * 1.5 + width
    x0, y0 = np.maximum(0, np.floor(np.minimum(start, end) - margin)).astype(int)
    x1, y1 = np.minimum(gray.shape[::-1], np.ceil(np.maximum(start, end) + margin)).astype(int)
    edges = cv2.Canny(gray[y0:y1, x0:x1], 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(10, int(length * .22)),
                            minLineLength=max(10, int(length * .6)), maxLineGap=3)
    if lines is None:
        return None
    evidence = []
    for raw in lines.reshape(-1, 4):
        a, b = raw.reshape(2, 2).astype(float) + [x0, y0]
        line_length = float(np.linalg.norm(b - a))
        if not .6 * length <= line_length <= 1.35 * length:
            continue
        for near, tip in ((a, b), (b, a)):
            hinge, closed = min(corners, key=lambda c: np.linalg.norm(c[0] - near))
            hinge_distance = float(np.linalg.norm(near - hinge))
            # Hough can truncate a leaf at its junction with the jamb. Its supporting
            # line must still pass through the hinge, with only a short missing end.
            line_axis = (tip - near) / line_length
            hinge_offset = hinge - near
            perpendicular_distance = abs(float(hinge_offset[0] * line_axis[1]
                                               - hinge_offset[1] * line_axis[0]))
            if (hinge_distance > max(3, width * .5, length * .18)
                    or perpendicular_distance > max(3, width * .4, length * .04)):
                continue
            radius = float(np.linalg.norm(tip - hinge))
            if not .65 * length <= radius <= 1.35 * length:
                continue
            leaf = (tip - hinge) / radius
            angle = math.atan2(float(closed[0] * leaf[1] - closed[1] * leaf[0]), float(closed @ leaf))
            if not math.radians(35) <= abs(angle) <= math.radians(145):
                continue
            # A perpendicular wall edge is not a leaf. Ignore its attached first quarter.
            free_leaf = LineString([hinge + (tip - hinge) * .25, tip])
            if free_leaf.intersection(walls.buffer(1)).length > free_leaf.length * .2:
                continue
            theta = math.atan2(closed[1], closed[0])
            hits = []
            tolerance = max(2, min(4, width * .18))
            for fraction in np.linspace(.18, .82, 9):
                t = theta + angle * fraction
                p = hinge + radius * np.array([math.cos(t), math.sin(t)])
                px, py = np.rint(p - [x0, y0]).astype(int)
                r = int(math.ceil(tolerance))
                patch = edges[max(0, py-r):min(edges.shape[0], py+r+1),
                              max(0, px-r):min(edges.shape[1], px+r+1)]
                # Known wall ink cannot count as a swing arc.
                hits.append(bool(patch.size and patch.any() and not walls.buffer(1).covers(Point(p))))
            if sum(hits) < 7 or not all(any(hits[i:i+3]) for i in (0, 3, 6)):
                continue
            evidence.append({"hinge": hinge.tolist(), "tip": tip.tolist(),
                             "leaf_length": round(line_length, 3),
                             "hinge_distance": round(hinge_distance, 3),
                             "swing_angle_degrees": round(math.degrees(angle), 3),
                             "arc_hits": sum(hits), "arc_samples": len(hits)})
    return max(evidence, key=lambda e: (e["arc_hits"], -e["hinge_distance"])) if evidence else None


def recover_openings(classified, rooms, rgb):
    height, width = rgb.shape[:2]
    for data in (classified, rooms):
        if (data.get("meta", {}).get("width"), data.get("meta", {}).get("height")) != (width, height):
            raise ValueError("Recovery coordinates must refer to the original raster dimensions")
    recovered_classified, recovered_rooms = copy.deepcopy(classified), copy.deepcopy(rooms)
    walls = structural_union(classified.get("walls", []))
    known = structural_union(classified.get("doors", []) + classified.get("windows", []))
    gray = rgb.min(axis=2)
    decisions = []
    existing_ids = [o.get("id") for o in classified.get("doors", []) + classified.get("windows", [])]
    next_id = max([i for i in existing_ids if isinstance(i, int)] + [-1]) + 1
    for index, closure in enumerate(rooms.get("diagnostics", {}).get("temporary_closures", [])):
        decision = {"closure_index": index, "start": closure["start"], "end": closure["end"],
                    "width": closure["width"], "status": "rejected"}
        decisions.append(decision)
        a, b = np.array(closure["start"], float), np.array(closure["end"], float)
        length, width = float(np.linalg.norm(b - a)), float(closure["width"])
        if closure.get("evidence") != "door_leaf" or width <= 0 or length < max(15, width * 1.5):
            decision["reason"] = "no_leaf_candidate"
            continue
        axis = (b - a) / length
        if max(abs(axis)) < math.cos(math.radians(12)):
            decision["reason"] = "diagonal_opening_not_supported"
            continue
        shape = gap_polygon(a, b, width)
        if known.intersection(shape).area > shape.area * .2:
            decision["reason"] = "existing_opening"
            continue
        jambs = walls.buffer(3)
        if any(sum(jambs.covers(Point(p + direction * width * t)) for t in (.25, .75, 1.25)) < 2
               for p, direction in ((a, -axis), (b, axis))):
            decision["reason"] = "missing_wall_jamb"
            continue
        if walls.intersection(shape).area > shape.area * .25:
            decision["reason"] = "gap_contains_wall"
            continue
        evidence = leaf_and_arc(gray, a, b, width, walls)
        if evidence is None:
            decision["reason"] = "leaf_and_arc_not_confirmed"
            continue
        opening = {"id": next_id, "type": "door", "outer": [list(p) for p in shape.exterior.coords[:-1]],
                   "holes": [], "orientation": "horizontal" if abs(axis[0]) >= abs(axis[1]) else "vertical",
                   "length": round(length, 3), "thickness": width,
                   "classification": {"method": "recovered_leaf_and_arc", "evidence": evidence},
                   "recovery": {"closure_index": index, "requires_review": True}}
        next_id += 1
        recovered_classified.setdefault("doors", []).append(opening)
        recovered_classified.setdefault("openings", []).append(copy.deepcopy(opening))
        recovered_rooms.setdefault("doors", []).append(copy.deepcopy(opening))
        known = known.union(shape)
        decision.update(status="recovered", reason="leaf_and_arc_confirmed", opening_id=opening["id"],
                        outer=opening["outer"], evidence=evidence)
    count = sum(d["status"] == "recovered" for d in decisions)
    report = {"version": 1, "method": "jamb_leaf_and_arc", "recovered_doors": count,
              "candidates": decisions, "requires_review": bool(count)}
    recovered_rooms.setdefault("diagnostics", {})["opening_recovery"] = report
    if count:
        recovered_rooms["diagnostics"].setdefault("warnings", []).append(
            f"Восстановлены пропущенные двери ({count}) по полотну и дуге открывания; проверьте проёмы.")
    return recovered_classified, recovered_rooms, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("classified", type=Path)
    parser.add_argument("rooms", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    classified, rooms, report = recover_openings(
        json.loads(args.classified.read_text(encoding="utf-8")),
        json.loads(args.rooms.read_text(encoding="utf-8")), np.asarray(image))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, data in (("_recovered_classified.json", classified), ("_rooms_recovered.json", rooms),
                         ("_opening_recovery.json", report)):
        (args.output_dir / (args.image.stem + suffix)).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    draw = ImageDraw.Draw(image)
    for decision in report["candidates"]:
        confirmed = decision["status"] == "recovered"
        draw.line([tuple(decision["start"]), tuple(decision["end"])],
                  fill="lime" if confirmed else "orange", width=4)
        if confirmed:
            e = decision["evidence"]
            draw.line([tuple(e["hinge"]), tuple(e["tip"])], fill="blue", width=3)
    image.save(args.output_dir / (args.image.stem + "_opening_recovery_overlay.png"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
