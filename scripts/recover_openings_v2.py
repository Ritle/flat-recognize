"""Conservative recovery from accepted topology gaps; never invent unmarked passages.

An arc-less diagonal leaf can only promote an accepted topology gap when both
wall jambs and a clear leaf tip are present; every recovered door requires review.
Short segmentation window candidates can be restored only where a raster-line
closure, polygon overlap, and structural jamb evidence agree.
Original segmentation, classification and room extraction artifacts remain intact.
"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
from door_leaf_evidence import diagonal_leaf_with_jambs, leaf_and_arc
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


def recover_openings(classified, rooms, rgb, prediction=None):
    height, width = rgb.shape[:2]
    for data in (classified, rooms, prediction):
        if data is None:
            continue
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
        method = "recovered_leaf_and_arc"
        if evidence is None:
            evidence = diagonal_leaf_with_jambs(gray, a, b, width, walls)
            method = "recovered_two_jamb_diagonal_leaf"
            if evidence is None:
                decision["reason"] = "leaf_and_arc_not_confirmed"
                continue
        opening = {"id": next_id, "type": "door", "outer": [list(p) for p in shape.exterior.coords[:-1]],
                   "holes": [], "orientation": "horizontal" if abs(axis[0]) >= abs(axis[1]) else "vertical",
                   "length": round(length, 3), "thickness": width,
                   "classification": {"method": method, "requires_review": True,
                                      "evidence": evidence},
                   "recovery": {"closure_index": index, "requires_review": True}}
        next_id += 1
        recovered_classified.setdefault("doors", []).append(opening)
        recovered_classified.setdefault("openings", []).append(copy.deepcopy(opening))
        recovered_rooms.setdefault("doors", []).append(copy.deepcopy(opening))
        known = known.union(shape)
        reason = ("leaf_and_arc_confirmed" if method == "recovered_leaf_and_arc" else
                  "two_jamb_diagonal_leaf_confirmed")
        decision.update(status="recovered", reason=reason, method=method, opening_id=opening["id"],
                        outer=opening["outer"], evidence=evidence)
    count = sum(d["status"] == "recovered" for d in decisions)
    window_decisions = []
    for candidate in (prediction or {}).get("diagnostics", {}).get("rejected", []):
        decision = {"source_id": candidate.get("id"), "status": "rejected"}
        window_decisions.append(decision)
        if (candidate.get("reject_reason") != ["not_elongated"]
                or candidate.get("source_class") not in ("door", "window")
                or candidate.get("aspect_ratio", 0) < 1.4):
            decision["reason"] = "not_short_window_candidate"
            continue
        shape = Polygon(candidate.get("outer", []))
        if not shape.is_valid:
            shape = shape.buffer(0)
        if shape.is_empty or shape.geom_type != "Polygon" or shape.area <= 0:
            decision["reason"] = "invalid_candidate_polygon"
            continue
        if known.intersection(shape).area > shape.area * .2:
            decision["reason"] = "existing_opening"
            continue
        matched = None
        for index, closure in enumerate(rooms.get("diagnostics", {}).get("temporary_closures", [])):
            a, b = np.array(closure["start"], float), np.array(closure["end"], float)
            length, gap_width = float(np.linalg.norm(b - a)), float(closure["width"])
            if (closure.get("evidence") != "raster_line" or gap_width <= 0
                    or length < max(30, gap_width * 1.5)):
                continue
            axis = (b - a) / length
            if max(abs(axis)) < math.cos(math.radians(12)):
                continue
            bounds = shape.bounds
            horizontal = abs(axis[0]) >= abs(axis[1])
            if horizontal != (bounds[2] - bounds[0] >= bounds[3] - bounds[1]):
                continue
            strip = gap_polygon(a, b, gap_width)
            overlap = shape.intersection(strip).area / min(shape.area, strip.area)
            if overlap < .8 or shape.distance(walls) > max(6, gap_width * .25):
                continue
            jambs = walls.buffer(3)
            support = [sum(jambs.covers(Point(p + direction * gap_width * t))
                           for t in (.25, .75, 1.25))
                       for p, direction in ((a, -axis), (b, axis))]
            if min(support) < 1 or max(support) < 2:
                continue
            matched = (index, round(overlap, 3), support)
            break
        if matched is None:
            decision["reason"] = "no_supported_raster_line_gap"
            continue
        closure_index, overlap, jamb_support = matched
        opening = copy.deepcopy(candidate)
        opening.pop("reject_reason", None)
        opening.update(id=next_id, type="window", outer=[list(p) for p in shape.exterior.coords[:-1]],
                       classification={"method": "recovered_raster_line_window",
                                       "requires_review": True, "source_class": candidate["source_class"],
                                       "evidence": {"closure_index": closure_index,
                                                    "overlap": overlap, "jamb_support": jamb_support}},
                       recovery={"closure_index": closure_index, "requires_review": True})
        next_id += 1
        recovered_classified.setdefault("windows", []).append(opening)
        recovered_classified.setdefault("openings", []).append(copy.deepcopy(opening))
        recovered_rooms.setdefault("windows", []).append(copy.deepcopy(opening))
        known = known.union(shape)
        decision.update(status="recovered", reason="raster_line_and_rejected_polygon",
                        closure_index=closure_index, opening_id=opening["id"], outer=opening["outer"],
                        overlap=overlap, jamb_support=jamb_support)
    window_count = sum(d["status"] == "recovered" for d in window_decisions)
    report = {"version": 2, "method": "jamb_leaf_arc_or_reviewed_diagonal",
              "window_method": "rejected_polygon_raster_line_jambs", "recovered_doors": count,
              "candidates": decisions, "recovered_windows": window_count,
              "window_candidates": window_decisions, "requires_review": bool(count or window_count)}
    recovered_rooms.setdefault("diagnostics", {})["opening_recovery"] = report
    if count:
        recovered_rooms["diagnostics"].setdefault("warnings", []).append(
            f"Восстановлены пропущенные двери ({count}) по полотну и дуге открывания; проверьте проёмы.")
    if window_count:
        recovered_rooms["diagnostics"].setdefault("warnings", []).append(
            f"Оконные кандидаты, отклонённые по форме, восстановлены по растровой линии "
            f"и стеновым торцам ({window_count}); проверьте окна.")
    return recovered_classified, recovered_rooms, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("classified", type=Path)
    parser.add_argument("rooms", type=Path)
    parser.add_argument("--prediction", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    classified, rooms, report = recover_openings(
        json.loads(args.classified.read_text(encoding="utf-8")),
        json.loads(args.rooms.read_text(encoding="utf-8")), np.asarray(image),
        json.loads(args.prediction.read_text(encoding="utf-8")) if args.prediction else None)
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
    for decision in report["window_candidates"]:
        if decision["status"] == "recovered":
            points = [tuple(p) for p in decision["outer"]]
            draw.line(points + points[:1], fill="cyan", width=4)
    image.save(args.output_dir / (args.image.stem + "_opening_recovery_overlay.png"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
