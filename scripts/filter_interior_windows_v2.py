"""Reject weak interior windows and rebuild rooms without them.

The topology is accepted only when the room polygons survive re-extraction.
"""

import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import Point, Polygon

from classify_openings import opening_axis, opening_geometry, structural_union
from door_leaf_evidence import leaf_and_arc, wall_jamb_support
from extract_rooms_v2 import extract_rooms


def same_room_topology(before, after):
    if len(before) != len(after):
        return False
    unmatched = [Polygon(room["polygon"]) for room in after]
    for room in before:
        source = Polygon(room["polygon"])
        if not unmatched or not source.is_valid or source.area <= 0:
            return False
        scores = [source.intersection(shape).area / source.union(shape).area
                  if shape.is_valid and shape.area > 0 else 0 for shape in unmatched]
        index = int(np.argmax(scores))
        if scores[index] < .98:
            return False
        unmatched.pop(index)
    return True


def same_room_across_opening(start, end, wall_width, room_shapes):
    """Find a room present on both sides along the full candidate span."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1:
        return None
    normal = np.array([-axis[1], axis[0]]) / length
    offset = max(12, wall_width * 2)
    for room_id, shape in room_shapes:
        if all(shape.covers(Point(start + axis * fraction + normal * offset * sign))
               for fraction in (.25, .5, .75) for sign in (-1, 1)):
            return room_id
    return None


def without_windows(classified, candidates):
    result = copy.deepcopy(classified)
    reasons = {candidate["id"]: candidate["reason"] for candidate in candidates}
    removed = [window for window in result["windows"] if window.get("id") in reasons]
    result["windows"] = [window for window in result["windows"] if window.get("id") not in reasons]
    result["openings"] = [opening for opening in result["openings"]
                          if opening.get("id") not in reasons]
    for window in removed:
        window["reject_reason"] = [reasons[window["id"]]]
        result.setdefault("rejected_openings", []).append(window)
    return result


def filter_interior_windows(classified, rooms, rgb):
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    for name, data in (("classified", classified), ("rooms", rooms)):
        meta = data.get("meta", {})
        if (meta.get("width"), meta.get("height")) != (width, height):
            raise ValueError(f"{name} coordinates must match the source raster")
    wall_shapes = structural_union(classified.get("walls", []))
    room_shapes = []
    for room in rooms.get("rooms", []):
        polygon = Polygon(room.get("polygon", []))
        if polygon.is_valid and polygon.area > 0:
            room_shapes.append((room.get("id"), polygon))
    candidates = []
    for window in classified.get("windows", []):
        _, _, _, _, _, thickness, _ = opening_geometry(window)
        point = Point(window.get("center", Polygon(window["outer"]).centroid.coords[0]))
        deep_rooms = [(room_id, round(shape.boundary.distance(point), 2))
                      for room_id, shape in room_shapes
                      if shape.covers(point) and shape.boundary.distance(point) > max(12, thickness * 2)]
        start, end, wall_width = opening_axis(window)
        across_room = same_room_across_opening(start, end, wall_width, room_shapes)
        if not deep_rooms and across_room is None:
            continue
        support = wall_jamb_support(start, end, wall_width, wall_shapes)
        if min(support) != 0 or leaf_and_arc(gray, start, end, wall_width, wall_shapes):
            continue
        reason = ("deep_interior_weak_jamb_window" if deep_rooms else
                  "same_room_both_sides_weak_jamb_window")
        candidates.append({"id": window.get("id"), "bbox": window.get("bbox"),
                         "room_id": deep_rooms[0][0] if deep_rooms else across_room,
                         "interior_distance_px": deep_rooms[0][1] if deep_rooms else None,
                         "across_distance_px": max(12, wall_width * 2) if across_room is not None else None,
                         "jamb_support": support, "door_leaf_arc": False, "reason": reason})

    filtered_classified = copy.deepcopy(classified)
    filtered_rooms = copy.deepcopy(rooms)
    report = {"version": 3, "method": "deep_filter_and_same_room_observation",
              "examined_windows": len(classified.get("windows", [])),
              "rejected_windows": 0, "candidates": [], "ambiguous_windows": 0,
              "ambiguous_candidates": [],
              "topology_stable": True}
    deep = [candidate for candidate in candidates
            if candidate["reason"] == "deep_interior_weak_jamb_window"]
    across = [candidate for candidate in candidates
              if candidate["reason"] == "same_room_both_sides_weak_jamb_window"]
    report.update(rejected_windows=len(deep), candidates=deep,
                  ambiguous_windows=len(across), ambiguous_candidates=across)
    if deep:
        filtered_classified = without_windows(filtered_classified, deep)
        rebuilt, _, _ = extract_rooms(filtered_classified, rgb)
        stable = same_room_topology(rooms.get("rooms", []), rebuilt.get("rooms", []))
        report["topology_stable"] = stable
        report.update(rooms_before=len(rooms.get("rooms", [])),
                      rooms_after=len(rebuilt.get("rooms", [])))
        if stable:
            filtered_rooms = rebuilt
        else:
            ids = {candidate["id"] for candidate in deep}
            filtered_rooms["windows"] = [window for window in filtered_rooms["windows"]
                                         if window.get("id") not in ids]
        if "opening_recovery" in rooms.get("diagnostics", {}):
            filtered_rooms.setdefault("diagnostics", {})["opening_recovery"] = (
                rooms["diagnostics"]["opening_recovery"])
        filtered_rooms.setdefault("diagnostics", {}).setdefault("warnings", []).append(
            f"Отклонено оконных кандидатов внутри комнат: {len(deep)}; "
            "контуры комнат пересчитаны и требуют проверки." if report["topology_stable"] else
            f"Отклонено оконных кандидатов внутри комнат: {len(deep)}; "
            "повторное выделение изменило контуры комнат, экспорт заблокирован.")
    filtered_rooms.setdefault("diagnostics", {})["interior_window_filter"] = report
    return filtered_classified, filtered_rooms, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("classified", type=Path)
    parser.add_argument("rooms", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    classified, rooms, report = filter_interior_windows(
        json.loads(args.classified.read_text(encoding="utf-8")),
        json.loads(args.rooms.read_text(encoding="utf-8")),
        np.asarray(image))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    for suffix, data in (("_interior_filtered_classified.json", classified),
                         ("_rooms_filtered.json", rooms),
                         ("_interior_windows.json", report)):
        (args.output_dir / (stem + suffix)).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for key, color in (("candidates", (255, 100, 0)),
                       ("ambiguous_candidates", (255, 0, 180))):
        for candidate in report[key]:
            bbox = candidate["bbox"]
            draw.rectangle(bbox, outline=color, width=5)
            draw.text((bbox[0], max(0, bbox[1] - 15)), str(candidate["id"]), fill=color)
    overlay.save(args.output_dir / (stem + "_interior_windows_overlay.png"))
    print(f"Interior window filter: {report['rejected_windows']} rejected, "
          f"{report['ambiguous_windows']} ambiguous; topology stable: {report['topology_stable']}")


if __name__ == "__main__":
    main()
