"""Reject weak interior windows and unsupported perpendicular endcaps.

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


def perpendicular_weak_endcaps(windows, wall_shapes):
    """Reject unsupported fragments crossing a longer perpendicular window."""
    geometry = [opening_geometry(window) for window in windows]
    support = [wall_jamb_support(*opening_axis(window), wall_shapes) for window in windows]
    candidates = []
    for index, window in enumerate(windows):
        cx1, cy1, cx2, cy2, length, thickness, orientation = geometry[index]
        if support[index] != [0, 0]:
            continue
        for parent_index, parent in enumerate(windows):
            if index == parent_index:
                continue
            px1, py1, px2, py2, parent_length, parent_thickness, parent_orientation = geometry[parent_index]
            if (orientation == parent_orientation or parent_length < length * 1.2
                    or max(support[parent_index]) < 2):
                continue
            margin = max(4, parent_thickness * .2)
            if orientation == "vertical":
                aligned = px1 - margin <= (cx1 + cx2) / 2 <= px2 + margin
                crossed = min(cy2, py2) - max(cy1, py1)
            else:
                aligned = py1 - margin <= (cy1 + cy2) / 2 <= py2 + margin
                crossed = min(cx2, px2) - max(cx1, px1)
            if aligned and crossed >= max(4, min(thickness, parent_thickness) * .15):
                candidates.append({"id": window.get("id"), "bbox": window.get("bbox"),
                                   "parent_id": parent.get("id"), "jamb_support": [0, 0],
                                   "reason": "perpendicular_weak_jamb_window_endcap"})
                break
    return candidates


def parallel_weak_window_pairs(windows, wall_shapes):
    """Surface overlapping parallel candidates with incomplete jambs for review."""
    geometry = [opening_geometry(window) for window in windows]
    support = [wall_jamb_support(*opening_axis(window), wall_shapes) for window in windows]
    ambiguous = {}
    for i, first in enumerate(geometry):
        for j in range(i + 1, len(geometry)):
            second = geometry[j]
            if first[-1] != second[-1]:
                continue
            vertical = first[-1] == "vertical"
            axis = (1, 3) if vertical else (0, 2)
            transverse = (0, 2) if vertical else (1, 3)
            overlap = max(0, min(first[axis[1]], second[axis[1]])
                          - max(first[axis[0]], second[axis[0]]))
            shorter = min(first[axis[1]] - first[axis[0]],
                          second[axis[1]] - second[axis[0]])
            if shorter <= 0 or overlap / shorter < .75:
                continue
            gap = max(0, first[transverse[0]] - second[transverse[1]],
                      second[transverse[0]] - first[transverse[1]])
            if gap > max(4, min(first[-2], second[-2]) * .5):
                continue
            for index, partner in ((i, j), (j, i)):
                if min(support[index]) >= 2:
                    continue
                uid = windows[index].get("id")
                if not isinstance(uid, int):
                    continue
                ambiguous.setdefault(uid, {"id": uid, "bbox": windows[index].get("bbox"),
                                           "paired_with": [], "jamb_support": support[index],
                                           "reason": "parallel_weak_jamb_window_pair"})
                ambiguous[uid]["paired_with"].append(windows[partner].get("id"))
    return list(ambiguous.values())


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


def interior_window_candidates(classified, rooms, rgb):
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
    return candidates, wall_shapes


def reviewed_window_report(classified, rooms, rgb):
    """Re-evaluate uncertainty after explicit choices without rejecting any choice."""
    candidates, wall_shapes = interior_window_candidates(classified, rooms, rgb)
    windows = classified.get("windows", [])
    endcaps = perpendicular_weak_endcaps(windows, wall_shapes)
    endcap_ids = {item["id"] for item in endcaps}
    across = [item for item in candidates
              if item["reason"] == "same_room_both_sides_weak_jamb_window"
              and item["id"] not in endcap_ids]
    deep = [item for item in candidates
            if item["reason"] == "deep_interior_weak_jamb_window"
            and item["id"] not in endcap_ids]
    return {"version": 4, "method": "reviewed_window_uncertainty",
            "examined_windows": len(windows), "rejected_windows": 0,
            "candidates": [], "ambiguous_windows": len(across),
            "ambiguous_candidates": across, "deep_interior_candidates": deep,
            "perpendicular_endcaps": endcaps,
            "parallel_window_candidates": parallel_weak_window_pairs(
                [window for window in windows if window.get("id") not in endcap_ids],
                wall_shapes),
            "topology_stable": None}


def filter_interior_windows(classified, rooms, rgb):
    candidates, wall_shapes = interior_window_candidates(classified, rooms, rgb)

    filtered_classified = copy.deepcopy(classified)
    filtered_rooms = copy.deepcopy(rooms)
    endcaps = perpendicular_weak_endcaps(classified.get("windows", []), wall_shapes)
    endcap_ids = {candidate["id"] for candidate in endcaps}
    report = {"version": 4, "method": "interior_and_perpendicular_endcap_filter",
              "examined_windows": len(classified.get("windows", [])),
              "rejected_windows": 0, "candidates": [], "ambiguous_windows": 0,
              "ambiguous_candidates": [], "perpendicular_endcaps": endcaps,
              "topology_stable": True}
    deep = [candidate for candidate in candidates
            if candidate["reason"] == "deep_interior_weak_jamb_window"
            and candidate["id"] not in endcap_ids]
    across = [candidate for candidate in candidates
              if candidate["reason"] == "same_room_both_sides_weak_jamb_window"
              and candidate["id"] not in endcap_ids]
    rejected = deep + endcaps
    report.update(rejected_windows=len(rejected), candidates=rejected,
                  ambiguous_windows=len(across), ambiguous_candidates=across)
    rejected_ids = {candidate["id"] for candidate in rejected}
    report["parallel_window_candidates"] = parallel_weak_window_pairs(
        [window for window in classified.get("windows", [])
         if window.get("id") not in rejected_ids],
        wall_shapes)
    if rejected:
        filtered_classified = without_windows(filtered_classified, rejected)
        rebuilt, _, _ = extract_rooms(filtered_classified, rgb)
        stable = same_room_topology(rooms.get("rooms", []), rebuilt.get("rooms", []))
        report["topology_stable"] = stable
        report.update(rooms_before=len(rooms.get("rooms", [])),
                      rooms_after=len(rebuilt.get("rooms", [])))
        if stable:
            if endcaps:
                # Endcaps are opening duplicates, not room-boundary repairs.
                # Re-extraction verifies topology, but its slightly shifted
                # contours can perturb downstream wall serialization.
                if deep:
                    baseline_classified = without_windows(classified, deep)
                    baseline_rooms, _, _ = extract_rooms(baseline_classified, rgb)
                else:
                    baseline_rooms = copy.deepcopy(rooms)
                report["endcap_geometry_preserved"] = True
                filtered_rooms = baseline_rooms
                ids = {candidate["id"] for candidate in endcaps}
                filtered_rooms["windows"] = [window for window in filtered_rooms["windows"]
                                             if window.get("id") not in ids]
                source_windows = {window.get("id"): window for window in classified.get("windows", [])}
                support = [{"outer": copy.deepcopy(source_windows[item["id"]]["outer"]),
                            "holes": copy.deepcopy(source_windows[item["id"]].get("holes", [])),
                            "source_opening_id": item["id"], "parent_window_id": item["parent_id"]}
                           for item in endcaps]
                filtered_rooms.setdefault("reclassified_wall_support", []).extend(support)
                report["reclassified_wall_support"] = [item["id"] for item in endcaps]
            else:
                filtered_rooms = rebuilt
        else:
            ids = {candidate["id"] for candidate in rejected}
            filtered_rooms["windows"] = [window for window in filtered_rooms["windows"]
                                         if window.get("id") not in ids]
        if "opening_recovery" in rooms.get("diagnostics", {}):
            filtered_rooms.setdefault("diagnostics", {})["opening_recovery"] = (
                rooms["diagnostics"]["opening_recovery"])
        filtered_rooms.setdefault("diagnostics", {}).setdefault("warnings", []).append(
            f"Отклонено оконных кандидатов без опоры на стены: {len(rejected)}; "
            "контуры комнат пересчитаны и требуют проверки." if report["topology_stable"] else
            f"Отклонено оконных кандидатов без опоры на стены: {len(rejected)}; "
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
