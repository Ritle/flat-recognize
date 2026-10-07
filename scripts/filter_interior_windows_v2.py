"""Reject weak window candidates inside a room after provisional room extraction.

Some candidates currently help the global gap search find unrelated wall closures.
Room geometry is therefore retained, but explicitly marked invalid when it depended
on an opening that is removed from the final project.
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


def filter_interior_windows(classified, rooms, geometry, gray):
    height, width = gray.shape
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
    rejected = []
    for window in classified.get("windows", []):
        _, _, _, _, _, thickness, _ = opening_geometry(window)
        point = Point(window.get("center", Polygon(window["outer"]).centroid.coords[0]))
        deep_rooms = [(room_id, round(shape.boundary.distance(point), 2))
                      for room_id, shape in room_shapes
                      if shape.covers(point) and shape.boundary.distance(point) > max(12, thickness * 2)]
        if not deep_rooms:
            continue
        start, end, wall_width = opening_axis(window)
        support = wall_jamb_support(start, end, wall_width, wall_shapes)
        if min(support) != 0 or leaf_and_arc(gray, start, end, wall_width, wall_shapes):
            continue
        rejected.append({"id": window.get("id"), "bbox": window.get("bbox"),
                         "room_id": deep_rooms[0][0], "interior_distance_px": deep_rooms[0][1],
                         "jamb_support": support, "door_leaf_arc": False})

    filtered_classified = copy.deepcopy(classified)
    filtered_rooms = copy.deepcopy(rooms)
    filtered_geometry = copy.deepcopy(geometry)
    report = {"version": 1, "method": "deep_interior_weak_jamb_window",
              "examined_windows": len(classified.get("windows", [])),
              "rejected_windows": len(rejected), "candidates": rejected,
              "room_geometry_provisional": bool(rejected)}
    if rejected:
        ids = {candidate["id"] for candidate in rejected}
        removed = [window for window in filtered_classified["windows"] if window.get("id") in ids]
        filtered_classified["windows"] = [window for window in filtered_classified["windows"]
                                          if window.get("id") not in ids]
        filtered_classified["openings"] = [opening for opening in filtered_classified["openings"]
                                           if opening.get("id") not in ids]
        for window in removed:
            window["reject_reason"] = ["deep_interior_weak_jamb_window"]
            filtered_classified.setdefault("rejected_openings", []).append(window)
        filtered_rooms["windows"] = [window for window in filtered_rooms["windows"]
                                     if window.get("id") not in ids]
        warning = (f"Отклонено оконных кандидатов внутри комнат: {len(rejected)}. "
                   "Геометрия комнат построена с их участием и требует проверки.")
        filtered_rooms.setdefault("diagnostics", {}).setdefault("warnings", []).append(warning)
        filtered_geometry.setdefault("errors", []).append(warning)
    filtered_rooms.setdefault("diagnostics", {})["interior_window_filter"] = report
    return filtered_classified, filtered_rooms, filtered_geometry, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("classified", type=Path)
    parser.add_argument("rooms", type=Path)
    parser.add_argument("geometry", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    args = parser.parse_args()
    image = Image.open(args.image).convert("RGB")
    classified, rooms, geometry, report = filter_interior_windows(
        json.loads(args.classified.read_text(encoding="utf-8")),
        json.loads(args.rooms.read_text(encoding="utf-8")),
        json.loads(args.geometry.read_text(encoding="utf-8")),
        cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    for suffix, data in (("_interior_filtered_classified.json", classified),
                         ("_rooms_filtered.json", rooms),
                         ("_geometry_filtered.json", geometry),
                         ("_interior_windows.json", report)):
        (args.output_dir / (stem + suffix)).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for candidate in report["candidates"]:
        bbox = candidate["bbox"]
        draw.rectangle(bbox, outline=(255, 100, 0), width=5)
        draw.text((bbox[0], max(0, bbox[1] - 15)), str(candidate["id"]), fill=(255, 100, 0))
    overlay.save(args.output_dir / (stem + "_interior_windows_overlay.png"))
    print(f"Interior window filter: {report['rejected_windows']} rejected; "
          f"geometry provisional: {report['room_geometry_provisional']}")


if __name__ == "__main__":
    main()
