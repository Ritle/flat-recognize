"""Propose raster-supported exterior enclosures without changing recognition."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Polygon, box
from shapely.ops import unary_union


def rail_bands(dark, first, last, lo, hi):
    """Find narrow dark bands across a longitudinal sample."""
    first, last = max(0, int(first)), min(dark.shape[0], int(last) + 1)
    lo, hi = max(0, int(lo)), min(dark.shape[1], int(hi) + 1)
    if hi - lo < 15 or last - first < 4:
        return []
    support = dark[first:last, lo:hi].mean(axis=1)
    indices = np.flatnonzero(support >= .8)
    bands = []
    for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1):
        if 0 < len(group) <= 6:
            bands.append((first + int(group[0]), first + int(group[-1])))
    return bands


def unique_pair(bands):
    pairs = [(a, b) for a, b in zip(bands, bands[1:])
             if 3 <= (sum(b) - sum(a)) / 2 <= 18 and b[0] - a[1] >= 2]
    return pairs[0] if len(pairs) == 1 else None


def first_room_crossing(rooms, x, y, direction, height):
    end = height - 1 if direction > 0 else 0
    cut = rooms.intersection(LineString([(x, y), (x, end)]))
    parts = [cut] if cut.geom_type == "LineString" else list(getattr(cut, "geoms", []))
    distances = [direction * (py - y) for part in parts if part.geom_type == "LineString"
                 for _, py in part.coords if direction * (py - y) > 0]
    return y + direction * min(distances) if distances else None


def seeded_enclosure(dark, opening, rooms, walls, windows):
    x0, y0, x1, y1 = opening["bbox"]
    length = x1 - x0
    if length < 30 or y1 - y0 > length * .3:
        return None
    pair = unique_pair(rail_bands(dark, y0 - 6, y1 + 6,
                                  x0 + length * .2, x1 - length * .2))
    if pair is None:
        return None
    along = np.logical_and(*[dark[a:b + 1].mean(axis=0) >= .6 for a, b in pair])
    along = cv2.morphologyEx(along.astype(np.uint8)[None, :], cv2.MORPH_CLOSE,
                             np.ones((1, 3), np.uint8))[0] > 0
    indices = np.flatnonzero(along)
    runs = [group for group in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            if len(group) >= 40 and group[0] <= (x0 + x1) / 2 <= group[-1]
            and min(x1, group[-1]) - max(x0, group[0]) >= length * .7]
    if len(runs) != 1:
        return None
    lo, hi = int(runs[0][0]), int(runs[0][-1])
    top = sum(sum(band) / 2 for band in pair) / 2
    proposals = []
    for direction in (-1, 1):
        base = first_room_crossing(rooms, (lo + hi) / 2, top, direction, dark.shape[0])
        if base is None or not 30 <= abs(base - top) <= min(dark.shape) * .25:
            continue
        a, b = sorted((top + (base - top) * .2, top + (base - top) * .75))
        sides = [unique_pair(rail_bands(dark.T, edge - 24, edge + 24, a, b))
                 for edge in (lo, hi)]
        if any(side is None for side in sides):
            continue
        left, right = [sum(sum(band) / 2 for band in side) / 2 for side in sides]
        if right - left < 40:
            continue
        first, last = sorted((top, base))
        side_support = []
        for side in sides:
            samples = np.logical_and(*[dark[int(first) + 8:int(last) - 7, a:b + 1].mean(axis=1) >= .6
                                       for a, b in side])
            gaps = np.flatnonzero(~samples)
            longest_gap = max((len(group) for group in
                np.split(gaps, np.flatnonzero(np.diff(gaps) > 1) + 1)), default=0)
            side_support.append(float(samples.mean()) if len(samples) and longest_gap <= 6 else 0)
        if min(side_support) < .85:
            continue
        crossings = [first_room_crossing(rooms, x, top, direction, dark.shape[0])
                     for x in (left, right)]
        if any(value is None for value in crossings):
            continue
        if max(crossings + [base]) - min(crossings + [base]) > 12:
            continue
        base = sum(crossings + [base]) / 3
        polygon = box(left, min(top, base), right, max(top, base))
        if polygon.intersection(rooms).area > polygon.area * .05:
            continue
        anchors = [walls.distance(LineString([(x, base - 1), (x, base + 1)]))
                   for x in (left, right)]
        if max(anchors) > 6:
            continue
        edges = [LineString([(left, top), (right, top)]),
                 LineString([(left, top), (left, base)]),
                 LineString([(right, top), (right, base)])]
        seed_ids, supported_edges = [], set()
        for window in windows:
            shape = Polygon(window["outer"])
            if not shape.is_valid or shape.is_empty:
                continue
            wx0, wy0, wx1, wy1 = shape.bounds
            horizontal = wx1 - wx0 >= wy1 - wy0
            for index, edge in enumerate(edges):
                if horizontal != (index == 0):
                    continue
                if edge.intersection(shape.buffer(6)).length >= min(edge.length * .3,
                        max(wx1 - wx0, wy1 - wy0) * .5):
                    seed_ids.append(window.get("id"))
                    supported_edges.add(index)
        if 0 not in supported_edges or len(supported_edges) < 2:
            continue
        proposals.append({"polygon": [list(p) for p in polygon.exterior.coords[:-1]],
            "bbox": list(polygon.bounds), "rail_spans": [[list(p) for p in edge.coords] for edge in edges],
            "seed_ids": sorted(set(seed_ids)), "anchor_distance_px": [round(v, 2) for v in anchors],
            "minimum_side_rail_support": round(min(side_support), 4),
            "outside_fraction": round(1 - polygon.intersection(rooms).area / polygon.area, 4),
            "requires_review": True})
    return proposals[0] if len(proposals) == 1 else None


def detect_exterior_enclosures(classified, room_data, rgb):
    height, width = rgb.shape[:2]
    for data in (classified, room_data):
        if (data.get("meta", {}).get("width"), data.get("meta", {}).get("height")) != (width, height):
            raise ValueError("Exterior enclosure coordinates must match the source raster")
    rooms = unary_union([Polygon(room["polygon"]).buffer(0) for room in room_data.get("rooms", [])])
    walls = unary_union([Polygon(wall["outer"], wall.get("holes", [])).buffer(0)
                         for wall in classified.get("walls", [])])
    report = {"version": 1, "method": "seeded_paired_raster_enclosure", "candidates": []}
    if rooms.is_empty or walls.is_empty:
        return report
    dark = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) < 140
    windows = classified.get("windows", [])
    for transpose in (False, True):
        matrix = [0, 1, 1, 0, 0, 0] if transpose else [1, 0, 0, 1, 0, 0]
        local_windows = []
        for window in windows:
            shape = affine_transform(Polygon(window["outer"]), matrix)
            local_windows.append({"id": window.get("id"), "outer": list(shape.exterior.coords),
                                  "bbox": list(shape.bounds)})
        for window in local_windows:
            proposal = seeded_enclosure(dark.T if transpose else dark, window,
                affine_transform(rooms, matrix), affine_transform(walls, matrix), local_windows)
            if proposal is None:
                continue
            shape = affine_transform(Polygon(proposal["polygon"]), matrix)
            if any(shape.intersection(Polygon(item["polygon"])).area /
                   shape.union(Polygon(item["polygon"])).area > .8 for item in report["candidates"]):
                continue
            proposal["polygon"] = [list(p) for p in shape.exterior.coords[:-1]]
            proposal["bbox"] = list(shape.bounds)
            if transpose:
                proposal["rail_spans"] = [[[y, x] for x, y in edge] for edge in proposal["rail_spans"]]
            proposal["id"] = len(report["candidates"])
            report["candidates"].append(proposal)
    return report


def save_overlay(image_path, report, output):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    for candidate in report["candidates"]:
        points = [tuple(p) for p in candidate["polygon"]]
        draw.line(points + points[:1], fill="orange", width=4)
        draw.text(points[0], f"Exterior contour {candidate['id'] + 1}: review", fill="red")
    image.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("classified", type=Path)
    parser.add_argument("rooms", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    args = parser.parse_args()
    with Image.open(args.image) as source:
        rgb = np.asarray(source.convert("RGB"))
    report = detect_exterior_enclosures(json.loads(args.classified.read_text(encoding="utf8")),
        json.loads(args.rooms.read_text(encoding="utf8")), rgb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
    if args.overlay:
        args.overlay.parent.mkdir(parents=True, exist_ok=True)
        save_overlay(args.image, report, args.overlay)
    print(f"Exterior enclosure proposals: {len(report['candidates'])}")


if __name__ == "__main__":
    main()
