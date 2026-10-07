import copy
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from door_leaf_evidence import diagonal_leaf_with_jambs, leaf_and_arc, wall_jamb_support
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
from shapely.ops import unary_union


def opening_geometry(opening):
    pts = np.array(opening["outer"], dtype=np.float32)

    minx = float(pts[:, 0].min())
    miny = float(pts[:, 1].min())
    maxx = float(pts[:, 0].max())
    maxy = float(pts[:, 1].max())

    width = maxx - minx
    height = maxy - miny

    orientation = "horizontal" if width >= height else "vertical"
    length = max(width, height)
    thickness = min(width, height)

    return minx, miny, maxx, maxy, length, thickness, orientation


def opening_axis(opening):
    minx, miny, maxx, maxy, _, thickness, orientation = opening_geometry(opening)
    if orientation == "horizontal":
        return np.array([minx, (miny + maxy) / 2]), np.array([maxx, (miny + maxy) / 2]), thickness
    return np.array([(minx + maxx) / 2, miny]), np.array([(minx + maxx) / 2, maxy]), thickness


def structural_union(items):
    shapes = []
    for item in items:
        shape = Polygon(item.get("outer", []), item.get("holes", []))
        if not shape.is_valid:
            shape = shape.buffer(0)
        if not shape.is_empty:
            shapes.append(shape)
    return unary_union(shapes)


def detached_thin_wall_labels(items, height, width):
    """Find sizable isolated linework that is thin relative to the main walls."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for item in items:
        outer = np.rint(np.asarray(item.get("outer", []))).astype(np.int32)
        if len(outer) < 3:
            continue
        polygon_mask = np.zeros_like(mask)
        cv2.fillPoly(polygon_mask, [outer], 255)
        for hole in item.get("holes", []):
            points = np.rint(np.asarray(hole)).astype(np.int32)
            if len(points) >= 3:
                cv2.fillPoly(polygon_mask, [points], 0)
        mask |= polygon_mask
    distance = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    positive = distance[distance > 1]
    if not positive.size:
        return np.zeros_like(mask, dtype=np.int32), set()
    thickness = max(2.0, float(np.percentile(positive, 75)) * 2)
    radius = max(1, round(thickness * .125))
    joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                              np.ones((radius * 2 + 1,) * 2, np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (joined > 0).astype(np.uint8), connectivity=8)
    if count <= 1:
        return labels, set()
    largest = max(int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, count))
    joined_distance = cv2.distanceTransform((joined > 0).astype(np.uint8), cv2.DIST_L2, 5)
    detached = {index for index in range(1, count)
                if (max(64, largest * .01) <= stats[index, cv2.CC_STAT_AREA] < largest * .05
                    and np.percentile(joined_distance[labels == index], 90) < thickness * .25)}
    return labels, detached


def weak_window_on_detached_linework(opening, walls, labels, detached):
    if opening.get("source_class") != "window" or not detached:
        return None
    start, end, width = opening_axis(opening)
    support = wall_jamb_support(start, end, width, walls)
    if min(support) >= 2:
        return None
    polygon = np.rint(np.asarray(opening["outer"])).astype(np.int32)
    neighborhood = np.zeros(labels.shape, dtype=np.uint8)
    cv2.fillPoly(neighborhood, [polygon], 255)
    neighborhood = cv2.dilate(neighborhood, np.ones((7, 7), np.uint8))
    indices, counts = np.unique(labels[neighborhood > 0], return_counts=True)
    contacts = [{"component": int(index), "pixels": int(pixels)}
                for index, pixels in zip(indices, counts)
                if index in detached and pixels >= 20]
    return {"jamb_support": support, "detached_linework": contacts} if contacts else None


def classify_opening(gray, opening):
    minx, miny, maxx, maxy, length, thickness, orientation = \
        opening_geometry(opening)

    # Ищем створку двери вокруг проёма.
    margin = int(max(20, min(100, length * 1.2)))

    x1 = max(0, int(minx - margin))
    y1 = max(0, int(miny - margin))
    x2 = min(gray.shape[1], int(maxx + margin))
    y2 = min(gray.shape[0], int(maxy + margin))

    roi = gray[y1:y2, x1:x2]

    edges = cv2.Canny(
        roi,
        50,
        150,
        apertureSize=3
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(12, int(length * 0.2)),
        minLineLength=max(10, int(length * 0.35)),
        maxLineGap=6
    )

    corners = [
        (minx, miny),
        (minx, maxy),
        (maxx, miny),
        (maxx, maxy),
    ]

    candidates = []

    if lines is not None:
        for line in np.asarray(lines).reshape(-1, 4):
            ax, ay, bx, by = map(float, line)

            ax += x1
            bx += x1
            ay += y1
            by += y1

            dx = bx - ax
            dy = by - ay

            line_length = math.hypot(dx, dy)

            angle = abs(
                math.degrees(
                    math.atan2(dy, dx)
                )
            ) % 180

            # Для горизонтального проёма створка вертикальная.
            if orientation == "horizontal":
                perpendicular = 60 <= angle <= 120
            else:
                # Для вертикального проёма створка горизонтальная.
                perpendicular = angle <= 30 or angle >= 150

            if not perpendicular:
                continue

            # Створка должна начинаться рядом с краем проёма.
            endpoint_distance = min(
                math.hypot(px - cx, py - cy)
                for px, py in [(ax, ay), (bx, by)]
                for cx, cy in corners
            )

            max_distance = max(
                18,
                thickness * 2.5,
                length * 0.25
            )

            if endpoint_distance > max_distance:
                continue

            # Створка примерно сопоставима по длине
            # с шириной дверного проёма.
            if not (
                length * 0.30
                <= line_length
                <= length * 1.70
            ):
                continue

            candidates.append({
                "length": round(line_length, 2),
                "angle": round(angle, 2),
                "distance": round(endpoint_distance, 2),
            })

    opening_type = "door" if candidates else "window"

    return opening_type, candidates


def classify_opening_with_leaf(gray, opening, walls):
    """Require a jamb-anchored leaf and matching swing arc for a door."""
    start, end, width = opening_axis(opening)
    evidence = leaf_and_arc(gray, start, end, width, walls)
    if evidence is not None:
        support = wall_jamb_support(start, end, width, walls)
        evidence = evidence | {"jamb_support": support} if min(support) >= 2 else None
    if evidence is None and opening.get("source_class") == "door":
        evidence = diagonal_leaf_with_jambs(gray, start, end, width, walls)
    return ("door", evidence) if evidence is not None else ("window", None)


def classify_openings(data, rgb):
    height, width = rgb.shape[:2]
    meta = data.get("meta", {})
    if (meta.get("width"), meta.get("height")) != (width, height):
        raise ValueError("Opening coordinates must refer to the original raster dimensions")
    source = data.get("openings")
    if source is None:
        source = data.get("doors", []) + data.get("windows", [])
    openings = copy.deepcopy(source)
    # The darkest RGB channel preserves colored architectural ink better than
    # luminance conversion and matches topology/recovery door evidence.
    gray = rgb.min(axis=2)
    walls = structural_union(data.get("walls", []))
    wall_labels, detached = detached_thin_wall_labels(data.get("walls", []), height, width)
    doors, windows, accepted, rejected = [], [], [], []
    for index, opening in enumerate(openings):
        opening_type, evidence = classify_opening_with_leaf(gray, opening, walls)
        opening["id"] = index
        opening["type"] = opening_type
        _, _, _, _, length, thickness, orientation = opening_geometry(opening)
        opening["orientation"] = orientation
        opening["length"] = round(length, 2)
        opening["thickness"] = round(thickness, 2)
        fallback = evidence is not None and evidence.get("arc_confirmed") is False
        opening["classification"] = {
            "method": "source_door_two_jamb_diagonal_leaf" if fallback else "jamb_leaf_and_arc",
            "confirmed": evidence is not None,
            "requires_review": fallback,
            "evidence": evidence,
            "source_class": opening.get("source_class"),
        }
        rejection = (weak_window_on_detached_linework(opening, walls, wall_labels, detached)
                     if opening_type == "window" else None)
        if rejection:
            opening["classification"].update(
                method="rejected_detached_thin_wall_window", rejection=rejection)
            rejected.append(opening)
            continue
        (doors if opening_type == "door" else windows).append(opening)
        accepted.append(opening)
    return {
        "meta": copy.deepcopy(meta),
        "walls": copy.deepcopy(data.get("walls", [])),
        "doors": doors,
        "windows": windows,
        "openings": accepted,
        "rejected_openings": rejected,
    }


def main():
    if len(sys.argv) != 3:
        print(
            "Usage: python scripts/classify_openings.py "
            "image.jpg result.json"
        )
        sys.exit(1)

    image_path = Path(sys.argv[1])
    json_path = Path(sys.argv[2])

    if not image_path.exists():
        print("Image not found:", image_path)
        sys.exit(1)

    if not json_path.exists():
        print("JSON not found:", json_path)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image = Image.open(image_path).convert("RGB")
    result = classify_openings(data, np.asarray(image))
    doors, windows = result["doors"], result["windows"]
    openings = result["openings"]

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    output_json = (
        output_dir /
        f"{image_path.stem}_classified.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            result,
            f,
            indent=2,
            ensure_ascii=False
        )

    # -----------------------
    # Overlay
    # -----------------------

    draw = ImageDraw.Draw(image)

    # Walls — red
    for wall in result["walls"]:
        points = [
            tuple(p)
            for p in wall["outer"]
        ]

        if len(points) >= 2:
            draw.line(
                points + [points[0]],
                fill="red",
                width=3
            )

    # Doors — green
    for opening in doors:
        points = [
            tuple(p)
            for p in opening["outer"]
        ]

        draw.line(
            points + [points[0]],
            fill="green",
            width=5
        )

    # Windows — blue
    for opening in windows:
        points = [
            tuple(p)
            for p in opening["outer"]
        ]

        draw.line(
            points + [points[0]],
            fill="blue",
            width=5
        )

    for opening in result["rejected_openings"]:
        points = [tuple(point) for point in opening["outer"]]
        draw.line(points + [points[0]], fill="orange", width=5)

    overlay_path = (
        output_dir /
        f"{image_path.stem}_classified_overlay.png"
    )

    image.save(overlay_path)

    print()
    print("Classification done")
    print(f"JSON:    {output_json}")
    print(f"Overlay: {overlay_path}")
    print()
    print(f"Total openings: {len(openings)}")
    print(f"Doors:          {len(doors)}")
    print(f"Windows:        {len(windows)}")
    print(f"Rejected:       {len(result['rejected_openings'])}")
    print()


if __name__ == "__main__":
    main()
