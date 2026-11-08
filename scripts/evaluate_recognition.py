"""Evaluate recognition artifacts against partial or complete ground truth.

The evaluator deliberately keeps accuracy separate from export validity.  A
ground-truth category must declare either ``count`` or ``geometry`` coverage;
missing categories are reported as unannotated and never treated as empty.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import LineString, Polygon, box as shapely_box
from shapely.ops import unary_union

CATEGORIES = ("rooms", "walls", "doors", "windows", "junctions")


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_ratio(numerator, denominator):
    if denominator == 0:
        return 1.0 if numerator == 0 else 0.0
    return numerator / denominator


def detection_metrics(tp, predicted, expected):
    precision = safe_ratio(tp, predicted)
    recall = safe_ratio(tp, expected)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    return {
        "expected": expected,
        "predicted": predicted,
        "true_positive": tp,
        "false_positive": predicted - tp,
        "false_negative": expected - tp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def polygon(item):
    shape = Polygon(item.get("polygon", item.get("outer", [])), item.get("holes", []))
    return shape if shape.is_valid else shape.buffer(0)


def greedy_matches(scores, threshold):
    matches, left, right = [], set(), set()
    for score, i, j in sorted(scores, reverse=True):
        if score < threshold or i in left or j in right:
            continue
        left.add(i)
        right.add(j)
        matches.append((i, j, score))
    return matches


def unmatched_objects(matches, expected, predicted):
    matched_truth = {i for i, _, _ in matches}
    matched_predictions = {j for _, j, _ in matches}
    return {
        "unmatched_truth": [item.get("id", str(i)) for i, item in enumerate(expected)
                            if i not in matched_truth],
        "unmatched_predictions": [j for j in range(len(predicted))
                                  if j not in matched_predictions],
    }


def evaluate_rooms(expected, predicted, tolerance):
    truth = [polygon(item) for item in expected]
    found = [polygon(item) for item in predicted]
    scores = []
    for i, a in enumerate(truth):
        for j, b in enumerate(found):
            union = a.union(b).area
            scores.append((a.intersection(b).area / union if union else 0.0, i, j))
    threshold = tolerance.get("room_iou", 0.5)
    matches = greedy_matches(scores, threshold)
    result = detection_metrics(len(matches), len(found), len(truth))
    result.update(
        mean_iou=round(sum(score for _, _, score in matches) / max(1, len(matches)), 4),
        iou_threshold=threshold,
        **unmatched_objects(matches, expected, predicted),
        matches=[{
            "truth": expected[i].get("id", str(i)),
            "prediction": j,
            "iou": round(score, 4),
        } for i, j, score in matches],
    )
    return result


def wall_shape(item):
    centerline = item.get("centerline", [])
    if len(centerline) < 2:
        return Polygon()
    thickness = float(item.get("thickness_px", 1))
    return LineString(centerline).buffer(thickness / 2, cap_style="square", join_style="mitre")


def wall_overlap_metrics(expected, found_shapes, regions=None, region_details=True):
    source_shapes = found_shapes
    truth_shapes = [wall_shape(item) for item in expected]
    if regions:
        annotated = unary_union([shapely_box(*bounds) for bounds in regions])
        truth_shapes = [shape.intersection(annotated) for shape in truth_shapes]
        found_shapes = [shape.intersection(annotated) for shape in found_shapes]
    truth = unary_union(truth_shapes) if truth_shapes else Polygon()
    found = unary_union(found_shapes) if found_shapes else Polygon()
    intersection = truth.intersection(found).area
    precision = safe_ratio(intersection, found.area)
    recall = safe_ratio(intersection, truth.area)
    union = truth.union(found).area
    metrics = {
        "expected": len(expected),
        "predicted": sum(not shape.is_empty and shape.area > 0 for shape in found_shapes),
        "area_precision": round(precision, 4),
        "area_recall": round(recall, 4),
        "area_f1": round(safe_ratio(2 * precision * recall, precision + recall), 4),
        "area_iou": round(safe_ratio(intersection, union), 4),
    }
    if regions:
        metrics.update(scope="annotation_regions", annotation_regions=regions)
        if region_details:
            metrics["per_region"] = [
                {"region": region,
                 **{key: local[key] for key in
                    ("area_precision", "area_recall", "area_f1", "area_iou")}}
                for region in regions
                for local in [wall_overlap_metrics(expected, source_shapes,
                                                   [region], region_details=False)]
            ]
    return metrics


def evaluate_walls(expected, predicted, regions=None):
    return wall_overlap_metrics(expected, [polygon(item) for item in predicted], regions)


def exported_wall_shapes(project, geometry):
    if geometry.get("adapter") != "polygon":
        raise ValueError("Export wall evaluation needs polygon adapter coordinates")
    transform = geometry.get("pixel_to_world", {})
    try:
        sx, sy = float(transform["scale_x"]), float(transform["scale_y"])
        ox, oy = float(transform["origin_x"]), float(transform["origin_y"])
        if not all(math.isfinite(value) for value in (sx, sy, ox, oy)) or min(sx, sy) <= 0:
            raise ValueError("Invalid pixel-to-world transform")
        level = project["levels"][0]
        connectors = {item["uuid"]: item for item in level["connectors"]}
        shapes = []
        for segment in level["segments"]:
            a, b = connectors[segment["start"]], connectors[segment["end"]]
            thickness = float(segment["thickness"])
            coords = [(float(a["x"]), float(a["y"])),
                      (float(b["x"]), float(b["y"]))]
            if (not math.isfinite(thickness) or thickness <= 0
                    or not all(math.isfinite(v) for point in coords for v in point)
                    or coords[0] == coords[1]):
                raise ValueError("Invalid exported wall segment")
            world = LineString(coords).buffer(thickness / 2, cap_style="square", join_style="mitre")
            shapes.append(affine_transform(world,
                                           [1 / sx, 0, 0, 1 / sy, -ox / sx, -oy / sy]))
        return shapes
    except (KeyError, IndexError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid exported wall geometry: {exc}") from exc


def line_description(points):
    line = LineString(points)
    a, b = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180
    return np.asarray(line.centroid.coords[0]), angle, line.length


def prediction_span(item):
    if "span" in item:
        return item["span"]
    shape = polygon(item)
    rectangle = shape.minimum_rotated_rectangle
    if rectangle.is_empty or rectangle.geom_type != "Polygon":
        return []
    points = [np.asarray(point, dtype=float) for point in list(rectangle.exterior.coords)[:4]]
    if len(points) != 4:
        return []
    index = max(range(4), key=lambda i: np.linalg.norm(points[(i + 1) % 4] - points[i]))
    # Either long rectangle edge is an arbitrary wall face. Match the opening
    # axis halfway between those faces, so polygon thickness cannot shift the
    # measured center to one side of the annotated span.
    start = (points[index] + points[(index + 3) % 4]) / 2
    end = (points[(index + 1) % 4] + points[(index + 2) % 4]) / 2
    return [start.tolist(), end.tolist()]


def prediction_thickness(item):
    if "thickness_px" in item:
        return item["thickness_px"]
    rectangle = polygon(item).minimum_rotated_rectangle
    if rectangle.is_empty or rectangle.geom_type != "Polygon":
        return None
    points = list(rectangle.exterior.coords)
    return min(LineString([points[i], points[i + 1]]).length for i in range(4))


def angle_difference(a, b):
    difference = abs(a - b) % 180
    return min(difference, 180 - difference)


def opening_scores(expected, predicted, tolerance):
    maximum_angle = tolerance.get("opening_angle_deg", 25)
    minimum_length_ratio = tolerance.get("opening_length_ratio", 0.45)
    scores = []
    for i, truth in enumerate(expected):
        truth_center, truth_angle, truth_length = line_description(truth["span"])
        distance_limit = tolerance.get("opening_center_px", max(12, truth_length * 0.3))
        for j, item in enumerate(predicted):
            span = prediction_span(item)
            if len(span) != 2:
                continue
            center, angle, length = line_description(span)
            distance = float(np.linalg.norm(center - truth_center))
            length_ratio = min(length, truth_length) / max(length, truth_length)
            angle_delta = angle_difference(angle, truth_angle)
            if distance > distance_limit or angle_delta > maximum_angle \
                    or length_ratio < minimum_length_ratio:
                continue
            score = (1 - distance / distance_limit) * (1 - angle_delta / 180) * length_ratio
            scores.append((score, i, j))
    return scores


def evaluate_openings(expected, predicted, tolerance):
    matches = greedy_matches(opening_scores(expected, predicted, tolerance), 1e-9)
    result = detection_metrics(len(matches), len(predicted), len(expected))
    result.update(unmatched_objects(matches, expected, predicted))
    result["matches"] = []
    for i, j, score in matches:
        truth_center, truth_angle, truth_length = line_description(expected[i]["span"])
        center, angle, length = line_description(prediction_span(predicted[j]))
        thickness = prediction_thickness(predicted[j])
        result["matches"].append({
            "truth": expected[i].get("id", str(i)), "prediction": j,
            "prediction_id": predicted[j].get("id"), "score": round(score, 4),
            "predicted_thickness_px": round(thickness, 2) if thickness is not None else None,
            "center_error_px": round(float(np.linalg.norm(center - truth_center)), 2),
            "length_error_px": round(abs(length - truth_length), 2),
            "angle_error_deg": round(angle_difference(angle, truth_angle), 2),
        })
    return result


def exported_openings(project, geometry):
    """Measure actual element spans from their carrier, position and size."""
    if geometry.get("adapter") != "polygon":
        raise ValueError("Export opening evaluation needs polygon adapter coordinates")
    result = {"doors": [], "windows": []}
    try:
        transform = geometry["pixel_to_world"]
        sx, sy = float(transform["scale_x"]), float(transform["scale_y"])
        ox, oy = float(transform["origin_x"]), float(transform["origin_y"])
        if not all(math.isfinite(v) for v in (sx, sy, ox, oy)) or min(sx, sy) <= 0:
            raise ValueError("Invalid pixel-to-world transform")
        level = project["levels"][0]
        elements = [item for item in level.get("elements", [])
                    if item.get("type") in ("doorItem", "windowItem")]
        if not elements:
            return result
        nodes = {item["uuid"]: item for item in level["connectors"]}
        segments = {item["uuid"]: item for item in level["segments"]}
        for element in elements:
            segment = segments[element["wall"]["uuid"]]
            a, b = nodes[segment["start"]], nodes[segment["end"]]
            ax, ay, bx, by = (float(v) for v in (a["x"], a["y"], b["x"], b["y"]))
            position = float(element["wall"]["position"])
            width, thickness = float(element["size"]["x"]), float(segment["thickness"])
            length = math.hypot(bx - ax, by - ay)
            if (not all(math.isfinite(v) for v in
                        (ax, ay, bx, by, position, width, thickness, length))
                    or not 0 <= position <= 100 or min(width, thickness, length) <= 0):
                raise ValueError("Invalid exported opening dimensions or carrier")
            ux, uy = (bx - ax) / length, (by - ay) / length
            cx, cy = ax + (bx - ax) * position / 100, ay + (by - ay) * position / 100
            span = [[(cx + sign * ux * width / 2 - ox) / sx,
                     (cy + sign * uy * width / 2 - oy) / sy] for sign in (-1, 1)]
            category = "doors" if element["type"] == "doorItem" else "windows"
            result[category].append({"id": element["uuid"], "span": span,
                "thickness_px": thickness / (sx * sy * math.hypot(ux / sx, uy / sy))})
        return result
    except (KeyError, IndexError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid exported opening geometry: {exc}") from exc


def rasterize_walls(walls, width, height):
    mask = np.zeros((height, width), np.uint8)
    for item in walls:
        outer = np.rint(np.asarray(item.get("outer", []))).astype(np.int32)
        if len(outer) >= 3:
            cv2.fillPoly(mask, [outer], 255)
            for hole in item.get("holes", []):
                points = np.rint(np.asarray(hole)).astype(np.int32)
                if len(points) >= 3:
                    cv2.fillPoly(mask, [points], 0)
    return mask


def skeletonize(mask):
    image = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.any(image):
        eroded = cv2.erode(image, element)
        skeleton |= image & ~cv2.dilate(eroded, element)
        image = eroded
    return skeleton > 0


def predicted_junctions(walls, width, height):
    pixels = skeletonize(rasterize_walls(walls, width, height))
    points = {(int(x), int(y)) for y, x in np.argwhere(pixels)}

    def neighbors(point):
        x, y = point
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidate = (x + dx, y + dy)
                if (dx or dy) and candidate in points:
                    if dx and dy and ((x + dx, y) in points or (x, y + dy) in points):
                        continue
                    result.append(candidate)
        return result

    adjacency = {point: neighbors(point) for point in points}
    candidates = {point for point in points if len(adjacency[point]) >= 3}
    groups = []
    while candidates:
        pending, group = [candidates.pop()], []
        while pending:
            point = pending.pop()
            group.append(point)
            attached = {neighbor for neighbor in adjacency[point] if neighbor in candidates}
            candidates.difference_update(attached)
            pending.extend(attached)
        groups.append(group)
    result = []
    for group in groups:
        group_set = set(group)
        branches = {neighbor for point in group for neighbor in adjacency[point]
                    if neighbor not in group_set}
        if len(branches) < 3:
            continue
        result.append({
            "point": np.mean(np.asarray(group), axis=0).tolist(),
            "type": "X" if len(branches) >= 4 else "T",
        })
    return result


def evaluate_junctions(expected, predicted, tolerance):
    distance_limit = tolerance.get("junction_center_px", 12)
    scores = []
    for i, truth in enumerate(expected):
        point = np.asarray(truth["point"], dtype=float)
        for j, found in enumerate(predicted):
            if truth["type"] != found["type"]:
                continue
            distance = float(np.linalg.norm(point - np.asarray(found["point"])))
            if distance <= distance_limit:
                scores.append((1 - distance / distance_limit, i, j))
    matches = greedy_matches(scores, 0)
    result = detection_metrics(len(matches), len(predicted), len(expected))
    result.update(unmatched_objects(matches, expected, predicted))
    result["matches"] = [{
        "truth": expected[i].get("id", str(i)),
        "prediction": j,
        "distance_px": round((1 - score) * distance_limit, 2),
    } for i, j, score in matches]
    return result


def validate_ground_truth(data):
    if data.get("version") != 1:
        raise ValueError("Ground truth version must be 1")
    image = data.get("image", {})
    if not all(isinstance(image.get(key), int) and image[key] > 0 for key in ("width", "height")):
        raise ValueError("image.width and image.height must be positive integers")
    width, height = image["width"], image["height"]

    def point(value, label):
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ValueError(f"{label} must be an [x, y] point")
        x, y = value
        if (not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                    and math.isfinite(v) for v in (x, y))
                or not (0 <= x <= width and 0 <= y <= height)):
            raise ValueError(f"{label} must be finite and inside the source image")

    def points(value, label, minimum):
        if not isinstance(value, list) or len(value) < minimum:
            raise ValueError(f"{label} needs at least {minimum} points")
        for index, value_point in enumerate(value):
            point(value_point, f"{label}[{index}]")

    coverage = data.get("coverage", {})
    unknown = set(coverage) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown coverage categories: {sorted(unknown)}")
    regions = data.get("annotation_regions", {})
    if not isinstance(regions, dict) or set(regions) - {"rooms", "walls", "doors", "windows"}:
        raise ValueError("annotation_regions supports only rooms, walls, doors and windows")
    for category, boxes in regions.items():
        if coverage.get(category) != "geometry" or not isinstance(boxes, list) or not boxes:
            raise ValueError(f"annotation_regions.{category} needs geometry coverage and boxes")
        for index, box in enumerate(boxes):
            label = f"annotation_regions.{category}[{index}]"
            if not isinstance(box, list) or len(box) != 4:
                raise ValueError(f"{label} needs [x1, y1, x2, y2]")
            x1, y1, x2, y2 = box
            point([x1, y1], label)
            point([x2, y2], label)
            if x1 >= x2 or y1 >= y2:
                raise ValueError(f"{label} must have positive area")
    for category, mode in coverage.items():
        if mode not in ("count", "geometry"):
            raise ValueError(f"Coverage for {category} must be count or geometry")
        if mode == "count" and category not in data.get("expected_counts", {}):
            raise ValueError(f"Missing expected_counts.{category}")
        if mode == "count":
            count = data["expected_counts"][category]
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise ValueError(f"expected_counts.{category} must be a nonnegative integer")
            continue
        items = data.get(category)
        if not isinstance(items, list):
            raise ValueError(f"Missing geometry array: {category}")
        ids = set()
        for index, item in enumerate(items):
            label = f"{category}[{index}]"
            if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"]:
                raise ValueError(f"{label} needs a nonempty id")
            if item["id"] in ids:
                raise ValueError(f"Duplicate {category} id: {item['id']}")
            ids.add(item["id"])
            if category == "rooms":
                outer = item.get("polygon", item.get("outer"))
                points(outer, f"{label}.polygon", 3)
                holes = item.get("holes", [])
                if not isinstance(holes, list):
                    raise ValueError(f"{label}.holes must be a list")
                for hole_index, hole in enumerate(holes):
                    points(hole, f"{label}.holes[{hole_index}]", 3)
                shape = Polygon(outer, holes)
                if not shape.is_valid or shape.area <= 0:
                    raise ValueError(f"{label} has an invalid or empty polygon")
                if category in regions:
                    annotated = unary_union([shapely_box(*bounds) for bounds in regions[category]])
                    if shape.difference(annotated).area > shape.area * .01:
                        raise ValueError(f"{label} is outside annotation_regions")
            elif category == "walls":
                points(item.get("centerline"), f"{label}.centerline", 2)
                thickness = item.get("thickness_px")
                if (not isinstance(thickness, (int, float)) or isinstance(thickness, bool)
                        or not math.isfinite(thickness) or thickness <= 0
                        or LineString(item["centerline"]).length <= 0):
                    raise ValueError(f"{label} needs positive thickness and length")
                if category in regions:
                    annotated = unary_union([shapely_box(*bounds) for bounds in regions[category]])
                    shape = wall_shape(item)
                    if shape.difference(annotated).area > shape.area * .01:
                        raise ValueError(f"{label} is outside annotation_regions")
            elif category in ("doors", "windows"):
                span = item.get("span")
                if not isinstance(span, list) or len(span) != 2:
                    raise ValueError(f"{label}.span needs exactly two points")
                points(span, f"{label}.span", 2)
                if span[0] == span[1]:
                    raise ValueError(f"{label}.span has zero length")
                if category in regions:
                    center = LineString(span).centroid
                    if not any(x1 <= center.x <= x2 and y1 <= center.y <= y2
                               for x1, y1, x2, y2 in regions[category]):
                        raise ValueError(f"{label}.span center is outside annotation_regions")
            else:
                point(item.get("point"), f"{label}.point")
                if item.get("type") not in ("T", "X"):
                    raise ValueError(f"{label}.type must be T or X")


def evaluate(ground_truth, recognition):
    validate_ground_truth(ground_truth)
    image = ground_truth["image"]
    meta = recognition.get("meta", {})
    if (meta.get("width"), meta.get("height")) != (image["width"], image["height"]):
        raise ValueError("Recognition and ground truth image dimensions differ")
    coverage = ground_truth.get("coverage", {})
    tolerance = ground_truth.get("tolerance", {})
    result = {"version": 1, "coverage": coverage, "metrics": {}, "unannotated": []}
    for category in CATEGORIES:
        mode = coverage.get(category)
        if mode is None:
            result["unannotated"].append(category)
            continue
        predicted = recognition.get(category, [])
        if category == "junctions" and "junctions" not in recognition:
            predicted = predicted_junctions(
                recognition.get("walls", []), image["width"], image["height"])
        if mode == "count":
            expected = ground_truth["expected_counts"][category]
            result["metrics"][category] = {
                "coverage": "count",
                "expected": expected,
                "predicted": len(predicted),
                "count_error": len(predicted) - expected,
                "exact": len(predicted) == expected,
            }
        elif category == "rooms":
            regions = ground_truth.get("annotation_regions", {}).get(category)
            if regions:
                annotated = unary_union([shapely_box(*bounds) for bounds in regions])
                indices = [index for index, item in enumerate(predicted)
                           if (shape := polygon(item)).area > 0
                           and shape.intersection(annotated).area >= shape.area * .5]
                metrics = evaluate_rooms(ground_truth[category],
                                         [predicted[index] for index in indices], tolerance)
                metrics.update(scope="annotation_regions", annotation_regions=regions,
                               scoped_prediction_indices=indices,
                               excluded_prediction_count=len(predicted) - len(indices))
                metrics["unmatched_predictions"] = [
                    indices[index] for index in metrics["unmatched_predictions"]]
                for match in metrics["matches"]:
                    match["prediction"] = indices[match["prediction"]]
            else:
                metrics = evaluate_rooms(ground_truth[category], predicted, tolerance)
            result["metrics"][category] = {"coverage": "geometry", **metrics}
        elif category == "walls":
            result["metrics"][category] = {
                "coverage": "geometry", **evaluate_walls(
                    ground_truth[category], predicted,
                    ground_truth.get("annotation_regions", {}).get(category))}
        elif category in ("doors", "windows"):
            regions = ground_truth.get("annotation_regions", {}).get(category)
            if regions:
                indices = []
                for index, item in enumerate(predicted):
                    span = prediction_span(item)
                    if len(span) != 2:
                        continue
                    center = LineString(span).centroid
                    if any(x1 <= center.x <= x2 and y1 <= center.y <= y2
                           for x1, y1, x2, y2 in regions):
                        indices.append(index)
                scoped = [predicted[index] for index in indices]
                metrics = evaluate_openings(ground_truth[category], scoped, tolerance)
                metrics["scope"] = "annotation_regions"
                metrics["annotation_regions"] = regions
                metrics["scoped_prediction_indices"] = indices
                metrics["excluded_prediction_count"] = len(predicted) - len(scoped)
                metrics["unmatched_predictions"] = [
                    indices[index] for index in metrics["unmatched_predictions"]]
                for match in metrics["matches"]:
                    match["prediction"] = indices[match["prediction"]]
            else:
                metrics = evaluate_openings(ground_truth[category], predicted, tolerance)
            result["metrics"][category] = {
                "coverage": "geometry",
                **metrics,
            }
        else:
            result["metrics"][category] = {
                "coverage": "geometry",
                **evaluate_junctions(ground_truth[category], predicted, tolerance),
            }
    result["all_annotated_counts_exact"] = bool(result["metrics"]) and all(
        (metric["exact"] if metric["coverage"] == "count" else
         metric["area_iou"] >= .999 if "area_iou" in metric else
         metric["true_positive"] == metric["expected"] and metric["false_positive"] == 0)
        for metric in result["metrics"].values())
    return result


def evaluate_files(ground_truth_path, recognition_path, image_path=None,
                   project_path=None, geometry_path=None):
    ground_truth = load_json(ground_truth_path)
    recognition = load_json(recognition_path)
    if image_path is not None:
        expected_hash = ground_truth.get("image", {}).get("sha256")
        actual_hash = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ValueError("Input image SHA-256 does not match ground truth")
    result = evaluate(ground_truth, recognition)
    if (project_path is None) != (geometry_path is None):
        raise ValueError("Export evaluation needs both project and geometry report")
    if project_path is not None:
        project, geometry = load_json(project_path), load_json(geometry_path)
        coverage = ground_truth.get("coverage", {})
        opening_coverage = {key: coverage[key] for key in ("doors", "windows") if key in coverage}
        export_truth = {**ground_truth, "coverage": opening_coverage,
                        "annotation_regions": {key: value for key, value in
                            ground_truth.get("annotation_regions", {}).items()
                            if key in opening_coverage}}
        result["export_geometry"] = evaluate(export_truth, {
            "meta": ground_truth["image"], **exported_openings(project, geometry)})["metrics"]
        if coverage.get("walls") == "geometry":
            result["export_geometry"]["walls"] = {
                "coverage": "geometry",
                **wall_overlap_metrics(
                    ground_truth["walls"], exported_wall_shapes(project, geometry),
                    ground_truth.get("annotation_regions", {}).get("walls"))}
        result["project"] = str(Path(project_path))
        result["geometry_report"] = str(Path(geometry_path))
    result["ground_truth"] = str(Path(ground_truth_path))
    result["recognition"] = str(Path(recognition_path))
    result["ground_truth_sha256"] = hashlib.sha256(
        Path(ground_truth_path).read_bytes()).hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("recognition", type=Path,
                        help="Rooms artifact containing walls/rooms/doors/windows")
    parser.add_argument("--image", type=Path, help="Verify source image SHA-256 when annotated")
    parser.add_argument("--project", type=Path,
                        help="Optional exported project for wall and opening comparison")
    parser.add_argument("--geometry", type=Path,
                        help="Polygon adapter report with pixel-to-world transform")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_files(args.ground_truth, args.recognition, args.image,
                            args.project, args.geometry)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
