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
from shapely.geometry import LineString, Polygon
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


def evaluate_walls(expected, predicted):
    truth = unary_union([wall_shape(item) for item in expected]) if expected else Polygon()
    found_shapes = [polygon(item) for item in predicted]
    found = unary_union(found_shapes) if found_shapes else Polygon()
    intersection = truth.intersection(found).area
    precision = safe_ratio(intersection, found.area)
    recall = safe_ratio(intersection, truth.area)
    union = truth.union(found).area
    return {
        "expected": len(expected),
        "predicted": len(predicted),
        "area_precision": round(precision, 4),
        "area_recall": round(recall, 4),
        "area_f1": round(safe_ratio(2 * precision * recall, precision + recall), 4),
        "area_iou": round(safe_ratio(intersection, union), 4),
    }


def line_description(points):
    line = LineString(points)
    a, b = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
    angle = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 180
    return np.asarray(line.centroid.coords[0]), angle, line.length


def prediction_span(item):
    shape = polygon(item)
    rectangle = shape.minimum_rotated_rectangle
    points = list(rectangle.exterior.coords) if not rectangle.is_empty else []
    edges = [(points[i], points[i + 1]) for i in range(max(0, len(points) - 1))]
    return max(edges, key=lambda edge: LineString(edge).length) if edges else []


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
    result["matches"] = [{
        "truth": expected[i].get("id", str(i)),
        "prediction": j,
        "score": round(score, 4),
    } for i, j, score in matches]
    return result


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
    coverage = data.get("coverage", {})
    unknown = set(coverage) - set(CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown coverage categories: {sorted(unknown)}")
    for category, mode in coverage.items():
        if mode not in ("count", "geometry"):
            raise ValueError(f"Coverage for {category} must be count or geometry")
        if mode == "count" and category not in data.get("expected_counts", {}):
            raise ValueError(f"Missing expected_counts.{category}")
        if mode == "geometry" and not isinstance(data.get(category), list):
            raise ValueError(f"Missing geometry array: {category}")


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
            result["metrics"][category] = {
                "coverage": "geometry", **evaluate_rooms(ground_truth[category], predicted, tolerance)}
        elif category == "walls":
            result["metrics"][category] = {
                "coverage": "geometry", **evaluate_walls(ground_truth[category], predicted)}
        elif category in ("doors", "windows"):
            result["metrics"][category] = {
                "coverage": "geometry",
                **evaluate_openings(ground_truth[category], predicted, tolerance),
            }
        else:
            result["metrics"][category] = {
                "coverage": "geometry",
                **evaluate_junctions(ground_truth[category], predicted, tolerance),
            }
    result["all_annotated_counts_exact"] = all(
        metric.get("exact", True) for metric in result["metrics"].values())
    return result


def evaluate_files(ground_truth_path, recognition_path, image_path=None):
    ground_truth = load_json(ground_truth_path)
    recognition = load_json(recognition_path)
    if image_path is not None:
        expected_hash = ground_truth.get("image", {}).get("sha256")
        actual_hash = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        if expected_hash and actual_hash != expected_hash:
            raise ValueError("Input image SHA-256 does not match ground truth")
    result = evaluate(ground_truth, recognition)
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
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate_files(args.ground_truth, args.recognition, args.image)
    rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
