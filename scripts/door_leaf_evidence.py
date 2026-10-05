"""Source-coordinate, jamb-anchored leaf evidence shared by V2 topology and recovery."""

import math

import cv2
import numpy as np
from shapely.geometry import Point


def extend_ink_line(a, b, edges, origin, maximum):
    direction = (b - a) / np.linalg.norm(b - a)
    ends = []
    for end, sign in ((a, -1), (b, 1)):
        last, missing = end, 0
        for distance in range(1, int(maximum) + 1):
            point = end + sign * direction * distance
            x, y = np.rint(point - origin).astype(int)
            if not (1 <= x < edges.shape[1] - 1 and 1 <= y < edges.shape[0] - 1):
                break
            if edges[y-1:y+2, x-1:x+2].any():
                last, missing = point, 0
            else:
                missing += 1
                if missing > 3:
                    break
        ends.append(last)
    return ends


def leaf_candidates(gray, start, end, width):
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    if length < 15 or width <= 0:
        return [], None, None
    axis = (end - start) / length
    normal = np.array([-axis[1], axis[0]])
    margin = length * 1.5 + width
    x0, y0 = np.maximum(0, np.floor(np.minimum(start, end) - margin)).astype(int)
    x1, y1 = np.minimum(gray.shape[::-1], np.ceil(np.maximum(start, end) + margin)).astype(int)
    if x1 <= x0 or y1 <= y0:
        return [], None, None
    edges = cv2.Canny(gray[y0:y1, x0:x1], 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=max(10, int(length * .22)),
                            minLineLength=max(10, int(length * .35)), maxLineGap=3)
    candidates = []
    if lines is None:
        return candidates, edges, np.array([x0, y0])
    for raw in lines.reshape(-1, 4):
        a, b = raw.reshape(2, 2).astype(float) + [x0, y0]
        a, b = extend_ink_line(a, b, edges, np.array([x0, y0]), length * .4)
        line_length = float(np.linalg.norm(b - a))
        if not .6 * length <= line_length <= 1.35 * length:
            continue
        for near, tip in ((a, b), (b, a)):
            line_axis = (tip - near) / line_length
            for jamb, closed in ((start, axis), (end, -axis)):
                # Search across the wall face and a bounded shift along the gap:
                # inferred wall width is not an exact hinge coordinate.
                seeds = [near]
                for lateral in (-.5, -.25, 0, .25, .5):
                    for longitudinal in (-.7, -.35, 0, .35, .7):
                        nominal = jamb + normal * width * lateral + axis * width * longitudinal
                        projected = near + line_axis * float((nominal - near) @ line_axis)
                        if np.linalg.norm(projected - nominal) <= max(3, width * .3):
                            seeds.append(projected)
                for hinge in seeds:
                    offset = hinge - jamb
                    if (abs(float(offset @ axis)) > width * .8 + 3
                            or abs(float(offset @ normal)) > width * .7 + 3
                            or np.linalg.norm(near - hinge) > max(4, min(length * .3, width * 1.5 + 3))):
                        continue
                    radius = float(np.linalg.norm(tip - hinge))
                    if not .65 * length <= radius <= 1.35 * length:
                        continue
                    leaf = (tip - hinge) / radius
                    angle = math.atan2(float(closed[0] * leaf[1] - closed[1] * leaf[0]), float(closed @ leaf))
                    if not math.radians(20) <= abs(angle) <= math.radians(145):
                        continue
                    # Broad wall ink cannot supply a leaf. Sample the actual raster,
                    # even if segmentation accidentally includes a thin leaf in walls.
                    leaf_normal = np.array([-leaf[1], leaf[0]])
                    distance = max(3, width * .4)
                    clear = 0
                    for fraction in (.3, .55, .8):
                        point = hinge + (tip - hinge) * fraction
                        probes = [point + sign * leaf_normal * distance for sign in (-1, 1)]
                        positions = np.rint(probes).astype(int)
                        if all(0 <= x < gray.shape[1] and 0 <= y < gray.shape[0] and gray[y, x] > 170
                               for x, y in positions):
                            clear += 1
                    if clear < 2:
                        continue
                    candidates.append({"hinge": hinge.tolist(), "tip": tip.tolist(),
                                       "leaf_length": round(line_length, 3),
                                       "hinge_distance": round(float(np.linalg.norm(near - hinge)), 3),
                                       "swing_angle_degrees": round(math.degrees(angle), 3),
                                       "jamb_index": 0 if np.array_equal(jamb, start) else 1,
                                       "radius": radius, "closed_direction": closed.tolist()})
    return candidates, edges, np.array([x0, y0])


def leaf_and_arc(gray, start, end, width, walls):
    candidates, edges, origin = leaf_candidates(gray, start, end, width)
    evidence = []
    wall_ink = walls.buffer(1)
    tolerance = int(math.ceil(max(2, min(4, width * .18))))
    for candidate in candidates:
        angle = math.radians(candidate["swing_angle_degrees"])
        if abs(angle) < math.radians(35):
            continue
        hinge = np.asarray(candidate["hinge"])
        closed = candidate["closed_direction"]
        theta = math.atan2(closed[1], closed[0])
        hits = []
        for fraction in np.linspace(.18, .82, 9):
            t = theta + angle * fraction
            point = hinge + candidate["radius"] * np.array([math.cos(t), math.sin(t)])
            px, py = np.rint(point - origin).astype(int)
            patch = edges[max(0, py-tolerance):min(edges.shape[0], py+tolerance+1),
                          max(0, px-tolerance):min(edges.shape[1], px+tolerance+1)]
            hits.append(bool(0 <= px < edges.shape[1] and 0 <= py < edges.shape[0]
                             and patch.size and patch.any() and not wall_ink.covers(Point(point))))
        if sum(hits) >= 7 and all(any(hits[i:i+3]) for i in (0, 3, 6)):
            evidence.append({k: v for k, v in candidate.items() if k not in ("radius", "closed_direction")}
                            | {"arc_hits": sum(hits), "arc_samples": len(hits)})
    return max(evidence, key=lambda e: (e["arc_hits"], -e["hinge_distance"])) if evidence else None
