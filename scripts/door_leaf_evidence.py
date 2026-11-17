"""Source-coordinate, jamb-anchored leaf evidence shared by V2 topology and recovery."""

import math

import cv2
import numpy as np
from shapely.geometry import LineString, Point


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


def leaf_candidates(gray, start, end, width, *, sensitive=False):
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
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=max(10, int(length * (.15 if sensitive else .22))),
                            minLineLength=max(10, int(length * .35)),
                            maxLineGap=8 if sensitive else 3)
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
                    if not math.radians(15 if sensitive else 20) <= abs(angle) <= math.radians(145):
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
    tolerance = math.ceil(max(2, min(4, width * .18)))
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


def perpendicular_leaf_and_arc(gray, start, end, width):
    """Test an axis-aligned quarter swing, including pale dashed arcs beside a wall."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = np.linalg.norm(end - start)
    if length < max(20, width * 2) or width <= 0:
        return None
    axis = (end - start) / length
    normal = np.array([-axis[1], axis[0]])
    angles = np.radians(np.linspace(6, 84, 48))
    leaf_samples = np.linspace(.18, .82, 25)
    margin = max(3, min(5, width * .3))

    def sample(points):
        xy = np.rint(points).astype(int)
        return gray[xy[..., 1], xy[..., 0]].astype(float)

    best = None
    for index, (jamb, closed) in enumerate(((start, axis), (end, -axis))):
        for side in (normal, -normal):
            directions = np.cos(angles)[:, None] * closed + np.sin(angles)[:, None] * side
            for lateral in np.linspace(-.6, .6, 7) * width:
                for shift in np.linspace(-.5, .5, 7) * width:
                    hinge = jamb + lateral * normal + shift * closed
                    radii = length - shift + np.linspace(-.5, .5, 9) * width
                    points = hinge + radii[:, None, None] * directions
                    leaf = hinge + radii[:, None, None] * leaf_samples[None, :, None] * side
                    # Missing samples outside the sheet must never count as arc evidence.
                    if (np.any(np.minimum(points.min(axis=(0, 1)), leaf.min(axis=(0, 1))) < margin + 1)
                            or np.any(np.maximum(points.max(axis=(0, 1)), leaf.max(axis=(0, 1)))
                                      >= np.array(gray.shape[::-1]) - margin - 1)):
                        continue
                    core = np.minimum.reduce([sample(points + d * directions) for d in (-.6, 0, .6)])
                    background = np.minimum(sample(points - margin * directions), sample(points + margin * directions))
                    hits = (background - core >= 9) & (core <= 245)
                    consistent = hits.reshape(len(radii), 4, -1).mean(axis=2).min(axis=1) >= .5
                    candidates = np.flatnonzero(consistent & (hits.mean(axis=1) >= .70))
                    for i in candidates:
                        stroke = np.minimum.reduce([sample(leaf[i] + d * closed) for d in (-1, 0, 1)])
                        beside = np.maximum(sample(leaf[i] - margin * closed), sample(leaf[i] + margin * closed))
                        support = float(np.mean(beside - stroke >= 12))
                        if support < .8:
                            continue
                        score = int(hits[i].sum()), support
                        if best is None or score > best[0]:
                            best = (score, {"hinge": hinge.tolist(), "tip": (hinge + radii[i] * side).tolist(),
                                            "jamb_index": index, "arc_hits": score[0], "arc_samples": len(angles),
                                            "leaf_support": support, "method": "quarter_swing_raster"})
    return best[1] if best else None


def wall_jamb_support(start, end, width, walls):
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    if length <= 0:
        return [0, 0]
    axis = (end - start) / length
    jambs = walls.buffer(3)
    return [sum(jambs.covers(Point(point + direction * width * step))
                for step in (.25, .75, 1.25))
            for point, direction in ((start, -axis), (end, axis))]


def shallow_leaf_hint(gray, start, end, width, walls):
    """Flag a near-closed raster leaf for review without promoting it to a door."""
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    if (length < max(15, width * 1.5)
            or max(abs((end - start) / length)) < math.cos(math.radians(12))):
        return None
    support = wall_jamb_support(start, end, width, walls)
    if min(support) < 2:
        return None
    opening = LineString([start, end]).buffer(width / 2, cap_style=2)
    if walls.intersection(opening).area > opening.area * .25:
        return None
    evidence = []
    for candidate in leaf_candidates(gray, start, end, width, sensitive=True)[0]:
        if not 15 <= abs(candidate["swing_angle_degrees"]) < 25:
            continue
        if candidate["hinge_distance"] > max(3, width * .5):
            continue
        clearance = float(walls.distance(Point(candidate["tip"])))
        if clearance < max(3, width * .35):
            continue
        evidence.append({"hinge": candidate["hinge"], "tip": candidate["tip"],
                         "swing_angle_degrees": candidate["swing_angle_degrees"],
                         "jamb_support": support, "tip_clearance": round(clearance, 3)})
    return max(evidence, key=lambda item: item["tip_clearance"]) if evidence else None


def dark_stroke_support(gray, hinge, tip):
    """Fraction of an interior stroke backed by original dark pixels (1 px tolerance)."""
    hinge,tip = np.asarray(hinge),np.asarray(tip)
    count = max(6, math.ceil(np.linalg.norm(tip-hinge)*.7))
    points = hinge + np.linspace(.15,.85,count)[:,None]*(tip-hinge)
    xy = np.rint(points).astype(int)
    if (xy[:,0].min() < 0 or xy[:,0].max() >= gray.shape[1]
            or xy[:,1].min() < 0 or xy[:,1].max() >= gray.shape[0]):
        return 0.0
    dark = cv2.dilate((gray < 140).astype(np.uint8), np.ones((3,3),np.uint8))
    return float(dark[xy[:,1],xy[:,0]].mean())


def dark_leaf_with_jambs(gray, start, end, width, walls, *, clear_span=None):
    """Stronger evidence for doors proposed from a continued wall axis.

    Pale watermark edges cannot define the Hough leaf. Verify the returned
    stroke against the original dark raster as well, since Hough can bridge
    gaps or extrapolate into unrelated furniture.
    """
    ink_gray = np.where(gray < 140, gray, 255).astype(np.uint8)
    evidence = diagonal_leaf_with_jambs(ink_gray, start, end, width, walls,
                                       clear_span=clear_span, inward_only=True)
    if evidence is None:
        return None
    support = dark_stroke_support(gray,evidence['hinge'],evidence['tip'])
    if support < .8:
        return None
    return {**evidence, 'dark_raster_support': round(support,3)}


def diagonal_leaf_with_jambs(gray, start, end, width, walls, *,
                             clear_span=None, inward_only=False):
    """Accept an arc-less leaf with two jambs, or a strong hinge and partial far jamb.

    Graph callers may supply a verified clear_span between jamb faces for the
    occupancy check, while retaining the axis endpoints for leaf search. The
    optional inward_only mode excludes a neighboring door's backwards leaf.
    Existing callers retain their original occupancy and swing-angle behavior.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(end - start))
    if length < max(15, width * 1.5):
        return None
    axis = (end - start) / length
    if max(abs(axis)) < math.cos(math.radians(12)):
        return None
    support = wall_jamb_support(start, end, width, walls)
    if min(support) < 1 or (min(support) < 2 and max(support) < 3):
        return None
    opening = LineString(clear_span if clear_span is not None else [start, end]).buffer(
        width / 2, cap_style=2)
    if walls.intersection(opening).area > opening.area * .25:
        return None
    evidence = []
    for candidate in leaf_candidates(gray, start, end, width)[0]:
        if min(support) < 2 and support[candidate["jamb_index"]] < 3:
            continue
        angle = abs(candidate["swing_angle_degrees"])
        if not (25 <= angle <= 75 or 105 <= angle <= 145):
            continue
        if inward_only and angle > 75:
            continue
        clearance = float(walls.distance(Point(candidate["tip"])))
        if clearance < max(3, width * .35):
            continue
        evidence.append({k: v for k, v in candidate.items() if k not in ("radius", "closed_direction")}
                        | {"jamb_support": support, "tip_clearance": round(clearance, 3),
                           "arc_confirmed": False,
                           "partial_far_jamb": min(support) < 2})
    return max(evidence, key=lambda e: (e["tip_clearance"], -e["hinge_distance"])) \
        if evidence else None
