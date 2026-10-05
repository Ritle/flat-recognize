"""Extend exported room boundaries with evidenced internal wall centerlines.

Works on source segmentation only; topology closures are never rasterized as walls.
The target schema and room count remain unchanged. A separate report records evidence.
"""

import copy
import math
import uuid

import cv2
import numpy as np
from extract_rooms_v2 import skeletonize
from polygon_geometry import parts
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.ops import nearest_points, unary_union


def skeleton_paths(mask):
    """Trace eight-connected centerlines, grouping adjacent junction pixels."""
    skeleton = skeletonize(mask)
    pixels = {(int(x), int(y)) for y, x in np.argwhere(skeleton)}

    def neighbors(p):
        x, y = p
        return [(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (dx or dy) and (x + dx, y + dy) in pixels
                and not (dx and dy and ((x + dx, y) in pixels or (x, y + dy) in pixels))]

    adjacency = {p: neighbors(p) for p in pixels}
    joints = {p for p in pixels if len(adjacency[p]) != 2}
    groups, group_at = [], {}
    for seed in sorted(joints):
        if seed in group_at:
            continue
        pending, group = [seed], []
        group_at[seed] = len(groups)
        while pending:
            p = pending.pop()
            group.append(p)
            for q in adjacency[p]:
                if q in joints and q not in group_at:
                    group_at[q] = len(groups)
                    pending.append(q)
        groups.append(group)
    centers = [tuple(np.mean(group, axis=0)) for group in groups]
    visited, paths = set(), []

    def trace(a, b):
        path = [centers[group_at[a]] if a in group_at else a]
        previous, current = a, b
        visited.add(frozenset((a, b)))
        while current not in group_at:
            path.append(current)
            choices = [q for q in adjacency[current] if q != previous]
            if not choices:
                break
            following = choices[0]
            pair = frozenset((current, following))
            if pair in visited:
                break
            visited.add(pair)
            previous, current = current, following
        path.append(centers[group_at[current]] if current in group_at else current)
        if len(path) >= 2:
            paths.append(path)

    for group in groups:
        for a in sorted(group):
            for b in adjacency[a]:
                if b not in group_at and frozenset((a, b)) not in visited:
                    trace(a, b)
    # Closed components have no endpoint or junction.
    for a in sorted(pixels):
        for b in adjacency[a]:
            if a not in group_at and b not in group_at and frozenset((a, b)) not in visited:
                trace(a, b)
    return paths, skeleton


def source_shapes(source):
    polygons = [Polygon(item["outer"], item.get("holes", []))
                for item in source.get("walls", []) + source.get("doors", []) + source.get("windows", [])
                if len(item.get("outer", [])) >= 3]
    return unary_union([p if p.is_valid else p.buffer(0) for p in polygons])


def internal_lines(source, room_shapes, existing, half_wall):
    structure = source_shapes(source)
    report = {"version": 1, "method": "segmentation_skeleton", "lines": [], "rejected": []}
    interior = unary_union(room_shapes)
    structure = structure.intersection(interior.buffer(half_wall + 3))
    if structure.is_empty or existing.is_empty:
        return [], report
    x0, y0, x1, y1 = structure.bounds
    step = max(1., max(x1 - x0, y1 - y0) / 1020)
    origin = np.array([math.floor(x0) - 2 * step, math.floor(y0) - 2 * step])
    width, height = math.ceil((x1 - origin[0]) / step) + 3, math.ceil((y1 - origin[1]) / step) + 3
    mask = np.zeros((height, width), np.uint8)
    for shape in parts(structure, "Polygon"):
        local = np.zeros_like(mask)
        def raster(ring):
            return np.rint((np.asarray(ring.coords) - origin) / step).astype(np.int32)
        cv2.fillPoly(local, [raster(shape.exterior)], 255)
        for hole in shape.interiors:
            cv2.fillPoly(local, [raster(hole)], 0)
        mask |= local
    paths, skeleton = skeleton_paths(mask)
    distances = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    radii = distances[skeleton > 0]
    wall_width = max(2 * step, float(np.median(radii)) * 2 * step) if radii.size else 2 * step
    report.update(raster_step=step, wall_width_px=wall_width, traced_paths=len(paths))
    # Suppress wall axes already represented by a room boundary.
    corridor = existing.buffer(max(half_wall + 3, wall_width * .65))
    domain = interior.difference(corridor)
    raw_support = source_shapes(source)
    wall_support = unary_union([Polygon(item["outer"], item.get("holes", [])).buffer(0)
                               for item in source.get("walls", [])])
    support = raw_support.buffer(step * 1.5)
    unknown = []
    for closure in source.get("diagnostics", {}).get("temporary_closures", []):
        strip = LineString([closure["start"], closure["end"]]).buffer(closure["width"] / 2 + 3)
        unknown.append(strip.difference(raw_support))
    unknown = unary_union(unknown) if unknown else GeometryCollection()
    accepted = []
    accepted_entries = []
    for path in paths:
        coordinates = np.asarray(path) * step + origin
        if len(np.unique(coordinates, axis=0)) < 2:
            continue
        line = LineString(coordinates).simplify(max(step * .75, wall_width * .12))
        for candidate in parts(line.intersection(domain), "LineString"):
            if candidate.length < max(8 * step, wall_width * 1.5):
                continue
            # Some wall masks contain a thin door leaf. Require sustained wall
            # width before converting a branch to a physical partition.
            widths = [2 * wall_support.boundary.distance(candidate.interpolate(t, normalized=True))
                      if wall_support.covers(candidate.interpolate(t, normalized=True)) else 0
                      for t in np.linspace(.15, .85, 9)]
            if sum(w >= max(3, wall_width * .25) for w in widths) < 6:
                report["rejected"].append({"points": [list(p) for p in candidate.coords],
                                           "reason": "insufficient_wall_width"})
                continue
            direction = np.array(candidate.coords[-1]) - np.array(candidate.coords[0])
            direction_length = float(np.linalg.norm(direction))
            possible_leaf = False
            for door in source.get("doors", []):
                door_shape = Polygon(door["outer"])
                bx0, by0, bx1, by1 = door_shape.bounds
                gap_length, gap_width = max(bx1 - bx0, by1 - by0), min(bx1 - bx0, by1 - by0)
                axis = np.array([1., 0.]) if bx1 - bx0 >= by1 - by0 else np.array([0., 1.])
                if (direction_length > 0 and abs(direction @ axis) / direction_length <= .55
                        and .35 * gap_length <= candidate.length <= 1.6 * gap_length
                        and np.median(widths) < gap_width * .75
                        and min(door_shape.distance(Point(candidate.coords[i])) for i in (0, -1))
                        <= max(4, gap_width)):
                    possible_leaf = True
                    break
            if possible_leaf:
                report["rejected"].append({"points": [list(p) for p in candidate.coords],
                                           "reason": "possible_door_leaf"})
                continue
            points = list(candidate.coords)
            for index in (0, -1):
                end = Point(points[index])
                nearest = nearest_points(end, existing)[1]
                extension = LineString([end, nearest])
                if (extension.length <= max(half_wall + wall_width + 4, 8 * step)
                        and extension.difference(support).length < step):
                    points[index] = tuple(nearest.coords[0])
            candidate = LineString(points)
            reason = None
            if candidate.intersection(unknown).length > 1e-4:
                reason = "temporary_closure_without_opening"
            elif candidate.difference(support).length > max(step, candidate.length * .02):
                reason = "missing_structural_support"
            entry = {"points": [list(p) for p in candidate.coords], "length_px": round(candidate.length, 3)}
            if reason:
                report["rejected"].append({**entry, "reason": reason})
            else:
                accepted.append(candidate)
                accepted_entries.append(entry)

    # A physical partition must join the room-boundary graph, directly or via
    # another accepted centerline. Isolated wall-like components are commonly
    # furniture rectangles (beds, cabinets and sanitary fixtures).
    remaining = set(range(len(accepted)))
    components = []
    connection_tolerance = max(step * 2, wall_width * .15)
    while remaining:
        pending, component = [remaining.pop()], []
        while pending:
            index = pending.pop()
            component.append(index)
            attached = {other for other in remaining
                        if accepted[index].distance(accepted[other]) <= connection_tolerance}
            remaining.difference_update(attached)
            pending.extend(attached)
        components.append(component)
    filtered = []
    attachment_tolerance = max(step * 3, half_wall + wall_width * 1.5 + 4)
    for component in components:
        shape = unary_union([accepted[index] for index in component])
        distance = float(shape.distance(existing))
        lengths = []
        orthogonal_lengths = []
        for index in component:
            for a, b in zip(accepted[index].coords, list(accepted[index].coords)[1:]):
                delta = np.asarray(b) - np.asarray(a)
                length = float(np.linalg.norm(delta))
                if length <= 1e-6:
                    continue
                lengths.append(length)
                if max(abs(delta / length)) >= math.cos(math.radians(12)):
                    orthogonal_lengths.append(length)
        orthogonal = sum(orthogonal_lengths) >= sum(lengths) * .85
        if orthogonal and distance > attachment_tolerance:
            for index in component:
                report["rejected"].append({
                    **accepted_entries[index],
                    "reason": "isolated_wall_component",
                    "boundary_distance_px": round(distance, 3),
                })
            continue
        for index in component:
            filtered.append(accepted[index])
            report["lines"].append(accepted_entries[index])
    return filtered, report


def extend_project_walls(project, source, geometry, thickness=20, height=270):
    """Node new centerlines with old edges; insert split nodes into area cycles."""
    level = project["levels"][0]
    transform = geometry["pixel_to_world"]
    sx, sy = transform["scale_x"], transform["scale_y"]
    if abs(sx - sy) > 1e-6 or sx <= 0:
        raise ValueError("Internal wall graph needs a uniform positive raster scale")
    coords = {c["uuid"]: ((c["x"] - transform["origin_x"]) / sx,
                           (c["y"] - transform["origin_y"]) / sy) for c in level["connectors"]}
    originals = [(s, LineString([coords[s["start"]], coords[s["end"]]])) for s in level["segments"]]
    existing = unary_union([line for _, line in originals])
    rooms = [Polygon([coords[c] for c in area["connectors"]]) for area in level["areas"]]
    lines, report = internal_lines(source, rooms, existing, thickness / sx / 2)
    report.update(added_segments=[], split_existing_segments=0)
    geometry["internal_wall_graph"] = report
    if not lines:
        return project, geometry
    project = copy.deepcopy(project)
    level = project["levels"][0]
    nodes = level["connectors"]
    by_point = {tuple(round(v, 4) for v in p): uid for uid, p in coords.items()}
    by_id = {c["uuid"]: c for c in nodes}

    def connector(point):
        key = tuple(round(float(v), 4) for v in point)
        if key not in by_point:
            uid = f"c{len(nodes)}"
            while uid in by_id:
                uid = f"c{int(uid[1:]) + 1}"
            by_point[key] = uid
            node = {"uuid": uid, "x": round(transform["origin_x"] + key[0] * sx, 4),
                    "y": round(transform["origin_y"] + key[1] * sy, 4),
                    "z": 0, "adjacency": []}
            nodes.append(node)
            by_id[uid], coords[uid] = node, key
        return by_point[key]

    # Explicitly insert attachment coordinates into old edges: floating-point
    # projections a few ulps off an edge must not leave a disconnected T-junction.
    anchors = [Point(p) for line in lines for p in (line.coords[0], line.coords[-1])]
    noded_old = []
    for _, line in originals:
        on_line = [(line.project(p), tuple(p.coords[0])) for p in anchors if line.distance(p) < 1e-5]
        points = [tuple(line.coords[0])]
        points.extend(p for distance, p in sorted(on_line) if 1e-5 < distance < line.length - 1e-5)
        points.append(tuple(line.coords[-1]))
        noded_old.append(LineString(points))
    network = unary_union([*noded_old, *lines])
    segments, pairs = [], set()
    template = level["segments"][0] if level["segments"] else {}
    used_originals = set()
    for line in parts(network, "LineString"):
        for a, b in zip(line.coords, list(line.coords)[1:]):
            edge = LineString([a, b])
            start, end = connector(a), connector(b)
            pair = frozenset((start, end))
            if start == end or pair in pairs:
                continue
            pairs.add(pair)
            old = next(((s, old_line) for s, old_line in originals
                        if edge.difference(old_line.buffer(1e-5)).length < 1e-5), None)
            segment = copy.deepcopy(old[0] if old else template)
            if old and abs(edge.length - old[1].length) < 1e-5:
                uid = old[0]["uuid"]
            else:
                uid = str(uuid.uuid4())
                if old:
                    used_originals.add(old[0]["uuid"])
            segment.update(uuid=uid, start=start, end=end, left=int(start[1:]), right=int(end[1:]))
            if not old:
                segment.update(thickness=thickness, height=height, alignment=.5,
                               half_width_left=thickness / 2, half_width_right=thickness / 2,
                               cross_left="sides", cross_right="sides", align_left="center", align_right="center")
                report["added_segments"].append({"uuid": uid, "start": list(a), "end": list(b)})
            segments.append(segment)
    for node in nodes:
        node["adjacency"] = []
    for segment in segments:
        a, b = segment["start"], segment["end"]
        by_id[a]["adjacency"].append(b)
        by_id[b]["adjacency"].append(a)
    for area in level["areas"]:
        old_cycle = area["connectors"]
        if old_cycle[0] == old_cycle[-1]:
            old_cycle = old_cycle[:-1]
        cycle = []
        for a, b in zip(old_cycle, old_cycle[1:] + old_cycle[:1]):
            edge = LineString([coords[a], coords[b]])
            on_edge = [(edge.project(Point(p)), uid) for uid, p in coords.items()
                       if edge.distance(Point(p)) < 1e-3]
            cycle.extend(uid for distance, uid in sorted(on_edge) if distance < edge.length - 1e-4)
        if len(set(cycle)) != len(cycle):
            raise ValueError("Internal wall graph introduced repeated area connectors")
        area["connectors"] = cycle
    level["segments"] = segments
    report["split_existing_segments"] = len(used_originals)
    if report["added_segments"]:
        geometry["warnings"].append("Внутренние стены восстановлены по сегментации; проверьте перегородки и соединения.")
    return project, geometry
