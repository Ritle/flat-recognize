"""Polygon-based export with shared, noded boundaries and separate evidence checks."""

import copy
import math
import uuid
from itertools import pairwise

from shapely import set_precision
from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, polygonize, unary_union


def parts(geometry, kind):
    if geometry.is_empty:
        return []
    if geometry.geom_type == kind:
        return [geometry]
    return [part for child in getattr(geometry, "geoms", []) for part in parts(child, kind)]


def requires_polygons(rooms):
    shapes = [Polygon(r["polygon"]) for r in rooms]
    if any(p.area / max(p.envelope.area, 1) < 0.9 for p in shapes):
        return True
    return any(a.envelope.intersection(b.envelope).area / min(a.envelope.area, b.envelope.area) >= 0.2
               for i, a in enumerate(shapes) for b in shapes[i + 1:])


def supported_offset_sliver(edge, walls, half_wall):
    """Allow only tiny offset fragments beside a recognized wall, not a gap."""
    return (edge.length <= max(2, half_wall * .5)
            and edge.difference(walls.buffer(half_wall)).length < 1e-4)


def align_window_boundaries(shapes, windows, walls, half_wall):
    """Straighten small raster kinks only where a long window supports the chord.

    Room extraction can follow the two sides of a window frame instead of its
    wall axis. Keep this correction local to the opening, preserve room topology,
    and report every changed contour for review.
    """
    result = list(shapes)
    changes = []
    for window_index, opening in enumerate(windows):
        points = opening.get("outer", [])
        if len(points) < 3:
            continue
        window = Polygon(points)
        if not window.is_valid or window.is_empty:
            continue
        x0, y0, x1, y1 = window.bounds
        vertical = y1 - y0 > x1 - x0
        lo, hi = (y0, y1) if vertical else (x0, x1)
        side_lo, side_hi = (x0, x1) if vertical else (y0, y1)
        length, width = hi - lo, side_hi - side_lo
        if walls.is_empty or length < max(40, half_wall * 8) or length < width * 3:
            continue
        cap = max(half_wall * 1.5, length * .1)
        side_margin = half_wall + 3
        # The source contour is an inner wall face; export moves it toward the
        # wall axis by about half a wall, where the tighter support gate runs.
        support = unary_union([walls, window]).buffer(2 * half_wall + 3)
        for room_index, shape in enumerate(result):
            if not shape.is_valid or shape.interiors:
                continue
            ring = list(shape.exterior.coords[:-1])
            n = len(ring)
            if n < 4:
                continue

            def axis(point):
                return point[1] if vertical else point[0]

            def side(point):
                return point[0] if vertical else point[1]

            eligible = [i for i, point in enumerate(ring)
                        if lo - cap <= axis(point) <= hi + cap
                        and side_lo - side_margin <= side(point) <= side_hi + side_margin]
            starts = sorted((i for i in eligible if abs(axis(ring[i]) - lo) <= cap),
                            key=lambda i: abs(axis(ring[i]) - lo))[:3]
            ends = sorted((i for i in eligible if abs(axis(ring[i]) - hi) <= cap),
                          key=lambda i: abs(axis(ring[i]) - hi))[:3]
            candidates = []
            for start in starts:
                for end in ends:
                    if start == end:
                        continue
                    for step in (1, -1):
                        chain = [start]
                        while chain[-1] != end and len(chain) <= n:
                            chain.append((chain[-1] + step) % n)
                        if chain[-1] != end or not 4 <= len(chain) <= n - 1:
                            continue
                        coords = [ring[i] for i in chain]
                        longitudinal = [axis(point) for point in coords]
                        if any(b < a - half_wall for a, b in pairwise(longitudinal)):
                            continue
                        chord = LineString([coords[0], coords[-1]])
                        if chord.length < length * .8 or abs(side(coords[-1]) - side(coords[0])) > chord.length * .12:
                            continue
                        if any(not (lo - cap <= axis(point) <= hi + cap
                                    and side_lo - side_margin <= side(point) <= side_hi + side_margin)
                               for point in coords):
                            continue
                        deviation = max(chord.distance(Point(point)) for point in coords[1:-1])
                        if not half_wall * .4 <= deviation <= half_wall + 6:
                            continue
                        if chord.difference(support).length > max(2, chord.length * .05):
                            continue
                        if any(walls.distance(Point(point)) > side_margin for point in (coords[0], coords[-1])):
                            continue
                        replacement = [point for i, point in enumerate(ring) if i not in chain[1:-1]]
                        candidate = Polygon(replacement)
                        if (not candidate.is_valid or candidate.interiors
                                or candidate.area <= 0 or
                                shape.symmetric_difference(candidate).area >
                                min(shape.area * .03, half_wall * length * 1.2)):
                            continue
                        if any(candidate.intersection(other).area > 1e-5
                               for j, other in enumerate(result) if j != room_index):
                            continue
                        candidates.append((abs(candidate.area - shape.area), candidate, deviation))
            if candidates:
                _, candidate, deviation = min(candidates, key=lambda item: item[0])
                changes.append({"room_id": room_index, "window_index": window_index,
                                "area_change_px2": round(candidate.area - shape.area, 2),
                                "max_deviation_px": round(deviation, 2)})
                result[room_index] = candidate
    return result, changes


def partition_rooms(shapes, offset, diagnostics=None):
    """Expand to wall axes, partition contested strips, retain original interiors.

    Local bisectors split overlapping expansion regions. Polygonizing the complete
    network and assigning its faces gives a disjoint partition, including at T/X
    junctions. No room is clipped to a rectangle.
    """
    if not shapes or any(not p.is_valid or p.area <= 0 or p.interiors for p in shapes):
        raise ValueError("Room geometry must consist of valid simple polygons without holes")
    if any(a.intersection(b).area > 1e-5 for i, a in enumerate(shapes) for b in shapes[i + 1:]):
        raise ValueError("Source room polygons overlap")
    # Preserve right-angle wall corners while capping acute raster spikes:
    # a square corner needs a mitre ratio of sqrt(2), but the default limit
    # of five can project a shallow notch several wall widths outward.
    expanded = [p.buffer(offset, join_style=2, mitre_limit=1.5) for p in shapes]
    lines = [p.boundary for p in shapes + expanded]
    for i, a in enumerate(expanded):
        for j in range(i + 1, len(expanded)):
            overlap = a.intersection(expanded[j])
            for region in parts(overlap, "Polygon"):
                vicinity = region.buffer(offset + 1)
                sa, sb = shapes[i].intersection(vicinity), shapes[j].intersection(vicinity)
                pa, pb = nearest_points(sa if not sa.is_empty else shapes[i], sb if not sb.is_empty else shapes[j])
                dx, dy = pb.x - pa.x, pb.y - pa.y
                length = math.hypot(dx, dy)
                if length < 1e-6:
                    continue
                mx, my = (pa.x + pb.x) / 2, (pa.y + pb.y) / 2
                span = math.hypot(region.bounds[2] - region.bounds[0], region.bounds[3] - region.bounds[1]) * 2 + offset * 4
                ux, uy = -dy / length * span, dx / length * span
                divider = LineString([(mx - ux, my - uy), (mx + ux, my + uy)]).intersection(region)
                lines.extend(parts(divider, "LineString"))
    assigned = [[] for _ in shapes]
    for face in polygonize(unary_union([set_precision(line, 0.0001) for line in lines])):
        point = face.representative_point()
        owners = [i for i, p in enumerate(expanded) if p.covers(point)]
        if owners:
            owner = min(owners, key=lambda i: (shapes[i].distance(point), i))
            assigned[owner].append(face)
    result = []
    for source, faces in zip(shapes, assigned):
        merged = unary_union(faces)
        if merged.geom_type == "MultiPolygon":
            main = max(merged.geoms, key=lambda p: p.area)
            if source.difference(main).area <= max(0.05, source.area * 1e-6):
                # Expansion strips can contain isolated slivers outside every
                # original room. They are not part of this room's boundary.
                merged = main
                if diagnostics is not None:
                    diagnostics.append("Отдельные фрагменты полосы смещения вне исходных комнат исключены.")
        result.append(merged)
    for index, (source, merged) in enumerate(zip(shapes, result)):
        if merged.geom_type == "Polygon" and merged.interiors:
            holes = [Polygon(r) for r in merged.interiors]
            if (all(p.area <= max(1, offset ** 2) for p in holes)
                    and all(p.intersection(other).area < 1e-5 for p in holes
                            for j, other in enumerate(result) if j != index)):
                # Voids smaller than one wall-offset cross-section are raster
                # junction artifacts created solely by expansion and belong to
                # no source room.
                # Keep larger holes blocked: the target contract cannot encode them.
                merged = Polygon(merged.exterior)
                result[index] = merged
                if diagnostics is not None:
                    diagnostics.append("Малые пустоты, возникшие при смещении границ, устранены; проверьте форму комнат.")
        if merged.geom_type != "Polygon" or not merged.is_valid or merged.interiors:
            components = parts(merged, "Polygon")
            details = [(round(p.area, 3), [round(Polygon(r).area, 3) for r in p.interiors]) for p in components]
            raise ValueError(f"Expanded room has holes or disconnected parts; target area cannot represent it: {details}")
        if source.difference(merged).area > max(0.05, source.area * 1e-6):
            raise ValueError("Polygon partition lost part of a source room")
    return result


def build_polygon_project(source, template, scale=1, thickness=20, height=270):
    shapes = [Polygon(r["polygon"]) for r in source["rooms"]]
    half_wall = thickness / scale / 2
    wall_shapes = [Polygon(p["outer"], p.get("holes", [])) for p in
                   source.get("walls", []) + source.get("reclassified_wall_support", [])]
    openings = [Polygon(p["outer"]) for p in source.get("doors", []) + source.get("windows", [])
                if len(p.get("outer", [])) >= 3]
    repaired = sum(not p.is_valid for p in wall_shapes + openings)
    walls = unary_union([p if p.is_valid else p.buffer(0) for p in wall_shapes]) if wall_shapes else GeometryCollection()
    shapes, aligned_windows = align_window_boundaries(shapes, source.get("windows", []), walls, half_wall)
    adjustments = []
    shapes = [orient(p, sign=1.0) for p in partition_rooms(shapes, half_wall, adjustments)]
    level_template = template["levels"][0]
    old = level_template.get("connectors", [])
    cx = (min(c["x"] for c in old) + max(c["x"] for c in old)) / 2 if old else 0
    cy = (min(c["y"] for c in old) + max(c["y"] for c in old)) / 2 if old else 0
    bounds = unary_union(shapes).bounds
    ox = cx - (bounds[0] + bounds[2]) / 2 * scale
    oy = cy - (bounds[1] + bounds[3]) / 2 * scale
    transform = {"scale_x": scale, "scale_y": scale, "origin_x": ox, "origin_y": oy}

    known = unary_union([p if p.is_valid else p.buffer(0) for p in openings]) if openings else GeometryCollection()
    support = unary_union([walls, known]).buffer(half_wall + 3)
    unknown = []
    # Room axes may be half a wall away from an opening polygon and temporary
    # raster closures can extend slightly beyond its caps. Use the same support
    # distance as physical openings, while keeping truly unmarked gaps unknown.
    known_tolerance = known.buffer(half_wall + 3) if not known.is_empty else known
    narrow_known_tolerance = known.buffer(3) if not known.is_empty else known
    supported_closure_area = unary_union([walls, known_tolerance])
    narrow_supported_closure_area = unary_union([walls, narrow_known_tolerance])
    narrow_unknown = []
    for closure in source.get("diagnostics", {}).get("temporary_closures", []):
        strip = LineString([closure["start"], closure["end"]]).buffer(closure["width"] / 2 + 3)
        # A topology-only closure may never silently become a physical wall.
        # Opening polygons and room contours are derived by different raster
        # operations, so a small tolerance prevents sub-pixel cap remnants.
        unknown.append(strip.difference(supported_closure_area))
        narrow_unknown.append(strip.difference(narrow_supported_closure_area))
    unknown = unary_union(unknown) if unknown else GeometryCollection()
    narrow_unknown = unary_union(narrow_unknown) if narrow_unknown else GeometryCollection()
    boundary = unary_union([p.boundary for p in shapes])
    network = unary_union([boundary, unknown.boundary]) if not unknown.is_empty else boundary
    raw_edges = []
    for line in parts(network, "LineString"):
        for a, b in zip(line.coords, list(line.coords)[1:]):
            edge = LineString([a, b])
            if edge.length > 1e-5 and edge.difference(boundary.buffer(1e-6)).length < 1e-5:
                raw_edges.append((a, b, edge))
    coords = {}
    nodes = []

    def connector(point):
        key = tuple(round(float(v), 4) for v in point)
        if key not in coords:
            uid = f"c{len(nodes)}"
            coords[key] = uid
            nodes.append({"uuid": uid, "x": round(ox + key[0] * scale, 4),
                          "y": round(oy + key[1] * scale, 4), "z": 0, "adjacency": []})
        return coords[key]

    segment_template = level_template.get("segments", [{}])[0]
    area_template = level_template.get("areas", [{}])[0]
    segments, unsupported = [], []
    matched_closure_edges = 0
    offset_sliver_edges = 0
    unique = set()
    for a, b, edge in raw_edges:
        start, end = connector(a), connector(b)
        pair = frozenset((start, end))
        if start == end or pair in unique:
            continue
        unique.add(pair)
        reason = None
        if edge.intersection(unknown).length > 1e-4:
            if supported_offset_sliver(edge, walls, half_wall):
                offset_sliver_edges += 1
            else:
                reason = "temporary_closure_without_opening"
        elif edge.difference(support).length > max(2, edge.length * 0.15):
            reason = "missing_structural_support"
        if reason:
            unsupported.append({"start": list(a), "end": list(b), "reason": reason})
            continue
        if edge.intersection(narrow_unknown).length > 1e-4:
            matched_closure_edges += 1
        segment = copy.deepcopy(segment_template)
        segment.update(uuid=str(uuid.uuid4()), start=start, end=end, thickness=thickness, height=height,
                       alignment=0.5, left=int(start[1:]), right=int(end[1:]),
                       half_width_left=thickness / 2, half_width_right=thickness / 2,
                       cross_left="sides", cross_right="sides", align_left="center", align_right="center",
                       userChangeBack=False, userChangeFront=False)
        segments.append(segment)
        nodes[int(start[1:])]["adjacency"].append(end)
        nodes[int(end[1:])]["adjacency"].append(start)

    areas = []
    for index, shape in enumerate(shapes):
        cycle = []
        ring = list(shape.exterior.coords)
        for a, b in pairwise(ring):
            edge = LineString([a, b])
            on_edge = [(edge.project(Point(p)), uid) for p, uid in coords.items()
                       if edge.distance(Point(p)) < 1e-3]
            for _, uid in sorted(on_edge):
                if not cycle or cycle[-1] != uid:
                    cycle.append(uid)
        if cycle and cycle[-1] == cycle[0]:
            cycle.pop()
        if len(set(cycle)) != len(cycle):
            raise ValueError("Noded area cycle contains repeated connectors")
        area = copy.deepcopy(area_template)
        area.update(uuid=str(uuid.uuid4()), name=f"Room {index + 1}", data="", connectors=cycle,
                    area_value=round(shape.area * scale ** 2, 4))
        areas.append(area)
    project = copy.deepcopy(template)
    project["editable"] = True
    project["levels"][0].update(height=height, connectors=nodes, segments=segments, areas=areas, elements=[], tapes=[])
    report = {"adapter": "polygon", "pixel_to_world": transform,
              "rooms": [{"id": i, "polygon": [list(p) for p in shape.exterior.coords[:-1]]}
                        for i, shape in enumerate(shapes)],
              "unsupported_segments": unsupported,
              "aligned_window_boundaries": aligned_windows,
              "matched_closure_edges": matched_closure_edges,
              "offset_sliver_edges": offset_sliver_edges,
              "warnings": list(dict.fromkeys(adjustments)), "errors": []}
    if aligned_windows:
        report["warnings"].append(
            f"Границы комнат у распознанных окон выпрямлены ({len(aligned_windows)}); "
            "проверьте положение проёмов и форму комнат.")
    if matched_closure_edges:
        report["warnings"].append(
            f"Границы временных замыканий сопоставлены с распознанными проёмами с допуском "
            f"половины толщины стены ({matched_closure_edges}); проверьте проёмы.")
    if offset_sliver_edges:
        report["warnings"].append(
            f"Короткие остатки смещения границы у распознанной стены приняты "
            f"как участки стены ({offset_sliver_edges}); проверьте по исходному плану.")
    if unsupported:
        report["errors"].append("Есть границы без подтверждённой стены или проёма; они не добавлены как физические стены.")
    if repaired:
        report["warnings"].append("Некорректные контуры стен/проёмов исправлены только для проверки опоры; проверьте геометрию.")
    return project, report


def render_geometry(project, report, image_path, output_path):
    from PIL import Image, ImageDraw
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    draw = ImageDraw.Draw(image)
    transform = report["pixel_to_world"]
    level = project["levels"][0]
    points = {c["uuid"]: ((c["x"] - transform["origin_x"]) / transform["scale_x"],
                           (c["y"] - transform["origin_y"]) / transform["scale_y"]) for c in level["connectors"]}
    for area in level["areas"]:
        cycle = [points[c] for c in area["connectors"]]
        if cycle:
            draw.line(cycle + cycle[:1], fill="magenta", width=2)
    internal = {s["uuid"] for s in report.get("internal_wall_graph", {}).get("added_segments", [])}
    for segment in level["segments"]:
        draw.line([points[segment["start"]], points[segment["end"]]],
                  fill="#00a99d" if segment["uuid"] in internal else "blue", width=3)
    for segment in report.get("unsupported_segments", []):
        draw.line([tuple(segment["start"]), tuple(segment["end"])], fill="orange", width=4)
    image.save(output_path)
