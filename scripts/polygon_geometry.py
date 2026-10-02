"""Polygon-based export with shared, noded boundaries and separate evidence checks."""

import copy
import math
import uuid

from shapely.geometry import GeometryCollection, LineString, Point, Polygon
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, polygonize, unary_union
from shapely import set_precision


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
    expanded = [p.buffer(offset, join_style=2) for p in shapes]
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
            if (all(p.area <= max(1, offset ** 2 * 0.25) for p in holes)
                    and all(p.intersection(other).area < 1e-5 for p in holes
                            for j, other in enumerate(result) if j != index)):
                # Tiny voids created solely by expansion belong to no source room.
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
    adjustments = []
    shapes = [orient(p, sign=1.0) for p in partition_rooms(shapes, half_wall, adjustments)]
    level_template = template["levels"][0]
    old = level_template.get("connectors", [])
    cx = (min(c["x"] for c in old) + max(c["x"] for c in old)) / 2 if old else 0
    cy = (min(c["y"] for c in old) + max(c["y"] for c in old)) / 2 if old else 0
    bounds = unary_union(shapes).bounds
    ox = cx - (bounds[0] + bounds[2]) / 2 * scale
    oy = cy - (bounds[1] + bounds[3]) / 2 * scale
    transform = dict(scale_x=scale, scale_y=scale, origin_x=ox, origin_y=oy)

    wall_shapes = [Polygon(p["outer"], p.get("holes", [])) for p in source.get("walls", [])]
    openings = [Polygon(p["outer"]) for p in source.get("doors", []) + source.get("windows", [])
                if len(p.get("outer", [])) >= 3]
    repaired = sum(not p.is_valid for p in wall_shapes + openings)
    walls = unary_union([p if p.is_valid else p.buffer(0) for p in wall_shapes]) if wall_shapes else GeometryCollection()
    known = unary_union([p if p.is_valid else p.buffer(0) for p in openings]) if openings else GeometryCollection()
    support = unary_union([walls, known]).buffer(half_wall + 3)
    unknown = []
    for closure in source.get("diagnostics", {}).get("temporary_closures", []):
        strip = LineString([closure["start"], closure["end"]]).buffer(closure["width"] / 2 + 3)
        # A topology-only closure may never silently become a physical wall.
        unknown.append(strip.difference(unary_union([walls, known])))
    unknown = unary_union(unknown) if unknown else GeometryCollection()
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
            nodes.append(dict(uuid=uid, x=round(ox + key[0] * scale, 4),
                              y=round(oy + key[1] * scale, 4), z=0, adjacency=[]))
        return coords[key]

    segment_template = level_template.get("segments", [{}])[0]
    area_template = level_template.get("areas", [{}])[0]
    segments, unsupported = [], []
    unique = set()
    for a, b, edge in raw_edges:
        start, end = connector(a), connector(b)
        pair = frozenset((start, end))
        if start == end or pair in unique:
            continue
        unique.add(pair)
        reason = None
        if edge.intersection(unknown).length > 1e-4:
            reason = "temporary_closure_without_opening"
        elif edge.difference(support).length > max(2, edge.length * 0.15):
            reason = "missing_structural_support"
        if reason:
            unsupported.append(dict(start=list(a), end=list(b), reason=reason))
            continue
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
        for a, b in zip(ring, ring[1:]):
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
    report = dict(adapter="polygon", pixel_to_world=transform,
                  rooms=[dict(id=i, polygon=[list(p) for p in shape.exterior.coords[:-1]]) for i, shape in enumerate(shapes)],
                  unsupported_segments=unsupported, warnings=list(dict.fromkeys(adjustments)), errors=[])
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
