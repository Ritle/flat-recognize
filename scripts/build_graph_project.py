"""Experimental physical graph -> application JSON, with explicit review diagnostics.

Topology-only gaps and annotated room faces are never used to create walls.
No inference, nearest-wall guessing or ground-truth lookup takes place here.
"""

import argparse
import copy
import json
import math
from pathlib import Path
import uuid

from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

from add_openings import build_hole_shape, find_door_template
from add_windows import find_window_template, scale_hole_shape, scale_sill_shape
from graph_export_geometry import carrier_geometry, merge_straight_chains, regularize_opening_carriers
from graph_topology import unresolved_gap_items
from review_wall_axes import apply_wall_axes
from validate_recognition import finite, validate_recognition
from window_exterior_context import window_exterior_context


def point(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or not all(finite(v) for v in value)):
        raise ValueError("Expected a finite source-raster point")
    return tuple(map(float, value))


def span(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("Expected an opening span with two endpoints")
    line = LineString([point(p) for p in value])
    if line.length <= 1e-6:
        raise ValueError("Zero-length opening span")
    return line


def opening_element(template, kind, segment, position, width):
    """Clone actual application serialization; width is an already measured hole.

    The old segmentation-width correction (x1.12) is intentionally inapplicable
    to clear spans. Vertical dimensions remain those of the real templates.
    """
    item = copy.deepcopy(template)
    uid = str(uuid.uuid4())
    old_width, height = (float(template["size"][axis]) for axis in ("x", "y"))
    if min(old_width, height) <= 0:
        raise ValueError("Invalid opening template size")
    item.update(uuid=uid, type=kind, name=f"graph_{kind}_{uid}")
    for material in item.get("materials", []):
        material["itemUuid"] = uid
    item["size"] = dict(x=round(width, 4), y=height)
    wall = item.setdefault("wall", {})
    wall.update(uuid=segment["uuid"], position=position, alignment=.5, offset=0, side="left")
    if kind in ("doorItem", "wallOpening"):
        if kind == "doorItem":
            item["size"]["z"] = segment["thickness"]
        item["position"] = {"z": 0}
        item["recess"] = 0
        wall["holeShape"] = build_hole_shape(width, height)
        if kind == "wallOpening":
            if item.get('arch', {}).get('type', 'none') != 'none':
                raise ValueError('Only a rectangular real wallOpening template is supported')
            for key in ('model', 'scale', 'sill', 'materials'):
                item.pop(key, None)
            return item
    else:
        original = template["wall"]["holeShape"]
        lo, hi = min(p["x"] for p in original), max(p["x"] for p in original)
        if not finite(lo) or not finite(hi) or hi <= lo:
            raise ValueError("Invalid window template hole width")
        # clear_span measures the hole, not the model body. Preserve the
        # template's installation allowance inside that measured aperture.
        sx = width / ((hi - lo) * 100)
        centered = [dict(p, x=p["x"] - (lo + hi) / 2) for p in original]
        wall["holeShape"] = scale_hole_shape(centered, sx, 1)
        item["size"]["x"] = round(old_width * sx, 4)
        if template.get("sill", {}).get("shape"):
            item["sill"]["shape"] = scale_sill_shape(template["sill"]["shape"], sx)
    size = item.get("model", {}).get("size", {})
    item["scale"] = {"x": item["size"]["x"] / float(size.get("x") or old_width),
                     "y": height / float(size.get("y") or height),
                     "z": (segment["thickness"] / float(size.get("z") or segment["thickness"]))
                     if kind == "doorItem" else 1}
    return item


def opening_binding(bridge, lines, tolerance=2):
    """Bind only within the recorded carrier, never across a room to another wall."""
    carrier = span(bridge["span"])
    clear = span(bridge.get("clear_span", bridge["span"]))
    if clear.difference(carrier.buffer(tolerance)).length > 1e-3:
        return None, "clear_span_outside_carrier"
    candidates = []
    for index, line in enumerate(lines):
        # Both ends of the clear span must belong to the same physical edge.
        ends = [Point(p) for p in clear.coords]
        distances = [line.distance(p) for p in ends]
        if max(distances) > tolerance:
            continue
        a, b = line.coords
        c, d = clear.coords
        dot = ((b[0] - a[0]) * (d[0] - c[0]) + (b[1] - a[1]) * (d[1] - c[1]))
        if abs(dot) / (line.length * clear.length) < math.cos(math.radians(12)):
            continue
        # Unclamped projection is required: shapely.project clamps endpoints,
        # which would silently shrink an aperture extending past the wall.
        unit = ((b[0] - a[0]) / line.length, (b[1] - a[1]) / line.length)
        projected = [sum((p[k] - a[k]) * unit[k] for k in (0, 1)) for p in (c, d)]
        lo, hi = sorted(projected)
        if lo < -1e-4 or hi > line.length + 1e-4:
            continue
        center = (lo + hi) / 2
        candidates.append((max(distances), index, center / line.length * 100, hi - lo))
    if not candidates:
        return None, "no_single_segment_contains_clear_span"
    candidates.sort()
    if len(candidates) > 1:
        return None, "ambiguous_carrier_segments"
    distance, index, position, width = candidates[0]
    return dict(index=index, position=position, width_px=width,
                distance_px=distance, center=list(clear.interpolate(.5, normalized=True).coords[0])), None


def simplify_nominal_walls(graph, thickness_px):
    """Join sampling kinks within half a wall while retaining each bound aperture."""
    points = {n["id"]: n["point"] for n in graph["nodes"]}
    lines = [LineString([points[e["start"]], points[e["end"]]]) for e in graph["edges"]]
    openings = {}
    for bridge in graph.get("diagnostics", {}).get("opening_bridges", []):
        binding, _ = opening_binding(bridge, lines)
        if binding is None:
            # Keep the original diagnostics for an unresolved carrier.
            return graph, []
        openings.setdefault(binding["index"], []).append((bridge, binding["width_px"]))

    def preserves_holes(chord, indices):
        intervals = []
        for index in indices:
            for bridge, width in openings.get(index, []):
                binding, _ = opening_binding(bridge, [chord])
                if binding is None or abs(binding["width_px"] - width) > .25:
                    return False
                center = chord.length * binding["position"] / 100
                radius = binding["width_px"] / 2
                intervals.append((center - radius, center + radius))
        intervals.sort()
        return all(a[1] <= b[0] + 1e-4 for a, b in zip(intervals, intervals[1:]))

    return merge_straight_chains(graph, max_deviation=min(2.5, thickness_px / 2),
                                 max_angle=60, allow_chord=preserves_holes, step_tolerance=1)


def build_graph_project(graph, template, window_template, *, scale=None, origin=(0, 0), height=270,
                        wall_thickness_cm=None, wall_axis_review=None,
                        allow_single_door_for_paired_leaves=False, passage_template=None, _simplify=True):
    if graph.get("coordinate_space") != "source_raster":
        raise ValueError("Graph must be in source_raster coordinates")
    scale = graph.get("scale_cm_per_px") if scale is None else scale
    if not finite(scale) or scale <= 0 or not finite(height) or height <= 0:
        raise ValueError("Scale and wall height must be finite and positive")
    if wall_thickness_cm is not None and (not finite(wall_thickness_cm) or wall_thickness_cm <= 0):
        raise ValueError("Nominal wall thickness must be finite and positive")
    origin = point(origin)
    if len(template.get("levels", [])) != 1:
        raise ValueError("Expected a single-level serialization template")
    base = template["levels"][0]
    if not base.get("segments") or not base.get("areas"):
        raise ValueError("Template must contain a segment and area")
    nodes, positions, ids = [], {}, {}
    for node in graph["nodes"]:
        key, xy = node["id"], point(node["point"])
        if key in ids or xy in positions:
            raise ValueError("Duplicate graph node ID or position")
        uid = f"c{len(nodes)}"
        ids[key] = uid
        positions[xy] = uid
        nodes.append(dict(uuid=uid, x=origin[0] + xy[0] * scale,
                          y=origin[1] + xy[1] * scale, z=0, adjacency=[]))
    xy_by_id = {ids[n["id"]]: point(n["point"]) for n in graph["nodes"]}
    node_by_id = {n["uuid"]: n for n in nodes}
    segments, lines, unique = [], [], set()
    for edge in graph["edges"]:
        if edge["start"] not in ids or edge["end"] not in ids:
            raise ValueError("Graph edge references a missing node")
        start, end = ids[edge["start"]], ids[edge["end"]]
        pair = frozenset((start, end))
        width = edge.get("thickness_px")
        if start == end or pair in unique or not finite(width) or width <= 0:
            raise ValueError("Duplicate/degenerate edge or invalid wall thickness")
        unique.add(pair)
        line = LineString([xy_by_id[start], xy_by_id[end]])
        segment = copy.deepcopy(base["segments"][0])
        thickness = width * scale if wall_thickness_cm is None else wall_thickness_cm
        segment.update(uuid=str(uuid.uuid4()), start=start, end=end,
                       thickness=thickness, height=height, alignment=.5,
                       left=int(start[1:]), right=int(end[1:]),
                       half_width_left=thickness / 2, half_width_right=thickness / 2,
                       cross_left="sides", cross_right="sides", align_left="center", align_right="center",
                       userChangeBack=False, userChangeFront=False)
        segments.append(segment)
        lines.append(line)
        node_by_id[start]["adjacency"].append(end)
        node_by_id[end]["adjacency"].append(start)
    if not lines:
        raise ValueError("Cannot export an empty physical graph")
    # Check intersections rather than silently renoding and invalidating provenance.
    for i, line in enumerate(lines):
        for j in range(i):
            crossing = line.intersection(lines[j])
            common = set(line.coords) & set(lines[j].coords)
            if not crossing.is_empty and (crossing.geom_type != "Point" or tuple(crossing.coords[0]) not in common):
                raise ValueError("Physical graph contains an unnoded intersection")
    # Validation above deliberately runs before simplification can hide damage.
    window = find_window_template(window_template)
    regularized = []
    if _simplify:
        unbound = [i for i, bridge in enumerate(graph.get("diagnostics", {}).get("opening_bridges", []))
                   if opening_binding(carrier_geometry(bridge), lines)[0] is None]
        refined, regularized = regularize_opening_carriers(graph, unbound)
        protected = {uid for r in regularized if r["accepted"] for uid in r["protected_nodes"]}
        simplified, merged = merge_straight_chains(refined, protected_nodes=protected)
    else:
        simplified, merged = graph, []
    wall_chains = []
    if _simplify and wall_thickness_cm is not None:
        simplified, wall_chains = simplify_nominal_walls(simplified, wall_thickness_cm / scale)
    axis_changes = []
    if _simplify and wall_axis_review is not None:
        if wall_thickness_cm is None:
            raise ValueError("Reviewed wall axes require a known wall thickness")
        simplified, axis_changes = apply_wall_axes(simplified, wall_axis_review, opening_binding,
                                                   wall_thickness_cm / scale)
    if merged or wall_chains or (_simplify and wall_axis_review is not None) or any(r["accepted"] for r in regularized):
        project, report = build_graph_project(simplified, template, window_template,
                                              scale=scale, origin=origin, height=height, wall_thickness_cm=wall_thickness_cm,
                                              allow_single_door_for_paired_leaves=allow_single_door_for_paired_leaves,
                                              passage_template=passage_template, wall_axis_review=wall_axis_review,
                                              _simplify=False)
        report["merged_chains"] = merged
        report["nominal_wall_chains"] = wall_chains
        if wall_axis_review is not None:
            report["reviewed_wall_axes"] = dict(image_sha256=wall_axis_review["image_sha256"], openings=axis_changes)
            if wall_axis_review.get("offset_junctions"):
                report["reviewed_wall_axes"]["offset_junctions"] = copy.deepcopy(wall_axis_review["offset_junctions"])
        report["regularized_carriers"] = regularized
        report["source_graph_counts"] = {k: len(graph.get(k, [])) for k in ("nodes", "edges", "faces")}
        return project, report
    diagnostics = graph.get("diagnostics", {})
    report = dict(adapter="physical_graph", pixel_to_world=dict(scale=scale, origin=list(origin)),
                  topology_gaps_omitted=[copy.deepcopy(gap) for _, gap in unresolved_gap_items(
                      diagnostics.get("topology_gaps", []), diagnostics.get("opening_bridges", []))],
                  unbound_openings=[], omitted_faces=[], rooms=[], bindings={"doors": [], "windows": [], "passages": []},
                  warnings=["Experimental graph export: opening types, wall branches and target-app import require review."],
                  errors=[])
    report["regularized_carriers"] = regularized
    if wall_thickness_cm is not None:
        report["wall_thickness_cm"] = wall_thickness_cm
    if wall_axis_review and wall_axis_review.get("dimension_lines"):
        report["dimension_lines"] = copy.deepcopy(wall_axis_review["dimension_lines"])
    wall_footprint = unary_union([line.buffer(segment["thickness"] / (2 * scale), cap_style=2)
                                 for line, segment in zip(lines, segments)])
    areas = []
    for shape in polygonize(lines):
        ring = list(shape.exterior.coords[:-1])
        cycle = [positions[p] for p in ring]
        if shape.interiors or len(set(cycle)) != len(cycle) or shape.area * scale ** 2 < 5000:
            report["omitted_faces"].append(dict(polygon=ring, reason="holes_repeated_nodes_or_small_area"))
            continue
        area = copy.deepcopy(base["areas"][0])
        inner = shape.difference(wall_footprint)
        area.update(uuid=str(uuid.uuid4()), name=f"Room {len(areas) + 1}", data="", connectors=cycle,
                    area_value=round(inner.area * scale ** 2, 4))
        areas.append(area)
        report["rooms"].append(dict(polygon=ring, uncertain=True, inner_area_cm2=area["area_value"]))
    door = find_door_template(template)
    passage = next((e for level in (passage_template or {}).get('levels', [])
                    for e in level.get('elements', []) if e.get('type') == 'wallOpening'), None)
    templates = dict(doors=door, windows=window, passages=passage)
    element_types = dict(doors='doorItem', windows='windowItem', passages='wallOpening')
    index_keys = dict(doors='door', windows='window', passages='passage')
    elements, occupied = [], {}
    recognition = dict(rooms=report["rooms"], doors=[], windows=[], passages=[])
    kinds = {"doors": "doors", "recovered_door": "doors", "windows": "windows", "recovered_window": "windows", "passages": "passages"}
    for index, bridge in enumerate(diagnostics.get("opening_bridges", [])):
        kind = kinds.get(bridge.get("kind"))
        if kind is None:
            report["unbound_openings"].append(dict(bridge_index=index, reason="unknown_opening_kind"))
            continue
        clear = span(bridge.get("clear_span", bridge["span"]))
        recognition[kind].append(dict(id=index, outer=list(clear.buffer(.5).exterior.coords[:-1])))
        binding, reason = opening_binding(bridge, lines)
        if binding is not None:
            edge_index = binding["index"]
            interval = (binding["position"] / 100 * lines[edge_index].length - binding["width_px"] / 2,
                        binding["position"] / 100 * lines[edge_index].length + binding["width_px"] / 2)
            if any(min(interval[1], hi) - max(interval[0], lo) > 1e-4
                   for lo, hi in occupied.get(edge_index, [])):
                reason = "overlapping_openings"
            elif templates[kind] is None:
                raise ValueError(f"Missing real {element_types[kind]} template")
        if reason:
            report["unbound_openings"].append(dict(bridge_index=index, bridge=bridge, reason=reason))
            continue
        segment = segments[binding["index"]]
        paired_leaves = kind == "doors" and bridge.get("type_review", {}).get("evidence", {}).get("leaf_count") == 2
        model_selection = ("single_template_user_approved" if allow_single_door_for_paired_leaves
                           else "single_template_requires_review") if paired_leaves else "default_template"
        if paired_leaves and not allow_single_door_for_paired_leaves:
            report["warnings"].append(f"Opening {index}: paired leaves detected, but the export uses a single-door template.")
        item = opening_element(templates[kind], element_types[kind], segment,
                               binding["position"], binding["width_px"] * scale)
        if kind == 'passages':
            report['warnings'].append(f'Opening {index}: passage height comes from the serialization template and requires review.')
        if item.get("position", {}).get("z", 0) + item["size"]["y"] > height:
            raise ValueError("Opening exceeds wall height")
        elements.append(item)
        occupied.setdefault(binding["index"], []).append(interval)
        report["bindings"][kind].append(dict(
            **{index_keys[kind]: len(report["bindings"][kind]) + 1},
            opening_id=index, bridge_index=index, element_uuid=item["uuid"], wall_uuid=segment["uuid"],
            position_percent=binding["position"], width_cm=binding["width_px"] * scale,
            element_width_cm=item["size"]["x"],
            model_selection=model_selection,
            distance_to_wall_cm=binding["distance_px"] * scale, pixel_center=binding["center"],
            width_source="clear_span" if "clear_span" in bridge else "carrier_span_requires_review",
            source=copy.deepcopy(bridge), requires_review=True))
        if kind == 'windows':
            report['bindings'][kind][-1]['exterior_context'] = window_exterior_context(
                list(clear.coords), segment['thickness']/scale, report['rooms'])
    # Unbound carrier edges would otherwise masquerade as solid walls in the app.
    if report["unbound_openings"]:
        report["errors"].append("Unbound opening carriers remain: this draft is not a final export.")
    project = copy.deepcopy(template)
    project["editable"] = True
    project["levels"][0].update(height=height, connectors=nodes, segments=segments,
                                 areas=areas, elements=elements, tapes=[])
    report["dangling_node_ids"] = [n["uuid"] for n in nodes if len(n["adjacency"]) <= 1]
    report["quality"] = validate_recognition(recognition, project, report["bindings"], scale,
                                              report, allow_empty_rooms=True)
    return project, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph", type=Path)
    parser.add_argument("--template", type=Path, default=Path("test/template.json"))
    parser.add_argument("--window-template", type=Path, default=Path("test/template_window.json"))
    parser.add_argument("--passage-template", type=Path, default=Path("test/template_passage.json"),
                        help="Real application JSON containing a rectangular wallOpening")
    parser.add_argument("--scale", type=float)
    parser.add_argument("--wall-thickness-cm", type=float, help="Use a known nominal wall thickness for export")
    parser.add_argument("--allow-single-door-for-paired-leaves", action="store_true",
                        help="Explicitly approve the standard door template for detected paired leaves")
    parser.add_argument("--output", type=Path, required=True, help="Review candidate; inspect accompanying report")
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))
    project, report = build_graph_project(load(args.graph), load(args.template), load(args.window_template), scale=args.scale,
                                        passage_template=load(args.passage_template) if args.passage_template else None,
                                        wall_thickness_cm=args.wall_thickness_cm,
                                        allow_single_door_for_paired_leaves=args.allow_single_door_for_paired_leaves)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(project, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    args.output.with_suffix(".report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    print(json.dumps(dict(status=report["quality"]["status"],
                          **{k: len(project["levels"][0][k]) for k in ("connectors", "segments", "areas", "elements")},
                          unbound_openings=len(report["unbound_openings"]))))
    return 0 if report["quality"]["export_allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
