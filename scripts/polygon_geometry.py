"""Polygon-based export with shared, noded boundaries and separate evidence checks."""

import copy
import math
import uuid
from itertools import pairwise

import numpy as np
from add_openings import choose_segment_binding, opening_export_geometry
from recover_windows_topology_v2 import measured_window_axis
from shapely import contains_xy, set_precision
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import nearest_points, polygonize, unary_union
from wall_band_measurement import flanking_wall_band, sample_straight_wall_band


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


def unsupported_boundary_evidence(edge, pre_partition_boundary, structural_support,
                                  half_wall, inferred_envelope=None):
    """Locate an unsupported project edge against the room contour before offset."""
    nearby = pre_partition_boundary.intersection(
        edge.buffer(half_wall + 2, cap_style="flat"))
    (ax, ay), (bx, by) = edge.coords
    dx, dy = bx - ax, by - ay
    parallel = []
    for line in parts(nearby, "LineString"):
        (x0, y0), (x1, y1) = line.coords[0], line.coords[-1]
        if line.length > 1e-5 and abs((x1 - x0) * dx + (y1 - y0) * dy) >= line.length * edge.length * .9:
            parallel.append(line)
    matched = unary_union(parallel) if parallel else GeometryCollection()
    source_length = matched.length
    unsupported_length = matched.difference(structural_support.buffer(3)).length
    if source_length >= edge.length * .55 and unsupported_length >= source_length * .5:
        envelope_near = (inferred_envelope is not None and not inferred_envelope.is_empty
                         and edge.intersection(inferred_envelope.buffer(half_wall + 3)).length
                         >= edge.length * .6)
        origin = "inferred_exterior_envelope" if envelope_near else "pre_partition_unconfirmed"
    elif source_length >= edge.length * .55 and unsupported_length <= source_length * .15:
        origin = "offset_or_support_mismatch"
    else:
        origin = "undetermined"
    return {"origin": origin,
            "pre_partition_parallel_px": round(source_length, 2),
            "pre_partition_unsupported_px": round(unsupported_length, 2)}


def matching_recovery_decision(closure, candidates):
    """Recovery indices may change when rooms are extracted again; match the gap itself."""
    start = np.asarray(closure["start"], float)
    end = np.asarray(closure["end"], float)
    width = float(closure.get("width", 0))
    length = float(np.linalg.norm(end - start))
    if length <= 0 or width <= 0:
        return None
    matches = []
    for decision in candidates:
        if not all(key in decision for key in ("start", "end", "width")):
            continue
        other_start = np.asarray(decision["start"], float)
        other_end = np.asarray(decision["end"], float)
        other_width = float(decision["width"])
        other_length = float(np.linalg.norm(other_end - other_start))
        if (other_length <= 0 or other_width <= 0
                or not .75 <= other_length / length <= 1.3
                or not .5 <= other_width / width <= 2):
            continue
        tolerance = max(3, min(width, other_width) * 1.5)
        direct = (float(np.linalg.norm(start - other_start)),
                  float(np.linalg.norm(end - other_end)))
        reversed_ends = (float(np.linalg.norm(start - other_end)),
                         float(np.linalg.norm(end - other_start)))
        distances = min((direct, reversed_ends), key=lambda pair: max(pair))
        if max(distances) <= tolerance:
            matches.append((sum(distances), decision))
    return min(matches, key=lambda item: item[0])[1] if matches else None


def unsupported_closure_groups(source, unsupported):
    """Group unsupported project edges by the temporary raster closure behind them.

    These are inspection targets, not openings or evidence that export is valid.
    """
    diagnostics = source.get("diagnostics", {})
    closures = diagnostics.get("temporary_closures", [])
    recovery = diagnostics.get("opening_recovery", {}).get("candidates", [])
    grouped = {}
    for edge in unsupported:
        index = edge.get("temporary_closure_index")
        if (edge.get("origin") != "temporary_topology_closure"
                or not isinstance(index, int) or not 0 <= index < len(closures)):
            continue
        grouped.setdefault(index, []).append(edge)
    result = []
    for index, edges in sorted(grouped.items()):
        closure = closures[index]
        decision = matching_recovery_decision(closure, recovery) or {}
        start, end = closure["start"], closure["end"]
        half_width = max(0, float(closure.get("width", 0))) / 2
        result.append({
            "closure_index": index, "start": start, "end": end,
            "center": [round((start[0] + end[0]) / 2, 2),
                       round((start[1] + end[1]) / 2, 2)],
            "bbox": [round(min(start[0], end[0]) - half_width, 2),
                     round(min(start[1], end[1]) - half_width, 2),
                     round(max(start[0], end[0]) + half_width, 2),
                     round(max(start[1], end[1]) + half_width, 2)],
            "closure_evidence": closure.get("evidence"),
            "recovery_closure_index": decision.get("closure_index"),
            "recovery_status": decision.get("status"),
            "recovery_reason": decision.get("reason"),
            "jamb_support": decision.get("jamb_support"),
            "unsupported_edge_count": len(edges),
            "unsupported_edges": [{"start": edge["start"], "end": edge["end"]}
                                  for edge in edges],
        })
    return result


def straighten_shared_door_kinks(nodes, segments, areas, doors, walls, half_wall, scale, ox, oy):
    """Remove a tiny degree-two bend only when a confirmed door needs its full span.

    Both adjacent rooms must share the same two wall endpoints. Replacing their
    common bent edge then changes ownership of a tiny strip, not the floorplan
    footprint or its room topology.
    """
    changes = []
    if walls.is_empty:
        return changes
    for door in doors:
        classification = door.get("classification") or {}
        evidence = classification.get("evidence") or {}
        if (classification.get("method") not in ("jamb_leaf_and_arc", "recovered_leaf_and_arc")
                or evidence.get("arc_hits", 0) < 5 or len(door.get("outer", [])) < 3):
            continue
        opening = Polygon(door["outer"])
        if not opening.is_valid or opening.is_empty:
            continue
        x0, y0, x1, y1 = opening.bounds
        horizontal = x1 - x0 >= y1 - y0
        width = float(door.get("length") or max(x1 - x0, y1 - y0)) * 1.12
        center = Point(sum(p[0] for p in door["outer"]) / len(door["outer"]),
                       sum(p[1] for p in door["outer"]) / len(door["outer"]))
        coordinates = {node["uuid"]: ((node["x"] - ox) / scale,
                                      (node["y"] - oy) / scale) for node in nodes}
        incident = {uid: [] for uid in coordinates}
        for segment in segments:
            incident[segment["start"]].append(segment)
            incident[segment["end"]].append(segment)
        support = unary_union([walls, opening]).buffer(half_wall + 3)
        for uid, attached in incident.items():
            if len(attached) != 2 or not opening.buffer(half_wall * .4).covers(Point(coordinates[uid])):
                continue
            ends = [segment["end"] if segment["start"] == uid else segment["start"]
                    for segment in attached]
            a, p, b = (coordinates[ends[0]], coordinates[uid], coordinates[ends[1]])
            chord = LineString([a, b])
            if (chord.length < width - 2 or chord.length > width / 1.12 + 3 * half_wall
                    or chord.distance(center) > half_wall * 1.5):
                continue
            dx, dy = b[0] - a[0], b[1] - a[1]
            if ((horizontal and abs(dy) > abs(dx) * .27)
                    or (not horizontal and abs(dx) > abs(dy) * .27)):
                continue
            deviation = chord.distance(Point(p))
            if not 1e-4 < deviation <= min(3, half_wall * .4):
                continue
            first = (p[0] - a[0], p[1] - a[1])
            second = (b[0] - p[0], b[1] - p[1])
            lengths = math.hypot(*first), math.hypot(*second)
            if (min(lengths) < half_wall * .5
                    or sum(x * y for x, y in zip(first, second)) /
                    (lengths[0] * lengths[1]) < math.cos(math.radians(10))):
                continue
            position = chord.project(center)
            overhang = max(0, width / 2 - position, position + width / 2 - chord.length)
            def fits(segment):
                line = LineString([coordinates[segment["start"]], coordinates[segment["end"]]])
                offset = line.project(center)
                return max(0, width / 2 - offset, offset + width / 2 - line.length) <= 5

            if overhang > 5 or any(fits(segment) for segment in attached):
                continue
            if (chord.difference(support).length > max(2, chord.length * .05)
                    or any(walls.distance(Point(point)) > half_wall * .5 + 2
                           for point in (a, b))):
                continue
            affected = [i for i, area in enumerate(areas) if uid in area["connectors"]]
            if len(affected) != 2:
                continue
            old_shapes, new_shapes = [], []
            for index in affected:
                cycle = areas[index]["connectors"]
                k = cycle.index(uid)
                if {cycle[k - 1], cycle[(k + 1) % len(cycle)]} != set(ends):
                    break
                old_shapes.append(Polygon([coordinates[item] for item in cycle]))
                new_shapes.append(Polygon([coordinates[item] for item in cycle if item != uid]))
            if (len(new_shapes) != 2 or any(not shape.is_valid or shape.area <= 0 for shape in new_shapes)
                    or new_shapes[0].intersection(new_shapes[1]).area > 1e-5
                    or unary_union(old_shapes).symmetric_difference(unary_union(new_shapes)).area > 1e-4
                    or any(a.symmetric_difference(b).area > min(a.area * .01, half_wall * width)
                           for a, b in zip(old_shapes, new_shapes))):
                continue
            survivor = attached[0]
            survivor["start"], survivor["end"] = ends
            survivor["left"], survivor["right"] = (int(item[1:]) for item in ends)
            segments.remove(attached[1])
            nodes[:] = [node for node in nodes if node["uuid"] != uid]
            for index, shape in zip(affected, new_shapes):
                areas[index]["connectors"].remove(uid)
                areas[index]["area_value"] = round(shape.area * scale ** 2, 4)
            changes.append({"door_id": door.get("id"), "removed_connector": uid,
                            "area_ids": affected, "deviation_px": round(deviation, 2),
                            "wall_length_px": round(chord.length, 2)})
            break
    if changes:
        by_id = {node["uuid"]: node for node in nodes}
        for node in nodes:
            node["adjacency"] = []
        for segment in segments:
            by_id[segment["start"]]["adjacency"].append(segment["end"])
            by_id[segment["end"]]["adjacency"].append(segment["start"])
    return changes


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
        recovery = opening.get("recovery") or {}
        thick_recovered = recovery.get("method") == "topology_recovered_thick_window"
        minimum_aspect = 1.75 if thick_recovered else 3
        if walls.is_empty or length < max(40, half_wall * 8) or length < width * minimum_aspect:
            continue
        cap = max(half_wall * 1.5, length * (.85 if thick_recovered else .1))
        side_margin = half_wall + 3
        # The source contour is an inner wall face; export moves it toward the
        # wall axis by about half a wall, where the tighter support gate runs.
        support = unary_union([walls, window]).buffer(2 * half_wall + 3)
        raw_support = unary_union([walls, window]) if thick_recovered else None
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

            # The ends of a window often fall inside long contour edges.
            # Supply exact candidate points without altering the source shape;
            # only an accepted, supported straightening keeps these vertices.
            split_ring = []
            for i, point in enumerate(ring):
                following = ring[(i + 1) % n]
                split_ring.append(point)
                span = axis(following) - axis(point)
                if abs(span) < 1e-6:
                    continue
                crossings = []
                for target in (lo, hi):
                    fraction = (target - axis(point)) / span
                    if 1e-6 < fraction < 1 - 1e-6:
                        crossing = (point[0] + (following[0] - point[0]) * fraction,
                                    point[1] + (following[1] - point[1]) * fraction)
                        if side_lo - side_margin <= side(crossing) <= side_hi + side_margin:
                            crossings.append((fraction, crossing))
                split_ring.extend(crossing for _, crossing in sorted(crossings))
            ring = split_ring
            n = len(ring)

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
                        max_deviation = max(half_wall + 6, width * .7) if thick_recovered else half_wall + 6
                        if not half_wall * .4 <= deviation <= max_deviation:
                            continue
                        if chord.difference(support).length > max(2, chord.length * .05):
                            continue
                        if any(walls.distance(Point(point)) > side_margin for point in (coords[0], coords[-1])):
                            continue
                        replacement = [point for i, point in enumerate(ring) if i not in chain[1:-1]]
                        candidate = Polygon(replacement)
                        change = shape.symmetric_difference(candidate) if candidate.is_valid else None
                        max_change = (min(shape.area * .09, width * length * 2.5)
                                      if thick_recovered else min(shape.area * .03, half_wall * length * 1.2))
                        if (not candidate.is_valid or candidate.interiors
                                or candidate.area <= 0 or
                                change.area > max_change):
                            continue
                        # A recovered thick window can justify moving the room
                        # face much farther than the nominal wall offset, but
                        # the changed strip must be mostly structural evidence.
                        support_fraction = (1 - change.difference(raw_support).area / change.area
                                            if thick_recovered and change.area > 0 else None)
                        if thick_recovered and (support_fraction is None or support_fraction < .7):
                            continue
                        if any(candidate.intersection(other).area > 1e-5
                               for j, other in enumerate(result) if j != room_index):
                            continue
                        candidates.append((abs(candidate.area - shape.area), candidate,
                                           deviation, support_fraction))
            if candidates:
                _, candidate, deviation, support_fraction = min(candidates, key=lambda item: item[0])
                change_report = {"room_id": room_index, "window_index": window_index,
                                 "area_change_px2": round(candidate.area - shape.area, 2),
                                 "max_deviation_px": round(deviation, 2)}
                if thick_recovered:
                    change_report.update(method="supported_thick_window_facade",
                                         support_fraction=round(support_fraction, 3))
                changes.append(change_report)
                result[room_index] = candidate
    return result, changes


def extend_window_endcap_corners(shapes, windows, gray, half_wall):
    """Extend a short room contour to a raster-bounded window jamb."""
    if gray is None:
        return list(shapes), []
    result, changes = list(shapes), []
    height, width = gray.shape
    for window_index, opening in enumerate(windows):
        if opening.get("recovery"):
            continue
        evidence = measured_window_axis(gray, opening)
        if evidence is None:
            continue
        (ax, ay), (bx, by) = evidence["span"]
        horizontal = abs(bx - ax) >= abs(by - ay)
        lo, hi = sorted((ax, bx) if horizontal else (ay, by))
        axis_side = (ay + by) / 2 if horizontal else (ax + bx) / 2
        length = hi - lo
        if length < half_wall * 6:
            continue

        along_index = 0 if horizontal else 1
        side_index = 1 - along_index

        def along(point, coordinate=along_index):
            return point[coordinate]

        def side(point, coordinate=side_index):
            return point[coordinate]

        proposals = []
        for room_index, shape in enumerate(result):
            if not shape.is_valid or shape.interiors:
                continue
            ring = list(shape.exterior.coords[:-1])
            for index in range(len(ring)):
                following = (index + 1) % len(ring)
                a, b = ring[index], ring[following]
                edge_length = abs(along(b) - along(a))
                overlap = max(0, min(max(along(a), along(b)), hi)
                              - max(min(along(a), along(b)), lo))
                if (edge_length < length * .4 or overlap < length * .45
                        or abs(side(a) - side(b)) > max(2, edge_length * .05)):
                    continue
                for endpoint, other, neighbor in ((index, following, (index - 1) % len(ring)),
                                                  (following, index, (following + 1) % len(ring))):
                    for target, opposite in ((hi, lo), (lo, hi)):
                        delta = target - along(ring[endpoint])
                        if (delta * (1 if target == hi else -1) < max(5, half_wall * .4)
                                or abs(delta) > min(half_wall * 4, length * .5)
                                or abs(along(ring[other]) - opposite) > max(10, half_wall * 1.5)
                                or abs(side(ring[endpoint]) - axis_side) > half_wall + 5):
                            continue
                        adjacent = ring[neighbor]
                        perpendicular = abs(side(adjacent) - side(ring[endpoint]))
                        if (perpendicular < max(20, half_wall * 2)
                                or abs(along(adjacent) - along(ring[endpoint])) >
                                max(half_wall, perpendicular * .35)):
                            continue
                        shifted = list(ring)
                        for point_index in (endpoint, neighbor):
                            u, v = along(shifted[point_index]) + delta, side(shifted[point_index])
                            shifted[point_index] = (u, v) if horizontal else (v, u)
                        candidate = Polygon(shifted)
                        if not candidate.is_valid or candidate.interiors or candidate.area <= 0:
                            continue
                        added = candidate.difference(shape)
                        removed = shape.difference(candidate)
                        if (added.area <= 0 or removed.area > added.area * .1
                                or added.area > min(shape.area * .01, half_wall * length * 4)
                                or any(candidate.intersection(other_shape).area > 1e-5
                                       for other_index, other_shape in enumerate(result)
                                       if other_index != room_index)):
                            continue
                        x0, y0, x1, y1 = added.bounds
                        xs = np.arange(max(0, math.floor(x0)), min(width, math.ceil(x1)), 3) + .5
                        ys = np.arange(max(0, math.floor(y0)), min(height, math.ceil(y1)), 3) + .5
                        if not len(xs) or not len(ys):
                            continue
                        sample_x, sample_y = np.meshgrid(xs, ys)
                        inside = contains_xy(added, sample_x, sample_y)
                        if np.count_nonzero(inside) < 10:
                            continue
                        bright = gray[sample_y.astype(int), sample_x.astype(int)] >= 190
                        bright_fraction = float(np.count_nonzero(bright & inside) /
                                                np.count_nonzero(inside))
                        if bright_fraction < .55:
                            continue
                        proposals.append((room_index, candidate, delta, added.area, bright_fraction))
        if len(proposals) != 1:
            continue
        room_index, candidate, delta, added_area, bright_fraction = proposals[0]
        result[room_index] = candidate
        changes.append({"room_id": room_index, "window_index": window_index,
                        "window_id": opening.get("id"), "corner_shift_px": round(abs(delta), 2),
                        "added_room_area_px2": round(added_area, 2),
                        "bright_fraction": round(bright_fraction, 3),
                        "raster_span": evidence["span"]})
    return result, changes


def bridge_recovered_window_bays(shapes, windows, walls, half_wall):
    """Add a small, supported room-face strip when a measured window crosses a wall junction."""
    result, changes = list(shapes), []
    if walls.is_empty:
        return result, changes
    for window_index, opening in enumerate(windows):
        recovery = opening.get("recovery") or {}
        if not isinstance(recovery, dict):
            continue
        evidence = recovery.get("raster_axis_evidence") or {}
        if not isinstance(evidence, dict):
            continue
        bay = evidence.get("bay_evidence") or {}
        if not isinstance(bay, dict):
            continue
        span = evidence.get("span")
        jamb_support = bay.get("jamb_support")
        if (recovery.get("method") != "topology_recovered_thick_window"
                or evidence.get("method") != "paired_raster_rails"
                or evidence.get("extent_basis") != "raster_clear_bay"
                or evidence.get("minimum_support", 0) < .8
                or not isinstance(jamb_support, (list, tuple)) or len(jamb_support) != 2
                or min(jamb_support) < .8
                or not isinstance(span, list) or len(span) != 2):
            continue
        try:
            ax, ay = map(float, span[0])
            bx, by = map(float, span[1])
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(v) for v in (ax, ay, bx, by)):
            continue
        dx, dy = bx - ax, by - ay
        horizontal = abs(dx) >= abs(dy)
        length = abs(dx) if horizontal else abs(dy)
        if length < half_wall * 8 or min(abs(dx), abs(dy)) > length * .1:
            continue
        center = Point((ax + bx) / 2, (ay + by) / 2)
        lo, hi = sorted((ax, bx) if horizontal else (ay, by))
        # An existing continuous facade edge already carries the measured bay.
        def fits(edge_start, edge_end):
            u, v = (edge_start[0], edge_end[0]) if horizontal else (edge_start[1], edge_end[1])
            p, q = (edge_start[1], edge_end[1]) if horizontal else (edge_start[0], edge_end[0])
            return (min(u, v) <= lo + half_wall / 2 and max(u, v) >= hi - half_wall / 2
                    and abs(p - q) <= abs(u - v) * .1
                    and LineString([edge_start, edge_end]).distance(center) <= half_wall * 2)
        if any(fits(a, b) for shape in result
               for a, b in pairwise(shape.exterior.coords)):
            continue
        opening_shape = Polygon(opening.get("outer", []))
        if not opening_shape.is_valid or opening_shape.is_empty:
            continue
        width = (opening_shape.bounds[3] - opening_shape.bounds[1] if horizontal else
                 opening_shape.bounds[2] - opening_shape.bounds[0])
        if width <= 0:
            continue
        support = unary_union([walls, opening_shape])
        pad = half_wall * 1.5
        reach = width + half_wall * 2
        candidates = []
        for side in (-1, 1):
            if horizontal:
                near = center.y + side * half_wall
                far = near + side * reach
                strip = box(lo - pad, min(near, far), hi + pad, max(near, far))
            else:
                near = center.x + side * half_wall
                far = near + side * reach
                strip = box(min(near, far), lo - pad, max(near, far), hi + pad)
            for room_index, shape in enumerate(result):
                candidate = shape.union(strip)
                if candidate.geom_type != "Polygon" or not candidate.is_valid or candidate.interiors:
                    continue
                added = candidate.difference(shape)
                if (added.area < half_wall * length * .2
                        or added.area > min(shape.area * .06, width * length * 1.5)):
                    continue
                support_fraction = added.intersection(support).area / added.area
                if support_fraction < .7:
                    continue
                if any(candidate.intersection(other).area > 1e-5
                       for i, other in enumerate(result) if i != room_index):
                    continue
                if not any(fits(a, b) for a, b in pairwise(candidate.exterior.coords)):
                    continue
                try:
                    partition_rooms([candidate if i == room_index else shape
                                     for i, shape in enumerate(result)], half_wall)
                except ValueError:
                    continue
                candidates.append((room_index, side, candidate, added.area, support_fraction))
        if len(candidates) != 1:
            continue
        room_index, side, candidate, area, support_fraction = candidates[0]
        result[room_index] = candidate
        changes.append({"room_id": room_index, "window_index": window_index,
                        "method": "supported_raster_window_bay",
                        "area_change_px2": round(area, 2),
                        "support_fraction": round(support_fraction, 3), "side": side})
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


def calibrate_recovered_facade_thickness(nodes, segments, areas, windows, walls,
                                         scale, origin_x, origin_y, default_thickness):
    """Set exterior segment width only when a recovered window has two wall flanks."""
    if walls.is_empty:
        return []
    by_id = {node["uuid"]: node for node in nodes}
    boundary_counts = {}
    for area in areas:
        ring = area["connectors"]
        for a, b in zip(ring, ring[1:] + ring[:1]):
            key = frozenset((a, b))
            boundary_counts[key] = boundary_counts.get(key, 0) + 1
    geometry = [{"segment": segment,
                 "ax": by_id[segment["start"]]["x"],
                 "ay": by_id[segment["start"]]["y"],
                 "bx": by_id[segment["end"]]["x"],
                 "by": by_id[segment["end"]]["y"]} for segment in segments]
    votes = {}
    for opening in windows:
        export = opening_export_geometry(opening)
        if export["source"] != "raster_clear_bay":
            continue
        orientation = export["orientation"]
        px, py = export["center"]
        binding = choose_segment_binding(
            geometry, origin_x + px * scale, origin_y + py * scale,
            export["length_px"] * scale, orientation)
        if (binding is None or not binding["aligned"] or binding["overhang"] > 2
                or binding["distance"] > max(default_thickness, 10 * scale)):
            continue
        segment = binding["segment"]
        key = frozenset((segment["start"], segment["end"]))
        if boundary_counts.get(key) != 1:
            continue
        evidence = opening["recovery"]["raster_axis_evidence"]
        band = flanking_wall_band(walls, evidence["span"])
        if band is None or not band["consistent_flanks"]:
            continue
        widths = list(band["sampled_sides_px"].values())
        axes = list(band["sampled_axes_px"].values())
        if (len(widths) != 2 or len(axes) != 2
                or max(widths) > min(widths) * 1.3
                or abs(axes[0] - axes[1]) > max(6, min(widths) * .15)):
            continue
        width_cm = min(widths) * scale
        if not default_thickness * 2.5 < width_cm < default_thickness * 6:
            continue
        normal_axis = ((binding["qy"] - origin_y) / scale if orientation == "horizontal" else
                       (binding["qx"] - origin_x) / scale)
        measured_axis = sum(axes) / 2
        if abs(normal_axis - measured_axis) > max(4, min(widths) * .1):
            continue
        votes.setdefault(segment["uuid"], []).append({
            "opening_id": opening.get("id"), "width_cm": width_cm,
            "axis_offset_px": round(normal_axis - measured_axis, 2)})
    changes = []
    for segment in segments:
        candidates = votes.get(segment["uuid"], [])
        if not candidates:
            continue
        widths = [item["width_cm"] for item in candidates]
        if max(widths) > min(widths) * 1.2:
            continue
        measured = round(min(widths), 2)
        original = segment["thickness"]
        segment.update(thickness=measured, half_width_left=measured / 2,
                       half_width_right=measured / 2)
        changes.append({"wall_uuid": segment["uuid"],
                        "opening_ids": [item["opening_id"] for item in candidates],
                        "original_thickness_cm": original,
                        "measured_thickness_cm": measured,
                        "axis_offset_px": max(abs(item["axis_offset_px"]) for item in candidates)})
    return changes


def realign_exterior_wall_axes(nodes, segments, areas, doors, windows, walls,
                               scale, origin_x, origin_y, default_thickness):
    """Move a straight exterior axis only into a stable raster wall band."""
    if walls.is_empty:
        return []
    by_id = {node["uuid"]: node for node in nodes}
    incident = {uid: [] for uid in by_id}
    boundary_counts = {}
    owners = {uid: set() for uid in by_id}
    for index, area in enumerate(areas):
        ring = area["connectors"]
        for uid in ring:
            owners[uid].add(index)
        for a, b in zip(ring, ring[1:] + ring[:1]):
            key = frozenset((a, b))
            boundary_counts[key] = boundary_counts.get(key, 0) + 1
    for segment in segments:
        incident[segment["start"]].append(segment)
        incident[segment["end"]].append(segment)
    opening_shapes = []
    for item in doors + windows:
        if len(item.get("outer", [])) >= 3:
            shape = Polygon(item["outer"])
            opening_shapes.append(shape if shape.is_valid else shape.buffer(0))
    openings = unary_union(opening_shapes) if opening_shapes else GeometryCollection()
    support = unary_union([walls, openings]).buffer(default_thickness / scale / 2 + 3)
    moved, changes = set(), []

    def point(uid):
        node = by_id[uid]
        return ((node["x"] - origin_x) / scale, (node["y"] - origin_y) / scale)

    for segment in segments:
        a, b = segment["start"], segment["end"]
        if (a in moved or b in moved or boundary_counts.get(frozenset((a, b))) != 1
                or len(incident[a]) != 2 or len(incident[b]) != 2
                or len(owners[a]) != 1 or owners[a] != owners[b]):
            continue
        old_line = LineString([point(a), point(b)])
        width_px = default_thickness / scale
        dx, dy = old_line.coords[1][0] - old_line.coords[0][0], \
            old_line.coords[1][1] - old_line.coords[0][1]
        if (old_line.length < 120 or min(abs(dx), abs(dy)) > old_line.length * .05
                or not openings.is_empty and any(
                    openings.distance(Point(point(uid))) < max(10, width_px * 1.5)
                    for uid in (a, b))):
            continue
        band = sample_straight_wall_band(walls, old_line, width_px)
        if band is None:
            continue
        measured, shift = band["median_width_px"], band["median_offset_px"]
        if (not max(6, measured * .2) < abs(shift) <= min(12, measured * .45)
                or not width_px * .7 <= measured <= width_px * 2):
            continue
        # One adjacent room is insufficient to call a wall exterior: missing
        # rooms can make an internal partition look like a facade.
        x0, y0, x1, y1 = walls.bounds
        middle = old_line.interpolate(.5, normalized=True)
        edge_distance = (min(abs(middle.x - x0), abs(x1 - middle.x))
                         if abs(dy) > abs(dx) else
                         min(abs(middle.y - y0), abs(y1 - middle.y)))
        if edge_distance > measured * 2:
            continue
        nx, ny = band["normal"]
        candidate = {uid: (point(uid)[0] + nx * shift,
                           point(uid)[1] + ny * shift) for uid in (a, b)}
        new_line = LineString([candidate[a], candidate[b]])
        old_band = old_line.buffer(width_px / 2, cap_style="flat")
        new_band = new_line.buffer(width_px / 2, cap_style="flat")
        before = old_band.intersection(walls).area / old_band.area
        after = new_band.intersection(walls).area / new_band.area
        if after < .9 or after - before < .08:
            continue
        owner = next(iter(owners[a]))
        ring = areas[owner]["connectors"]
        old_room = Polygon([point(uid) for uid in ring])
        new_room = Polygon([candidate.get(uid, point(uid)) for uid in ring])
        if not new_room.is_valid or new_room.area <= 0:
            continue
        changed = old_room.symmetric_difference(new_room)
        if (changed.is_empty or changed.area > old_room.area * .01
                or changed.intersection(walls).area / changed.area < .9):
            continue
        if any(new_room.intersection(Polygon([point(uid) for uid in area["connectors"]])).area > 1e-4
               for index, area in enumerate(areas) if index != owner):
            continue
        touched = {item["uuid"]: item for uid in (a, b) for item in incident[uid]}
        if any(LineString([candidate.get(item["start"], point(item["start"])),
                           candidate.get(item["end"], point(item["end"]))]).difference(support).length
               > max(2, LineString([point(item["start"]), point(item["end"])]).length * .15)
               for item in touched.values()):
            continue
        for uid, (x, y) in candidate.items():
            by_id[uid]["x"] = round(origin_x + x * scale, 4)
            by_id[uid]["y"] = round(origin_y + y * scale, 4)
        areas[owner]["area_value"] = round(new_room.area * scale ** 2, 4)
        moved.update((a, b))
        changes.append({"wall_uuid": segment["uuid"], "room_index": owner,
                        "axis_shift_px": round(abs(shift), 2),
                        "source_width_px": round(measured, 2),
                        "changed_room_area_px2": round(changed.area, 2),
                        "source_support": round(changed.intersection(walls).area / changed.area, 3),
                        "band_coverage_before": round(before, 3),
                        "band_coverage_after": round(after, 3)})
    return changes


def realign_project_exterior_wall_axes(project, source, report, default_thickness):
    """Apply guarded facade alignment after optional internal walls are noded."""
    report["realigned_exterior_wall_axes"] = []
    if report.get("unsupported_segments"):
        return project, report
    wall_shapes = []
    for item in source.get("walls", []) + source.get("reclassified_wall_support", []):
        shape = Polygon(item["outer"], item.get("holes", []))
        wall_shapes.append(shape if shape.is_valid else shape.buffer(0))
    walls = unary_union(wall_shapes) if wall_shapes else GeometryCollection()
    transform = report["pixel_to_world"]
    scale = transform["scale_x"]
    if abs(scale - transform["scale_y"]) > 1e-6:
        return project, report
    level = project["levels"][0]
    changes = realign_exterior_wall_axes(
        level["connectors"], level["segments"], level["areas"],
        source.get("doors", []), source.get("windows", []), walls,
        scale, transform["origin_x"], transform["origin_y"], default_thickness)
    report["realigned_exterior_wall_axes"] = changes
    if changes:
        positions = {node["uuid"]: ((node["x"] - transform["origin_x"]) / scale,
                                     (node["y"] - transform["origin_y"]) / scale)
                     for node in level["connectors"]}
        report["rooms"] = [{"id": i, "polygon": [list(positions[uid]) for uid in area["connectors"]]}
                           for i, area in enumerate(level["areas"])]
        report["warnings"].append(
            f"Оси наружных стен согласованы с устойчивой полосой на исходном плане "
            f"({len(changes)}); проверьте соединения стен и форму комнат.")
    return project, report


def build_polygon_project(source, template, scale=1, thickness=20, height=270,
                          raster_gray=None):
    shapes = [Polygon(r["polygon"]) for r in source["rooms"]]
    half_wall = thickness / scale / 2
    wall_shapes = [Polygon(p["outer"], p.get("holes", [])) for p in
                   source.get("walls", []) + source.get("reclassified_wall_support", [])]
    openings = [Polygon(p["outer"]) for p in source.get("doors", []) + source.get("windows", [])
                if len(p.get("outer", [])) >= 3]
    repaired = sum(not p.is_valid for p in wall_shapes + openings)
    walls = unary_union([p if p.is_valid else p.buffer(0) for p in wall_shapes]) if wall_shapes else GeometryCollection()
    shapes, aligned_windows = align_window_boundaries(shapes, source.get("windows", []), walls, half_wall)
    shapes, bridged_windows = bridge_recovered_window_bays(
        shapes, source.get("windows", []), walls, half_wall)
    shapes, extended_windows = extend_window_endcap_corners(
        shapes, source.get("windows", []), raster_gray, half_wall)
    pre_partition_boundary = unary_union([p.boundary for p in shapes])
    envelope_info = source.get("diagnostics", {}).get("exterior_envelope") or {}
    envelope_bbox = envelope_info.get("bbox")
    inferred_envelope = None
    if envelope_info.get("accepted"):
        run_data = envelope_info.get("inferred_boundary_runs")
        if isinstance(run_data, list):
            run_lines = []
            for run in run_data:
                if not isinstance(run, dict):
                    continue
                try:
                    line = LineString([run["start"], run["end"]])
                except (KeyError, TypeError, ValueError):
                    continue
                if line.length > 0 and all(math.isfinite(value) for point in line.coords
                                           for value in point):
                    run_lines.append(line)
            inferred_envelope = unary_union(run_lines) if run_lines else None
        elif (isinstance(envelope_bbox, list) and len(envelope_bbox) == 4
              and all(isinstance(value, (int, float)) and math.isfinite(value)
                      for value in envelope_bbox)
              and envelope_bbox[2] > envelope_bbox[0]
              and envelope_bbox[3] > envelope_bbox[1]):
            # Older room artifacts only recorded the accepted envelope box.
            inferred_envelope = box(*envelope_bbox).boundary
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
    structural_support = unary_union([walls, known])
    support = structural_support.buffer(half_wall + 3)
    unknown = []
    # Room axes may be half a wall away from an opening polygon and temporary
    # raster closures can extend slightly beyond its caps. Use the same support
    # distance as physical openings, while keeping truly unmarked gaps unknown.
    known_tolerance = known.buffer(half_wall + 3) if not known.is_empty else known
    narrow_known_tolerance = known.buffer(3) if not known.is_empty else known
    supported_closure_area = unary_union([walls, known_tolerance])
    narrow_supported_closure_area = unary_union([walls, narrow_known_tolerance])
    narrow_unknown = []
    closure_lines = []
    for closure_index, closure in enumerate(source.get("diagnostics", {}).get("temporary_closures", [])):
        line = LineString([closure["start"], closure["end"]])
        closure_lines.append((closure_index, line, closure))
        strip = line.buffer(closure["width"] / 2 + 3)
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
            evidence = unsupported_boundary_evidence(
                edge, pre_partition_boundary, structural_support,
                half_wall, inferred_envelope)
            if reason == "temporary_closure_without_opening":
                evidence["origin"] = "temporary_topology_closure"
                if closure_lines:
                    closure_index, line, closure = min(
                        closure_lines, key=lambda item: edge.distance(item[1]))
                    distance = edge.distance(line)
                    if distance <= closure["width"] / 2 + half_wall + 3:
                        evidence.update(temporary_closure_index=closure_index,
                                        temporary_closure_evidence=closure.get("evidence"),
                                        temporary_closure_distance_px=round(distance, 2))
            unsupported.append({"start": list(a), "end": list(b), "reason": reason,
                                **evidence})
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
    aligned_doors = straighten_shared_door_kinks(
        nodes, segments, areas, source.get("doors", []), walls, half_wall, scale, ox, oy)
    if aligned_doors:
        positions = {node["uuid"]: ((node["x"] - ox) / scale,
                                     (node["y"] - oy) / scale) for node in nodes}
        shapes = [Polygon([positions[uid] for uid in area["connectors"]]) for area in areas]
    calibrated_walls = calibrate_recovered_facade_thickness(
        nodes, segments, areas, source.get("windows", []), walls, scale, ox, oy, thickness)
    project = copy.deepcopy(template)
    project["editable"] = True
    project["levels"][0].update(height=height, connectors=nodes, segments=segments, areas=areas, elements=[], tapes=[])
    report = {"adapter": "polygon", "pixel_to_world": transform,
              "rooms": [{"id": i, "polygon": [list(p) for p in shape.exterior.coords[:-1]]}
                        for i, shape in enumerate(shapes)],
              "unsupported_segments": unsupported,
              "unsupported_closure_groups": unsupported_closure_groups(source, unsupported),
              "unsupported_origin_counts": {
                  origin: sum(item["origin"] == origin for item in unsupported)
                  for origin in ("inferred_exterior_envelope", "temporary_topology_closure",
                                 "pre_partition_unconfirmed",
                                 "offset_or_support_mismatch", "undetermined")},
              "aligned_window_boundaries": aligned_windows,
              "bridged_window_bays": bridged_windows,
              "extended_window_endcaps": extended_windows,
              "realigned_exterior_wall_axes": [],
              "calibrated_wall_thickness": calibrated_walls,
              "aligned_door_boundaries": aligned_doors,
              "matched_closure_edges": matched_closure_edges,
              "offset_sliver_edges": offset_sliver_edges,
              "warnings": list(dict.fromkeys(adjustments)), "errors": []}
    if aligned_windows:
        report["warnings"].append(
            f"Границы комнат у распознанных окон выпрямлены ({len(aligned_windows)}); "
            "проверьте положение проёмов и форму комнат.")
    if aligned_doors:
        report["warnings"].append(
            f"Малые изгибы общих границ комнат у подтверждённых дверей выпрямлены ({len(aligned_doors)}); "
            "проверьте привязку дверей и форму помещений.")
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
