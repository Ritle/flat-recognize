"""Small, bounded changes to physical graph geometry before serialization."""

import copy
import math

from shapely.geometry import LineString, Point


def carrier_geometry(bridge):
    # A cut trimmed inside an existing window must retain the original
    # carrier's alignment, including its junctions with neighbouring windows.
    if bridge.get('aperture_measurement', {}).get('source') == 'visible_window_jambs':
        return dict(bridge, clear_span=bridge['span'])
    return bridge


def regularize_opening_carriers(graph, bridge_indices, max_deviation=.5, *, end_margins=None):
    """Resegment a degree-two path beneath an opening, without narrowing its hole.

    Cuts lie on existing edges, beyond each aperture end by its template margin.
    A quarter-pixel guard covers rounding without moving the hole itself. The chord
    must remain within half a pixel of every replaced vertex, preserve all
    junctions and introduce no crossings. Uncertain cases are diagnostic only.
    """
    result = copy.deepcopy(graph)
    decisions = []
    protected = set()
    for index in bridge_indices:
        bridge = carrier_geometry(graph.get("diagnostics", {}).get("opening_bridges", [])[index])
        aperture = LineString(bridge.get("clear_span", bridge["span"]))
        record = dict(bridge_index=index, accepted=False)
        decisions.append(record)
        if aperture.length <= 1e-6:
            record["reason"] = "degenerate_aperture"
            continue
        p, q = aperture.coords
        unit = ((q[0] - p[0]) / aperture.length, (q[1] - p[1]) / aperture.length)
        margin = (end_margins or {}).get(index, .25)
        targets = [(p[0] - margin * unit[0], p[1] - margin * unit[1]),
                   (q[0] + margin * unit[0], q[1] + margin * unit[1])]
        record["end_margin_px"] = margin
        positions = {n["id"]: tuple(n["point"]) for n in result["nodes"]}
        edges = copy.deepcopy(result["edges"])
        anchors = []
        for side, target in enumerate(targets):
            candidates = []
            for i, edge in enumerate(edges):
                a, b = positions[edge["start"]], positions[edge["end"]]
                line = LineString([a, b])
                dot = sum((b[k] - a[k]) * unit[k] for k in (0, 1))
                if abs(dot) < line.length * math.cos(math.radians(25)):
                    continue
                distance = line.distance(Point(target))
                if distance <= 2:
                    candidates.append((distance, i, line.interpolate(line.project(Point(target)))))
            candidates.sort(key=lambda row: row[0])
            if not candidates:
                break
            distance, i, foot = candidates[0]
            xy = tuple(foot.coords[0])
            # Equidistant distinct axes cannot be disambiguated by geometry.
            if any(abs(other[0] - distance) < 1e-4 and other[2].distance(foot) > 1e-4
                   for other in candidates[1:]):
                break
            edge = edges[i]
            uid = next((edge[k] for k in ("start", "end")
                        if Point(positions[edge[k]]).distance(foot) < 1e-6), None)
            if uid is None:
                uid = f"opening_cut_{index}_{side}"
                while uid in positions:
                    uid += "_"
                positions[uid] = xy
                edges[i:i + 1] = [dict(edge, end=uid), dict(edge, start=uid)]
            anchors.append(uid)
        if len(anchors) != 2 or anchors[0] == anchors[1]:
            record["reason"] = "missing_unique_carrier_anchors"
            continue
        incident = {uid: [] for uid in positions}
        for i, edge in enumerate(edges):
            incident[edge["start"]].append(i)
            incident[edge["end"]].append(i)
        paths = []
        corridor = aperture.buffer(3)
        for initial in incident[anchors[0]]:
            current, edge_index = anchors[0], initial
            path, vertices = [], [current]
            while edge_index not in path:
                edge = edges[edge_index]
                other = edge["end"] if edge["start"] == current else edge["start"]
                line = LineString([positions[current], positions[other]])
                if line.difference(corridor).length > 1e-4:
                    break
                path.append(edge_index)
                vertices.append(other)
                if other == anchors[1]:
                    paths.append((path, vertices))
                    break
                if len(incident[other]) != 2 or other in protected:
                    break
                current, edge_index = other, next(i for i in incident[other] if i != edge_index)
        if len(paths) != 1:
            record["reason"] = "junction_or_missing_unique_path"
            continue
        path, vertices = paths[0]
        chord = LineString([positions[a] for a in anchors])
        samples = [positions[v] for v in vertices]
        deviation = max(chord.distance(Point(xy)) for xy in samples)
        record.update(path_points=samples, max_deviation_px=deviation)
        if deviation > max_deviation:
            record["reason"] = "axis_deviation_exceeds_budget"
            continue
        # Preserve the full aperture projected on its new axis, without clamps.
        a, b = chord.coords
        along = [(sum((xy[k] - a[k]) * (b[k] - a[k]) for k in (0, 1)) / chord.length)
                 for xy in aperture.coords]
        if min(along) < 0 or max(along) > chord.length:
            record["reason"] = "aperture_exceeds_chord"
            continue
        kept = [edge for i, edge in enumerate(edges) if i not in path]
        crossing = False
        for edge in kept:
            line = LineString([positions[edge["start"]], positions[edge["end"]]])
            intersection = chord.intersection(line)
            common = set(chord.coords) & set(line.coords)
            if not intersection.is_empty and (intersection.geom_type != "Point" or tuple(intersection.coords[0]) not in common):
                crossing = True
                break
        if crossing:
            record["reason"] = "new_intersection"
            continue
        lengths = [LineString([positions[edges[i][k]] for k in ("start", "end")]).length for i in path]
        width = sum(edges[i]["thickness_px"] * length for i, length in zip(path, lengths)) / sum(lengths)
        kept.append(dict(start=anchors[0], end=anchors[1], thickness_px=width))
        used = {e[k] for e in kept for k in ("start", "end")}
        result["nodes"] = [dict(id=uid, point=list(xy), adjacency=[]) for uid, xy in positions.items() if uid in used]
        result["edges"] = kept
        protected.update(anchors)
        record.update(accepted=True, span=list(chord.coords), protected_nodes=anchors)
    by_id = {n["id"]: n for n in result["nodes"]}
    for node in by_id.values():
        node["adjacency"] = []
    for edge in result["edges"]:
        by_id[edge["start"]]["adjacency"].append(edge["end"])
        by_id[edge["end"]]["adjacency"].append(edge["start"])
    return result, decisions


def merge_straight_chains(graph, max_deviation=.5, max_angle=8, *, protected_nodes=(),
                          allow_chord=None, step_tolerance=0):
    """Remove degree-two sampling nodes; retain junctions and real corners.

    All original samples, not just the latest join, must stay within the error
    budget. Thickness is length-weighted. No new junction/intersection is allowed.
    The input graph and its recognition diagnostics remain unmodified.
    """
    result = copy.deepcopy(graph)
    positions = {n["id"]: tuple(n["point"]) for n in graph["nodes"]}
    edges = []
    for index, edge in enumerate(graph["edges"]):
        line = LineString([positions[edge["start"]], positions[edge["end"]]])
        edges.append(dict(edge, samples=list(line.coords), source_edges=[index],
                          original_length=line.length,
                          thickness_sum=line.length * edge["thickness_px"]))
    changed = True
    while changed:
        changed = False
        incident = {uid: [] for uid in positions}
        for i, edge in enumerate(edges):
            incident[edge["start"]].append(i)
            incident[edge["end"]].append(i)
        for uid, indices in incident.items():
            if len(indices) != 2 or uid in protected_nodes:
                continue
            i, j = indices
            first, second = edges[i], edges[j]
            a = first["end"] if first["start"] == uid else first["start"]
            b = second["end"] if second["start"] == uid else second["start"]
            if a == b:
                continue
            p, q, r = positions[a], positions[uid], positions[b]
            u, v = (q[0] - p[0], q[1] - p[1]), (r[0] - q[0], r[1] - q[1])
            lengths = math.hypot(*u), math.hypot(*v)
            # Pixel stair-steps at a jamb can turn 90 degrees for less than
            # one pixel along an otherwise continuous wall.
            angle_limit = max_angle if min(lengths) > step_tolerance else max(max_angle, 100)
            if sum(x * y for x, y in zip(u, v)) < lengths[0] * lengths[1] * math.cos(math.radians(angle_limit)):
                continue
            line = LineString([p, r])
            samples = first["samples"] + second["samples"]
            if max(line.distance(Point(xy)) for xy in samples) > max_deviation:
                continue
            source_edges = first["source_edges"] + second["source_edges"]
            if allow_chord is not None and not allow_chord(line, source_edges):
                continue
            crossing = False
            for k, edge in enumerate(edges):
                if k in indices:
                    continue
                other = LineString([positions[edge["start"]], positions[edge["end"]]])
                intersection = line.intersection(other)
                common = set(line.coords) & set(other.coords)
                if not intersection.is_empty and (intersection.geom_type != "Point" or tuple(intersection.coords[0]) not in common):
                    crossing = True
                    break
            if crossing:
                continue
            length = first["original_length"] + second["original_length"]
            width_sum = first["thickness_sum"] + second["thickness_sum"]
            merged = dict(start=a, end=b, thickness_px=width_sum / length,
                          samples=samples, original_length=length, thickness_sum=width_sum,
                          source_edges=source_edges)
            edges = [edge for k, edge in enumerate(edges) if k not in indices] + [merged]
            changed = True
            break
    used = {edge[k] for edge in edges for k in ("start", "end")}
    result["nodes"] = [dict(n, adjacency=[]) for n in result["nodes"] if n["id"] in used]
    by_id = {n["id"]: n for n in result["nodes"]}
    result["edges"] = []
    merged = []
    for index, edge in enumerate(edges):
        result["edges"].append({k: edge[k] for k in ("start", "end", "thickness_px")})
        by_id[edge["start"]]["adjacency"].append(edge["end"])
        by_id[edge["end"]]["adjacency"].append(edge["start"])
        if len(edge["source_edges"]) > 1:
            line = LineString([positions[edge["start"]], positions[edge["end"]]])
            merged.append(dict(edge_index=index, source_edges=edge["source_edges"],
                               max_deviation_px=max(line.distance(Point(p)) for p in edge["samples"])))
    return result, merged
