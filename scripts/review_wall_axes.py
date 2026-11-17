"""Apply source-raster wall axes and jamb measurements after carrier identification."""

import copy
import math

from shapely.geometry import LineString
from shapely.ops import polygonize

from graph_export_geometry import merge_straight_chains
from graph_topology import unresolved_gap_items


def apply_offset_junction(graph, measurement, thickness_px):
    """Join two measured parallel runs through the footprint of their common stem."""
    ends = [tuple(p) for p in measurement["ends"]]
    center = tuple(measurement["junction"])
    normals = measurement["normals"]
    along = 0 if ends[0][1] == ends[1][1] else 1
    normal = 1 - along
    if not (ends[0][normal] == center[normal] == ends[1][normal]
            and ends[0][along] < center[along] < ends[1][along]
            and len(normals) == 2 and normals[0] != normals[1]
            and all(abs(v - center[normal]) <= thickness_px / 2 for v in normals)):
        raise ValueError("Invalid offset junction measurement")
    nodes = {n["id"]: n for n in graph["nodes"]}
    ids = {tuple(n["point"]): n["id"] for n in nodes.values()}
    if any(p not in ids for p in [*ends, center]):
        raise ValueError("Offset junction does not match the reviewed wall runs")
    joint = nodes[ids[center]]
    ends_ids = [ids[p] for p in ends]
    stems = set(joint["adjacency"]) - set(ends_ids)
    if len(stems) != 1 or len(joint["adjacency"]) != 3:
        raise ValueError("Expected a three-way wall junction")
    stem = nodes[stems.pop()]
    if stem["point"][along] != center[along]:
        raise ValueError("Offset junction requires a perpendicular wall")
    # Both ends may move along adjoining perpendicular walls without tilting them.
    for uid, end, value in zip(ends_ids, ends, normals):
        for neighbor in nodes[uid]["adjacency"]:
            if neighbor == joint["id"]:
                continue
            p = nodes[neighbor]["point"]
            if p[along] != end[along]:
                raise ValueError("Offset wall run continues past the measured endpoint")
            if (p[normal] - end[normal]) * (p[normal] - value) <= 0:
                raise ValueError("Offset would reverse or collapse an adjoining wall")
    # The end nearer the stem keeps the existing junction. Extending the stem
    # through the other run avoids a diagonal wall or two overlapping segments.
    forward = stem["point"][normal] > center[normal]
    near = normals.index(max(normals) if forward else min(normals))
    far = 1 - near
    if (stem["point"][normal] - normals[near]) * (1 if forward else -1) <= 0:
        raise ValueError("Offset exceeds the supporting wall")
    extra = f"{joint['id']}-offset"
    if extra in nodes:
        raise ValueError("Duplicate offset junction")
    for uid, value in zip(ends_ids, normals):
        nodes[uid]["point"][normal] = value
    joint["point"][normal] = normals[near]
    far_point = list(center)
    far_point[normal] = normals[far]
    start, end = joint["id"], ends_ids[far]
    arm = next(e for e in graph["edges"] if {start, end} == {e["start"], e["end"]})
    stem_edge = next(e for e in graph["edges"] if {start, stem["id"]} == {e["start"], e["end"]})
    arm["start" if arm["start"] == start else "end"] = extra
    graph["edges"].append(dict(start=start, end=extra, thickness_px=stem_edge["thickness_px"]))
    joint["adjacency"].remove(end)
    joint["adjacency"].append(extra)
    nodes[end]["adjacency"].remove(start)
    nodes[end]["adjacency"].append(extra)
    graph["nodes"].append(dict(id=extra, point=far_point, adjacency=[start, end]))


def apply_wall_axes(graph, review, bind, thickness_px):
    # A measurement is [normal coordinate, first station, last station] in pixels.
    # Limit snapping to the reviewed stretch, including its corner junctions.
    tolerance = 3.5
    axes = (review["vertical"], review["horizontal"])
    for runs in axes:
        for normal, lo, hi in runs:
            if not all(math.isfinite(v) for v in (normal, lo, hi)) or lo >= hi:
                raise ValueError("Invalid reviewed wall axis")

    def snap(point, dimension):
        matches = {normal for normal, lo, hi in axes[dimension]
                   if abs(point[dimension] - normal) <= tolerance
                   and lo - tolerance <= point[1-dimension] <= hi + tolerance}
        if len(matches) > 1:
            raise ValueError("Ambiguous reviewed wall axes")
        return next(iter(matches), point[dimension])

    original = {n["id"]: n["point"] for n in graph["nodes"]}
    moved = {uid: tuple(snap(p, d) for d in (0, 1)) for uid, p in original.items()}
    nodes, edges, old_lines, hosts = {}, [], [], []
    adjacent = {frozenset((e["start"], e["end"])) for e in graph["edges"]}
    for uid, xy in moved.items():
        if xy in nodes and frozenset((uid, nodes[xy])) not in adjacent:
            raise ValueError("Axis review would join unrelated junctions")
        nodes.setdefault(xy, uid)
    for edge in graph["edges"]:
        a, b = edge["start"], edge["end"]
        old_lines.append(LineString([original[a], original[b]]))
        p, q = moved[a], moved[b]
        hosts.append((p, q))
        if p == q:
            continue
        edges.append(dict(edge, start=nodes[p], end=nodes[q]))
    adjusted = dict(graph, nodes=[dict(id=uid, point=list(p), adjacency=[]) for p, uid in nodes.items()],
                    edges=edges, diagnostics=copy.deepcopy(graph["diagnostics"]))
    diagnostics = adjusted["diagnostics"]
    diagnostics["topology_gaps"] = [gap for _, gap in unresolved_gap_items(
        diagnostics.get("topology_gaps", []), diagnostics["opening_bridges"])]
    adjusted, _ = merge_straight_chains(adjusted, max_deviation=1e-6, max_angle=.0001)

    def edge_lines():
        points = {n["id"]: n["point"] for n in adjusted["nodes"]}
        return [LineString([points[e["start"]], points[e["end"]]]) for e in adjusted["edges"]]

    original_hosts = edge_lines()
    for measurement in review.get("offset_junctions", []):
        apply_offset_junction(adjusted, measurement, thickness_px)
    lines = edge_lines()
    if len(list(polygonize(lines))) != len(list(polygonize(old_lines))):
        raise ValueError("Reviewed wall axes changed the number of physical faces")

    jambs = review.get("jambs", [])
    used = set()
    changes = []
    for i, bridge in enumerate(adjusted["diagnostics"]["opening_bridges"]):
        old, _ = bind(bridge, old_lines)
        if old is None:
            raise ValueError("Cannot move an opening without its original carrier")
        a, b = hosts[old["index"]]
        if a == b:
            raise ValueError("Axis review collapsed an occupied carrier")
        clear = bridge.get("clear_span", bridge["span"])
        center = LineString(clear).centroid.coords[0]
        kind = {"recovered_door": "doors", "recovered_window": "windows"}.get(bridge["kind"], bridge["kind"])
        matches = [j for j, row in enumerate(jambs) if kind == row["kind"] and math.dist(center, row["near"]) <= 3]
        if len(matches) > 1 or used.intersection(matches):
            raise ValueError("Ambiguous reviewed jamb measurement")
        if a[0] != b[0] and a[1] != b[1]:
            if matches or not all(tuple(original[k]) == moved[k] for k in
                                  (graph["edges"][old["index"]]["start"], graph["edges"][old["index"]]["end"])):
                raise ValueError("Opening carrier only partially covered by axis review")
            continue
        along = 0 if a[1] == b[1] else 1
        limits = sorted(p[along] for p in clear)
        if matches:
            j = matches[0]
            used.add(j)
            lo, hi = jambs[j]["limits"]
            if not (lo < hi and max(abs(lo-limits[0]), abs(hi-limits[1])) <= 8):
                raise ValueError("Reviewed jambs do not match the source opening")
            limits = [lo, hi]
        # Keep the particular source wall. A nearby parallel wall cannot take
        # the binding simply because its center is closer to the moved opening.
        host = LineString([a, b])
        owners = [k for k, line in enumerate(original_hosts) if line.buffer(1e-6).covers(host)]
        if len(owners) != 1:
            raise ValueError("Missing or ambiguous wall after straightening")
        a = lines[owners[0]].coords[0]
        span = [[v, a[1]] if along == 0 else [a[0], v] for v in limits]
        stations = limits + [p[along] for p in bridge["span"]]
        bridge["span"] = [[v, a[1]] if along == 0 else [a[0], v] for v in (min(stations), max(stations))]
        bridge["clear_span"] = span
        check, _ = bind(bridge, lines)
        if check is None or check["index"] != owners[0]:
            raise ValueError("Reviewed opening left its original wall")
        # A hole can fit on a segment yet cut into the face of the
        # perpendicular wall at its end, as seen in the target editor.
        wall = lines[owners[0]]
        clearances = []
        for end in wall.coords:
            for other in lines:
                if end in other.coords and other.coords[0][along] == other.coords[1][along]:
                    clearance = min(abs(x-end[along]) for x in limits) - thickness_px / 2
                    clearances.append(clearance)
        if clearances and min(clearances) < -1e-4:
            raise ValueError(f"Reviewed opening {i} intersects a perpendicular wall")
        changes.append(dict(bridge_index=i, original_span=graph["diagnostics"]["opening_bridges"][i]["span"],
                            original_clear_span=clear, clear_span=span, measured_jambs=bool(matches),
                            corner_clearance_px=min(clearances, default=None)))
    if len(used) != len(jambs):
        raise ValueError("A reviewed opening was not found")
    return adjusted, changes
