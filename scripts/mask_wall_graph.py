"""Experimental room-independent wall graph, in source raster coordinates.

No target-project serialization here. Rejected evidence stays in the report.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon
from shapely import set_precision
from shapely.ops import polygonize, unary_union

from extract_rooms import fill_polygon
from internal_wall_graph import skeleton_paths
from polygon_geometry import parts
from classify_openings import opening_axis, structural_union
from door_leaf_evidence import (leaf_and_arc, diagonal_leaf_with_jambs,
                                dark_leaf_with_jambs, wall_jamb_support, dark_stroke_support)
from window_frame_evidence import narrow_frame_rails, bounded_frame_bay


def node_lines(lines, snap, *, straighten=True):
    """Snap endpoints onto supported neighboring edges, then node intersections."""
    lines = [line for line in lines if line.length > 1e-6]
    # Quantize only near-axis directions. True diagonals retain their orientation.
    straight = []
    for line in lines:
        a, b = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
        delta = b-a
        angle = abs(math.degrees(math.atan2(delta[1], delta[0]))) % 180
        if straighten and min(angle, 180-angle) <= 6:
            a[1] = b[1] = (a[1]+b[1])/2
        elif straighten and abs(angle-90) <= 6:
            a[0] = b[0] = (a[0]+b[0])/2
        straight.append(LineString([a,b]))
    # Cluster close endpoints once: independent pairwise snapping swaps endpoints
    # and leaves the very gaps it was intended to repair.
    points=[np.array(p) for line in straight for p in (line.coords[0],line.coords[-1])]
    parents=list(range(len(points)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,a in enumerate(points):
        for j in range(i):
            if i//2!=j//2 and np.linalg.norm(a-points[j])<=snap:
                parents[root(i)]=root(j)
    centers={}
    for i,p in enumerate(points):centers.setdefault(root(i),[]).append(p)
    centers={k:np.mean(v,axis=0) for k,v in centers.items()}
    adjusted=[LineString([centers[root(2*i)],centers[root(2*i+1)]]) for i in range(len(straight))]
    adjusted=[line for line in adjusted if line.length>1e-6]
    # Insert explicit T anchors in the target edge so projections remain exact.
    anchors=[[] for _ in adjusted]; endpoints=[list(line.coords) for line in adjusted]
    for i,line in enumerate(adjusted):
        for k,xy in enumerate(line.coords):
            p=Point(xy)
            candidates=[]
            for j,other in enumerate(adjusted):
                t=other.project(p)
                if j!=i and snap<t<other.length-snap:
                    candidates.append((other.distance(p),j,other.interpolate(t)))
            if candidates:
                distance,j,target=min(candidates,key=lambda row:row[0])
                if distance<=snap:
                    endpoints[i][k]=target.coords[0];anchors[j].append(target.coords[0])
    adjusted=[LineString([endpoints[i][0],*sorted(anchors[i],key=lambda p:line.project(Point(p))),endpoints[i][-1]])
              for i,line in enumerate(adjusted)]
    # unary_union nodes actual crossings; round to remove floating point duplicates.
    network=unary_union([set_precision(line,1e-4) for line in adjusted])
    return [LineString([a,b]) for line in parts(network,'LineString')
            for a,b in zip(line.coords,list(line.coords)[1:]) if Point(a).distance(Point(b))>1e-5]


def supported_wall_runs(line, raw_distance, radius, thickness):
    """Keep continuous thick-ink portions of a rejected mixed wall/opening axis.

    The neighborhood and ink threshold match the existing raster support test.
    No unsupported gap is filled, even when two supported runs are collinear.
    """
    height, width = raw_distance.shape
    count = max(3, math.ceil(line.length) + 1)
    distances = np.linspace(0, line.length, count)
    supported = []
    for distance in distances:
        x, y = np.rint(line.interpolate(distance).coords[0]).astype(int)
        x, y = np.clip(x, 0, width - 1), np.clip(y, 0, height - 1)
        patch = raw_distance[max(0, y-radius):min(height, y+radius+1),
                             max(0, x-radius):min(width, x+radius+1)]
        supported.append(bool(patch.max() >= 1.8))
    start = None
    for index, present in enumerate(supported + [False]):
        if present and start is None:
            start = index
        elif not present and start is not None:
            if distances[index - 1] - distances[start] >= max(3, thickness):
                yield LineString([line.interpolate(distances[start]).coords[0],
                                  line.interpolate(distances[index - 1]).coords[0]])
            start = None


def t_jamb_gaps(edges, scale):
    """Candidate gaps from a wall tip to the transverse wall at a T jamb.

    A transverse junction is not a dangling endpoint. Only an outward-facing
    tip and an existing transverse junction qualify; raster evidence is checked
    by the caller before any candidate becomes a physical opening.
    """
    incident = {}
    for edge in edges:
        a, b = map(tuple, (edge.coords[0], edge.coords[-1]))
        incident.setdefault(a, []).append(b)
        incident.setdefault(b, []).append(a)
    network = unary_union(edges)
    for tip, neighbors in incident.items():
        if len(neighbors) != 1:
            continue
        outward = np.asarray(tip) - neighbors[0]
        outward = outward / np.linalg.norm(outward)
        for junction, branches in incident.items():
            if len(branches) < 2:
                continue
            delta = np.asarray(junction) - tip
            length = float(np.linalg.norm(delta))
            if not 60 / scale <= length <= 150 / scale:
                continue
            direction = delta / length
            if float(outward @ direction) < .98:
                continue
            if any(abs(float((np.asarray(end) - junction) @ direction))
                   > np.linalg.norm(np.asarray(end) - junction) * .2
                   for end in branches):
                continue
            gap = LineString([tip, junction])
            # A candidate cannot traverse another wall or an existing opening.
            interior = LineString([gap.interpolate(.001).coords[0],
                                   gap.interpolate(length - .001).coords[0]])
            if interior.intersects(network):
                continue
            yield gap


def terminal_jamb_gaps(edges, scale):
    """Open spans between a wall tip and a collinear or transverse jamb.

    A transverse wall may end at two opposing doorways, so its node can have
    degree one. At least one endpoint must extend the carrier axis outward.
    """
    incident = {}
    for edge in edges:
        a, b = map(tuple, (edge.coords[0], edge.coords[-1]))
        incident.setdefault(a, []).append(b)
        incident.setdefault(b, []).append(a)
    network = unary_union(edges)
    nodes = list(incident)
    for i, a in enumerate(nodes):
        for b in nodes[:i]:
            if min(len(incident[a]), len(incident[b])) != 1:
                continue
            gap = LineString([a, b])
            if not 60 / scale <= gap.length <= 150 / scale:
                continue
            direction = (np.asarray(b) - a) / gap.length
            aligned, valid = [], True
            for node, sign in ((a, -1), (b, 1)):
                dots = [float((np.asarray(end) - node) @ direction) * sign
                        / np.linalg.norm(np.asarray(end) - node) for end in incident[node]]
                aligned.append(any(value > .98 for value in dots))
                if not (aligned[-1] or all(abs(value) < .2 for value in dots)):
                    valid = False
            if not valid or not any(aligned):
                continue
            interior = LineString([gap.interpolate(.001).coords[0],
                                   gap.interpolate(gap.length - .001).coords[0]])
            if not interior.intersects(network):
                yield gap


def projected_jamb_gaps(edges, scale):
    """Extend a long wall tip to the interior of a transverse carrier edge.

    Short spurs cannot define a long extrapolation: the supported incident axis
    must be at least as long as the proposed gap. Existing graph nodes are
    handled by terminal_jamb_gaps instead.
    """
    incident = {}
    for edge in edges:
        for a,b in ((edge.coords[0],edge.coords[-1]),(edge.coords[-1],edge.coords[0])):
            incident.setdefault(tuple(a),[]).append(tuple(b))
    network = unary_union(edges)
    for start, neighbors in incident.items():
        if len(neighbors) != 1:
            continue
        direction = np.asarray(start) - neighbors[0]
        supported_length = float(np.linalg.norm(direction))
        direction /= supported_length
        ray = LineString([start, np.asarray(start) + direction * min(150/scale, supported_length)])
        for edge in edges:
            vector = np.asarray(edge.coords[-1]) - edge.coords[0]
            if abs(float(vector @ direction)) > edge.length * .2:
                continue
            hit = ray.intersection(edge)
            if hit.geom_type != 'Point' or hit.distance(Point(start)) < 60/scale:
                continue
            if min(hit.distance(Point(edge.coords[0])),hit.distance(Point(edge.coords[-1]))) < .01:
                continue
            gap = LineString([start,hit.coords[0]])
            interior = LineString([gap.interpolate(.001).coords[0],
                                   gap.interpolate(gap.length-.001).coords[0]])
            if not interior.intersects(network):
                yield gap


def continued_jamb_gaps(edges, scale, widths):
    """Use a wall's straight continuation through junctions to orient its tip.

    A short skeleton tail can be skewed at a T junction. It cannot alone
    support extrapolation; a unique aligned chain, at least as long as the gap,
    must stay inside the local wall-width corridor. Raster door evidence is
    still required by the caller.
    """
    incident = {}
    for edge in edges:
        a, b = edge.coords[0], edge.coords[-1]
        incident.setdefault(a, []).append(b)
        incident.setdefault(b, []).append(a)
    network = unary_union(edges)
    seen = set()
    for tip, neighbors in incident.items():
        if len(neighbors) != 1:
            continue
        path = [tip, neighbors[0]]
        width = local_axis_width(LineString(path), widths)
        while True:
            previous, current = np.asarray(path[-2]), np.asarray(path[-1])
            direction = (current-previous)/np.linalg.norm(current-previous)
            choices = [p for p in incident[path[-1]] if p not in path
                       and float((np.asarray(p)-current) @ direction)
                       > .95*np.linalg.norm(np.asarray(p)-current)]
            if len(choices) != 1:
                break
            axis = LineString([tip, choices[0]])
            if any(axis.distance(Point(p)) > max(2, width/2) for p in path):
                break
            path.append(choices[0])
        if len(path) < 3:
            continue
        axis = np.asarray(tip)-path[-1]
        support_length = float(np.linalg.norm(axis))
        direction = axis/support_length
        proposals = []
        for target, branches in incident.items():
            delta = np.asarray(target)-tip
            length = float(np.linalg.norm(delta))
            if not 60/scale <= length <= min(150/scale, support_length):
                continue
            gap_axis = delta/length
            if float(gap_axis @ direction) < .98:
                continue
            dots = [float((np.asarray(p)-target) @ gap_axis)/np.linalg.norm(np.asarray(p)-target)
                    for p in branches]
            if any(dot > .98 for dot in dots) or all(abs(dot) < .2 for dot in dots):
                proposals.append(('existing_node', LineString([tip, target])))
        ray = LineString([tip, np.asarray(tip)+direction*min(150/scale, support_length)])
        for edge in edges:
            if abs(float((np.asarray(edge.coords[-1])-edge.coords[0]) @ direction)) > edge.length*.2:
                continue
            hit = ray.intersection(edge)
            if hit.geom_type != 'Point' or hit.distance(Point(tip)) < 60/scale:
                continue
            if min(hit.distance(Point(edge.coords[0])), hit.distance(Point(edge.coords[-1]))) < .01:
                continue
            proposals.append(('projected_jamb', LineString([tip, hit.coords[0]])))
        for target_kind, gap in proposals:
            key = tuple(sorted(gap.coords))
            interior = LineString([gap.interpolate(.001), gap.interpolate(gap.length-.001)])
            if key in seen or interior.intersects(network):
                continue
            seen.add(key)
            yield gap, dict(target_kind=target_kind, support_path=[list(p) for p in path],
                            support_length_px=round(support_length, 3), local_width_px=width)


def clear_jamb_span(gap, walls, width, scale):
    """Locate the free centerline between jamb faces without moving the axis."""
    free = list(parts(gap.difference(walls), 'LineString'))
    if not free:
        return None
    span = max(free, key=lambda line: line.length)
    if span.length < max(60 / scale, gap.length * .6):
        return None
    bounds = sorted(gap.project(Point(p)) for p in (span.coords[0], span.coords[-1]))
    if max(bounds[0], gap.length - bounds[1]) > width:
        return None
    band = span.buffer(width / 2, cap_style=2)
    if walls.intersection(band).area > band.area * .25:
        return None
    return span


def local_axis_width(edge, evidence):
    """Use the nearest measured source axis, as for final edge thickness."""
    return min(evidence, key=lambda item: LineString(
        [item['start'], item['end']]).distance(edge.centroid))['thickness_px']


def topology_gap_support(gap, left_width, right_width):
    """Bound an unconfirmed closure by both flanking wall widths.

    Six local wall widths is the existing relaxed room-repair gap limit.
    Confirmed door recovery is handled separately and does not use this limit.
    """
    local_width = min(left_width, right_width)
    return dict(local_width_px=local_width, max_gap_px=6 * local_width,
                accepted=gap.length <= 6 * local_width)


def raster_pier_runs(edges, windows, raw_distance, widths):
    """Recover thick raster piers inside a supported series of windows.

    Interpolate only between outward-facing wall tips. Two separate, aligned
    supplied windows must support the carrier. Never extrapolate beyond the
    tips, cross a graph edge, or fill a window with an inferred wall.
    """
    incident = {}
    for edge in edges:
        for point in (edge.coords[0], edge.coords[-1]):
            incident.setdefault(point, []).append(edge)
    ends = [(point, branches[0]) for point, branches in incident.items() if len(branches) == 1]
    network = unary_union(edges)
    for i, (a, ea) in enumerate(ends):
        for b, eb in ends[:i]:
            gap = LineString([a, b])
            if gap.length < .01:
                continue
            direction = (np.asarray(b) - a) / gap.length
            outward = []
            for tip, edge, sign in ((a, ea, 1), (b, eb, -1)):
                other = edge.coords[-1] if edge.coords[0] == tip else edge.coords[0]
                outward.append(float((np.asarray(tip)-other) @ direction)*sign/edge.length)
            if min(outward) < .98:
                continue
            interior = LineString([gap.interpolate(.001), gap.interpolate(gap.length-.001)])
            if interior.intersects(network):
                continue
            aligned = []
            for opening in windows:
                start, end, width = opening_axis(opening)
                axis = LineString([start, end])
                if (axis.length < 3 or abs(float((end-start) @ direction)) < .95*axis.length
                        or gap.distance(axis.centroid) > max(4, width/2)):
                    continue
                interval = sorted([gap.project(Point(start)), gap.project(Point(end))])
                if interval[1] - interval[0] >= axis.length * .6:
                    aligned.append((interval, opening))
            if not any(a1[1] < a2[0] or a2[1] < a1[0]
                       for i, (a1, _) in enumerate(aligned) for a2, _ in aligned[:i]):
                continue
            thickness = min(local_axis_width(ea, widths), local_axis_width(eb, widths))
            radius = max(1, min(5, round(thickness/2)))
            # Clip even thick raster support to the supplied window intervals:
            # dark frames are evidence of an opening, not a new solid pier.
            holes = unary_union([LineString([gap.interpolate(interval[0]), gap.interpolate(interval[1])])
                                 for interval, _ in aligned])
            for run in supported_wall_runs(gap, raw_distance, radius, thickness):
                for pier in parts(run.difference(holes), 'LineString'):
                    if pier.length < max(3, thickness):
                        continue
                    yield pier, dict(start=list(pier.coords[0]), end=list(pier.coords[-1]),
                                     thickness_px=thickness, raster_support=1.0,
                                     source='raster_pier', parent_span=list(gap.coords),
                                     opening_ids=[op.get('id') for _, op in aligned],
                                     requires_review=True)


def candidate_door_jamb_repairs(edges, widths, windows, prediction, walls, gray, raw_distance,
                                 scale, excluded_shape):
    """Restore a missed thick pier only with a model door and adjacent window.

    The raw pier lies inside an existing graph gap. Its opposite side must
    attach to a supplied window; the door side needs two jambs, a free span,
    an inward leaf and direct dark-pixel support. No candidate coordinates are
    used as a replacement for raster wall geometry.
    """
    candidates = [c for c in (prediction or {}).get('diagnostics',{}).get('rejected',[])
                  if c.get('source_class')=='door' and c.get('reject_reason')
                  and set(c['reject_reason']) <= {'not_elongated','too_short','too_far_from_wall'}]
    network = unary_union(edges)
    repairs,decisions = [],[]
    if not candidates or not windows:
        return repairs,decisions
    for gap in terminal_jamb_gaps(edges,scale):
        if gap.intersects(excluded_shape):
            continue
        width = min(local_axis_width(Point(p),widths) for p in gap.coords)
        direction = (np.asarray(gap.coords[-1])-gap.coords[0])/gap.length
        for candidate in candidates:
            try:
                polygon = Polygon(candidate.get('outer',[]))
            except (TypeError,ValueError):
                continue
            if polygon.is_empty or not polygon.is_valid:
                continue
            a,b,_ = opening_axis(candidate)
            length = float(np.linalg.norm(b-a))
            if (length<8 or abs(float((b-a) @ direction))<length*.95
                    or gap.distance(polygon.centroid)>max(3,width/2)):
                continue
            for core in supported_wall_runs(gap,raw_distance,0,max(3,width/2)):
                if core.distance(network)<.01:
                    continue
                for tip,near in ((gap.coords[0],core.coords[0]),(gap.coords[-1],core.coords[-1])):
                    door = LineString([tip,near])
                    projected = sorted(door.project(Point(p)) for p in (a,b))
                    if projected[1]-projected[0] < length*.7:
                        continue
                    record = dict(source_id=candidate.get('id'),span=list(door.coords),
                                  jamb_span=list(core.coords),accepted=False)
                    decisions.append(record)
                    window_ids = []
                    for window in windows:
                        x,y,window_width = opening_axis(window)
                        axis = LineString([x,y])
                        if (axis.length<3 or abs(float((y-x) @ direction))<axis.length*.95
                                or axis.intersects(door.buffer(window_width/2,cap_style=2))):
                            continue
                        window_near,window_far = sorted((Point(x),Point(y)),key=core.distance)
                        tolerance = max(4,min(8,window_width))
                        if core.distance(window_near)<=tolerance and network.distance(window_far)<=tolerance:
                            window_ids.append(window.get('id'))
                    if not window_ids:
                        record['reason'] = 'no_adjacent_supplied_window'
                        continue
                    augmented = walls.union(core.buffer(width/2,cap_style=2))
                    clear = clear_jamb_span(door,augmented,width,scale)
                    if clear is None:
                        record['reason'] = 'no_supported_clear_span'
                        continue
                    evidence = []
                    for search in (door,clear):
                        leaf = diagonal_leaf_with_jambs(
                            gray,*search.coords,width,augmented,clear_span=list(clear.coords),inward_only=True)
                        if leaf and min(leaf['jamb_support'])>=2:
                            support = dark_stroke_support(gray,leaf['hinge'],leaf['tip'])
                            if support>=.8:
                                evidence.append({**leaf,'dark_raster_support':round(support,3)})
                    if not evidence:
                        record['reason'] = 'missing_model_guided_dark_leaf'
                        continue
                    # Accept at most one repair of this physical span; later
                    # candidates cannot reuse the newly connected network.
                    interior = LineString([door.interpolate(.001),door.interpolate(door.length-.001)])
                    if interior.intersects(network):
                        record['reason'] = 'occupied_graph_span'
                        continue
                    best = max(evidence,key=lambda item:(item['dark_raster_support'],item['tip_clearance']))
                    record.update(accepted=True,reason='model_door_raw_pier_window_and_leaf',
                                  clear_span=list(clear.coords),window_ids=window_ids,evidence=best)
                    repairs.append(dict(jamb=dict(start=list(core.coords[0]),end=list(core.coords[-1]),
                                                  thickness_px=width,raster_support=1.0,
                                                  source='candidate_supported_jamb',requires_review=True),
                                        bridge=dict(kind='recovered_door',source='candidate_supported_jamb',
                                                    source_id=candidate.get('id'),span=list(door.coords),
                                                    clear_span=list(clear.coords),window_ids=window_ids,
                                                    evidence=best,requires_review=True)))
                    network = unary_union([network,core,door])
    return repairs,decisions


def bounded_window_attachment(opening, edges, widths, walls, gray, excluded_shape):
    """Attach a known frame whose aperture is shorter than its carrier span."""
    record = dict(opening_id=opening.get('id'),accepted=False)
    a,b,width = opening_axis(opening)
    length = float(np.linalg.norm(b-a))
    if length < 8 or width <= 0:
        record['reason'] = 'unresolved_frame_size'
        return None,record
    direction = (b-a)/length
    targets,distances = [],[]
    for p in (Point(a),Point(b)):
        choices = []
        for edge in edges:
            if abs(float((np.asarray(edge.coords[-1])-edge.coords[0]) @ direction)) < edge.length*.95:
                continue
            target = edge.interpolate(edge.project(p))
            choices.append((p.distance(target),target,edge))
        if not choices:
            break
        distance,target,edge = min(choices,key=lambda row:row[0])
        # Include half a pixel of raster quantization; keep the existing 8 px
        # ceiling, then independently verify the full aperture and both jambs.
        tolerance = max(4,min(8,(width+local_axis_width(edge,widths))/2+.5))
        if distance > tolerance:
            break
        targets.append(target.coords[0])
        distances.append(dict(distance_px=round(distance,3),tolerance_px=round(tolerance,3)))
    if len(targets)!=2:
        record['reason'] = 'missing_nearby_carrier'
        return None,record
    gap = LineString(targets)
    if gap.length < length*.6 or gap.intersects(excluded_shape):
        record['reason'] = 'invalid_or_excluded_span'
        return None,record
    interior = LineString([gap.interpolate(.001),gap.interpolate(gap.length-.001)])
    if interior.intersects(unary_union(edges)):
        record['reason'] = 'occupied_graph_span'
        return None,record
    bay = bounded_frame_bay(gray,a,b,width,targets)
    if bay is None:
        record['reason'] = 'missing_bounded_raster_bay'
        return None,record
    support = wall_jamb_support(*bay['clear_span'],width,walls)
    if min(support)<2:
        record['reason'] = 'missing_structural_jambs'
        return None,record
    record.update(accepted=True,reason='bounded_frame_and_local_jambs',span=targets,
                  bay_evidence=bay,jamb_support=support,attachment_evidence=distances)
    return dict(kind='windows',opening_id=opening.get('id'),span=targets,
                clear_span=bay['clear_span'],requires_review=True,bay_evidence=bay,
                attachment_method='bounded_frame_and_local_jambs',jamb_support=support),record


def raster_frame_gaps(edges, walls, known_openings, gray, raw_distance, width, scale, excluded_shape):
    """Recover small unclassified frame gaps with two model and raster jambs.

    At least one wall tip must extend outward along the carrier. A short skewed
    cap may form the other jamb, but thin linework cannot supply thick raster
    support on both sides. These openings always require review.
    """
    incident = {}
    for edge in edges:
        a,b = edge.coords[0],edge.coords[-1]
        incident.setdefault(a,[]).append(b)
        incident.setdefault(b,[]).append(a)
    tips = [(p,ends[0]) for p,ends in incident.items() if len(ends)==1]
    network = unary_union(edges)
    known = structural_union(known_openings)
    bridges,decisions = [],[]
    for i,(a,neighbor_a) in enumerate(tips):
        for b,neighbor_b in tips[:i]:
            gap = LineString([a,b])
            if not 8 <= gap.length <= 150/scale:
                continue
            direction = (np.asarray(b)-a)/gap.length
            if max(abs(direction)) < math.cos(math.radians(12)):
                continue
            outward = [float((np.asarray(a)-neighbor_a) @ direction)/np.linalg.norm(np.asarray(a)-neighbor_a),
                       float((np.asarray(neighbor_b)-b) @ direction)/np.linalg.norm(np.asarray(neighbor_b)-b)]
            if max(outward)<.95 or min(outward)<-.2:
                continue
            interior = LineString([gap.interpolate(.001),gap.interpolate(gap.length-.001)])
            if gap.intersects(excluded_shape) or interior.intersects(network):
                continue
            # Correct only small cap skew; true diagonal frames remain handled
            # by supplied segmentation candidates rather than this fallback.
            x,y = np.asarray(a).copy(),np.asarray(b).copy()
            normal_index = 1 if abs(direction[0])>abs(direction[1]) else 0
            x[normal_index] = y[normal_index] = (x[normal_index]+y[normal_index])/2
            bay = bounded_frame_bay(gray,x,y,width,[a,b])
            if bay is None:
                continue
            record = dict(span=[list(a),list(b)],accepted=False,bay_evidence=bay)
            decisions.append(record)
            clear = LineString(bay['clear_span'])
            band = clear.buffer(width/2,cap_style=2)
            if band.intersection(known).area > band.area*.2:
                record['reason'] = 'known_opening_requires_attachment'
                continue
            support = wall_jamb_support(*bay['clear_span'],width,walls)
            if min(support)<2:
                record['reason'] = 'missing_structural_jambs'
                continue
            axis = (np.asarray(clear.coords[-1])-clear.coords[0])/clear.length
            normal = np.array([-axis[1],axis[0]])
            thick = []
            for point,sign in ((clear.coords[0],-1),(clear.coords[-1],1)):
                samples = [np.asarray(point)+axis*sign*width*t+normal*width*s
                           for t in (.5,1) for s in (-.25,0,.25)]
                xy = np.rint(samples).astype(int)
                valid = ((xy[:,0]>=0)&(xy[:,0]<gray.shape[1])&(xy[:,1]>=0)&(xy[:,1]<gray.shape[0]))
                thick.append(float(raw_distance[xy[valid,1],xy[valid,0]].max()) if valid.any() else 0)
            if min(thick)<1.8:
                record['reason'] = 'missing_thick_raster_jambs'
                continue
            record.update(accepted=True,reason='bounded_frame_with_two_thick_jambs',
                          jamb_support=support,thick_jamb_radius_px=thick)
            bridges.append(dict(kind='recovered_window',source='bounded_raster_frame_gap',
                                span=record['span'],clear_span=bay['clear_span'],thickness_px=width,
                                requires_review=True,bay_evidence=bay,jamb_support=support,
                                thick_jamb_radius_px=thick))
            network = network.union(gap)
    return bridges,decisions


def rejected_window_bridges(edges, prediction, gray, excluded_shape):
    """Revisit rejected polygons only at two existing wall tips.

    Distance rejection requires a matching gap and raster rails. Short narrow
    candidates additionally require a measured clear bay with two dark jambs.
    Thickness rejection remains out of scope.
    """
    incident = {}
    for edge in edges:
        for p in (edge.coords[0], edge.coords[-1]):
            incident.setdefault(p, []).append(edge)
    ends = [(p, branches[0]) for p, branches in incident.items() if len(branches) == 1]
    network = unary_union(edges)
    decisions, bridges = [], []
    for candidate in (prediction or {}).get('diagnostics', {}).get('rejected', []):
        record = dict(source_id=candidate.get('id'), accepted=False)
        decisions.append(record)
        reasons = set(candidate.get('reject_reason',[]))
        short = bool(reasons) and reasons <= {'too_short','not_elongated'}
        if (reasons != {'too_far_from_wall'} and not short
                or candidate.get('source_class') not in ('door', 'window')):
            record['reason'] = 'unsupported_rejection_reason'
            continue
        try:
            shape = Polygon(candidate.get('outer', []))
        except (TypeError, ValueError):
            record['reason'] = 'invalid_candidate_polygon'
            continue
        if not shape.is_valid or shape.is_empty:
            record['reason'] = 'invalid_candidate_polygon'
            continue
        a, b, width = opening_axis(candidate)
        length = float(np.linalg.norm(b-a))
        if width <= 0 or length < (8 if short else max(8, width*1.8)):
            record['reason'] = 'unsupported_candidate_shape'
            continue
        axis = (b-a)/length
        tolerance = max(4, min(8, width))
        targets = []
        for point, sign in ((a, 1), (b, -1)):
            choices = []
            for tip, edge in ends:
                other = edge.coords[-1] if edge.coords[0] == tip else edge.coords[0]
                outward = (np.asarray(tip)-other)/edge.length
                distance = float(np.linalg.norm(point-tip))
                if distance <= tolerance and float(outward @ axis)*sign > .95:
                    choices.append((distance, tip))
            if not choices:
                break
            targets.append(min(choices, key=lambda row: row[0])[1])
        if len(targets) != 2:
            record['reason'] = 'missing_aligned_wall_tips'
            continue
        gap = LineString(targets)
        record['span'] = [list(p) for p in gap.coords]
        maximum_length = length+2*tolerance if short else length*1.5
        if not length*.6 <= gap.length <= maximum_length:
            record['reason'] = 'gap_length_mismatch'
            continue
        interior = LineString([gap.interpolate(.01), gap.interpolate(gap.length-.01)])
        if gap.intersects(excluded_shape):
            record['reason'] = 'excluded_region'
        elif interior.intersects(network):
            record['reason'] = 'occupied_graph_span'
        else:
            frame = narrow_frame_rails(gray, a, b, width)
            bay = bounded_frame_bay(gray,a,b,width,targets) if short else None
            if frame is None:
                record['reason'] = 'missing_parallel_frame_rails'
            elif short and bay is None:
                record['reason'] = 'missing_bounded_raster_bay'
            else:
                record.update(accepted=True, reason='supported_gap_and_raster_frame', evidence=frame)
                bridges.append(dict(kind='recovered_window', source_id=candidate.get('id'),
                                    source_class=candidate.get('source_class'), span=record['span'],
                                    thickness_px=width, evidence=frame, requires_review=True))
                if bay:
                    record.update(reason='short_candidate_with_bounded_raster_bay',bay_evidence=bay)
                    bridges[-1].update(clear_span=bay['clear_span'],bay_evidence=bay,
                                       source='short_candidate_raster_bay')
                network = network.union(gap)
    return bridges, decisions


def interior_line_closures(edges, topology_gaps, candidates, widths, walls, gray, scale):
    """Partition an existing interior at an uncertain, raster-marked passage.

    These spans never enter the physical wall graph or opening classifications.
    They require two structural jambs and must split one existing face into
    exactly two sufficiently large faces. Exterior closure is out of scope.
    """
    network = unary_union(edges+[LineString(r['span']) for r in topology_gaps])
    closures, decisions = [], []
    for candidate in candidates:
        if candidate.get('accepted') or candidate.get('reason') not in (
                'missing_inward_leaf_evidence', 'missing_dark_inward_leaf_evidence'):
            continue
        gap = LineString(candidate['span'])
        interior = LineString([gap.interpolate(.001), gap.interpolate(gap.length-.001)])
        if interior.intersects(network):
            continue
        record = dict(span=candidate['span'],accepted=False)
        decisions.append(record)
        local_widths = [local_axis_width(Point(p),widths) for p in gap.coords]
        support = topology_gap_support(gap,*local_widths)
        record.update(support)
        record['accepted'] = False
        if not support['accepted']:
            record['reason'] = 'gap_too_long_for_local_wall_width'
            continue
        width = support['local_width_px']
        jambs = wall_jamb_support(*gap.coords,width,walls)
        if min(jambs) < 2:
            record['reason'] = 'missing_structural_jambs'
            continue
        parents = [face for face in polygonize(network)
                   if face.contains(gap.interpolate(.5,normalized=True))
                   and interior.difference(face).length < .01]
        if len(parents) != 1:
            record['reason'] = 'not_inside_existing_face'
            continue
        clear = LineString(candidate['clear_span'])
        a,b = np.asarray(clear.coords)
        normal = np.array([-(b-a)[1],(b-a)[0]])/clear.length
        jitter = max(2,round(width*.2))
        evidence = []
        for offset in range(-jitter,jitter+1):
            samples = a + np.linspace(.2,.8,max(5,round(clear.length)))[:,None]*(b-a) + offset*normal
            xy = np.rint(samples).astype(int)
            if (xy[:,0].min()<0 or xy[:,0].max()>=gray.shape[1]
                    or xy[:,1].min()<0 or xy[:,1].max()>=gray.shape[0]):
                continue
            fraction = float(np.mean(gray[xy[:,1],xy[:,0]]<140))
            evidence.append((fraction,offset))
        fraction,offset = max(evidence,default=(0,0),key=lambda row:row[0])
        if fraction < .8:
            record['reason'] = 'missing_continuous_raster_line'
            continue
        trial = network.union(gap)
        children = [face for face in polygonize(trial)
                    if parents[0].contains(face.representative_point())]
        if (len(children)!=2 or any(face.area*scale*scale<5000 for face in children)
                or abs(sum(face.area for face in children)-parents[0].area)>.01):
            record['reason'] = 'no_two_valid_interior_faces'
            continue
        closure = dict(span=candidate['span'],reason='interior_raster_line_without_confirmed_type',
                       evidence='raster_line',raster_support=round(fraction,3),
                       raster_offset_px=offset,jamb_support=jambs,requires_review=True,
                       opening_type='uncertain',**support)
        closures.append(closure)
        record.update(closure)
        network = trial
    return closures,decisions


def build_graph(source, rgb, scale=1.0, excluded_regions=(), *, prediction=None):
    """Build axes without rooms. Explicit exclusions are in source pixel space.

    Exclusions are user/annotation scope, never inferred from a file name. They
    are optional: automatic detection of exterior railings is still separate.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be positive centimeters per source pixel')
    h,w=rgb.shape[:2]
    if (source['meta']['width'],source['meta']['height'])!=(w,h):
        raise ValueError('Source dimensions do not match image')
    if prediction is not None:
        meta = prediction.get('meta', {})
        if (meta.get('width'), meta.get('height')) != (w,h):
            raise ValueError('Prediction dimensions do not match image')
        if meta.get('coordinate_space', 'source_raster') != 'source_raster':
            raise ValueError('Prediction must use source raster coordinates')
        if meta.get('source') and source['meta'].get('source') and meta['source'] != source['meta']['source']:
            raise ValueError('Prediction and classified source names do not match')
    mask=np.zeros((h,w),np.uint8)
    for wall in source.get('walls',[]):
        local=np.zeros_like(mask);fill_polygon(local,wall);mask|=local
    # Only explicitly supplied/confirmed openings may support an axis gap.
    for opening in source.get('doors',[])+source.get('windows',[]):
        local=np.zeros_like(mask);fill_polygon(local,opening);mask|=local
    excluded=np.zeros_like(mask)
    excluded_shape=unary_union([Polygon(ring) for ring in excluded_regions])
    for ring in excluded_regions:
        cv2.fillPoly(excluded,[np.rint(ring).astype(np.int32)],255)
    mask[excluded>0]=0
    paths,skeleton=skeleton_paths(mask)
    distance=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)
    # Independent raw-raster evidence rejects broad segmentation furniture/fills.
    ink=(rgb.max(axis=2)<160).astype(np.uint8)
    raw_distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    accepted=[];rejected=[];widths=[];retained_runs=[]
    for path in paths:
        line=LineString(path).simplify(1.0)
        for a,b in zip(line.coords,list(line.coords)[1:]):
            edge=LineString([a,b])
            if edge.length<0.01:continue
            samples=np.array([edge.interpolate(t,normalized=True).coords[0]
                              for t in np.linspace(.1,.9,max(3,round(edge.length)))])
            xy=np.rint(samples).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
            radius=float(np.median(distance[xy[:,1],xy[:,0]]))
            support=[]
            for x,y in xy:
                r=max(1,min(5,round(radius)))
                support.append(float(raw_distance[max(0,y-r):min(h,y+r+1),max(0,x-r):min(w,x+r+1)].max()))
            support=float(np.mean(np.array(support)>=1.8))
            record=dict(start=list(a),end=list(b),thickness_px=round(radius*2,3),raster_support=round(support,3))
            if support<.55:
                rejected.append(dict(record,reason='weak_thick_raster_support'))
                for run in supported_wall_runs(edge, raw_distance,
                                               max(1, min(5, round(radius))), radius*2):
                    evidence = dict(start=list(run.coords[0]), end=list(run.coords[-1]),
                                    thickness_px=record['thickness_px'], raster_support=1.0,
                                    source='supported_subspan', parent_span=[list(a), list(b)])
                    accepted.append(run)
                    widths.append(evidence)
                    retained_runs.append(evidence)
                continue
            accepted.append(edge);widths.append(record)
    edges=node_lines(accepted,min(15/scale,5),straighten=False)
    original_network = unary_union(edges)
    recovered_piers = []
    pier_candidates = list(raster_pier_runs(edges, source.get('windows', []), raw_distance, widths))
    for pier, evidence in pier_candidates:
        if not pier.intersects(excluded_shape):
            edges.append(pier)
            widths.append(evidence)
            recovered_piers.append(evidence)
    if recovered_piers:
        edges=node_lines(edges,.01,straighten=False)
    unconfirmed_pier_gaps = []
    pier_network = unary_union([LineString([r['start'], r['end']]) for r in recovered_piers])
    def uses_unconfirmed_pier(gap):
        # Raster-only piers can bind supplied windows, but must not introduce
        # new door/closure hypotheses (nearby furniture can imitate a leaf).
        if any(pier_network.distance(Point(p)) < .01 and original_network.distance(Point(p)) > .01
               for p in (gap.coords[0], gap.coords[-1])):
            span = [list(p) for p in gap.coords]
            if not any(r['span'] == span or r['span'] == span[::-1] for r in unconfirmed_pier_gaps):
                unconfirmed_pier_gaps.append(dict(span=span,reason='unconfirmed_raster_jamb',requires_review=True))
            return True
        return False
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    wall_shapes=structural_union(source.get('walls',[]))
    opening_bridges=[]
    jamb_repairs,jamb_repair_candidates=candidate_door_jamb_repairs(
        edges,widths,source.get('windows',[]),prediction,wall_shapes,gray,raw_distance,scale,excluded_shape)
    for repair in jamb_repairs:
        jamb=repair['jamb']
        widths.append(jamb)
        edges.extend([LineString([jamb['start'],jamb['end']]),LineString(repair['bridge']['span'])])
        opening_bridges.append(repair['bridge'])
    if jamb_repairs:
        edges=node_lines(edges,.01,straighten=False)
    pending_windows=[]
    for kind in ('doors','windows'):
        for opening in source.get(kind,[]):
            a,b,width=opening_axis(opening)
            length=float(np.linalg.norm(b-a))
            if length<3:continue
            direction=(b-a)/length
            targets=[]
            bound=False
            reviewed_attachment = False
            attachment_evidence = []
            for p in (Point(a),Point(b)):
                candidates=[]
                for edge in edges:
                    vector=np.array(edge.coords[-1])-edge.coords[0]
                    if abs(np.dot(vector,direction))<edge.length*.95:continue
                    target=edge.interpolate(edge.project(p))
                    candidates.append((p.distance(target),target,edge))
                if not candidates:break
                dist,target,carrier=min(candidates,key=lambda r:r[0])
                tolerance=max(4,min(8,width))
                if dist>tolerance:
                    # A medial axis can retreat inside a thick jamb. Broaden
                    # attachment only by that local thickness and only when a
                    # window frame is independently visible in the raster.
                    local_width=local_axis_width(carrier,widths)
                    tolerance=max(4,min(8,(width+local_width)/2))
                    if (kind!='windows' or dist>tolerance
                            or narrow_frame_rails(gray,a,b,width) is None):break
                    reviewed_attachment=True
                    attachment_evidence.append(dict(distance_px=round(dist,3),
                                                    tolerance_px=round(tolerance,3),
                                                    local_width_px=local_width))
                targets.append(target.coords[0])
            if len(targets)==2:
                bridge=LineString(targets)
                if (bridge.length>=length*.6 and bridge.length<=length*1.5
                        and not bridge.intersects(excluded_shape)):
                    edges.append(bridge)
                    bound=True
                    opening_bridges.append(dict(kind=kind,opening_id=opening.get('id'),span=targets))
                    if reviewed_attachment:
                        opening_bridges[-1].update(requires_review=True,
                            attachment_method='local_jamb_width_and_raster_frame',
                            attachment_evidence=attachment_evidence)
            if kind=='windows' and not bound:
                pending_windows.append(opening)
    # Reuse the existing raw leaf evidence at collinear graph endpoints.
    edges=node_lines(edges,min(15/scale,5),straighten=False)
    frame_attachment_candidates=[]
    for opening in pending_windows:
        bridge,decision=bounded_window_attachment(opening,edges,widths,wall_shapes,gray,excluded_shape)
        frame_attachment_candidates.append(decision)
        if bridge:
            edges=node_lines(edges+[LineString(bridge['span'])],.01,straighten=False)
            opening_bridges.append(bridge)
    recovered_windows, window_candidates = rejected_window_bridges(edges, prediction, gray, excluded_shape)
    if recovered_windows:
        edges=node_lines(edges+[LineString(r['span']) for r in recovered_windows],.01,straighten=False)
        opening_bridges.extend(recovered_windows)
    # Partial runs restore geometry, not additional independent width samples.
    # Otherwise splitting one axis changes door-evidence tolerances everywhere.
    full_widths = [r['thickness_px'] for r in widths
                   if r.get('source') not in ('supported_subspan', 'raster_pier', 'candidate_supported_jamb')]
    gap_width = float(np.median(full_widths or [r['thickness_px'] for r in widths])) if widths else 5
    frame_gaps,frame_gap_candidates=raster_frame_gaps(
        edges,wall_shapes,source.get('doors',[])+source.get('windows',[]),gray,raw_distance,
        gap_width,scale,excluded_shape)
    if frame_gaps:
        edges=node_lines(edges+[LineString(r['span']) for r in frame_gaps],.01,straighten=False)
        opening_bridges.extend(frame_gaps)
    degree={}
    for edge in edges:
        for xy in (edge.coords[0],edge.coords[-1]):degree[xy]=degree.get(xy,0)+1
    ends=[(xy,edge) for edge in edges for xy in (edge.coords[0],edge.coords[-1]) if degree[xy]==1]
    topology_gaps=[];rejected_topology_gaps=[]
    for i,(a,ea) in enumerate(ends):
        for b,eb in ends[:i]:
            gap=LineString([a,b]);length=gap.length
            if not 60/scale<=length<=150/scale:continue
            if gap.intersects(excluded_shape):continue
            direction=(np.array(b)-a)/length
            if any(abs(np.dot(np.array(e.coords[-1])-e.coords[0],direction))<e.length*.98 for e in (ea,eb)):continue
            # Require the two segments to extend away from the gap, not overlap
            # it or be unrelated parallel fragments inside the same room.
            if any(gap.intersection(e).length>1e-5 for e in (ea,eb)):continue
            if uses_unconfirmed_pier(gap):continue
            width=gap_width
            evidence=leaf_and_arc(gray,a,b,width,wall_shapes) or diagonal_leaf_with_jambs(gray,a,b,width,wall_shapes)
            if evidence:
                edges.append(gap);opening_bridges.append(dict(kind='recovered_door',span=[a,b],evidence=evidence))
            else:
                support = topology_gap_support(gap, local_axis_width(ea, widths),
                                               local_axis_width(eb, widths))
                record = dict(span=[a,b], reason='collinear_gap_without_door_evidence',
                              **support)
                if support['accepted']:
                    topology_gaps.append(record)
                else:
                    record['reason'] = 'gap_too_long_for_local_wall_width'
                    rejected_topology_gaps.append(record)
    # The original endpoint-pair search cannot reach a jamb on a T junction.
    # Unlike topology-only closures, these extra candidates require physical
    # leaf evidence and an open span in the original wall mask.
    t_jamb_candidates = []
    for gap in t_jamb_gaps(edges, scale):
        if uses_unconfirmed_pier(gap):continue
        a, b = list(gap.coords)
        record = dict(span=[list(a), list(b)], accepted=False)
        width = gap_width
        band = gap.buffer(width / 2, cap_style=2)
        if gap.intersects(excluded_shape):
            record['reason'] = 'excluded_region'
        elif wall_shapes.intersection(band).area > band.area * .25:
            record['reason'] = 'span_occupied_by_wall'
        else:
            evidence = (leaf_and_arc(gray, a, b, width, wall_shapes)
                        or diagonal_leaf_with_jambs(gray, a, b, width, wall_shapes))
            if evidence:
                record.update(accepted=True, evidence=evidence)
                edges.append(gap)
                opening_bridges.append(dict(kind='recovered_door', span=[a, b],
                                            evidence=evidence, source='t_jamb_gap'))
            else:
                record['reason'] = 'missing_door_leaf_evidence'
        t_jamb_candidates.append(record)
    terminal_candidates = []
    gap_proposals = [('existing_node', gap) for gap in terminal_jamb_gaps(edges, scale)]
    gap_proposals += [('projected_jamb', gap) for gap in projected_jamb_gaps(edges, scale)]
    continued_support = {}
    def terminal_proposals():
        yield from gap_proposals
        # Existing confirmed doors can complete a carrier through a junction.
        # Recompute tips after processing them so the same door is not recovered
        # again with a slightly shifted projection.
        for gap, support in continued_jamb_gaps(edges, scale, widths):
            continued_support[tuple(sorted(gap.coords))] = support
            yield 'continued_axis', gap
    for method, gap in terminal_proposals():
        if uses_unconfirmed_pier(gap):continue
        record = dict(span=[list(p) for p in gap.coords], accepted=False, method=method)
        if method == 'continued_axis':
            record['axis_support'] = continued_support[tuple(sorted(gap.coords))]
        clear = clear_jamb_span(gap, wall_shapes, gap_width, scale)
        if gap.intersects(excluded_shape):
            record['reason'] = 'excluded_region'
        elif method == 'continued_axis' and LineString([
                gap.interpolate(.001), gap.interpolate(gap.length-.001)]).intersects(unary_union(edges)):
            record['reason'] = 'span_occupied_by_graph'
        elif clear is None:
            record['reason'] = 'no_supported_clear_span'
        else:
            record['clear_span'] = [list(p) for p in clear.coords]
            # Use both axis endpoints and jamb faces to locate the leaf, but
            # validate occupancy on the free span. Require an inward leaf to
            # avoid borrowing a neighboring doorway's backwards-pointing leaf.
            evidence = []
            for search in (gap, clear):
                if method == 'continued_axis':
                    item = dark_leaf_with_jambs(
                        gray, search.coords[0], search.coords[-1], gap_width, wall_shapes,
                        clear_span=list(clear.coords))
                else:
                    item = diagonal_leaf_with_jambs(
                        gray, search.coords[0], search.coords[-1], gap_width, wall_shapes,
                        clear_span=list(clear.coords), inward_only=True)
                if item and (method == 'existing_node' or min(item['jamb_support']) >= 2):
                    evidence.append(item)
            if evidence:
                best = max(evidence, key=lambda item: item['tip_clearance'])
                record.update(accepted=True, evidence=best)
                edges.append(gap)
                opening_bridges.append(dict(kind='recovered_door', span=list(gap.coords),
                                            clear_span=list(clear.coords), evidence=best,
                                            source='terminal_jamb_gap', method=method,
                                            requires_review=True))
                if method == 'continued_axis':
                    opening_bridges[-1]['axis_support'] = record['axis_support']
            else:
                record['reason'] = ('missing_dark_inward_leaf_evidence' if method == 'continued_axis'
                                    else 'missing_inward_leaf_evidence')
        terminal_candidates.append(record)
    interior_closures, interior_candidates = interior_line_closures(
        edges,topology_gaps,terminal_candidates,widths,wall_shapes,gray,scale)
    topology_gaps.extend(interior_closures)
    edges=node_lines(edges,0.01,straighten=False)
    nodes=[];index={};segments=[]
    def vertex(xy):
        key=tuple(round(float(v),4) for v in xy)
        if key not in index:
            index[key]=len(nodes);nodes.append(dict(id=len(nodes),point=list(key),adjacency=[]))
        return index[key]
    for edge in edges:
        a,b=vertex(edge.coords[0]),vertex(edge.coords[-1])
        if a==b:continue
        evidence=min(widths,key=lambda r:LineString([r['start'],r['end']]).distance(edge.centroid))
        segments.append(dict(start=a,end=b,thickness_px=evidence['thickness_px']))
        nodes[a]['adjacency'].append(b);nodes[b]['adjacency'].append(a)
    topology=node_lines(edges+[LineString(g['span']) for g in topology_gaps],.01,straighten=False)
    review_spans = [LineString(g['span']) for g in topology_gaps]
    review_spans.extend(LineString(g['span']) for g in opening_bridges if g.get('requires_review'))
    review_spans.extend(LineString([r['start'], r['end']]) for r in recovered_piers)
    faces=[dict(polygon=[list(xy) for xy in poly.exterior.coords[:-1]],area_px2=poly.area,
                requires_review=any(poly.boundary.intersection(span).length>1 for span in review_spans))
           for poly in polygonize(topology) if poly.area*scale*scale>=5000]
    return dict(version=1,coordinate_space='source_raster',scale_cm_per_px=scale,
                nodes=nodes,edges=segments,faces=faces,
                diagnostics=dict(accepted=widths,rejected=rejected,opening_bridges=opening_bridges,topology_gaps=topology_gaps,
                                 excluded_regions=list(excluded_regions),
                                 retained_wall_runs=retained_runs,
                                 recovered_raster_piers=recovered_piers,
                                 unconfirmed_pier_gaps=unconfirmed_pier_gaps,
                                 rejected_window_candidates=window_candidates,
                                 frame_attachment_candidates=frame_attachment_candidates,
                                 candidate_jamb_repairs=jamb_repair_candidates,
                                 raster_frame_gap_candidates=frame_gap_candidates,
                                 interior_line_candidates=interior_candidates,
                                 terminal_jamb_candidates=terminal_candidates,
                                 rejected_topology_gaps=rejected_topology_gaps,
                                 t_jamb_candidates=t_jamb_candidates))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('image',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scale',type=float,default=1)
    parser.add_argument('--prediction',type=Path,help='Optional raw prediction with rejected opening candidates')
    parser.add_argument('--scope',type=Path,help='JSON containing exclude_polygons in source pixels')
    args=parser.parse_args()
    rgb=np.asarray(Image.open(args.image).convert('RGB'))
    scope=json.loads(args.scope.read_text(encoding='utf-8')) if args.scope else {}
    prediction=json.loads(args.prediction.read_text(encoding='utf-8')) if args.prediction else None
    graph=build_graph(json.loads(args.source.read_text(encoding='utf-8')),rgb,args.scale,
                      scope.get('exclude_polygons',[]),prediction=prediction)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(graph,indent=2),encoding='utf-8')
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
    for edge in graph['edges']:
        draw.line([tuple(graph['nodes'][edge[k]]['point']) for k in ['start','end']],fill='cyan',width=2)
    im.save(args.output.with_suffix('.png'))
    print({k:len(graph[k]) for k in ['nodes','edges','faces']})


if __name__=='__main__':main()
