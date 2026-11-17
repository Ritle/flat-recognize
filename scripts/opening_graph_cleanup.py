"""Raster-confirmed cleanup of known, blank apertures in a physical wall graph.

This is a separate postprocessor, not an opening detector or JSON adapter.
Missing model-supplied windows require continuous frames as well as jambs.
Ambiguous gaps retain their original graph and diagnostic rejection.
"""
import copy
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union, polygonize

from classify_openings import opening_axis
from window_frame_evidence import bounded_frame_bay, narrow_frame_rails


def trim_window_cuts(graph, source, rgb):
    """Trim cuts inferred from whole carriers when both raster jambs are visible."""
    result = copy.deepcopy(graph)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    changes = []
    for index, bridge in enumerate(result['diagnostics']['opening_bridges']):
        if bridge['kind'] != 'windows' or 'clear_span' in bridge or bridge.get('opening_id') is None:
            continue
        matches = [w for w in source.get('windows', []) if w.get('id') == bridge['opening_id']]
        if len(matches) != 1:
            continue
        a, b, width = opening_axis(matches[0])
        start, end = np.asarray(bridge['span'], dtype=float)
        length = np.linalg.norm(end - start)
        if length < 8 or width < 3 or np.linalg.norm(b-a) < 5:
            continue
        direction = (end - start) / length
        if abs((b-a) @ direction) < .98 * np.linalg.norm(b-a):
            continue
        bay = bounded_frame_bay(gray, a, b, width, (start-width*direction, end+width*direction))
        if bay is None:
            continue
        measured = np.asarray(bay['clear_span'])
        offsets = (measured - start) @ direction
        projected = start + offsets[:, None] * direction
        lo, hi = sorted(offsets)
        if (max(np.linalg.norm(measured-projected, axis=1)) > 2 or lo < -1 or hi > length+1
                or hi-lo < .55*length or max(lo, length-hi) < max(2.5, width*.5)):
            continue
        # Only trim the cut within its own carrier. Keep the wall graph intact.
        bridge['clear_span'] = [(start+t*direction).tolist() for t in (max(0, lo), min(length, hi))]
        bridge['aperture_measurement'] = dict(source='visible_window_jambs', **bay)
        changes.append(dict(bridge_index=index, previous_width_px=float(length),
                            measured_width_px=float(min(length, hi)-max(0, lo))))
    return result, changes


def blank_aperture(gray, opening):
    """A continuous white core bounded by thick, dark jambs on both sides."""
    a, b, width = opening_axis(opening)
    length = float(np.linalg.norm(b - a))
    if width < 4 or length < 8 or length < width * 2:
        return None
    axis = (b - a) / length
    normal = np.array([-axis[1], axis[0]])
    along = np.arange(-math.ceil(width), math.ceil(length + width) + .5, .5)
    offsets = np.linspace(-width * .2, width * .2, max(3, math.ceil(width * .4) + 1))
    points = a + along[None, :, None] * axis + offsets[:, None, None] * normal
    if (points[..., 0].min() < 0 or points[..., 1].min() < 0
            or points[..., 0].max() > gray.shape[1] - 1 or points[..., 1].max() > gray.shape[0] - 1):
        return None
    strip = cv2.remap(gray, points[..., 0].astype('float32'), points[..., 1].astype('float32'), cv2.INTER_LINEAR)
    white = (strip >= 200).mean(axis=0) >= .9
    middle = int(np.argmin(abs(along - length / 2)))
    if not white[middle]:
        return None
    left = right = middle
    while left > 0 and white[left - 1]:
        left -= 1
    while right + 1 < len(white) and white[right + 1]:
        right += 1
    lo, hi = along[left] - .25, along[right] + .25
    if not .75 * length <= hi - lo <= 1.25 * length or abs(lo) > width or abs(hi - length) > width:
        return None
    # Require a full dark jamb band, not a single door leaf or dimension line.
    bands = [(along >= lo - width / 2) & (along <= lo - 1.5),
             (along <= hi + width / 2) & (along >= hi + 1.5)]
    if any(not band.any() or np.mean(strip[:, band] < 140) < .85 for band in bands):
        return None
    return dict(start=a + lo * axis, end=a + hi * axis, axis=axis, normal=normal,
                width=float(width), length=float(hi - lo), white_fraction=float((strip[:, left:right + 1] >= 200).mean()))


def line_parts(shape):
    if shape.is_empty:
        return []
    if shape.geom_type == 'LineString':
        return [shape] if shape.length > 1e-5 else []
    return [line for part in getattr(shape, 'geoms', []) for line in line_parts(part)]


def ink_radii(line, distance):
    points = np.array([line.interpolate(t, normalized=True).coords[0]
                       for t in np.linspace(.05, .95, max(5, math.ceil(line.length)))])
    x, y = np.rint(points).astype(int).T
    if min(x) < 0 or min(y) < 0 or max(x) >= distance.shape[1] or max(y) >= distance.shape[0]:
        return None
    return distance[y, x]


def cleanup_one(lines, bridge, opening, gray, distance, excluded=None, *, evidence=None):
    attempts = []
    for factor in (.5, 1., 1.5):
        refined, updated, report = cleanup_at_padding(lines, bridge, opening, gray, distance, factor, excluded, evidence=evidence)
        attempts.append(dict(pad_factor=factor, reason=report.get('reason'), accepted=report['accepted']))
        if report['accepted'] or report.get('reason') != 'missing_unique_jamb_ports':
            return refined, updated, dict(report, attempts=attempts)
    return None, None, dict(report, attempts=attempts)


def cleanup_at_padding(lines, bridge, opening, gray, distance, pad_factor, excluded=None, *, evidence=None):
    evidence = blank_aperture(gray, opening) if evidence is None else evidence
    report = dict(accepted=False)
    if evidence is None:
        return None, None, dict(report, reason='missing_blank_aperture_and_dark_jambs')
    a, b, axis, normal = (evidence[k] for k in ('start', 'end', 'axis', 'normal'))
    width, length = evidence['width'], evidence['length']
    pad, half = width * pad_factor, width
    start, end = a - axis * pad, b + axis * pad
    domain = Polygon([start - normal * half, end - normal * half,
                      end + normal * half, start + normal * half])
    if excluded is not None and domain.intersects(excluded):
        return None, None, dict(report, reason='excluded_region')
    clear_spans = evidence.get('clear_spans', [(a, b)])
    core = unary_union([LineString([np.asarray(p) + axis, np.asarray(q) - axis]).buffer(half, cap_style=2)
                        for p, q in clear_spans])
    report.update(clear_span=[a.tolist(), b.tolist()], domain=list(domain.exterior.coords),
                  white_fraction=evidence['white_fraction'], jamb_padding_px=pad)
    # A real wall crossing the aperture cannot be discarded, even if most of
    # the surrounding rectangle is white. Test direct ink, not nearby pixels.
    for line, _ in lines:
        for part in line_parts(line.intersection(core)):
            radii = ink_radii(part, distance)
            if radii is None or (part.length >= 3 and np.mean(radii >= 1.8) > .25):
                delta = np.array(part.coords[-1]) - part.coords[0]
                offset = abs(float((np.array(part.centroid.coords[0]) - a) @ normal))
                frame_rail = (bridge['kind'] == 'windows' and radii is not None
                              and abs(float(delta @ axis)) >= part.length * .95
                              and offset >= width * .3)
                if frame_rail:
                    continue
                return None, None, dict(report, reason='thick_structure_inside_aperture', conflicting_part=list(part.coords))
    outside, removed = [], []
    for line, thickness in lines:
        inside = line.intersection(domain)
        if inside.length > 1e-5:
            removed.extend(list(p.coords) for p in line_parts(inside))
            outside.extend((p, thickness) for p in line_parts(line.difference(domain)))
        else:
            outside.append((line, thickness))
    # Exact boundary ports in the clipped graph. Real external junctions must
    # survive; only an isolated thin leaf or a very short spur may be removed.
    def key(p):
        return tuple(round(float(v), 7) for v in p)
    incident = {}
    for i, (line, _) in enumerate(outside):
        for p in (line.coords[0], line.coords[-1]):
            incident.setdefault(key(p), []).append(i)
    ports = [p for p in incident if domain.boundary.distance(Point(p)) < 1e-5]
    end_ports = [[], []]
    side_ports = []
    for p in ports:
        along = float((np.array(p) - a) @ axis)
        if abs(along + pad) < 1e-4:
            end_ports[0].append(p)
        elif abs(along - length - pad) < 1e-4:
            end_ports[1].append(p)
        else:
            side_ports.append(p)
    if any(len(ports) != 1 for ports in end_ports):
        return None, None, dict(report, reason='missing_unique_jamb_ports', port_counts=list(map(len, end_ports)),
                               jamb_ports=end_ports, side_ports=side_ports)
    drop = set()
    leaves = []
    for port in side_ports:
        if len(incident[port]) != 1:
            return None, None, dict(report, reason='external_junction_at_side_port')
        path, current, previous = [], port, None
        while True:
            options = [i for i in incident[current] if i != previous]
            if not options:
                break
            if len(options) != 1 or options[0] in path:
                return None, None, dict(report, reason='side_branch_has_junction_or_loop')
            i = options[0]
            path.append(i)
            line = outside[i][0]
            ends = [key(line.coords[0]), key(line.coords[-1])]
            current = ends[1] if ends[0] == current else ends[0]
            previous = i
            if current in ports:
                return None, None, dict(report, reason='side_branch_rejoins_graph')
        total = sum(outside[i][0].length for i in path)
        radii = [ink_radii(outside[i][0], distance) for i in path]
        if any(r is None for r in radii):
            return None, None, dict(report, reason='side_branch_outside_raster')
        radii = np.concatenate(radii)
        delta = np.array(current) - port
        perpendicular = abs(float(delta @ axis)) <= max(2, total * .2)
        near_jamb = min(abs(float((np.array(port) - a) @ axis)),
                        abs(float((np.array(port) - b) @ axis))) <= width
        leaf = bridge['kind'] == 'doors' and perpendicular and near_jamb and total <= length * 1.25
        if ((not leaf and total > width * .5) or np.mean(radii >= 1.8) > .1
                or float(np.median(radii)) > 1.5):
            return None, None, dict(report, reason='side_branch_not_confirmed_thin_artifact')
        drop.update(path)
        leaves.append(dict(points=[list(outside[i][0].coords) for i in path], length_px=total,
                           kind='door_leaf' if leaf else 'short_thin_spur'))
    outside = [item for i, item in enumerate(outside) if i not in drop]
    additions = [(LineString([start, end]), width)]
    for port, anchor in zip((end_ports[0][0], end_ports[1][0]), (start, end)):
        connection = LineString([port, anchor])
        if connection.length < 1e-5:
            continue
        radii = ink_radii(connection, distance)
        if radii is None or np.mean(radii >= 1.8) < .8:
            return None, None, dict(report, reason='jamb_connection_not_in_thick_ink')
        additions.append((connection, width))
    for i, (line, _) in enumerate(additions):
        for other, _ in outside + additions[:i]:
            crossing = line.intersection(other)
            if crossing.is_empty:
                continue
            common = [Point(p) for p in (line.coords[0], line.coords[-1])
                      if min(Point(p).distance(Point(q)) for q in (other.coords[0], other.coords[-1])) < 1e-5]
            if crossing.geom_type != 'Point' or not any(crossing.distance(p) < 1e-5 for p in common):
                return None, None, dict(report, reason='replacement_crosses_other_structure')
    updated = dict(bridge, span=[start.tolist(), end.tolist()], clear_span=[a.tolist(), b.tolist()],
                   requires_review=True, carrier_cleanup='raster_confirmed_blank_aperture')
    report.update(accepted=True, removed_edges=removed, removed_side_branches=leaves,
                  replacement_span=updated['span'], previous_span=bridge['span'])
    return outside + additions, updated, report


def overlaps_existing_carrier(opening, bridges):
    a, b, width = opening_axis(opening)
    axis = LineString([a, b])
    if axis.length <= 1e-5:
        return True
    direction = (b - a) / axis.length
    for bridge in bridges:
        line = LineString(bridge.get('clear_span', bridge['span']))
        delta = np.asarray(line.coords[-1]) - line.coords[0]
        if (abs(float(delta @ direction)) >= line.length * .95
                and axis.intersection(line.buffer(max(2, width / 2), cap_style=2)).length >= axis.length * .5):
            return True
    return False


def paired_aperture(first, second, distance):
    """Two measured white gaps separated by a directly supported solid pier."""
    axis = first['axis']
    if abs(float(axis @ second['axis'])) < .995:
        return None
    width = max(first['width'], second['width'])
    if width > min(first['width'], second['width']) * 1.5:
        return None
    if abs(float((second['start'] - first['start']) @ first['normal'])) > min(first['width'], second['width']) * .25:
        return None
    ordered = sorted((first, second), key=lambda e: float(e['start'] @ axis))
    left, right = ordered
    gap = float((right['start'] - left['end']) @ axis)
    if not 3 <= gap <= width * 3:
        return None
    pier = LineString([left['end'] + axis * 1.5, right['start'] - axis * 1.5])
    radii = ink_radii(pier, distance)
    if radii is None or np.mean(radii >= 1.8) < .8:
        return None
    return dict(start=left['start'], end=right['end'], axis=axis, normal=first['normal'], width=width,
                length=float((right['end'] - left['start']) @ axis),
                white_fraction=min(first['white_fraction'], second['white_fraction']),
                clear_spans=[(e['start'], e['end']) for e in ordered],
                pier_span=list(pier.coords), pier_min_radius_px=float(radii.min()))


def rejected_frame_candidates(prediction, gray):
    """Reconsider only geometry-filtered model candidates with continuous rails.

    Their semantic class is merely a proposal. Frame spacing supplies a measured
    thickness; the original rejected polygon and reasons remain in provenance.
    These candidates still have to pass the aperture/ports/pier recovery checks.
    """
    candidates, rejections = [], []
    rejected = prediction.get('diagnostics', {}).get('rejected', [])
    allowed = {'too_short', 'not_elongated', 'too_far_from_wall'}
    for item in rejected:
        reasons = item.get('reject_reason', [])
        oid = item.get('id')
        record = dict(source='rejected_model_frame', source_id=oid, accepted=False)
        if (item.get('source_class') not in ('door', 'window') or not reasons
                or not set(reasons) <= allowed):
            rejections.append(dict(record, reason='ineligible_model_rejection'))
            continue
        if oid is None or sum(o.get('id') == oid for o in rejected) != 1:
            rejections.append(dict(record, reason='ambiguous_raw_candidate_id'))
            continue
        a, b, width = opening_axis(item)
        frame = narrow_frame_rails(gray, a, b, width)
        if frame is None:
            rejections.append(dict(record, reason='missing_continuous_frame_rails'))
            continue
        axis = (b - a) / np.linalg.norm(b - a)
        normal = np.array([-axis[1], axis[0]])
        lo, hi = frame['rail_offsets_px'][0], frame['rail_offsets_px'][-1]
        outer = [(p + normal * offset).tolist() for p, offset in ((a, lo), (b, lo), (b, hi), (a, hi))]
        # Do not carry a stale bbox/thickness: opening_axis must use the measured
        # frame geometry rather than the rejected segmentation rectangle.
        candidates.append(dict(id=f'raw-window-{oid}', outer=outer, holes=[],
                               recovery_source='rejected_model_frame',
                               raw_recovery=dict(candidate=copy.deepcopy(item), frame_evidence=frame,
                                                 measured_frame_width_px=hi - lo)))
    return candidates, rejections


def recover_classified_windows(lines, diagnostics, source, gray, distance, excluded):
    """Attach classified or frame-verified model candidates to physical carriers.

    An absent carrier is not evidence of an opening. Both raster jambs and two
    existing graph ports are mandatory. A missing pier may be supported jointly
    with a neighbouring *already bound* window and a thick raster strip.
    """
    reports = []
    windows = source.get('windows', [])
    bridges = diagnostics['opening_bridges']
    for opening in windows:
        oid = opening.get('id')
        if any(b.get('kind') == 'windows' and b.get('opening_id') == oid for b in bridges):
            continue
        base = dict(source=opening.get('recovery_source', 'classified_window_without_carrier'),
                    opening_id=oid, accepted=False)
        if 'raw_recovery' in opening:
            base['raw_recovery'] = copy.deepcopy(opening['raw_recovery'])
        if oid is None or sum(o.get('id') == oid for o in windows) != 1:
            reports.append(dict(base, reason='ambiguous_source_opening_id'))
            continue
        if overlaps_existing_carrier(opening, bridges):
            reports.append(dict(base, reason='already_represented_by_existing_carrier'))
            continue
        a, b, _ = opening_axis(opening)
        frame = narrow_frame_rails(gray, *opening_axis(opening))
        if frame is None:
            reports.append(dict(base, reason='missing_continuous_frame_rails'))
            continue
        proposal = dict(kind='windows', opening_id=oid, span=[a.tolist(), b.tolist()], source=base['source'])
        updated_lines, updated, decision = cleanup_one(lines, proposal, opening, gray, distance, excluded)
        neighbour = None
        pair = None
        if updated_lines is None and decision.get('reason') == 'missing_unique_jamb_ports':
            first = blank_aperture(gray, opening)
            options = []
            for j, other_bridge in enumerate(bridges):
                if other_bridge.get('kind') != 'windows' or 'clear_span' in other_bridge:
                    continue
                matches = [o for o in windows if o.get('id') == other_bridge.get('opening_id')]
                if len(matches) != 1:
                    continue
                neighbour_frame = narrow_frame_rails(gray, *opening_axis(matches[0]))
                if neighbour_frame is None:
                    continue
                second = blank_aperture(gray, matches[0])
                combined = paired_aperture(first, second, distance) if first and second else None
                if combined is None:
                    continue
                trial_lines, trial_bridge, trial_report = cleanup_one(
                    lines, proposal, opening, gray, distance, excluded, evidence=combined)
                if trial_lines is not None:
                    options.append((j, second, combined, trial_lines, trial_bridge, trial_report, neighbour_frame))
            if len(options) == 1:
                neighbour, second, pair, updated_lines, updated, decision, neighbour_frame = options[0]
                updated['clear_span'] = [first['start'].tolist(), first['end'].tolist()]
                updated['carrier_cleanup'] = 'paired_raster_apertures'
            elif len(options) > 1:
                decision = dict(decision, reason='ambiguous_supported_window_pair')
        record = dict(base, **decision, frame_evidence=frame)
        reports.append(record)
        if updated_lines is None:
            continue
        # Existing, unrelated holes cannot be erased or covered by a new span.
        domain = Polygon(decision['domain'])
        if any(j != neighbour and domain.intersection(LineString(b.get('clear_span', b['span']))).length > .1
               for j, b in enumerate(bridges)):
            record.update(accepted=False, reason='overlaps_another_opening')
            continue
        lines = updated_lines
        if neighbour is not None:
            bridges[neighbour] = dict(bridges[neighbour], span=updated['span'],
                                      clear_span=[second['start'].tolist(), second['end'].tolist()],
                                      carrier_cleanup='paired_raster_apertures', requires_review=True)
            record.update(neighbour_opening_id=bridges[neighbour].get('opening_id'),
                          pier_span=pair['pier_span'], pier_min_radius_px=pair['pier_min_radius_px'],
                          neighbour_frame_evidence=neighbour_frame, clear_span=updated['clear_span'],
                          neighbour_clear_span=bridges[neighbour]['clear_span'])
        record['bridge_index'] = len(bridges)
        if 'raw_recovery' in opening:
            updated['raw_recovery'] = copy.deepcopy(opening['raw_recovery'])
        bridges.append(updated)
        # A proven aperture supersedes a fully enclosed temporary closure.
        # Partial/unrelated topology boundaries are never promoted to walls.
        resolved, remaining = [], []
        direction = np.asarray(updated['span'][1]) - updated['span'][0]
        direction /= np.linalg.norm(direction)
        for gap in diagnostics.get('topology_gaps', []):
            line = LineString(gap['span'])
            parallel = abs(float((np.asarray(line.coords[-1]) - line.coords[0]) @ direction)) >= line.length * .95
            (resolved if domain.covers(line) and parallel else remaining).append(gap)
        diagnostics['topology_gaps'] = remaining
        record['resolved_topology_gaps'] = resolved
    return lines, reports


def cleanup_opening_graph(graph, source, rgb, bridge_indices, *, recover_missing=False, prediction=None):
    """Refine failed bindings; optionally recover missing, raster-supported frames."""
    height, width = rgb.shape[:2]
    meta = source.get('meta', {})
    if (meta.get('width'), meta.get('height')) != (width, height):
        raise ValueError('Classified source dimensions do not match raster')
    if graph.get('coordinate_space') != 'source_raster' or meta.get('coordinate_space', 'source_raster') != 'source_raster':
        raise ValueError('Graph and source must use source_raster coordinates')
    if prediction is not None:
        raw_meta = prediction.get('meta', {})
        if not recover_missing:
            raise ValueError('Raw prediction requires recover_missing=True')
        if (raw_meta.get('width'), raw_meta.get('height')) != (width, height):
            raise ValueError('Raw prediction dimensions do not match raster')
        if raw_meta.get('coordinate_space', 'source_raster') != 'source_raster':
            raise ValueError('Raw prediction must use source_raster coordinates')
        if raw_meta.get('source') and meta.get('source') and raw_meta['source'] != meta['source']:
            raise ValueError('Raw prediction and classified source names do not match')
    result = copy.deepcopy(graph)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    distance = cv2.distanceTransform((rgb.max(axis=2) < 160).astype(np.uint8), cv2.DIST_L2, 5)
    nodes = {n['id']: n['point'] for n in graph['nodes']}
    lines = [(LineString([nodes[e[k]] for k in ('start', 'end')]), e['thickness_px']) for e in graph['edges']]
    reports = []
    excluded = unary_union([Polygon(p) for p in graph.get('diagnostics', {}).get('excluded_regions', [])])
    for index in dict.fromkeys(bridge_indices):
        bridge = result['diagnostics']['opening_bridges'][index]
        matches = [o for o in source.get(bridge.get('kind'), []) if o.get('id') == bridge.get('opening_id')]
        if bridge.get('kind') not in ('doors', 'windows') or len(matches) != 1 or 'clear_span' in bridge:
            reports.append(dict(bridge_index=index, accepted=False, reason='requires_unique_classified_unmeasured_opening'))
            continue
        updated_lines, updated_bridge, report = cleanup_one(lines, bridge, matches[0], gray, distance, excluded)
        reports.append(dict(report, bridge_index=index, opening_id=bridge.get('opening_id')))
        if updated_lines is not None:
            lines = updated_lines
            result['diagnostics']['opening_bridges'][index] = updated_bridge
    if recover_missing:
        recovery_source = source
        if prediction is not None:
            candidates, rejections = rejected_frame_candidates(prediction, gray)
            reports.extend(rejections)
            recovery_source = dict(source, windows=source.get('windows', []) + candidates)
        lines, missing_reports = recover_classified_windows(lines, result['diagnostics'], recovery_source, gray, distance, excluded)
        reports.extend(missing_reports)
    if not any(r['accepted'] for r in reports):
        return result, reports
    positions, vertices, edges = {}, [], []
    def vertex(point):
        xy = tuple(round(float(v), 7) for v in point)
        if xy not in positions:
            positions[xy] = len(vertices)
            vertices.append(dict(id=len(vertices), point=list(xy), adjacency=[]))
        return positions[xy]
    for line, thickness in lines:
        start, end = vertex(line.coords[0]), vertex(line.coords[-1])
        if start == end:
            continue
        edges.append(dict(start=start, end=end, thickness_px=thickness))
        vertices[start]['adjacency'].append(end)
        vertices[end]['adjacency'].append(start)
    result.update(nodes=vertices, edges=edges)
    network = [LineString([vertices[e[k]]['point'] for k in ('start', 'end')]) for e in edges]
    network += [LineString(g['span']) for g in result['diagnostics'].get('topology_gaps', [])]
    result['faces'] = [dict(polygon=list(p.exterior.coords)[:-1], area_px2=p.area, requires_review=True)
                       for p in polygonize(unary_union(network)) if p.area * graph['scale_cm_per_px'] ** 2 >= 5000]
    result['diagnostics']['opening_carrier_cleanup'] = reports
    return result, reports


def main():
    from PIL import Image, ImageDraw
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('image', type=Path)
    parser.add_argument('--bindings-report', type=Path, required=True)
    parser.add_argument('--recover-missing', action='store_true', help='Also attach classified windows without a carrier, using raster evidence')
    parser.add_argument('--prediction', type=Path, help='Optional matching raw prediction; reconsider geometry-filtered frame candidates')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    graph, source, report = load(args.graph), load(args.source), load(args.bindings_report)
    rows = report['unbound_openings']
    for row in rows:
        if row.get('bridge') != graph['diagnostics']['opening_bridges'][row['bridge_index']]:
            raise ValueError('Binding diagnostics do not match source graph')
    rgb = np.asarray(Image.open(args.image).convert('RGB'))
    refined, decisions = cleanup_opening_graph(graph, source, rgb, [r['bridge_index'] for r in rows],
                                               recover_missing=args.recover_missing,
                                               prediction=load(args.prediction) if args.prediction else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(refined, indent=2, allow_nan=False), encoding='utf-8')
    args.output.with_suffix('.report.json').write_text(json.dumps(decisions, indent=2, allow_nan=False), encoding='utf-8')
    overlay = Image.fromarray(rgb)
    draw = ImageDraw.Draw(overlay)
    for edge in refined['edges']:
        draw.line([tuple(refined['nodes'][edge[k]]['point']) for k in ('start', 'end')], fill='cyan', width=2)
    overlay.save(args.output.with_suffix('.png'))
    print(dict(accepted=sum(r['accepted'] for r in decisions), attempted=len(decisions)))


if __name__ == '__main__':
    main()
