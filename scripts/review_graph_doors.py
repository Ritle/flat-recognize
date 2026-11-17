"""Apply image-bound user door choices to existing uncertain graph boundaries.

The user supplies the type; source walls and raster must still support the gap.
This does not change automatic recognition or guess the hinge/swing direction.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from classify_openings import structural_union
from door_leaf_evidence import wall_jamb_support
from mask_wall_graph import clear_jamb_span
from opening_graph_cleanup import ink_radii
from review_decisions import MAX_ANSWERS, image_sha256


def gap_fingerprint(gap):
    geometry = dict(span=sorted(gap['span']), width=gap['local_width_px'])
    return hashlib.sha256(json.dumps(geometry, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]


def apply_graph_door_decisions(graph, source, rgb, decisions, source_sha256):
    if decisions.get('version') != 1 or decisions.get('image_sha256') != source_sha256:
        raise ValueError('Door review format or source image hash does not match')
    answers = decisions.get('answers')
    if not isinstance(answers, list) or len(answers) > MAX_ANSWERS:
        raise ValueError('Invalid number of graph review answers')
    height, width = rgb.shape[:2]
    if (source.get('meta', {}).get('width'), source.get('meta', {}).get('height')) != (width, height):
        raise ValueError('Classified source dimensions do not match raster')
    if graph.get('coordinate_space') != 'source_raster' or source.get('meta', {}).get('coordinate_space', 'source_raster') != 'source_raster':
        raise ValueError('Door review requires source_raster coordinates')
    gaps = graph.get('diagnostics', {}).get('topology_gaps', [])
    applied = {b.get('review', {}).get('gap_fingerprint') for b in graph.get('diagnostics', {}).get('opening_bridges', [])
               if b.get('source') == 'explicit_user_review' and b.get('kind') == 'doors'}
    selected, seen = [], set()
    for answer in answers:
        fingerprint = answer.get('gap_fingerprint') if isinstance(answer, dict) else None
        if not fingerprint or fingerprint in seen or answer.get('choice') != 'door':
            raise ValueError('Graph door review needs unique explicit door choices')
        seen.add(fingerprint)
        if fingerprint in applied:
            continue
        matches = [g for g in gaps if gap_fingerprint(g) == fingerprint]
        if len(matches) != 1:
            raise ValueError('Reviewed gap is missing or its geometry has changed')
        selected.append((answer, matches[0]))
    result = copy.deepcopy(graph)
    if not selected:
        return result
    nodes = {n['id']: n for n in result['nodes']}
    walls = structural_union(source.get('walls', []))
    distance = cv2.distanceTransform((rgb.max(axis=2) < 160).astype(np.uint8), cv2.DIST_L2, 5)
    excluded = unary_union([Polygon(p) for p in result['diagnostics'].get('excluded_regions', [])])
    for answer, candidate in selected:
        line = LineString(candidate['span'])
        wall_width = candidate['local_width_px']
        endpoint_ids = []
        for p in line.coords:
            matches = [uid for uid, n in nodes.items() if Point(n['point']).distance(Point(p)) < 1e-5]
            if len(matches) != 1:
                raise ValueError('Reviewed gap no longer has two existing graph endpoints')
            endpoint_ids.append(matches[0])
        edges = [LineString([nodes[e[k]]['point'] for k in ('start', 'end')]) for e in result['edges']]
        interior = LineString([line.interpolate(.001), line.interpolate(line.length - .001)])
        if interior.intersects(unary_union(edges)) or line.intersects(excluded):
            raise ValueError('Reviewed door crosses existing structure or an excluded region')
        clear = clear_jamb_span(line, walls, wall_width, graph['scale_cm_per_px'])
        if clear is None or min(wall_jamb_support(*clear.coords, wall_width, walls)) < 2:
            raise ValueError('Reviewed door has no supported free aperture between jambs')
        for bridge in result['diagnostics'].get('opening_bridges', []):
            other = LineString(bridge.get('clear_span', bridge['span']))
            if clear.intersection(other.buffer(wall_width / 2, cap_style=2)).length > clear.length * .2:
                raise ValueError('Reviewed door overlaps an existing opening')
        radii = ink_radii(clear, distance)
        if radii is None or np.mean(radii >= 1.8) > .15:
            raise ValueError('Reviewed aperture is occupied by thick raster structure')
        a, b = np.asarray(clear.coords[0]), np.asarray(clear.coords[-1])
        axis = (b - a) / clear.length
        jamb_radii = []
        for p, direction in ((a, -axis), (b, axis)):
            probe = LineString([p + direction * wall_width * .25, p + direction * wall_width * .75])
            r = ink_radii(probe, distance)
            if r is None or np.mean(r >= 1.8) < .8:
                raise ValueError('Reviewed door jamb is not supported by thick raster ink')
            jamb_radii.append(float(r.min()))
        before = [p for p in polygonize(edges) if p.area * graph['scale_cm_per_px'] ** 2 >= 5000]
        after = [p for p in polygonize(edges + [line]) if p.area * graph['scale_cm_per_px'] ** 2 >= 5000]
        if len(after) != len(before) + 1 or abs(sum(p.area for p in after) - sum(p.area for p in before)) > .01:
            raise ValueError('Reviewed internal door does not split one existing area safely')
        result['edges'].append(dict(start=endpoint_ids[0], end=endpoint_ids[1], thickness_px=wall_width))
        nodes[endpoint_ids[0]]['adjacency'].append(endpoint_ids[1])
        nodes[endpoint_ids[1]]['adjacency'].append(endpoint_ids[0])
        fingerprint = answer['gap_fingerprint']
        result['diagnostics']['opening_bridges'].append(dict(
            kind='doors', opening_id=f'reviewed-door-{fingerprint}', source='explicit_user_review',
            span=list(line.coords), clear_span=list(clear.coords), requires_review=True,
            review=dict(gap_fingerprint=fingerprint, image_sha256=source_sha256, choice='door',
                        user_statement=answer.get('user_statement'), hinge_and_swing='unconfirmed_template_defaults'),
            evidence=dict(model_jamb_support=wall_jamb_support(*clear.coords, wall_width, walls),
                          raster_jamb_min_radius_px=jamb_radii, type_confirmed_by_user=True)))
        result['diagnostics']['topology_gaps'] = [g for g in result['diagnostics']['topology_gaps'] if gap_fingerprint(g) != fingerprint]
    network = [LineString([nodes[e[k]]['point'] for k in ('start', 'end')]) for e in result['edges']]
    network += [LineString(g['span']) for g in result['diagnostics']['topology_gaps']]
    result['faces'] = [dict(polygon=list(p.exterior.coords)[:-1], area_px2=p.area, requires_review=True)
                       for p in polygonize(unary_union(network)) if p.area * graph['scale_cm_per_px'] ** 2 >= 5000]
    return result


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('image', type=Path)
    parser.add_argument('--decisions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    result = apply_graph_door_decisions(load(args.graph), load(args.source),
                                       np.asarray(Image.open(args.image).convert('RGB')),
                                       load(args.decisions), image_sha256(args.image))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(dict(reviewed_doors=sum(b.get('source') == 'explicit_user_review' for b in result['diagnostics']['opening_bridges'])))


if __name__ == '__main__':
    main()
