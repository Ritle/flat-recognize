"""Report diagonal leaf candidates on existing openings and topology gaps.

This diagnostic never changes a graph or promotes an opening automatically.
Model-wall jambs and dark raster leaf geometry provide independent evidence;
the existing recognizer remains authoritative until regression is assessed.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from classify_openings import opening_axis, structural_union
from door_leaf_evidence import dark_stroke_support, leaf_and_arc, wall_jamb_support
from graph_topology import unresolved_gap_items
from mask_wall_graph import clear_jamb_span
from reclassify_graph_doors import anchored_leaves


def leaf_context(gray, leaf):
    """Surface ambiguous nearby geometry; never decide the opening type."""
    hinge, tip = np.asarray(leaf['hinge'], float), np.asarray(leaf['tip'], float)
    vector = tip-hinge
    length = float(np.linalg.norm(vector))
    unit = vector/length
    margin = max(12, length*.6)
    lo = np.maximum(0, np.floor(np.minimum(hinge,tip)-margin)).astype(int)
    hi = np.minimum(gray.shape[::-1], np.ceil(np.maximum(hinge,tip)+margin)).astype(int)
    patch = gray[lo[1]:hi[1],lo[0]:hi[0]]
    contours,_ = cv2.findContours((patch < 140).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    local_hinge, local_tip = hinge-lo, tip-lo
    closed=[]
    for contour in contours:
        area = cv2.contourArea(contour)
        if not length**2*.04 <= area <= length**2*4:
            continue
        probes=[local_hinge+(local_tip-local_hinge)*f for f in (.3,.7)]
        distances=[abs(cv2.pointPolygonTest(contour,tuple(map(float,p)),True)) for p in probes]
        if max(distances) <= 5:
            closed.append(dict(area_px2=round(area,2), max_distance_px=round(max(distances),2)))
    segments=cv2.createLineSegmentDetector().detect(patch)[0]
    crosses=[]
    if segments is not None:
        for raw in segments.reshape(-1,4):
            a,b=raw.reshape(2,2).astype(float)
            direction=b-a; size=float(np.linalg.norm(direction))
            if not max(3,length*.12) <= size <= length*.6:
                continue
            if abs(float(direction@unit))/size > .55:
                continue
            line=LineString([a,b])
            distance=line.distance(Point(local_tip))
            if distance <= max(3,length*.15):
                crosses.append(dict(length_px=round(size,2),tip_distance_px=round(distance,2)))
    return dict(near_closed_contours=closed, near_tip_cross_strokes=crosses,
                interpretation='ambiguous_visual_context_requires_review')


def diagnose_graph_door_leaves(graph, source, rgb, *, diagnostic_walls=None):
    meta = source.get('meta', {})
    if (meta.get('width'), meta.get('height')) != (rgb.shape[1], rgb.shape[0]):
        raise ValueError('Source dimensions do not match raster')
    if graph.get('coordinate_space') != 'source_raster' or meta.get('coordinate_space', 'source_raster') != 'source_raster':
        raise ValueError('Source-raster coordinates required')
    walls = structural_union(source.get('walls', [])) if diagnostic_walls is None else diagnostic_walls
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    excluded = unary_union([Polygon(p) for p in graph.get('diagnostics', {}).get('excluded_regions', [])])
    results = []
    candidates = [('opening', i, b) for i, b in enumerate(graph['diagnostics'].get('opening_bridges', []))
                  if b.get('kind') == 'windows']
    diagnostics = graph['diagnostics']
    candidates += [('gap', i, g) for i, g in unresolved_gap_items(
        diagnostics.get('topology_gaps', []), diagnostics.get('opening_bridges', []))]
    for kind, index, candidate in candidates:
        matches = [o for o in source.get('windows', []) if o.get('id') == candidate.get('opening_id')]
        width = candidate.get('local_width_px') if kind == 'gap' else opening_axis(matches[0])[2] if len(matches) == 1 else None
        row = dict(candidate_kind=kind, index=index, opening_id=candidate.get('opening_id'),
                   span=candidate['span'], accepted=False)
        results.append(row)
        if width is None or width <= 0:
            row['reason'] = 'missing_unique_width'; continue
        line = LineString(candidate.get('clear_span', candidate['span']))
        if line.length <= 0 or line.intersects(excluded):
            row['reason'] = 'invalid_or_excluded'; continue
        # Diagnostic only: raster scale is frequently provisional. The shared
        # helper's 60 cm floor must not discard a geometrically valid opening
        # solely because a graph happened to use a default scale of 1.
        diagnostic_scale = max(graph['scale_cm_per_px'], 60 / (line.length * .6))
        clear = clear_jamb_span(line, walls, width, diagnostic_scale)
        if clear is None:
            row['reason'] = 'missing_model_free_aperture'; continue
        support = wall_jamb_support(*line.coords, width, walls)
        if min(support) < 2:
            row['reason'] = 'insufficient_model_jambs'; continue
        leaves = anchored_leaves(gray, *line.coords, width, 1)
        options = leaves[0] + leaves[1]
        arc = None
        if options:
            leaf = min(options, key=lambda item: item['jamb_distance_px'])
            reason = 'model_jambs_and_diagonal_raster_leaf'
        else:
            arc = leaf_and_arc(gray, *line.coords, width, walls)
            if arc is not None and dark_stroke_support(gray, arc['hinge'], arc['tip']) >= .8:
                leaf = dict(hinge=arc['hinge'], tip=arc['tip'], length_px=arc['leaf_length'],
                            angle_degrees=arc['swing_angle_degrees'], jamb_index=arc['jamb_index'])
                reason = 'model_jambs_leaf_and_swing_arc'
            else:
                arc = None
                # Only a review hint: raster line detectors often stop short of
                # the jamb when the leaf joins a thick wall or is fragmented.
                # Do not widen the recognizer's actual classification threshold.
                nearby = anchored_leaves(gray, *line.coords, width, 1, jamb_tolerance_factor=1.15)
                options = nearby[0] + nearby[1]
                if not options:
                    row['reason'] = 'missing_anchored_leaf'; continue
                leaf = min(options, key=lambda item: item['jamb_distance_px'])
                reason = 'model_jambs_and_nearby_diagonal_review_hint'
        row.update(accepted=True, reason=reason,
                   model_jamb_support=support, model_clear_span=list(clear.coords),
                   physical_minimum_not_applied=diagnostic_scale > graph['scale_cm_per_px'],
                   leaf=leaf, context=leaf_context(gray, leaf),
                   arc_evidence=arc,
                   limitations=['furniture_or_dimension_line_can_imitate_leaf',
                                'furniture_or_dimension_arc_can_imitate_swing_arc',
                                'candidate_is_not_a_confirmed_door'],
                   requires_review=True)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('graph', type=Path); p.add_argument('source', type=Path); p.add_argument('image', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    load = lambda path: json.loads(path.read_text(encoding='utf-8'))
    with Image.open(args.image) as image:
        rows = diagnose_graph_door_leaves(load(args.graph), load(args.source), np.asarray(image.convert('RGB')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(dict(candidates=len(rows), flagged=sum(r['accepted'] for r in rows)))


if __name__ == '__main__':
    main()
