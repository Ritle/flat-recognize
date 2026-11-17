"""Apply recorded visual reviews of existing internal openings as open passages.

The choice is external review data, not inferred from being inside the house.
Geometry and original provenance remain intact; contradictory raster evidence
rejects a choice instead of silently deleting a window or adding a solid wall.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from classify_openings import opening_axis, structural_union
from door_leaf_evidence import dark_leaf_with_jambs, leaf_and_arc
from opening_graph_cleanup import ink_radii
from review_decisions import image_sha256, MAX_ANSWERS
from window_exterior_context import window_exterior_context
from window_frame_evidence import narrow_frame_rails


def passage_fingerprint(bridge):
    data = dict(opening_id=bridge.get('opening_id'), span=sorted(bridge['span']),
                clear_span=sorted(bridge.get('clear_span', bridge['span'])))
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:20]


def apply_passage_reviews(graph, source, rgb, review, source_sha256):
    if review.get('version') != 1 or review.get('image_sha256') != source_sha256:
        raise ValueError('Passage review format or image hash does not match')
    answers = review.get('answers')
    if not isinstance(answers, list) or len(answers) > MAX_ANSWERS:
        raise ValueError('Invalid passage review answers')
    if (source.get('meta', {}).get('width'), source.get('meta', {}).get('height')) != rgb.shape[1::-1]:
        raise ValueError('Source dimensions do not match raster')
    if graph.get('coordinate_space') != 'source_raster' or source.get('meta', {}).get('coordinate_space', 'source_raster') != 'source_raster':
        raise ValueError('Source-raster coordinates required')
    result = copy.deepcopy(graph)
    nodes = {n['id']: n['point'] for n in graph['nodes']}
    lines = [LineString([nodes[e[k]] for k in ('start', 'end')]) for e in graph['edges']]
    rooms = [dict(polygon=list(p.exterior.coords)) for p in polygonize(lines)
             if not p.interiors and p.area*graph['scale_cm_per_px']**2 >= 5000]
    excluded = unary_union([Polygon(p) for p in graph['diagnostics'].get('excluded_regions', [])])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    distance = cv2.distanceTransform((rgb.max(axis=2) < 160).astype(np.uint8), cv2.DIST_L2, 5)
    walls = structural_union(source.get('walls', []))
    seen = set()
    for answer in answers:
        fingerprint = answer.get('fingerprint') if isinstance(answer, dict) else None
        if (not fingerprint or fingerprint in seen or answer.get('choice') != 'open_passage'
                or answer.get('reviewer') not in ('assistant_visual_review', 'user') or not answer.get('basis')):
            raise ValueError('Unique passage choice, reviewer and review basis required')
        seen.add(fingerprint)
        matches = [b for b in result['diagnostics']['opening_bridges'] if passage_fingerprint(b) == fingerprint]
        if len(matches) != 1:
            raise ValueError('Reviewed opening is missing or its geometry changed')
        bridge = matches[0]
        if bridge.get('kind') == 'passages' and bridge.get('passage_review', {}).get('image_sha256') == source_sha256:
            continue
        if bridge.get('kind') != 'windows':
            raise ValueError('Only an existing window candidate can use this review mode')
        candidates = [o for o in source.get('windows', []) if o.get('id') == bridge.get('opening_id')]
        if len(candidates) != 1:
            raise ValueError('Reviewed opening has no unique source candidate')
        width = opening_axis(candidates[0])[2]
        span = bridge.get('clear_span', bridge['span'])
        if window_exterior_context(span, width, rooms)['status'] != 'interior':
            raise ValueError('Passage review requires an internal opening')
        if LineString(span).intersects(excluded):
            raise ValueError('Reviewed passage intersects excluded geometry')
        radii = ink_radii(LineString(span), distance)
        if radii is None or np.mean(radii >= 1.8) > .15:
            raise ValueError('Thick raster structure occupies the passage')
        if narrow_frame_rails(gray, *span, width):
            raise ValueError('Continuous window frame contradicts passage review')
        if leaf_and_arc(gray, *span, width, walls) or dark_leaf_with_jambs(gray, *span, width, walls):
            raise ValueError('Visible door leaf contradicts passage review')
        bridge.update(kind='passages', requires_review=True,
                      passage_review=dict(answer, image_sha256=source_sha256, previous_kind='windows',
                                          height_source='serialization_template_not_raster'))
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
    graph = apply_passage_reviews(load(args.graph), load(args.source), np.asarray(Image.open(args.image).convert('RGB')),
                                 load(args.decisions), image_sha256(args.image))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(dict(passages=sum(b.get('kind') == 'passages' for b in graph['diagnostics']['opening_bridges'])))


if __name__ == '__main__':
    main()
