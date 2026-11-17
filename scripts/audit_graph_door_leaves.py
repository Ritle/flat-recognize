"""Run leaf diagnostics on already recorded graph doors for recall analysis.

These are prior predictions, not ground truth. Reusing them as diagnostic
candidates does not change their classification or measure precision.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import cv2
from PIL import Image, ImageDraw
from shapely.geometry import LineString

from classify_openings import structural_union
from diagnose_graph_door_leaves import diagnose_graph_door_leaves
from opening_graph_cleanup import ink_radii


def audit_existing_graph_doors(graph, source, rgb):
    result = copy.deepcopy(graph)
    nodes = {node['id']: node['point'] for node in graph['nodes']}
    edges = [(LineString([nodes[e[k]] for k in ('start', 'end')]), e['thickness_px']) for e in graph['edges']]
    if not edges:
        raise ValueError('Existing-door audit needs graph edges')
    doors = [(i, b) for i, b in enumerate(graph['diagnostics'].get('opening_bridges', []))
             if b.get('kind') in ('doors', 'recovered_door')]
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    distance = cv2.distanceTransform((rgb.max(axis=2) < 160).astype(np.uint8), cv2.DIST_L2, 5)
    walls = structural_union(source.get('walls', []))
    rows = []
    for index, bridge in doors:
        line = LineString(bridge['span'])
        center = line.interpolate(.5, normalized=True)
        width = min(edges, key=lambda edge: edge[0].distance(center))[1]
        result['diagnostics']['opening_bridges'] = []
        result['diagnostics']['topology_gaps'] = [dict(span=bridge['span'], local_width_px=width)]
        # Use a local cut only for an already raster-verified cleanup, and
        # independently recheck current pixels before trusting its provenance.
        candidate_walls = walls
        evidence = 'original_model_walls'
        if bridge.get('carrier_cleanup') == 'raster_confirmed_blank_aperture' and 'clear_span' in bridge:
            clear = LineString(bridge['clear_span'])
            radii = ink_radii(clear, distance) if clear.length > 0 else None
            jambs = []
            if clear.length > 0:
                a, b = np.asarray(clear.coords[0]), np.asarray(clear.coords[-1])
                axis = (b-a)/clear.length
                for point, direction in ((a, -axis), (b, axis)):
                    probe = LineString([point+direction*width*.25, point+direction*width*.75])
                    samples = ink_radii(probe, distance)
                    jambs.append(samples is not None and np.mean(samples >= 1.8) >= .8)
            if (clear.length > 0 and clear.difference(line.buffer(1)).length < .01
                    and radii is not None and np.mean(radii >= 1.8) <= .15 and all(jambs)):
                candidate_walls = walls.difference(clear.buffer(width*.75, cap_style=2))
                evidence = 'raster_rechecked_previously_cleaned_carrier'
        row = diagnose_graph_door_leaves(result, source, rgb, diagnostic_walls=candidate_walls)[0]
        row.update(original_bridge_index=index, original_kind=bridge['kind'])
        row['wall_evidence'] = evidence
        if evidence == 'raster_rechecked_previously_cleaned_carrier':
            row['raster_verified_clear_span'] = copy.deepcopy(bridge['clear_span'])
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path); parser.add_argument('source', type=Path)
    parser.add_argument('image', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    with Image.open(args.image) as image:
        rgb = np.asarray(image.convert('RGB'))
    rows = audit_existing_graph_doors(load(args.graph), load(args.source), rgb)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    overlay = Image.fromarray(rgb)
    draw = ImageDraw.Draw(overlay)
    for row in rows:
        if row['accepted']:
            leaf = row['leaf']
            draw.line([tuple(leaf['hinge']), tuple(leaf['tip'])],
                      fill='red' if row['context']['near_tip_cross_strokes'] else 'lime', width=2)
    overlay.save(args.output.with_suffix('.png'))
    print(dict(existing_door_candidates=len(rows), detected=sum(r['accepted'] for r in rows),
               note='Existing predictions are not ground truth'))


if __name__ == '__main__':
    main()
