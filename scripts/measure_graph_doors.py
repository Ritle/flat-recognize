"""Expand truncated, reclassified door apertures on existing wall paths only."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from classify_openings import opening_axis
from graph_export_geometry import regularize_opening_carriers
from reclassify_graph_doors import door_evidence


def measure_graph_doors(graph, source, rgb):
    height, width = rgb.shape[:2]
    meta = source.get('meta', {})
    if (meta.get('width'), meta.get('height')) != (width, height):
        raise ValueError('Source dimensions do not match raster')
    if graph.get('coordinate_space') != 'source_raster' or meta.get('coordinate_space', 'source_raster') != 'source_raster':
        raise ValueError('Source-raster coordinates required')
    result = copy.deepcopy(graph)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    distance = cv2.distanceTransform((rgb.max(axis=2) < 160).astype(np.uint8), cv2.DIST_L2, 5)
    excluded = unary_union([Polygon(p) for p in graph['diagnostics'].get('excluded_regions', [])])
    reports = []
    for index, original in enumerate(graph['diagnostics']['opening_bridges']):
        review = original.get('type_review', {})
        if (original.get('kind') != 'doors' or review.get('source') != 'raster_leaf_reclassification'
                or original.get('aperture_measurement')):
            continue
        record = dict(bridge_index=index, accepted=False)
        reports.append(record)
        matches = [o for o in source.get('windows', []) if o.get('id') == original.get('opening_id')]
        if len(matches) != 1 or matches[0].get('source_class') != 'door':
            record['reason'] = 'requires_unique_model_door'; continue
        old_span = original.get('clear_span', original['span'])
        evidence = door_evidence(gray, distance, old_span, opening_axis(matches[0])[2])
        if evidence is None:
            record['reason'] = 'missing_current_raster_evidence'; continue
        aperture = LineString(evidence['measured_jamb_span'])
        if aperture.length <= LineString(old_span).length + 1:
            record['reason'] = 'no_significant_truncation'; continue
        if aperture.intersects(excluded):
            record['reason'] = 'excluded_region'; continue
        if any(aperture.intersection(LineString(b.get('clear_span', b['span'])).buffer(1, cap_style=2)).length > .1
               for j, b in enumerate(result['diagnostics']['opening_bridges']) if j != index):
            record['reason'] = 'overlaps_existing_opening'; continue
        trial = copy.deepcopy(result)
        bridge = trial['diagnostics']['opening_bridges'][index]
        bridge.update(span=evidence['measured_jamb_span'], clear_span=evidence['measured_jamb_span'])
        trial, changes = regularize_opening_carriers(trial, [index])
        change = changes[0]
        record['carrier'] = change
        if not change['accepted']:
            record['reason'] = change['reason']; continue
        bridge = trial['diagnostics']['opening_bridges'][index]
        bridge['span'] = change['span']
        bridge['aperture_measurement'] = dict(source='raster_jambs_on_existing_path',
                                             previous_span=old_span, evidence=evidence)
        bridge['type_review']['evidence']['exported_aperture_unchanged'] = False
        # The path helper preserves junctions and crossings; check bounded faces too.
        def network(g):
            nodes = {n['id']: n['point'] for n in g['nodes']}
            return [LineString([nodes[e[k]] for k in ('start', 'end')]) for e in g['edges']]
        before, after = [list(polygonize(network(g))) for g in (result, trial)]
        if len(before) != len(after) or any(not any(p.intersection(q).area/p.union(q).area >= .995 for q in after) for p in before):
            record['reason'] = 'bounded_faces_changed'; continue
        lines = network(trial) + [LineString(g['span']) for g in trial['diagnostics'].get('topology_gaps', [])]
        trial['faces'] = [dict(polygon=list(p.exterior.coords)[:-1], area_px2=p.area, requires_review=True)
                          for p in polygonize(unary_union(lines)) if p.area*graph['scale_cm_per_px']**2 >= 5000]
        result = trial
        record.update(accepted=True, old_width_px=LineString(old_span).length, measured_width_px=aperture.length)
    return result, reports


def main():
    from PIL import Image
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    load = lambda p: json.loads(p.read_text(encoding='utf-8'))
    graph, report = measure_graph_doors(load(args.graph), load(args.source), np.asarray(Image.open(args.image).convert('RGB')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(graph, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    args.output.with_suffix('.report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(dict(measured=sum(r['accepted'] for r in report)))


if __name__ == '__main__':
    main()
