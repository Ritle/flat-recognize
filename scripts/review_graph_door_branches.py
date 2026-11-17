"""Apply a visually reviewed door between existing graph nodes.

Short dangling graph spurs attributed to door leaves may be removed only when
listed explicitly. The normal reviewed-door safety checks validate the new
aperture, jambs, collisions and room split. This is not automatic recognition.
"""
import copy
import json
from pathlib import Path
import argparse

import numpy as np
from PIL import Image
from shapely.geometry import LineString

from review_decisions import image_sha256
from review_graph_doors import apply_graph_door_decisions, gap_fingerprint


def apply_branch_door_review(graph, source, rgb, review, digest):
    if review.get('version') != 1 or review.get('image_sha256') != digest:
        raise ValueError('Review image hash or format does not match')
    if review.get('reviewer') != 'assistant_visual_review' or not review.get('basis'):
        raise ValueError('Attributed visual review required')
    span = review.get('span')
    width = review.get('local_width_px')
    if not isinstance(width, (int, float)) or not 1 <= width <= 100:
        raise ValueError('Invalid local wall width')
    gap = dict(span=span, local_width_px=width)
    fingerprint = gap_fingerprint(gap)
    branches = review.get('remove_dangling_branches', [])
    if not isinstance(branches, list) or len(branches) > 2:
        raise ValueError('At most two reviewed branches may be removed')
    for index, bridge in enumerate(graph['diagnostics']['opening_bridges']):
        if bridge.get('source') == 'assistant_visual_review' and bridge.get('review', {}).get('gap_fingerprint') == fingerprint:
            return copy.deepcopy(graph)
        if (bridge.get('source') == 'paired_raster_leaf_branches'
                and bridge['evidence'].get('gap_fingerprint') == fingerprint
                and sorted(map(tuple, bridge['span'])) == sorted(map(tuple, span))
                and sorted(sorted(b) for b in branches) == sorted(sorted(b) for b in bridge['evidence']['removed_dangling_branches'])):
            result = copy.deepcopy(graph)
            result['diagnostics']['opening_bridges'][index]['review'] = dict(
                reviewer=review['reviewer'], basis=review['basis'], image_sha256=digest,
                gap_fingerprint=fingerprint, choice='door', removed_dangling_branches=copy.deepcopy(branches))
            return result
    result = remove_jamb_branches(graph, span, width, branches)
    result['diagnostics'].setdefault('topology_gaps', []).append(gap)
    decisions = dict(version=1,image_sha256=digest,answers=[dict(gap_fingerprint=fingerprint,choice='door')])
    result = apply_graph_door_decisions(result,source,rgb,decisions,digest)
    bridge = result['diagnostics']['opening_bridges'][-1]
    bridge['source'] = 'assistant_visual_review'
    bridge['review'].update(reviewer='assistant_visual_review',basis=review['basis'],
                            removed_dangling_branches=copy.deepcopy(branches))
    bridge['evidence'].pop('type_confirmed_by_user',None)
    if review.get('visible_leaf_count') == 2:
        bridge['type_review'] = dict(evidence=dict(leaf_count=2,source='assistant_visual_review'))
    return result


def remove_jamb_branches(graph, span, width, branches):
    result = copy.deepcopy(graph)
    nodes = {n['id']: n for n in result['nodes']}
    endpoints = []
    for point in span:
        found = [n['id'] for n in result['nodes'] if np.linalg.norm(np.asarray(n['point'])-point) < 1e-5]
        if len(found) != 1:
            raise ValueError('Door endpoints must be unique existing graph nodes')
        endpoints.append(found[0])
    if len(endpoints) != 2 or endpoints[0] == endpoints[1]:
        raise ValueError('Expected two distinct endpoints')
    if not isinstance(branches, list) or len(branches) > 2:
        raise ValueError('At most two reviewed branches may be removed')
    for branch in branches:
        if not isinstance(branch, list) or len(branch) != 2:
            raise ValueError('Branch needs two endpoint coordinates')
        ids = []
        for point in branch:
            found = [n['id'] for n in result['nodes'] if np.linalg.norm(np.asarray(n['point'])-point) < 1e-5]
            if len(found) != 1:
                raise ValueError('Reviewed branch endpoint missing')
            ids.append(found[0])
        junction = next((i for i in ids if i in endpoints), None)
        terminal = next((i for i in ids if i != junction), None)
        if junction is None or terminal is None or terminal in endpoints:
            raise ValueError('Branch must start at a door jamb')
        matches = [e for e in result['edges'] if {e['start'],e['end']} == set(ids)]
        incident = [e for e in result['edges'] if terminal in (e['start'],e['end'])]
        if len(matches) != 1 or len(incident) != 1:
            raise ValueError('Branch must be unique and dangling')
        if LineString(branch).length > width*1.5:
            raise ValueError('Branch exceeds local jamb length')
        # Do not remove an edge that is itself part of another opening.
        branch_line = LineString(branch)
        for other in result['diagnostics'].get('opening_bridges', []):
            if branch_line.intersection(LineString(other.get('clear_span',other['span'])).buffer(1)).length > 1:
                raise ValueError('Branch overlaps an existing opening')
        result['edges'].remove(matches[0])
        nodes[junction]['adjacency'].remove(terminal)
        result['nodes'] = [n for n in result['nodes'] if n['id'] != terminal]
        nodes.pop(terminal)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('graph',type=Path);p.add_argument('source',type=Path);p.add_argument('image',type=Path)
    p.add_argument('--decision',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();load=lambda x:json.loads(x.read_text(encoding='utf-8'))
    with Image.open(a.image) as im:
        result=apply_branch_door_review(load(a.graph),load(a.source),np.asarray(im.convert('RGB')),load(a.decision),image_sha256(a.image))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Visual door review applied')


if __name__=='__main__':main()
