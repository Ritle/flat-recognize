"""Remove image-reviewed dangling furniture strokes from a wall graph.

Candidates are selected externally; image hash, topology, room boundaries and
opening carriers are checked. This is not an automatic furniture detector.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point
from shapely.ops import polygonize, unary_union

from review_decisions import image_sha256


def check_reviewed_path(choice, removed_edges):
    if 'expected_path' not in choice:
        return
    actual = [choice['terminal_point']]
    for edge in removed_edges:
        a, b = edge['start'], edge['end']
        if actual[-1] not in (a, b):
            raise ValueError('Disconnected reviewed branch')
        actual.append(b if a == actual[-1] else a)
    if actual != choice['expected_path']:
        raise ValueError('Reviewed branch path has changed')


def duplicates_reviewed_window_jamb(choice, path, line, bridge, incident_edges, nodes):
    if (choice.get('coincident_window_jamb') is not True or not choice.get('expected_path')
            or len(path) != 1 or bridge.get('kind') not in ('windows', 'recovered_window')):
        return False
    carrier = LineString(bridge['span'])
    if not carrier.buffer(.5).covers(line):
        return False
    # The original window must have its own surviving edge at this junction.
    for edge in incident_edges:
        if edge is path[0]:
            continue
        axis = LineString([nodes[edge[key]]['point'] for key in ('start', 'end')])
        if axis.hausdorff_distance(carrier) < .001:
            return True
    return False


def apply_furniture_review(graph, source, rgb, review, digest):
    if not isinstance(review, dict) or review.get('version')!=1 or review.get('image_sha256')!=digest:
        raise ValueError('Furniture review image hash does not match')
    if graph.get('coordinate_space')!='source_raster' or source.get('meta',{}).get('coordinate_space','source_raster')!='source_raster':
        raise ValueError('Source-raster coordinates required')
    if (source.get('meta',{}).get('width'),source.get('meta',{}).get('height'))!=(rgb.shape[1],rgb.shape[0]):
        raise ValueError('Source dimensions do not match image')
    choices=review.get('candidates')
    if not isinstance(choices,list) or len(choices)>50:
        raise ValueError('Invalid furniture review')
    result=copy.deepcopy(graph)
    if not choices:
        return result
    records=result['diagnostics'].setdefault('reviewed_furniture_removals',[])
    already={r['fingerprint']:r for r in records}
    nodes={n['id']:n for n in result['nodes']}
    def lines():return [LineString([nodes[e[k]]['point'] for k in ('start','end')]) for e in result['edges']]
    before=unary_union(list(polygonize(lines())))
    seen=set()
    for choice in choices:
        if not isinstance(choice,dict) or choice.get('reviewer')!='assistant_visual_review' or not choice.get('basis'):
            raise ValueError('Explicit attributed visual review required')
        point=choice.get('terminal_point')
        if not isinstance(point,list) or len(point)!=2 or any(not isinstance(v,(int,float)) or not np.isfinite(v) for v in point):
            raise ValueError('Invalid terminal point')
        fingerprint=hashlib.sha256(json.dumps(point,separators=(',',':')).encode()).hexdigest()[:20]
        if fingerprint in seen:raise ValueError('Duplicate reviewed terminal')
        seen.add(fingerprint)
        if fingerprint in already:
            check_reviewed_path(choice, already[fingerprint]['removed_edges'])
            continue
        matches=[n['id'] for n in result['nodes'] if Point(n['point']).distance(Point(point))<1e-5]
        if len(matches)!=1:raise ValueError('Reviewed terminal missing or ambiguous')
        terminal=matches[0];incident={uid:[] for uid in nodes}
        for edge in result['edges']:
            incident[edge['start']].append(edge);incident[edge['end']].append(edge)
        if len(incident[terminal])!=1:raise ValueError('Reviewed node must be dangling')
        path=[];current=terminal;previous=None;visited={terminal}
        while True:
            options=[e for e in incident[current] if e is not previous]
            if len(options)!=1:break
            edge=options[0];nxt=edge['end'] if edge['start']==current else edge['start']
            if nxt in visited:raise ValueError('Furniture path cannot form a cycle')
            path.append(edge);visited.add(nxt);previous=edge;current=nxt
            if len(incident[current])!=2:break
        chain=[LineString([nodes[e[k]]['point'] for k in ('start','end')]) for e in path]
        length=sum(line.length for line in chain)
        if not path or length>40:raise ValueError('Reviewed branch is missing or too long')
        for line in chain:
            for bridge in result['diagnostics'].get('opening_bridges',[]):
                if line.intersection(LineString(bridge.get('clear_span',bridge['span'])).buffer(1,cap_style=2)).length > .1:
                    if duplicates_reviewed_window_jamb(choice, path, line, bridge, incident[current], nodes):
                        continue
                    raise ValueError('Furniture branch overlaps an opening')
        removed_edges = [dict(start=nodes[e['start']]['point'], end=nodes[e['end']]['point'],
                              thickness_px=e['thickness_px']) for e in path]
        check_reviewed_path(choice, removed_edges)
        records.append(dict(fingerprint=fingerprint,terminal_point=point,reviewer=choice['reviewer'],
                            basis=choice['basis'],image_sha256=digest,length_px=length,
                            removed_edges=removed_edges))
        for edge in path:result['edges'].remove(edge)
        used={e[k] for e in result['edges'] for k in ('start','end')}
        result['nodes']=[n for n in result['nodes'] if n['id'] in used]
        nodes={n['id']:n for n in result['nodes']}
    for node in result['nodes']:node['adjacency']=[]
    for edge in result['edges']:
        nodes[edge['start']]['adjacency'].append(edge['end'])
        nodes[edge['end']]['adjacency'].append(edge['start'])
    after=unary_union(list(polygonize(lines())))
    if before.symmetric_difference(after).area>.01:
        raise ValueError('Removing furniture changed physical room faces')
    network=lines()+[LineString(g['span']) for g in result['diagnostics'].get('topology_gaps',[])]
    result['faces']=[dict(polygon=list(p.exterior.coords)[:-1],area_px2=p.area,requires_review=True)
                     for p in polygonize(unary_union(network)) if p.area*graph['scale_cm_per_px']**2>=5000]
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('graph',type=Path);p.add_argument('source',type=Path);p.add_argument('image',type=Path)
    p.add_argument('--decisions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    with Image.open(a.image) as im:
        result=apply_furniture_review(load(a.graph),load(a.source),np.asarray(im.convert('RGB')),load(a.decisions),image_sha256(a.image))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(dict(reviewed_furniture_branches=len(result['diagnostics']['reviewed_furniture_removals'])))


if __name__=='__main__':main()
