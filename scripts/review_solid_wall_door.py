"""Apply an image-bound, reviewed solid partition with a door.

The endpoints must be on existing graph edges; no coordinates are inferred
from ground truth. The new boundary must split exactly one physical area.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import polygonize, unary_union

from review_decisions import image_sha256


def partition_fingerprint(review):
    geometry = dict(partition=review['partition_span'],door=review['door_span'],width=review['thickness_px'])
    return hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]


def apply_reviewed_partition(graph, source, rgb, review, digest):
    if review.get('version') != 1 or review.get('image_sha256') != digest:
        raise ValueError('Partition review does not match image')
    if review.get('wall_choice') != 'solid' or not review.get('user_statement') or not review.get('visual_basis'):
        raise ValueError('Explicit solid-wall choice and door review basis required')
    if graph.get('coordinate_space') != 'source_raster' or source.get('meta', {}).get('coordinate_space','source_raster') != 'source_raster':
        raise ValueError('Source-raster coordinates required')
    if (source.get('meta',{}).get('width'),source.get('meta',{}).get('height')) != (rgb.shape[1],rgb.shape[0]):
        raise ValueError('Source dimensions do not match image')
    metadata = dict(image_sha256=digest,wall_choice='solid',user_statement=review['user_statement'],
                    door_basis=review['visual_basis'],door_reviewer='assistant_visual_review',
                    width_source='approximate_visual',hinge_and_swing='template_defaults')
    return insert_partition(graph, review, metadata)


def insert_partition(graph, review, metadata):
    """Insert a wall and its measured opening only if they safely divide an existing area."""
    width=review.get('thickness_px')
    if not isinstance(width,(int,float)) or not np.isfinite(width) or not 1 <= width <= 100:
        raise ValueError('Invalid wall thickness')
    partition=LineString(review['partition_span'])
    door=LineString(review['door_span'])
    if partition.length <= 0 or door.length < 8 or door.difference(partition.buffer(.25)).length > .01:
        raise ValueError('Door must lie within the partition')
    projected=[partition.project(Point(p)) for p in door.coords]
    if min(projected) < width or max(projected) > partition.length-width:
        raise ValueError('Door must leave wall at both ends')
    key=hashlib.sha256(json.dumps(dict(partition=review['partition_span'],door=review['door_span']),
                                  sort_keys=True,separators=(',',':')).encode()).hexdigest()[:20]
    result=copy.deepcopy(graph)
    if any(b.get('review',{}).get('partition_fingerprint')==key for b in result['diagnostics'].get('opening_bridges',[])):
        return result
    nodes={n['id']:n for n in result['nodes']}
    def edge_line(edge):return LineString([nodes[edge[k]]['point'] for k in ('start','end')])
    original_lines=[edge_line(e) for e in result['edges']]
    excluded=unary_union([Polygon(p) for p in result['diagnostics'].get('excluded_regions',[])])
    if partition.intersects(excluded):raise ValueError('Partition intersects excluded region')
    if any(partition.intersection(LineString(b.get('clear_span',b['span'])).buffer(1,cap_style=2)).length > .1
           for b in result['diagnostics'].get('opening_bridges',[])):
        raise ValueError('Partition overlaps another opening')
    interior=LineString([partition.interpolate(.001),partition.interpolate(partition.length-.001)])
    if any(interior.intersects(LineString(g['span'])) for g in result['diagnostics'].get('topology_gaps',[])):
        raise ValueError('Partition intersects an unresolved topology gap')
    if interior.intersects(unary_union(original_lines)):
        raise ValueError('Partition crosses existing graph')
    endpoints=[]
    for point in partition.coords:
        pt=Point(point)
        hits=[(edge_line(e).distance(pt),i) for i,e in enumerate(result['edges']) if edge_line(e).distance(pt)<1e-5]
        if not hits:raise ValueError('Partition endpoint must lie on an existing wall axis')
        existing=[n['id'] for n in result['nodes'] if Point(n['point']).distance(pt)<1e-5]
        if len(existing)>1:raise ValueError('Ambiguous partition endpoint')
        if existing:
            uid=existing[0]
        else:
            if len(hits)!=1:raise ValueError('Ambiguous edge split')
            i=hits[0][1];edge=result['edges'][i];uid='reviewed_partition_'+key+'_'+str(len(endpoints))
            if uid in nodes:raise ValueError('Partition node id conflict')
            node=dict(id=uid,point=list(point),adjacency=[]);result['nodes'].append(node);nodes[uid]=node
            result['edges'][i:i+1]=[dict(edge,end=uid),dict(edge,start=uid)]
        endpoints.append(uid)
    if endpoints[0]==endpoints[1]:raise ValueError('Partition endpoints must differ')
    result['edges'].append(dict(start=endpoints[0],end=endpoints[1],thickness_px=float(width)))
    for node in result['nodes']:node['adjacency']=[]
    for edge in result['edges']:
        nodes[edge['start']]['adjacency'].append(edge['end'])
        nodes[edge['end']]['adjacency'].append(edge['start'])
    after_lines=[edge_line(e) for e in result['edges']]
    scale=graph['scale_cm_per_px']
    before=[p for p in polygonize(original_lines) if p.area*scale**2>=5000]
    after=[p for p in polygonize(after_lines) if p.area*scale**2>=5000]
    if len(after)!=len(before)+1 or abs(sum(p.area for p in before)-sum(p.area for p in after))>.01:
        raise ValueError('Partition must safely split one existing bounded area')
    result['diagnostics']['opening_bridges'].append(dict(
        kind='doors',opening_id='reviewed-partition-door-'+key,source='reviewed_solid_partition',
        span=list(partition.coords),clear_span=list(door.coords),requires_review=True,
        review=dict(metadata,partition_fingerprint=key)))
    network=after_lines+[LineString(g['span']) for g in result['diagnostics'].get('topology_gaps',[])]
    result['faces']=[dict(polygon=list(p.exterior.coords)[:-1],area_px2=p.area,requires_review=True)
                     for p in polygonize(unary_union(network)) if p.area*scale**2>=5000]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph',type=Path);parser.add_argument('source',type=Path);parser.add_argument('image',type=Path)
    parser.add_argument('--decision',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    with Image.open(args.image) as im:
        result=apply_reviewed_partition(load(args.graph),load(args.source),np.asarray(im.convert('RGB')),load(args.decision),image_sha256(args.image))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Reviewed solid partition and door applied')


if __name__=='__main__':main()
