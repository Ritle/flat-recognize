"""Opt-in raster leaf checks for model doors currently exported as windows.

Only opening types change. Measured jamb spans are evidence, not a replacement
for graph geometry; width/hinge/swing remain reviewable in the export.
"""
import argparse
import copy
import json
import math
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from classify_openings import opening_axis, structural_union
from door_leaf_evidence import dark_stroke_support, wall_jamb_support
from window_exterior_context import window_exterior_context


def raster_aperture(distance, start, end, width):
    a, b = np.asarray(start, float), np.asarray(end, float)
    length = float(np.linalg.norm(b-a))
    if length < 10 or width < 3:
        return None
    axis = (b-a)/length
    ts = np.arange(-width, length+width, .5)
    xy = np.rint(a+ts[:, None]*axis).astype(int)
    if (xy[:, 0].min() < 0 or xy[:, 1].min() < 0 or
            xy[:, 0].max() >= distance.shape[1] or xy[:, 1].max() >= distance.shape[0]):
        return None
    radii = distance[xy[:, 1], xy[:, 0]]
    middle = int(np.argmin(abs(ts-length/2)))
    if radii[middle] >= 1.8:
        return None
    lo = hi = middle
    while lo > 0 and radii[lo-1] < 1.8:
        lo -= 1
    while hi+1 < len(ts) and radii[hi+1] < 1.8:
        hi += 1
    if lo == 0 or hi == len(ts)-1:
        return None
    for mask in ((ts >= ts[lo]-width*.5) & (ts < ts[lo]),
                 (ts > ts[hi]) & (ts <= ts[hi]+width*.5)):
        if mask.sum() < 3 or float(np.mean(radii[mask] >= 1.8)) < .8:
            return None
    if not .75*length <= ts[hi]-ts[lo] <= 1.5*length:
        return None
    return [a+ts[lo]*axis, a+ts[hi]*axis]


def anchored_leaves(gray, start, end, width, leaf_fraction, *, jamb_tolerance_factor=.75):
    """Short dark strokes anchored at real jambs, without line extrapolation."""
    a, b = np.asarray(start), np.asarray(end)
    length = float(np.linalg.norm(b-a)); axis = (b-a)/length
    normal = np.array([-axis[1], axis[0]])
    target = length*leaf_fraction
    margin = target+width
    lo = np.maximum(0, np.floor(np.minimum(a,b)-margin)).astype(int)
    hi = np.minimum(gray.shape[::-1], np.ceil(np.maximum(a,b)+margin)).astype(int)
    lines = cv2.createLineSegmentDetector().detect(gray[lo[1]:hi[1],lo[0]:hi[0]])[0]
    result = [[], []]
    if lines is None:
        return result
    for raw in lines.reshape(-1,4):
        p,q=raw.reshape(2,2).astype(float)+lo
        for hinge,tip in ((p,q),(q,p)):
            vector=tip-hinge; radius=float(np.linalg.norm(vector))
            if not .65*target <= radius <= 1.3*target:
                continue
            for index,(jamb,direction) in enumerate(((a,axis),(b,-axis))):
                if np.linalg.norm(hinge-jamb) > max(3,width*jamb_tolerance_factor):
                    continue
                angle=math.degrees(math.acos(float(np.clip(vector@direction/radius,-1,1))))
                if not 25 <= angle <= 70:
                    continue
                if dark_stroke_support(gray,hinge,tip) < .9:
                    continue
                # Reject broad wall/furniture strokes: both sides must be clear.
                n=np.array([-vector[1],vector[0]])/radius
                probes=np.array([hinge+vector*t+n*s*max(4,width*.6)
                                 for t in (.3,.55,.8) for s in (-1,1)])
                xy=np.rint(probes).astype(int)
                if (xy[:,0].min()<0 or xy[:,1].min()<0 or xy[:,0].max()>=gray.shape[1]
                        or xy[:,1].max()>=gray.shape[0] or np.mean(gray[xy[:,1],xy[:,0]]>170)<.83):
                    continue
                result[index].append(dict(hinge=hinge.tolist(),tip=tip.tolist(),length_px=radius,
                                          side=float(vector@normal),angle_degrees=angle,
                                          jamb_distance_px=float(np.linalg.norm(hinge-jamb))))
    return result


def door_evidence(gray, distance, span, width, *, paired_leaf_filter=None):
    measured=raster_aperture(distance,*span,width)
    if measured is None:
        return None
    halves=anchored_leaves(gray,*measured,width,.5)
    if paired_leaf_filter is not None:
        halves = [[leaf for leaf in leaves if paired_leaf_filter(index, leaf)]
                  for index, leaves in enumerate(halves)]
    pairs=[(a,b) for a in halves[0] for b in halves[1]
           if a['side']*b['side']>0 and min(a['length_px'],b['length_px'])/max(a['length_px'],b['length_px'])>=.7
           and abs(a['angle_degrees']-b['angle_degrees'])<=20]
    if pairs:
        leaves=min(pairs,key=lambda pair:sum(v['jamb_distance_px'] for v in pair))
        kind='paired_inward_dark_leaves'
    elif paired_leaf_filter is not None:
        return None
    else:
        singles=anchored_leaves(gray,*measured,width,1)
        candidates=singles[0]+singles[1]
        if not candidates:
            return None
        leaves=[min(candidates,key=lambda leaf:leaf['jamb_distance_px'])]
        kind='jamb_anchored_dark_leaf'
    return dict(method=kind,leaves=leaves,measured_jamb_span=[p.tolist() for p in measured],
                leaf_count=len(leaves),hinge_and_swing='unconfirmed_template_defaults',
                exported_aperture_unchanged=True, model_leaf_count_requires_review=len(leaves)>1)


def connected_leaf_ink(gray, jamb, tip):
    """Require one thin corridor of connected dark pixels from jamb to tip."""
    ends = np.rint([jamb, tip]).astype(int)
    lo = np.maximum(0, ends.min(axis=0) - 3)
    hi = np.minimum(gray.shape[::-1], ends.max(axis=0) + 4)
    patch = gray[lo[1]:hi[1], lo[0]:hi[0]]
    a, b = ends - lo
    corridor = np.zeros(patch.shape, np.uint8)
    cv2.line(corridor, a, b, 1, 5)
    _, components = cv2.connectedComponents(corridor * (patch < 140))
    def labels(point):
        x, y = point
        return set(components[max(0,y-2):y+3, max(0,x-2):x+3].ravel()) - {0}
    return bool(labels(a) & labels(b))


def free_leaf_tip(gray, hinge, tip, width):
    """Look for light space ahead and on both sides of the end of the stroke."""
    direction = np.asarray(tip) - hinge
    tip_radius = max(4, width*.7, np.linalg.norm(direction)*.15)
    angles = np.linspace(-np.pi/2, np.pi/2, math.ceil(np.pi*tip_radius)+1)
    angles += np.arctan2(direction[1], direction[0])
    probes = tip + tip_radius * np.column_stack((np.cos(angles), np.sin(angles)))
    x, y = np.rint(probes).astype(int).T
    return (x.min() >= 0 and x.max() < gray.shape[1] and y.min() >= 0 and y.max() < gray.shape[0]
            and np.all(gray[y, x] >= 160))


def jamb_face_leaves(gray, measured, normal, width):
    """Yield free-ended strokes attached to the corresponding face of a measured jamb."""
    for sign in (-1, 1):
        corners = measured + sign*normal*width/2
        for jamb, leaves in zip(corners, anchored_leaves(gray, *corners, width, 1)):
            for leaf in leaves:
                tip = np.asarray(leaf['tip'])
                if (sign*leaf['side'] > 0 and connected_leaf_ink(gray, jamb, tip)
                        and free_leaf_tip(gray, jamb, tip, width)):
                    yield jamb, leaf


def outward_leaf_evidence(gray, distance, span, width, rooms, walls):
    """Look for a leaf fixed to the exterior face of a measured jamb."""
    context = window_exterior_context(span, width, rooms)
    if context['status'] != 'exterior' or min(wall_jamb_support(*span, width, walls)) < 2:
        return None
    span = np.asarray(span, float)
    axis = span[1] - span[0]
    normal = np.array([-axis[1], axis[0]]) / np.linalg.norm(axis)
    sign = 1 if context['sides'][1] == 'outside' else -1
    # A frame rail can obscure the axis. Look for the same two thick jambs
    # on independent parallel scanlines and project them back to the axis.
    radius = int(width / 2)
    measurements = []
    for offset in np.unique(np.linspace(-radius, radius, min(9, 2*radius+1)).round()):
        shift = normal * offset
        measured = raster_aperture(distance, *(span + shift), width)
        if measured is not None:
            measurements.append(np.asarray(measured) - shift)
    if len(measurements) < 2:
        return None
    middle = np.median(measurements, axis=0)
    agreeing = [row for row in measurements if np.linalg.norm(row-middle, axis=1).max() <= max(2, width*.35)]
    if len(agreeing) < 2:
        return None
    measured = np.median(agreeing, axis=0)
    corners = measured + sign * normal * width/2
    candidates = anchored_leaves(gray, *corners, width, 1)
    valid = []
    for jamb, leaves in zip(corners, candidates):
        for leaf in leaves:
            if leaf['side']*sign <= 0 or leaf['jamb_distance_px'] > max(2, width*.4):
                continue
            tip = np.asarray(leaf['tip'])
            if not connected_leaf_ink(gray, jamb, tip):
                continue
            if not free_leaf_tip(gray, jamb, tip, width):
                continue
            valid.append(leaf)
    if not valid:
        return None
    return dict(method='outward_leaf_at_exterior_jamb', leaves=[min(valid, key=lambda l: l['jamb_distance_px'])],
                measured_jamb_span=measured.tolist(), agreeing_scanlines=len(agreeing), leaf_count=1,
                hinge_and_swing='unconfirmed_template_defaults', exported_aperture_unchanged=True,
                model_leaf_count_requires_review=False)


def reclassify_graph_doors(graph, source, rgb):
    height,width=rgb.shape[:2]
    if (source.get('meta',{}).get('width'),source.get('meta',{}).get('height'))!=(width,height):
        raise ValueError('Source dimensions do not match raster')
    if graph.get('coordinate_space')!='source_raster' or source.get('meta',{}).get('coordinate_space','source_raster')!='source_raster':
        raise ValueError('Source-raster coordinates required')
    result=copy.deepcopy(graph); decisions=[]
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    distance=cv2.distanceTransform((rgb.max(axis=2)<160).astype(np.uint8),cv2.DIST_L2,5)
    excluded=unary_union([Polygon(p) for p in graph.get('diagnostics',{}).get('excluded_regions',[])])
    walls = structural_union(source.get('walls', []))
    nodes = {n['id']: n['point'] for n in graph['nodes']}
    lines = [LineString([nodes[e['start']], nodes[e['end']]]) for e in graph['edges']]
    # Temporary topology closures cannot establish the side facing outside.
    rooms = [dict(polygon=list(p.exterior.coords), holes=[list(r.coords) for r in p.interiors])
             for p in polygonize(lines) if p.area * graph.get('scale_cm_per_px', 1)**2 >= 5000]
    for index,bridge in enumerate(result['diagnostics']['opening_bridges']):
        if bridge.get('kind')!='windows' or bridge.get('window_review', {}).get('choice') == 'window':
            continue
        matches=[o for o in source.get('windows',[]) if o.get('id')==bridge.get('opening_id')]
        if len(matches)!=1 or matches[0].get('source_class')!='door':
            continue
        span=bridge.get('clear_span',bridge['span']); local_width=opening_axis(matches[0])[2]
        record=dict(bridge_index=index,opening_id=bridge.get('opening_id'),accepted=False)
        decisions.append(record)
        if LineString(span).intersects(excluded):
            record['reason']='excluded_region';continue
        evidence=door_evidence(gray,distance,span,local_width)
        if evidence is None and rooms and not walls.is_empty:
            evidence=outward_leaf_evidence(gray,distance,span,local_width,rooms,walls)
        if evidence is None:
            record['reason']='missing_supported_dark_leaf';continue
        bridge.update(kind='doors',requires_review=True,
                      type_review=dict(previous_kind='windows',source='raster_leaf_reclassification',evidence=evidence))
        record.update(accepted=True,evidence=evidence)
    return result,decisions


def main():
    from PIL import Image
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph',type=Path);parser.add_argument('source',type=Path);parser.add_argument('image',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();load=lambda p:json.loads(p.read_text(encoding='utf-8'))
    graph,report=reclassify_graph_doors(load(args.graph),load(args.source),np.asarray(Image.open(args.image).convert('RGB')))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(graph,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    args.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(dict(reclassified=sum(r['accepted'] for r in report)))


if __name__=='__main__':main()
