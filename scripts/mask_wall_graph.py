"""Experimental room-independent wall graph, in source raster coordinates.

No target-project serialization here. Rejected evidence stays in the report.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from shapely.geometry import LineString, Point, Polygon
from shapely import set_precision
from shapely.ops import polygonize, unary_union

from extract_rooms import fill_polygon
from internal_wall_graph import skeleton_paths
from polygon_geometry import parts
from classify_openings import opening_axis, structural_union
from door_leaf_evidence import leaf_and_arc, diagonal_leaf_with_jambs


def node_lines(lines, snap, *, straighten=True):
    """Snap endpoints onto supported neighboring edges, then node intersections."""
    lines = [line for line in lines if line.length > 1e-6]
    # Quantize only near-axis directions. True diagonals retain their orientation.
    straight = []
    for line in lines:
        a, b = np.asarray(line.coords[0]), np.asarray(line.coords[-1])
        delta = b-a
        angle = abs(math.degrees(math.atan2(delta[1], delta[0]))) % 180
        if straighten and min(angle, 180-angle) <= 6:
            a[1] = b[1] = (a[1]+b[1])/2
        elif straighten and abs(angle-90) <= 6:
            a[0] = b[0] = (a[0]+b[0])/2
        straight.append(LineString([a,b]))
    # Cluster close endpoints once: independent pairwise snapping swaps endpoints
    # and leaves the very gaps it was intended to repair.
    points=[np.array(p) for line in straight for p in (line.coords[0],line.coords[-1])]
    parents=list(range(len(points)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]];i=parents[i]
        return i
    for i,a in enumerate(points):
        for j in range(i):
            if i//2!=j//2 and np.linalg.norm(a-points[j])<=snap:
                parents[root(i)]=root(j)
    centers={}
    for i,p in enumerate(points):centers.setdefault(root(i),[]).append(p)
    centers={k:np.mean(v,axis=0) for k,v in centers.items()}
    adjusted=[LineString([centers[root(2*i)],centers[root(2*i+1)]]) for i in range(len(straight))]
    adjusted=[line for line in adjusted if line.length>1e-6]
    # Insert explicit T anchors in the target edge so projections remain exact.
    anchors=[[] for _ in adjusted]; endpoints=[list(line.coords) for line in adjusted]
    for i,line in enumerate(adjusted):
        for k,xy in enumerate(line.coords):
            p=Point(xy)
            candidates=[]
            for j,other in enumerate(adjusted):
                t=other.project(p)
                if j!=i and snap<t<other.length-snap:
                    candidates.append((other.distance(p),j,other.interpolate(t)))
            if candidates:
                distance,j,target=min(candidates,key=lambda row:row[0])
                if distance<=snap:
                    endpoints[i][k]=target.coords[0];anchors[j].append(target.coords[0])
    adjusted=[LineString([endpoints[i][0],*sorted(anchors[i],key=lambda p:line.project(Point(p))),endpoints[i][-1]])
              for i,line in enumerate(adjusted)]
    # unary_union nodes actual crossings; round to remove floating point duplicates.
    network=unary_union([set_precision(line,1e-4) for line in adjusted])
    return [LineString([a,b]) for line in parts(network,'LineString')
            for a,b in zip(line.coords,list(line.coords)[1:]) if Point(a).distance(Point(b))>1e-5]


def build_graph(source, rgb, scale=1.0, excluded_regions=()):
    """Build axes without rooms. Explicit exclusions are in source pixel space.

    Exclusions are user/annotation scope, never inferred from a file name. They
    are optional: automatic detection of exterior railings is still separate.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('scale must be positive centimeters per source pixel')
    h,w=rgb.shape[:2]
    if (source['meta']['width'],source['meta']['height'])!=(w,h):
        raise ValueError('Source dimensions do not match image')
    mask=np.zeros((h,w),np.uint8)
    for wall in source.get('walls',[]):
        local=np.zeros_like(mask);fill_polygon(local,wall);mask|=local
    # Only explicitly supplied/confirmed openings may support an axis gap.
    for opening in source.get('doors',[])+source.get('windows',[]):
        local=np.zeros_like(mask);fill_polygon(local,opening);mask|=local
    excluded=np.zeros_like(mask)
    excluded_shape=unary_union([Polygon(ring) for ring in excluded_regions])
    for ring in excluded_regions:
        cv2.fillPoly(excluded,[np.rint(ring).astype(np.int32)],255)
    mask[excluded>0]=0
    paths,skeleton=skeleton_paths(mask)
    distance=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)
    # Independent raw-raster evidence rejects broad segmentation furniture/fills.
    ink=(rgb.max(axis=2)<160).astype(np.uint8)
    raw_distance=cv2.distanceTransform(ink,cv2.DIST_L2,5)
    accepted=[];rejected=[];widths=[]
    for path in paths:
        line=LineString(path).simplify(1.0)
        for a,b in zip(line.coords,list(line.coords)[1:]):
            edge=LineString([a,b])
            if edge.length<0.01:continue
            samples=np.array([edge.interpolate(t,normalized=True).coords[0]
                              for t in np.linspace(.1,.9,max(3,round(edge.length)))])
            xy=np.rint(samples).astype(int);xy[:,0]=np.clip(xy[:,0],0,w-1);xy[:,1]=np.clip(xy[:,1],0,h-1)
            radius=float(np.median(distance[xy[:,1],xy[:,0]]))
            support=[]
            for x,y in xy:
                r=max(1,min(5,round(radius)))
                support.append(float(raw_distance[max(0,y-r):min(h,y+r+1),max(0,x-r):min(w,x+r+1)].max()))
            support=float(np.mean(np.array(support)>=1.8))
            record=dict(start=list(a),end=list(b),thickness_px=round(radius*2,3),raster_support=round(support,3))
            if support<.55:
                rejected.append(dict(record,reason='weak_thick_raster_support'));continue
            accepted.append(edge);widths.append(record)
    edges=node_lines(accepted,min(15/scale,5),straighten=False)
    opening_bridges=[]
    for kind in ('doors','windows'):
        for opening in source.get(kind,[]):
            a,b,width=opening_axis(opening)
            length=float(np.linalg.norm(b-a))
            if length<3:continue
            direction=(b-a)/length
            targets=[]
            for p in (Point(a),Point(b)):
                candidates=[]
                for edge in edges:
                    vector=np.array(edge.coords[-1])-edge.coords[0]
                    if abs(np.dot(vector,direction))<edge.length*.95:continue
                    target=edge.interpolate(edge.project(p))
                    candidates.append((p.distance(target),target))
                if not candidates:break
                dist,target=min(candidates,key=lambda r:r[0])
                if dist>max(4,min(8,width)):break
                targets.append(target.coords[0])
            if len(targets)==2:
                bridge=LineString(targets)
                if (bridge.length>=length*.6 and bridge.length<=length*1.5
                        and not bridge.intersects(excluded_shape)):
                    edges.append(bridge)
                    opening_bridges.append(dict(kind=kind,opening_id=opening.get('id'),span=targets))
    # Reuse the existing raw leaf evidence at collinear graph endpoints.
    edges=node_lines(edges,min(15/scale,5),straighten=False)
    degree={}
    for edge in edges:
        for xy in (edge.coords[0],edge.coords[-1]):degree[xy]=degree.get(xy,0)+1
    ends=[(xy,edge) for edge in edges for xy in (edge.coords[0],edge.coords[-1]) if degree[xy]==1]
    wall_shapes=structural_union(source.get('walls',[]))
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    topology_gaps=[]
    for i,(a,ea) in enumerate(ends):
        for b,eb in ends[:i]:
            gap=LineString([a,b]);length=gap.length
            if not 60/scale<=length<=150/scale:continue
            if gap.intersects(excluded_shape):continue
            direction=(np.array(b)-a)/length
            if any(abs(np.dot(np.array(e.coords[-1])-e.coords[0],direction))<e.length*.98 for e in (ea,eb)):continue
            # Require the two segments to extend away from the gap, not overlap
            # it or be unrelated parallel fragments inside the same room.
            if any(gap.intersection(e).length>1e-5 for e in (ea,eb)):continue
            width=float(np.median([r['thickness_px'] for r in widths])) if widths else 5
            evidence=leaf_and_arc(gray,a,b,width,wall_shapes) or diagonal_leaf_with_jambs(gray,a,b,width,wall_shapes)
            if evidence:
                edges.append(gap);opening_bridges.append(dict(kind='recovered_door',span=[a,b],evidence=evidence))
            else:
                topology_gaps.append(dict(span=[a,b],reason='collinear_gap_without_door_evidence'))
    edges=node_lines(edges,0.01,straighten=False)
    nodes=[];index={};segments=[]
    def vertex(xy):
        key=tuple(round(float(v),4) for v in xy)
        if key not in index:
            index[key]=len(nodes);nodes.append(dict(id=len(nodes),point=list(key),adjacency=[]))
        return index[key]
    for edge in edges:
        a,b=vertex(edge.coords[0]),vertex(edge.coords[-1])
        if a==b:continue
        evidence=min(widths,key=lambda r:LineString([r['start'],r['end']]).distance(edge.centroid))
        segments.append(dict(start=a,end=b,thickness_px=evidence['thickness_px']))
        nodes[a]['adjacency'].append(b);nodes[b]['adjacency'].append(a)
    topology=node_lines(edges+[LineString(g['span']) for g in topology_gaps],.01,straighten=False)
    faces=[dict(polygon=[list(xy) for xy in poly.exterior.coords[:-1]],area_px2=poly.area,
                requires_review=any(poly.boundary.intersection(LineString(g['span'])).length>1 for g in topology_gaps))
           for poly in polygonize(topology) if poly.area*scale*scale>=5000]
    return dict(version=1,coordinate_space='source_raster',scale_cm_per_px=scale,
                nodes=nodes,edges=segments,faces=faces,
                diagnostics=dict(accepted=widths,rejected=rejected,opening_bridges=opening_bridges,topology_gaps=topology_gaps,
                                 excluded_regions=list(excluded_regions)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('image',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--scale',type=float,default=1)
    parser.add_argument('--scope',type=Path,help='JSON containing exclude_polygons in source pixels')
    args=parser.parse_args()
    rgb=np.asarray(Image.open(args.image).convert('RGB'))
    scope=json.loads(args.scope.read_text(encoding='utf-8')) if args.scope else {}
    graph=build_graph(json.loads(args.source.read_text(encoding='utf-8')),rgb,args.scale,scope.get('exclude_polygons',[]))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(graph,indent=2),encoding='utf-8')
    im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
    for edge in graph['edges']:
        draw.line([tuple(graph['nodes'][edge[k]]['point']) for k in ['start','end']],fill='cyan',width=2)
    im.save(args.output.with_suffix('.png'))
    print({k:len(graph[k]) for k in ['nodes','edges','faces']})


if __name__=='__main__':main()
