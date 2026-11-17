"""Independently check exported wall cuts against their real segments.

This audit checks serialized holeShape geometry, not recognition accuracy or
compatibility with the target application. It never modifies the project.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from shapely.geometry import LineString, Polygon
from shapely.ops import unary_union


def audit_project_opening_holes(project, *, tolerance_cm=.002):
    if not isinstance(tolerance_cm,(int,float)) or not math.isfinite(tolerance_cm) or tolerance_cm < 0:
        raise ValueError('Invalid tolerance')
    levels=project.get('levels')
    if not isinstance(levels,list):raise ValueError('Project levels must be a list')
    issues=[]; rows=[]; counts={}
    def issue(code,level,uid,**context):
        issues.append(dict(code=code,level=level,element_uuid=uid,**context))
    def number(v):return isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)
    for level_index,level in enumerate(levels):
        connectors={n['uuid']:n for n in level.get('connectors',[])}
        segments={e['uuid']:e for e in level.get('segments',[])}
        intervals={}
        for item in level.get('elements',[]):
            kind=item.get('type')
            if kind not in ('doorItem','windowItem','wallOpening'):continue
            uid=item.get('uuid');counts[kind]=counts.get(kind,0)+1
            wall=item.get('wall',{});wall_id=wall.get('uuid') if isinstance(wall,dict) else None
            segment=segments.get(wall_id)
            if segment is None:
                issue('missing_wall',level_index,uid,wall_uuid=wall_id);continue
            a,b=connectors.get(segment.get('start')),connectors.get(segment.get('end'))
            if not a or not b or not all(number(p.get(k)) for p in (a,b) for k in ('x','y')):
                issue('invalid_wall_endpoints',level_index,uid);continue
            length=math.hypot(b['x']-a['x'],b['y']-a['y'])
            position=wall.get('position');shape=wall.get('holeShape');z=item.get('position',{}).get('z',0)
            if length <= 0 or not number(position) or not 0 <= position <= 100:
                issue('invalid_wall_position',level_index,uid);continue
            if not isinstance(shape,list) or len(shape)<3 or any(not isinstance(p,dict) or not number(p.get('x')) or not number(p.get('y')) for p in shape):
                issue('invalid_hole_shape',level_index,uid);continue
            poly=Polygon([(p['x'],p['y']) for p in shape])
            if not poly.is_valid or poly.area<=0 or not number(z):
                issue('invalid_hole_polygon',level_index,uid);continue
            x0,y0,x1,y1=poly.bounds
            center=length*position/100
            start,end=center+100*x0,center+100*x1
            lower,upper=z+100*y0,z+100*y1
            height=segment.get('height')
            row=dict(level=level_index,element_uuid=uid,type=kind,wall_uuid=wall_id,
                     wall_length_cm=length,hole_start_cm=start,hole_end_cm=end,
                     hole_bottom_cm=lower,hole_top_cm=upper)
            rows.append(row)
            if start < -tolerance_cm or end > length+tolerance_cm:
                issue('hole_outside_wall',level_index,uid,start_cm=start,end_cm=end,wall_length_cm=length)
            if not number(height) or lower < -tolerance_cm or upper > height+tolerance_cm:
                issue('hole_outside_wall_height',level_index,uid,bottom_cm=lower,top_cm=upper,wall_height_cm=height)
            for previous in intervals.setdefault(wall_id,[]):
                if min(end,previous['end'])-max(start,previous['start'])>tolerance_cm and min(upper,previous['upper'])-max(lower,previous['lower'])>tolerance_cm:
                    issue('overlapping_holes',level_index,uid,other_element_uuid=previous['uid'],wall_uuid=wall_id)
            intervals[wall_id].append(dict(start=start,end=end,lower=lower,upper=upper,uid=uid))
            for material in item.get('materials',[]):
                if material.get('itemUuid') != uid:
                    issue('invalid_material_reference',level_index,uid)
    return dict(status='pass' if not issues else 'invalid',counts=counts,openings=rows,issues=issues,
                tolerance_cm=tolerance_cm,note='Structural check only; target application import is not verified.')


def measure_topology_gaps(project, report, audit):
    """Partition each diagnostic span in plan view using the serialized wall cuts."""
    if report.get('adapter') != 'physical_graph' or len(project['levels']) != 1:
        raise ValueError('Expected a single-level physical graph export')
    if audit['status'] != 'pass':
        raise ValueError('Topology gap measurement requires a valid wall-hole audit')
    transform = report['pixel_to_world']
    scale = transform['scale']
    ox, oy = transform['origin']
    if not math.isfinite(scale) or scale <= 0 or not all(map(math.isfinite, (ox, oy))):
        raise ValueError('Invalid pixel-to-world transform')
    expected = {(b['element_uuid'], b['wall_uuid']) for group in report['bindings'].values() for b in group}
    if expected != {(h['element_uuid'], h['wall_uuid']) for h in audit['openings']}:
        raise ValueError('Opening bindings in graph report do not match project')

    by_wall = defaultdict(list)
    for hole in audit['openings']:
        by_wall[hole['wall_uuid']].append(hole)
    level = project['levels'][0]
    points = {n['uuid']: (n['x'], n['y']) for n in level['connectors']}
    walls, solids = [], []
    for segment in level['segments']:
        axis = LineString([points[segment['start']], points[segment['end']]])
        radius = segment['thickness'] / 2
        footprint = axis.buffer(radius, cap_style=2)
        walls.append(footprint)
        cuts = []
        for hole in by_wall[segment['uuid']]:
            # The hole audit allows tiny endpoint roundoff. Negative distances
            # would interpolate from the opposite end of the segment.
            a = axis.interpolate(max(0., hole['hole_start_cm']))
            b = axis.interpolate(max(0., hole['hole_end_cm']))
            cuts.append(LineString([a, b]).buffer(radius, cap_style=2))
        solids.append(footprint.difference(unary_union(cuts)))
    wall = unary_union(walls)
    solid = unary_union(solids)
    results = []
    for gap in report['topology_gaps_omitted']:
        line = LineString([(ox + x * scale, oy + y * scale) for x, y in gap['span']])
        covered = line.intersection(wall)
        results.append(dict(span=gap['span'], length_px=line.length / scale,
                            solid_wall_px=covered.intersection(solid).length / scale,
                            opening_px=covered.difference(solid).length / scale,
                            uncovered_px=line.difference(wall).length / scale))
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('project',type=Path);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--graph-report', type=Path,
                        help='Measure omitted topology spans against this graph export in plan view')
    args=parser.parse_args()
    if args.output.resolve() in {path.resolve() for path in (args.project, args.graph_report) if path}:
        parser.error('Output must not overwrite an input')
    project = json.loads(args.project.read_text(encoding='utf-8'))
    result=audit_project_opening_holes(project)
    if args.graph_report and result['status'] == 'pass':
        report = json.loads(args.graph_report.read_text(encoding='utf-8'))
        result['topology_gap_coverage'] = measure_topology_gaps(project, report, result)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=result['status'],counts=result['counts'],issues=len(result['issues']))))
    return 0 if result['status']=='pass' else 2


if __name__=='__main__':raise SystemExit(main())
