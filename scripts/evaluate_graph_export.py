"""Evaluate actual physical-graph export areas and hole shapes against GT.

Ground truth is read only after recognition/export. Openings are measured from
wall.holeShape and segment coordinates, not from model-body width or carrier span.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from audit_project_opening_holes import audit_project_opening_holes
from evaluate_recognition import evaluate


def graph_export_geometry(project, report, image):
    if report.get('adapter')!='physical_graph':raise ValueError('Physical graph adapter report required')
    transform=report.get('pixel_to_world',{})
    scale=transform.get('scale');origin=transform.get('origin')
    if (not isinstance(scale,(int,float)) or isinstance(scale,bool) or not math.isfinite(scale) or scale<=0
            or not isinstance(origin,list) or len(origin)!=2 or not all(isinstance(v,(int,float)) and math.isfinite(v) for v in origin)):
        raise ValueError('Invalid graph pixel-to-world transform')
    if audit_project_opening_holes(project)['status']!='pass':raise ValueError('Project hole audit failed')
    if len(project.get('levels',[]))!=1:raise ValueError('Expected one project level')
    level=project['levels'][0];ox,oy=origin
    points={n['uuid']:((n['x']-ox)/scale,(n['y']-oy)/scale) for n in level['connectors']}
    edges={e['uuid']:(points[e['start']],points[e['end']]) for e in level['segments']}
    data=dict(meta=dict(width=image['width'],height=image['height']),
              rooms=[dict(polygon=[points[c] for c in a['connectors']]) for a in level['areas']],doors=[],windows=[],passages=[])
    for item in level['elements']:
        if item['type'] not in ('doorItem','windowItem','wallOpening'):continue
        wall=item['wall'];a,b=edges[wall['uuid']];dx,dy=b[0]-a[0],b[1]-a[1]
        length=math.hypot(dx,dy)
        if length<=0:raise ValueError('Zero-length wall')
        ux,uy=dx/length,dy/length;center=length*wall['position']/100
        xs=[v['x']*100/scale for v in wall['holeShape']]
        span=[[a[0]+ux*(center+offset),a[1]+uy*(center+offset)] for offset in (min(xs),max(xs))]
        kind={'doorItem':'doors','windowItem':'windows','wallOpening':'passages'}[item['type']]
        data[kind].append(dict(id=item['uuid'],span=span))
    return data


def evaluate_graph_export(ground_truth,project,report):
    truth=dict(ground_truth)
    truth['coverage']={k:v for k,v in ground_truth.get('coverage',{}).items() if k in ('rooms','doors','windows')}
    truth['annotation_regions']={k:v for k,v in ground_truth.get('annotation_regions',{}).items() if k in truth['coverage']}
    result=evaluate(truth,graph_export_geometry(project,report,truth['image']))
    result['interpretation']='Graph export areas and actual hole shapes; partial ground truth is not full-plan accuracy.'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('ground_truth',type=Path);parser.add_argument('project',type=Path);parser.add_argument('report',type=Path)
    parser.add_argument('--image',type=Path,help='Verify image SHA-256 against ground truth')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    gt=load(args.ground_truth)
    if args.image and gt.get('image',{}).get('sha256') and hashlib.sha256(args.image.read_bytes()).hexdigest()!=gt['image']['sha256']:
        raise ValueError('Input image SHA-256 does not match ground truth')
    if args.output.resolve() in {p.resolve() for p in (args.ground_truth,args.project,args.report,args.image) if p}:
        raise ValueError('Output must not overwrite an input')
    result=evaluate_graph_export(gt,load(args.project),load(args.report))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:{a:m[a] for a in ('true_positive','false_positive','false_negative') if a in m}
                      for k,m in result['metrics'].items()}))


if __name__=='__main__':main()
