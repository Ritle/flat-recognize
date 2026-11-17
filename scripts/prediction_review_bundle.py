"""Bundle saved model results and image-bound graph reviews for local replay.

Contains the user's raster and derived data; store outside Git. The bundle
starts from saved prediction/classification and does not include inference,
weights, source code or Python dependencies. Hashes detect corruption only.
"""
import argparse
import io
import json
import math
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

from audit_project_opening_holes import audit_project_opening_holes
from build_graph_project import build_graph_project
from graph_review_bundle import MAX_FILE_BYTES, MAX_TOTAL_BYTES, digest, json_bytes, read_bundle
from prepare_review_graph import prepare_review_graph
from replay_graph_reviews import replay_reviews


def build_files(source,image,prediction,manifest,templates,*,scale,door_review=None,passage_review=None,
                scope=None,recover_missing=False,reconsider_rejected=False,refine_door_types=False,
                measure_door_widths=False):
    config=dict(version=1,scale=scale,recover_missing=recover_missing,reconsider_rejected=reconsider_rejected,
                refine_door_types=refine_door_types,measure_door_widths=measure_door_widths,
                door_review=door_review is not None,passage_review=passage_review is not None,scope=scope is not None)
    if not isinstance(scale,(int,float)) or isinstance(scale,bool) or not math.isfinite(scale) or scale<=0:
        raise ValueError('Scale must be positive and finite')
    if manifest.get('version')!=1 or not isinstance(manifest.get('steps'),list) or len(manifest['steps'])>100:
        raise ValueError('Invalid review manifest')
    allow=manifest.get('allow_single_door_for_paired_leaves',False)
    if not isinstance(allow,bool):raise ValueError('Door template approval must be boolean')
    files={'source.json':source,'image.bin':image,'prediction.json':prediction,
           'template.json':templates[0],'window-template.json':templates[1],'passage-template.json':templates[2]}
    for name,data in (('initial-door-review.json',door_review),('initial-passage-review.json',passage_review),('scope.json',scope)):
        if data is not None:files[name]=data
    steps=[]
    for i,step in enumerate(manifest['steps']):
        if not isinstance(step,dict) or not isinstance(step.get('decision_bytes'),bytes):
            raise ValueError('Expected resolved review bytes')
        path=f'reviews/{i:03d}.json';files[path]=step['decision_bytes'];steps.append(dict(kind=step.get('kind'),decision=path))
    files['manifest.json']=json_bytes(dict(version=1,allow_single_door_for_paired_leaves=allow,steps=steps))
    files['config.json']=json_bytes(config)
    if any(len(v)>MAX_FILE_BYTES for v in files.values()) or sum(map(len,files.values()))>MAX_TOTAL_BYTES:
        raise ValueError('Bundle exceeds size limits')
    files['checksums.json']=json_bytes(dict(version=1,sha256={k:digest(v) for k,v in files.items()}))
    return files


def replay_prediction_files(files):
    base={'source.json','image.bin','prediction.json','template.json','window-template.json',
          'passage-template.json','manifest.json','config.json','checksums.json'}
    if not base <= set(files):raise ValueError('Bundle missing required files')
    load=lambda name:json.loads(files[name].decode('utf-8'))
    config=load('config.json');manifest=load('manifest.json')
    flags=('recover_missing','reconsider_rejected','refine_door_types','measure_door_widths','door_review','passage_review','scope')
    if config.get('version')!=1 or any(not isinstance(config.get(k),bool) for k in flags):
        raise ValueError('Invalid preparation config')
    scale=config.get('scale')
    if not isinstance(scale,(int,float)) or isinstance(scale,bool) or not math.isfinite(scale) or scale<=0:
        raise ValueError('Invalid scale')
    if manifest.get('version')!=1 or not isinstance(manifest.get('steps'),list) or len(manifest['steps'])>100:
        raise ValueError('Invalid review manifest')
    allow=manifest.get('allow_single_door_for_paired_leaves',False)
    if not isinstance(allow,bool):raise ValueError('Invalid door choice')
    optional={'initial-door-review.json':'door_review','initial-passage-review.json':'passage_review','scope.json':'scope'}
    expected=base|{name for name,key in optional.items() if config[key]}
    steps=[]
    for i,step in enumerate(manifest['steps']):
        path=f'reviews/{i:03d}.json'
        if not isinstance(step,dict) or step.get('decision')!=path or path not in files:
            raise ValueError('Invalid bundled review path')
        expected.add(path);steps.append(dict(kind=step.get('kind'),decision=load(path)))
    if set(files)!=expected:raise ValueError('Bundle inventory does not match configuration')
    with Image.open(io.BytesIO(files['image.bin'])) as im:rgb=np.asarray(im.convert('RGB'))
    source=load('source.json');template=load('template.json');windows=load('window-template.json')
    image_hash=digest(files['image.bin'])
    scope=load('scope.json') if config['scope'] else {}
    graph,stages=prepare_review_graph(source,rgb,scale,template,windows,prediction=load('prediction.json'),
        excluded_regions=scope.get('exclude_polygons',[]),recover_missing=config['recover_missing'],
        reconsider_rejected=config['reconsider_rejected'],refine_door_types=config['refine_door_types'],
        measure_door_widths=config['measure_door_widths'],
        door_review=load('initial-door-review.json') if config['door_review'] else None,
        passage_review=load('initial-passage-review.json') if config['passage_review'] else None,digest=image_hash)
    graph,reviews=replay_reviews(graph,source,rgb,image_hash,steps)
    project,report=build_graph_project(graph,template,windows,passage_template=load('passage-template.json'),
                                       allow_single_door_for_paired_leaves=allow)
    audit=audit_project_opening_holes(project)
    if not report['quality']['export_allowed'] or audit['status']!='pass':
        raise ValueError('Replayed prediction bundle failed export or hole audit')
    report['preparation_stages']=stages;report['replay_steps']=reviews
    return dict(project=project,graph=graph,report=report,audit=audit)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    create=sub.add_parser('create');create.add_argument('source',type=Path);create.add_argument('image',type=Path)
    create.add_argument('--prediction',type=Path,required=True);create.add_argument('--manifest',type=Path,required=True)
    create.add_argument('--scale',type=float,required=True);create.add_argument('--output',type=Path,required=True)
    create.add_argument('--door-review',type=Path);create.add_argument('--passage-review',type=Path);create.add_argument('--scope',type=Path)
    create.add_argument('--recover-missing',action='store_true');create.add_argument('--reconsider-rejected',action='store_true')
    create.add_argument('--refine-door-types',action='store_true');create.add_argument('--measure-door-widths',action='store_true')
    create.add_argument('--template',type=Path,default=Path('test/template.json'))
    create.add_argument('--window-template',type=Path,default=Path('test/template_window.json'))
    create.add_argument('--passage-template',type=Path,default=Path('test/template_passage.json'))
    verify=sub.add_parser('verify');verify.add_argument('bundle',type=Path)
    restore=sub.add_parser('restore');restore.add_argument('bundle',type=Path);restore.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='create':
        manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
        inputs=[x for x in (args.source,args.image,args.prediction,args.manifest,args.door_review,args.passage_review,args.scope,
                            args.template,args.window_template,args.passage_template) if x]
        for step in manifest.get('steps',[]):
            path=args.manifest.parent/step['decision'];inputs.append(path);step['decision_bytes']=path.read_bytes()
        if args.output.resolve() in {p.resolve() for p in inputs}:raise ValueError('Bundle cannot overwrite input')
        files=build_files(args.source.read_bytes(),args.image.read_bytes(),args.prediction.read_bytes(),manifest,
                          [args.template.read_bytes(),args.window_template.read_bytes(),args.passage_template.read_bytes()],
                          scale=args.scale,door_review=args.door_review.read_bytes() if args.door_review else None,
                          passage_review=args.passage_review.read_bytes() if args.passage_review else None,
                          scope=args.scope.read_bytes() if args.scope else None,recover_missing=args.recover_missing,
                          reconsider_rejected=args.reconsider_rejected,refine_door_types=args.refine_door_types,
                          measure_door_widths=args.measure_door_widths)
        result=replay_prediction_files(files)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with ZipFile(args.output,'x',compression=ZIP_DEFLATED) as archive:
            for name,data in files.items():archive.writestr(name,data)
    else:
        result=replay_prediction_files(read_bundle(args.bundle))
    if args.action=='restore':
        outputs=[args.output,args.output.with_suffix('.graph.json'),args.output.with_suffix('.report.json'),args.output.with_suffix('.audit.json')]
        if len({p.resolve() for p in outputs})!=len(outputs) or any(p.exists() or p.resolve()==args.bundle.resolve() for p in outputs):
            raise ValueError('Restore outputs must be new distinct files')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        for path,key in zip(outputs,('project','graph','report','audit')):path.write_bytes(json_bytes(result[key]))
    print(json.dumps(dict(status=result['report']['quality']['status'],areas=len(result['project']['levels'][0]['areas']),
                          openings=result['audit']['counts'],preparation_stages=len(result['report']['preparation_stages']),
                          review_steps=len(result['report']['replay_steps']))))


if __name__=='__main__':main()
