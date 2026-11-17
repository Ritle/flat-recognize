"""Create, verify and restore a local image-bound graph review bundle.

Bundles contain the source image and derived project data. They are private
local artifacts, not Git inputs. Checksums detect accidental corruption; they
are not signatures and do not authenticate the bundle's creator.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image

from audit_project_opening_holes import audit_project_opening_holes
from build_graph_project import build_graph_project
from replay_graph_reviews import replay_reviews


MAX_FILE_BYTES=100*1024*1024
MAX_TOTAL_BYTES=300*1024*1024


def digest(data):return hashlib.sha256(data).hexdigest()


def json_bytes(obj):return json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False).encode('utf-8')


def create_bundle(graph_path,source_path,image_path,manifest_path,templates,output):
    manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('version')!=1 or not isinstance(manifest.get('steps'),list) or len(manifest['steps'])>100:
        raise ValueError('Invalid review manifest')
    allow=manifest.get('allow_single_door_for_paired_leaves',False)
    if not isinstance(allow,bool):raise ValueError('Door template approval must be boolean')
    files={'graph.json':graph_path.read_bytes(),'source.json':source_path.read_bytes(),
           'image.bin':image_path.read_bytes(),
           'template.json':templates[0].read_bytes(),'window-template.json':templates[1].read_bytes(),
           'passage-template.json':templates[2].read_bytes()}
    steps=[];input_paths=[graph_path,source_path,image_path,manifest_path,*templates]
    for index,step in enumerate(manifest['steps']):
        if not isinstance(step,dict) or not isinstance(step.get('decision'),str):
            raise ValueError('Invalid review step')
        decision_path=manifest_path.parent/step['decision'];input_paths.append(decision_path)
        member=f'reviews/{index:03d}.json';files[member]=decision_path.read_bytes()
        steps.append(dict(kind=step.get('kind'),decision=member))
    if output.resolve() in {p.resolve() for p in input_paths}:
        raise ValueError('Bundle cannot overwrite an input')
    files['manifest.json']=json_bytes(dict(version=1,allow_single_door_for_paired_leaves=allow,steps=steps))
    if any(len(data)>MAX_FILE_BYTES for data in files.values()) or sum(map(len,files.values()))>MAX_TOTAL_BYTES:
        raise ValueError('Bundle exceeds size limits')
    files['checksums.json']=json_bytes(dict(version=1,sha256={name:digest(data) for name,data in files.items()}))
    # Replay before creating the archive, so invalid input cannot create a bundle.
    result=replay_bundle_files(files)
    output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(output,'x',compression=ZIP_DEFLATED) as archive:
        for name,data in files.items():archive.writestr(name,data)
    return result


def read_bundle(path):
    with ZipFile(path,'r') as archive:
        infos=archive.infolist();names=[info.filename for info in infos]
        if len(names)!=len(set(names)) or len(names)>115:
            raise ValueError('Duplicate or excessive bundle entries')
        if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts or '\\' in n or n.endswith('/') for n in names):
            raise ValueError('Unsafe bundle path')
        if any(i.file_size>MAX_FILE_BYTES for i in infos) or sum(i.file_size for i in infos)>MAX_TOTAL_BYTES:
            raise ValueError('Bundle exceeds size limits')
        files={info.filename:archive.read(info) for info in infos}
    if 'checksums.json' not in files:raise ValueError('Missing checksums')
    checks=json.loads(files['checksums.json'].decode('utf-8'))
    hashes=checks.get('sha256',{})
    if checks.get('version')!=1 or not isinstance(hashes,dict) or set(hashes)!=set(files)-{'checksums.json'}:
        raise ValueError('Bundle checksum inventory does not match')
    if any(digest(files[name])!=value for name,value in hashes.items()):
        raise ValueError('Bundle checksum mismatch')
    return files


def replay_bundle_files(files):
    required={'graph.json','source.json','image.bin','template.json','window-template.json','passage-template.json','manifest.json','checksums.json'}
    if not required <= set(files):raise ValueError('Bundle missing required files')
    load=lambda name:json.loads(files[name].decode('utf-8'))
    manifest=load('manifest.json')
    if manifest.get('version')!=1 or not isinstance(manifest.get('steps'),list) or len(manifest['steps'])>100:
        raise ValueError('Invalid bundled manifest')
    allow=manifest.get('allow_single_door_for_paired_leaves',False)
    if not isinstance(allow,bool):raise ValueError('Invalid bundled door choice')
    steps=[]
    for index,step in enumerate(manifest['steps']):
        expected=f'reviews/{index:03d}.json'
        if not isinstance(step,dict) or step.get('decision')!=expected or expected not in files:
            raise ValueError('Invalid bundled review path')
        steps.append(dict(kind=step.get('kind'),decision=load(expected)))
    if set(files)!=required|{step['decision'] for step in manifest['steps']}:
        raise ValueError('Unexpected bundle files')
    with Image.open(io.BytesIO(files['image.bin'])) as image:rgb=np.asarray(image.convert('RGB'))
    graph,applied=replay_reviews(load('graph.json'),load('source.json'),rgb,digest(files['image.bin']),steps)
    project,report=build_graph_project(graph,load('template.json'),load('window-template.json'),
                                       passage_template=load('passage-template.json'),
                                       allow_single_door_for_paired_leaves=allow)
    audit=audit_project_opening_holes(project)
    if not report['quality']['export_allowed'] or audit['status']!='pass':
        raise ValueError('Replayed bundle failed export or hole audit')
    report['replay_steps']=applied
    return dict(project=project,graph=graph,report=report,audit=audit)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    create=sub.add_parser('create');create.add_argument('graph',type=Path);create.add_argument('source',type=Path);create.add_argument('image',type=Path)
    create.add_argument('--manifest',type=Path,required=True);create.add_argument('--output',type=Path,required=True)
    create.add_argument('--template',type=Path,default=Path('test/template.json'))
    create.add_argument('--window-template',type=Path,default=Path('test/template_window.json'))
    create.add_argument('--passage-template',type=Path,default=Path('test/template_passage.json'))
    verify=sub.add_parser('verify');verify.add_argument('bundle',type=Path)
    restore=sub.add_parser('restore');restore.add_argument('bundle',type=Path);restore.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.action=='create':
        result=create_bundle(args.graph,args.source,args.image,args.manifest,
                             [args.template,args.window_template,args.passage_template],args.output)
    else:
        result=replay_bundle_files(read_bundle(args.bundle))
    if args.action=='restore':
        targets=[args.output,args.output.with_suffix('.graph.json'),args.output.with_suffix('.report.json'),args.output.with_suffix('.audit.json')]
        if len({p.resolve() for p in targets})!=len(targets) or any(p.exists() or p.resolve()==args.bundle.resolve() for p in targets):
            raise ValueError('Restore outputs must be new distinct files')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        for path,key in zip(targets,('project','graph','report','audit')):path.write_bytes(json_bytes(result[key]))
    print(json.dumps(dict(status=result['report']['quality']['status'],areas=len(result['project']['levels'][0]['areas']),
                          openings=result['audit']['counts'],steps=len(result['report']['replay_steps']))))


if __name__=='__main__':main()
