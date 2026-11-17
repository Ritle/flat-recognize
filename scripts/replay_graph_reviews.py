"""Replay explicit, image-bound graph reviews and export a review project.

This starts from an existing graph and classified source. It does not run
inference, invent decisions, or change the normal recognition pipeline.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from audit_project_opening_holes import audit_project_opening_holes
from build_graph_project import build_graph_project
from review_decisions import image_sha256
from review_graph_door_branches import apply_branch_door_review
from review_graph_doors import apply_graph_door_decisions
from review_graph_furniture import apply_furniture_review
from review_graph_passages import apply_passage_reviews
from review_solid_wall_door import apply_reviewed_partition
from review_window_as_door import apply_window_door_reviews


def replay_reviews(graph, source, rgb, digest, steps):
    if not isinstance(steps,list) or len(steps)>100:
        raise ValueError('Review steps must be a list with at most 100 entries')
    current=copy.deepcopy(graph)
    applied=[]
    for index,step in enumerate(steps):
        if not isinstance(step,dict) or not isinstance(step.get('decision'),dict):
            raise ValueError(f'Invalid review step {index}')
        kind=step.get('kind');decision=step['decision']
        if kind=='gap_door':current=apply_graph_door_decisions(current,source,rgb,decision,digest)
        elif kind=='window_door':current=apply_window_door_reviews(current,source,(rgb.shape[1],rgb.shape[0]),decision,digest)
        elif kind=='passage':current=apply_passage_reviews(current,source,rgb,decision,digest)
        elif kind=='branch_door':current=apply_branch_door_review(current,source,rgb,decision,digest)
        elif kind=='solid_partition_door':current=apply_reviewed_partition(current,source,rgb,decision,digest)
        elif kind=='furniture':current=apply_furniture_review(current,source,rgb,decision,digest)
        else:raise ValueError(f'Unknown review kind at step {index}: {kind}')
        applied.append(dict(index=index,kind=kind))
    return current,applied


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('graph',type=Path);p.add_argument('source',type=Path);p.add_argument('image',type=Path)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--template',type=Path,default=Path('test/template.json'))
    p.add_argument('--window-template',type=Path,default=Path('test/template_window.json'))
    p.add_argument('--passage-template',type=Path,default=Path('test/template_passage.json'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    manifest=load(args.manifest)
    if manifest.get('version')!=1 or not isinstance(manifest.get('steps'),list):
        raise ValueError('Invalid review manifest')
    allow=manifest.get('allow_single_door_for_paired_leaves',False)
    if not isinstance(allow,bool):raise ValueError('Door template approval must be boolean')
    inputs=[args.graph,args.source,args.image,args.manifest,args.template,args.window_template,args.passage_template]
    steps=[]
    for item in manifest['steps']:
        if not isinstance(item,dict) or not isinstance(item.get('decision'),str):
            raise ValueError('Manifest steps require a decision file path')
        path=args.manifest.parent/item['decision'];inputs.append(path)
        steps.append(dict(kind=item.get('kind'),decision=load(path)))
    paths=[args.output,args.output.with_suffix('.report.json'),args.output.with_suffix('.graph.json'),args.output.with_suffix('.audit.json')]
    if len({p.resolve() for p in paths})!=len(paths) or {p.resolve() for p in paths}&{p.resolve() for p in inputs}:
        raise ValueError('Output files must not overwrite input files')
    with Image.open(args.image) as im:rgb=np.asarray(im.convert('RGB'))
    graph,steps_applied=replay_reviews(load(args.graph),load(args.source),rgb,image_sha256(args.image),steps)
    project,report=build_graph_project(graph,load(args.template),load(args.window_template),
                                       passage_template=load(args.passage_template),
                                       allow_single_door_for_paired_leaves=allow)
    audit=audit_project_opening_holes(project)
    if not report['quality']['export_allowed'] or audit['status']!='pass':
        raise ValueError('Reviewed graph failed export or opening-hole validation')
    report['replay_steps']=steps_applied
    paths[0].parent.mkdir(parents=True,exist_ok=True)
    for path,obj in zip(paths,(project,report,graph,audit)):
        path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status=report['quality']['status'],steps=len(steps_applied),
                          areas=len(project['levels'][0]['areas']),openings=audit['counts'])))


if __name__=='__main__':main()
