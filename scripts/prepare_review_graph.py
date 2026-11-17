"""Prepare an experimental review graph from saved segmentation results.

Uses the existing graph, cleanup and review stages; does not run inference or
read ground truth. Optional steps are explicit and keep their own diagnostics.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from build_graph_project import build_graph_project
from mask_wall_graph import build_graph
from measure_graph_doors import measure_graph_doors
from opening_graph_cleanup import cleanup_opening_graph, trim_window_cuts
from reclassify_graph_doors import reclassify_graph_doors
from recover_graph_paired_doors import recover_graph_paired_doors
from recover_interior_gap_doors import recover_interior_gap_doors
from review_decisions import image_sha256
from review_graph_doors import apply_graph_door_decisions
from review_graph_passages import apply_passage_reviews


def prepare_review_graph(source,rgb,scale,template,windows,*,prediction=None,excluded_regions=(),
                         recover_missing=False,reconsider_rejected=False,door_review=None,
                         refine_door_types=False,measure_door_widths=False,passage_review=None,digest=None):
    if reconsider_rejected and (not recover_missing or prediction is None):
        raise ValueError('Reconsidering rejected candidates requires prediction and recover_missing')
    if measure_door_widths and not refine_door_types:
        raise ValueError('Measuring reclassified doors requires refine_door_types')
    if (door_review is not None or passage_review is not None) and not digest:
        raise ValueError('Image digest required for manual reviews')
    graph=build_graph(source,rgb,scale,excluded_regions,prediction=prediction)
    stages=[dict(stage='graph',nodes=len(graph['nodes']),edges=len(graph['edges']))]
    _,report=build_graph_project(graph,template,windows)
    graph,rows=cleanup_opening_graph(graph,source,rgb,[r['bridge_index'] for r in report['unbound_openings']])
    stages.append(dict(stage='failed_binding_cleanup',accepted=sum(r['accepted'] for r in rows),decisions=rows))
    if recover_missing:
        graph,rows=cleanup_opening_graph(graph,source,rgb,[],recover_missing=True)
        stages.append(dict(stage='missing_windows',accepted=sum(r['accepted'] for r in rows),decisions=rows))
    if reconsider_rejected:
        graph,rows=cleanup_opening_graph(graph,source,rgb,[],recover_missing=True,prediction=prediction)
        stages.append(dict(stage='rejected_frames',accepted=sum(r['accepted'] for r in rows),decisions=rows))
    if door_review is not None:
        graph=apply_graph_door_decisions(graph,source,rgb,door_review,digest)
        stages.append(dict(stage='door_review',answers=len(door_review['answers'])))
    if refine_door_types:
        graph,rows=reclassify_graph_doors(graph,source,rgb)
        stages.append(dict(stage='door_types',accepted=sum(r['accepted'] for r in rows),decisions=rows))
    if measure_door_widths:
        graph,rows=measure_graph_doors(graph,source,rgb)
        stages.append(dict(stage='door_widths',accepted=sum(r['accepted'] for r in rows),decisions=rows))
    if refine_door_types:
        graph,rows=recover_graph_paired_doors(graph,source,rgb)
        stages.append(dict(stage='paired_branch_doors',accepted=sum(r['accepted'] for r in rows),decisions=rows))
        graph,rows=recover_interior_gap_doors(graph,source,rgb)
        stages.append(dict(stage='interior_gap_doors',accepted=sum(r['accepted'] for r in rows),decisions=rows))
        graph, changes = trim_window_cuts(graph, source, rgb)
        stages.append(dict(stage='window_jamb_widths', accepted=len(changes), decisions=changes))
    if passage_review is not None:
        graph=apply_passage_reviews(graph,source,rgb,passage_review,digest)
        stages.append(dict(stage='passage_review',answers=len(passage_review['answers'])))
    return graph,stages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('image',type=Path)
    p.add_argument('--scale',type=float,required=True);p.add_argument('--prediction',type=Path)
    p.add_argument('--scope',type=Path,help='Optional source-pixel exclude_polygons JSON')
    p.add_argument('--recover-missing',action='store_true');p.add_argument('--reconsider-rejected',action='store_true')
    p.add_argument('--door-review',type=Path);p.add_argument('--refine-door-types',action='store_true')
    p.add_argument('--measure-door-widths',action='store_true');p.add_argument('--passage-review',type=Path)
    p.add_argument('--template',type=Path,default=Path('test/template.json'))
    p.add_argument('--window-template',type=Path,default=Path('test/template_window.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    inputs=[x for x in (a.source,a.image,a.prediction,a.scope,a.door_review,a.passage_review,a.template,a.window_template) if x]
    outputs=[a.output,a.output.with_suffix('.stages.json')]
    if len({x.resolve() for x in outputs})!=2 or {x.resolve() for x in outputs}&{x.resolve() for x in inputs}:
        raise ValueError('Outputs must not overwrite inputs')
    with Image.open(a.image) as image:rgb=np.asarray(image.convert('RGB'))
    scope=load(a.scope) if a.scope else {}
    graph,stages=prepare_review_graph(load(a.source),rgb,a.scale,load(a.template),load(a.window_template),
        prediction=load(a.prediction) if a.prediction else None,excluded_regions=scope.get('exclude_polygons',[]),
        recover_missing=a.recover_missing,reconsider_rejected=a.reconsider_rejected,
        door_review=load(a.door_review) if a.door_review else None,refine_door_types=a.refine_door_types,
        measure_door_widths=a.measure_door_widths,passage_review=load(a.passage_review) if a.passage_review else None,
        digest=image_sha256(a.image))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for path,obj in zip(outputs,(graph,stages)):
        path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(nodes=len(graph['nodes']),edges=len(graph['edges']),stages=[s['stage'] for s in stages])))


if __name__=='__main__':main()
