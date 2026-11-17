"""Ask explicit, image-bound questions for actionable uncertain graph openings.

Questions never change types automatically. Answers reuse the existing safety
checks and can leave uncertain candidates unresolved. This is an experimental
graph workflow; it does not modify the default recognition pipeline.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from build_graph_project import build_graph_project
from diagnose_graph_door_leaves import diagnose_graph_door_leaves
from review_decisions import MAX_ANSWERS, image_sha256
from review_graph_doors import apply_graph_door_decisions, gap_fingerprint
from review_graph_passages import apply_passage_reviews, passage_fingerprint
from review_window_as_door import apply_window_door_reviews
from raster_partition_candidates import raster_partitions
from review_solid_wall_door import insert_partition, partition_fingerprint


def make_graph_questions(graph,source,rgb,digest,template,windows,passages=None):
    if not isinstance(digest,str) or len(digest)!=64:
        raise ValueError('Image SHA-256 required')
    if (source.get('meta',{}).get('width'),source.get('meta',{}).get('height')) != (rgb.shape[1],rgb.shape[0]):
        raise ValueError('Source dimensions do not match raster')
    if graph.get('coordinate_space')!='source_raster' or source.get('meta',{}).get('coordinate_space','source_raster')!='source_raster':
        raise ValueError('Source-raster coordinates required')
    _,report=build_graph_project(graph,template,windows,passage_template=passages)
    leaf_candidates = {row['index']: row for row in diagnose_graph_door_leaves(graph,source,rgb)
                       if row['candidate_kind']=='opening' and row['accepted']}
    questions=[]
    for gap in report.get('topology_gaps_omitted', []):
        if gap.get('reason')!='interior_raster_line_without_confirmed_type':continue
        fingerprint=gap_fingerprint(gap)
        trial=dict(version=1,image_sha256=digest,answers=[dict(gap_fingerprint=fingerprint,choice='door')])
        try:apply_graph_door_decisions(graph,source,rgb,trial,digest)
        except ValueError:continue
        questions.append(dict(id='gap:'+fingerprint,kind='gap',fingerprint=fingerprint,
                              image_sha256=digest,span=gap['span'],current='unresolved',
                              reason='interior_boundary_with_uncertain_opening_type',
                              choices=['door','undecided']))
    for binding in report['bindings']['windows']:
        index=binding['bridge_index'];bridge=graph['diagnostics']['opening_bridges'][index]
        if binding.get('source') and passage_fingerprint(binding['source']) != passage_fingerprint(bridge):
            raise ValueError('Exported opening does not match its graph bridge')
        if bridge.get('kind')!='windows' or bridge.get('window_review',{}).get('image_sha256')==digest:continue
        interior = binding.get('exterior_context',{}).get('status')=='interior'
        leaf = leaf_candidates.get(index)
        if not interior and not leaf:continue
        fingerprint=passage_fingerprint(bridge)
        choices=['window']
        door_trial=dict(version=1,image_sha256=digest,answers=[dict(fingerprint=fingerprint,choice='door',
                                                                    user_statement='explicit choice')])
        try:apply_window_door_reviews(graph,source,(rgb.shape[1],rgb.shape[0]),door_trial,digest)
        except ValueError:pass
        else:choices.insert(0,'door')
        if interior:
            trial=dict(version=1,image_sha256=digest,answers=[dict(fingerprint=fingerprint,choice='open_passage',
                                                                   reviewer='user',basis='explicit choice')])
            try:apply_passage_reviews(graph,source,rgb,trial,digest)
            except ValueError:pass
            else:choices.insert(-1,'open_passage')
        choices.append('undecided')
        questions.append(dict(id='window:'+fingerprint,kind='window',fingerprint=fingerprint,
                              image_sha256=digest,span=bridge.get('clear_span',bridge['span']),
                              opening_id=bridge.get('opening_id'),current='window',
                              reason='window_between_interior_areas' if interior else 'possible_door_leaf_near_window',
                              leaf_evidence=leaf['leaf'] if leaf else None,
                              leaf_method=leaf['reason'] if leaf else None,
                              leaf_context=leaf['context'] if leaf else None,choices=choices))
    for candidate in raster_partitions(graph, rgb):
        fingerprint = partition_fingerprint(candidate)
        try:
            trial = insert_partition(graph, candidate, {})
        except ValueError:
            continue
        if len(trial['edges']) == len(graph['edges']):
            continue
        questions.append(dict(id='partition:'+fingerprint, kind='partition', fingerprint=fingerprint,
                              image_sha256=digest, span=candidate['partition_span'], candidate=candidate,
                              leaf_evidence=candidate['leaf'], reason='door_between_parallel_lined_panels',
                              choices=['solid_wall_with_door','undecided']))
    ids=[q['id'] for q in questions]
    if len(set(ids))!=len(ids):raise ValueError('Ambiguous graph review fingerprints')
    return dict(version=1,image_sha256=digest,image_size=[rgb.shape[1],rgb.shape[0]],questions=questions[:MAX_ANSWERS])


def apply_graph_answers(graph,source,rgb,digest,template,windows,answers,passages=None):
    if answers.get('version')!=1 or answers.get('image_sha256')!=digest:
        raise ValueError('Graph review image hash or format does not match')
    rows=answers.get('answers')
    if not isinstance(rows,list) or len(rows)>MAX_ANSWERS:raise ValueError('Invalid graph review answers')
    generated=make_graph_questions(graph,source,rgb,digest,template,windows,passages)
    by_id={q['id']:q for q in generated['questions']};seen=set()
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get('id'),str):raise ValueError('Invalid graph review answer')
        uid=row.get('id');question=by_id.get(uid)
        if uid in seen or question is None or row.get('choice') not in question['choices']:
            raise ValueError('Repeated, stale or unsupported graph review choice')
        seen.add(uid)
    result=copy.deepcopy(graph)
    for row in rows:
        question=by_id[row['id']];choice=row['choice'];fingerprint=question['fingerprint']
        if choice=='undecided':continue
        statement=row.get('user_statement') or choice
        if not isinstance(statement,str):raise ValueError('User statement must be text')
        if question['kind']=='gap':
            payload=dict(version=1,image_sha256=digest,answers=[dict(gap_fingerprint=fingerprint,choice='door',user_statement=statement)])
            result=apply_graph_door_decisions(result,source,rgb,payload,digest)
        elif question['kind']=='partition':
            candidate=question['candidate']
            result=insert_partition(result,candidate,dict(image_sha256=digest,wall_choice='solid',
                user_statement=statement,door_basis='jamb_anchored_raster_leaf',width_source='raster_jambs',
                leaf_evidence=candidate['leaf'],flank_evidence=candidate['flanks'],
                hinge_and_swing='template_defaults',question_fingerprint=fingerprint))
        elif choice=='door':
            payload=dict(version=1,image_sha256=digest,answers=[dict(fingerprint=fingerprint,choice='door',user_statement=statement)])
            result=apply_window_door_reviews(result,source,(rgb.shape[1],rgb.shape[0]),payload,digest)
        elif choice=='open_passage':
            payload=dict(version=1,image_sha256=digest,answers=[dict(fingerprint=fingerprint,choice='open_passage',reviewer='user',basis=statement)])
            result=apply_passage_reviews(result,source,rgb,payload,digest)
        elif choice=='window':
            matches=[b for b in result['diagnostics']['opening_bridges'] if passage_fingerprint(b)==fingerprint]
            if len(matches)!=1 or matches[0].get('kind')!='windows':raise ValueError('Reviewed window changed')
            matches[0]['window_review']=dict(image_sha256=digest,choice='window',user_statement=statement)
    return result


def draw_graph_questions(rgb, questions):
    overlay=Image.fromarray(rgb);draw=ImageDraw.Draw(overlay)
    for i,q in enumerate(questions['questions'],1):
        leaf=q.get('leaf_evidence')
        if leaf:
            draw.line([tuple(leaf['hinge']),tuple(leaf['tip'])],fill='blue',width=3)
        line=[tuple(p) for p in q['span']];draw.line(line,fill='orange' if q['kind']=='window' else 'magenta',width=4)
        if q['kind']=='partition':
            draw.line([tuple(p) for p in q['candidate']['door_span']],fill='limegreen',width=4)
        x=sum(p[0] for p in line)/2;y=sum(p[1] for p in line)/2
        draw.text((x+4,y+4),str(i),fill='red',stroke_width=1,stroke_fill='white')
    return overlay


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('graph',type=Path);p.add_argument('source',type=Path);p.add_argument('image',type=Path)
    p.add_argument('--answers',type=Path,help='Optional explicit answers JSON; absent generates questions')
    p.add_argument('--template',type=Path,default=Path('test/template.json'));p.add_argument('--window-template',type=Path,default=Path('test/template_window.json'))
    p.add_argument('--passage-template',type=Path,default=Path('test/template_passage.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    inputs=[p for p in (a.graph,a.source,a.image,a.answers,a.template,a.window_template,a.passage_template) if p]
    outputs=[a.output] if a.answers else [a.output,a.output.with_suffix('.png')]
    if len({p.resolve() for p in outputs})!=len(outputs) or {p.resolve() for p in outputs}&{p.resolve() for p in inputs}:
        raise ValueError('Outputs must not overwrite inputs')
    graph,source=load(a.graph),load(a.source)
    with Image.open(a.image) as image:rgb=np.asarray(image.convert('RGB'))
    digest=image_sha256(a.image);template=load(a.template);windows=load(a.window_template);passages=load(a.passage_template)
    if a.answers:
        result=apply_graph_answers(graph,source,rgb,digest,template,windows,load(a.answers),passages)
    else:
        result=make_graph_questions(graph,source,rgb,digest,template,windows,passages)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    if not a.answers:
        draw_graph_questions(rgb,result).save(a.output.with_suffix('.png'))
    print(json.dumps(dict(questions=len(result['questions'])) if not a.answers else dict(applied=len(load(a.answers)['answers']))))


if __name__=='__main__':main()
