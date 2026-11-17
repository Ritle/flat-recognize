"""Build a gated experimental graph job from existing recognition artifacts."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from build_graph_project import build_graph_project
from graph_review_questions import apply_graph_answers, draw_graph_questions, make_graph_questions
from graph_scope import validate_graph_scope
from graph_topology import unresolved_gap_items
from prepare_review_graph import prepare_review_graph
from review_decisions import image_sha256
from review_graph_furniture import apply_furniture_review
from render_graph_export import render_graph_export

ROOT = Path(__file__).resolve().parent.parent


class GraphAnswerError(ValueError):
    """Saved answers could not be applied to the current graph."""


def run_job(image, source_path, prediction_path, output_dir, *, graph_path=None, answers_path=None, scale=1.0,
            scope_path=None, furniture_review_path=None, allow_single_door_for_paired_leaves=False,
            wall_thickness_cm=None, wall_axis_review_path=None):
    output_dir = Path(output_dir)
    outputs = ('graph.json', 'graph_stages.json', 'graph_report.json', 'quality.json',
               'graph_questions.json', 'graph_questions.png', 'graph_export.png', 'project.json')
    if any((output_dir / name).exists() for name in outputs):
        raise ValueError('Graph job outputs already exist')
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError('Scale must be a positive finite number')
    load = lambda path: json.loads(Path(path).read_text(encoding='utf-8'))
    source, prediction = load(source_path), load(prediction_path)
    template = load(ROOT / 'test/template.json')
    windows = load(ROOT / 'test/template_window.json')
    passages = load(ROOT / 'test/template_passage.json')
    with Image.open(image) as raster:
        rgb = np.asarray(raster.convert('RGB'))
    digest = image_sha256(image)
    axis_review = load(wall_axis_review_path) if wall_axis_review_path is not None else None
    if axis_review is not None and axis_review.get('image_sha256') != digest:
        raise ValueError('Reviewed wall axes belong to another image')
    if graph_path and scope_path:
        raise ValueError('Excluded regions cannot change after graph construction')
    exclusions = validate_graph_scope(load(scope_path), (rgb.shape[1], rgb.shape[0])) if scope_path else []
    if graph_path:
        graph, stages = load(graph_path), []
        diagnostics = graph.get('diagnostics', {})
        if 'topology_gaps' in diagnostics:
            diagnostics['topology_gaps'] = [gap for _, gap in unresolved_gap_items(
                diagnostics['topology_gaps'], diagnostics.get('opening_bridges', []))]
    else:
        graph, stages = prepare_review_graph(
            source, rgb, scale, template, windows, prediction=prediction,
            excluded_regions=exclusions,
            recover_missing=True, reconsider_rejected=True,
            refine_door_types=True, measure_door_widths=True, digest=digest)
    if answers_path:
        try:
            answers = load(answers_path)
            if not isinstance(answers, dict):
                raise ValueError('Expected a graph answers object')
            graph = apply_graph_answers(graph, source, rgb, digest, template, windows, answers, passages)
        except ValueError as exc:
            raise GraphAnswerError(str(exc)) from exc
    if furniture_review_path is not None:
        graph = apply_furniture_review(graph, source, rgb, load(furniture_review_path), digest)
        stages.append(dict(stage='furniture_review',
                           reviewed_branches=len(graph['diagnostics'].get('reviewed_furniture_removals', []))))
    project, report = build_graph_project(graph, template, windows, passage_template=passages,
                                          wall_thickness_cm=wall_thickness_cm, wall_axis_review=axis_review,
                                          allow_single_door_for_paired_leaves=allow_single_door_for_paired_leaves)
    if not project['levels'][0]['areas']:
        quality = report['quality']
        message = 'Замкнутые помещения не найдены; экспорт заблокирован.'
        quality['issues'].append(dict(code='graph_service_no_rooms', severity='error', message=message))
        quality['errors'].append(message)
        quality['status'], quality['export_allowed'] = 'invalid', False
    questions = make_graph_questions(graph, source, rgb, digest, template, windows, passages)
    for question in questions['questions']:
        if question.get('reason') == 'door_between_parallel_lined_panels':
            code = 'graph_unconfirmed_partition'
            message = 'У двери видны параллельные линии. Уточните боковые участки, чтобы разделить помещения.'
        elif question.get('reason') == 'possible_door_leaf_near_window':
            code = 'graph_possible_door_leaf'
            message = 'У окна обнаружена линия, похожая на дверное полотно; проверьте тип проёма.'
        else:
            continue
        report['quality']['warnings'].append(message)
        report['quality']['issues'].append(dict(code=code, severity='warning',
                                               message=message, question_id=question['id'], span=question['span']))
        if report['quality']['status'] == 'good':
            report['quality']['status'] = 'review'
    output_dir.mkdir(parents=True, exist_ok=True)
    def write(name, value):
        (output_dir / name).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                                 allow_nan=False), encoding='utf-8')
    write('graph.json', graph)
    write('graph_stages.json', stages)
    write('graph_report.json', report)
    write('quality.json', report['quality'])
    write('graph_questions.json', questions)
    draw_graph_questions(rgb, questions).save(output_dir / 'graph_questions.png')
    if report['quality']['export_allowed'] and report['quality']['status'] in ('good', 'review'):
        overlay, _ = render_graph_export(Image.fromarray(rgb), project, report)
        overlay.save(output_dir / 'graph_export.png')
        write('project.json', project)
    return report['quality']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('source', type=Path)
    p.add_argument('prediction', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--graph', type=Path)
    p.add_argument('--answers', type=Path, help='Apply saved answers to matching questions on the current image')
    p.add_argument('--scale', type=float, default=1.0, help='Approximate centimeters per pixel')
    p.add_argument('--wall-thickness-cm', type=float, help='Known nominal wall thickness for this plan')
    p.add_argument('--wall-axis-review', type=Path, help='Image-bound wall axes and opening jamb measurements')
    p.add_argument('--scope', type=Path, help='Explicit excluded source-pixel polygons')
    p.add_argument('--furniture-review', type=Path, help='Apply image-bound, visually reviewed wall branch removals')
    p.add_argument('--allow-single-door-for-paired-leaves', action='store_true')
    args = p.parse_args()
    try:
        quality = run_job(args.image, args.source, args.prediction, args.output,
                          graph_path=args.graph, answers_path=args.answers, scale=args.scale,
                          scope_path=args.scope, furniture_review_path=args.furniture_review,
                          wall_thickness_cm=args.wall_thickness_cm, wall_axis_review_path=args.wall_axis_review,
                          allow_single_door_for_paired_leaves=args.allow_single_door_for_paired_leaves)
    except GraphAnswerError as exc:
        print(f'GRAPH_REVIEW_ERROR: {exc}')
        return 3
    except ValueError as exc:
        print(f'GRAPH_BUILD_ERROR: {exc}')
        return 4
    print(json.dumps({'status': quality['status'], 'export_allowed': quality['export_allowed']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
