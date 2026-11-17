"""Compare saved diagnostic flags with door GT, without feeding GT to recognition.

Inputs are precomputed diagnostic reports. The result measures only candidates
available to those reports and inherits any bias from manual graph edits.
"""
import argparse
import json
from pathlib import Path

from evaluate_recognition import evaluate


def evaluate_diagnostic_reports(ground_truth, reports, *, span_source='carrier'):
    has_doors = ground_truth.get('coverage', {}).get('doors') == 'geometry'
    has_windows = ground_truth.get('coverage', {}).get('windows') == 'geometry'
    if not (has_doors or has_windows):
        raise ValueError('Door or window geometry ground truth is required')
    truth = dict(ground_truth)
    truth['coverage'] = {'doors': 'geometry'} if has_doors else {}
    truth['annotation_regions'] = {k: v for k, v in ground_truth.get('annotation_regions', {}).items() if k == 'doors'}
    if span_source not in ('carrier', 'model-clear', 'verified-clear'):
        raise ValueError('Invalid diagnostic span source')
    spans, seen = [], set()
    source_counts = {'carrier': 0, 'model-clear': 0, 'verified-clear': 0}
    for report in reports:
        if not isinstance(report, list):
            raise ValueError('Diagnostic report must be a list')
        for row in report:
            if not row.get('accepted'):
                continue
            carrier = row['span']
            key = tuple(sorted(tuple(point) for point in carrier))
            if key not in seen:
                seen.add(key)
                field = {'model-clear': 'model_clear_span', 'verified-clear': 'raster_verified_clear_span'}.get(span_source)
                selected = span_source if field and row.get(field) else 'carrier'
                span = row[field] if selected != 'carrier' else carrier
                source_counts[selected] += 1
                spans.append(dict(span=span))
    result = evaluate(truth, dict(meta=ground_truth['image'], doors=spans))
    if has_windows:
        window_truth = dict(ground_truth)
        window_truth['coverage'] = {'windows': 'geometry'}
        window_truth['annotation_regions'] = {k: v for k, v in ground_truth.get('annotation_regions', {}).items() if k == 'windows'}
        window_eval = evaluate(window_truth, dict(meta=ground_truth['image'], windows=spans))['metrics']['windows']
        result['window_reference_overlap'] = dict(
            annotated_windows=window_eval['expected'], flagged_spans_in_scope=window_eval['predicted'],
            matched_flagged_spans=window_eval['true_positive'], matches=window_eval['matches'],
            excluded_flagged_spans=window_eval.get('excluded_prediction_count', 0),
            scope=window_eval.get('scope', 'entire_image'),
            interpretation=('One-to-one matches of diagnostic door hints to window annotations; '
                            'review against source, not automatic reclassification.'))
    result['interpretation'] = ('Evaluation of pre-existing diagnostic candidates only; '
                                'manual graph reviews and incomplete annotation may bias results.')
    result['span_selection'] = dict(requested=span_source, used=source_counts,
                                    note='Source counts include candidates outside annotated regions.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ground-truth',type=Path,required=True)
    parser.add_argument('--report',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--span-source',choices=('carrier','model-clear','verified-clear'),default='carrier')
    args=parser.parse_args()
    load=lambda path:json.loads(path.read_text(encoding='utf-8'))
    result=evaluate_diagnostic_reports(load(args.ground_truth),[load(p) for p in args.report],span_source=args.span_source)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps({'doors':result['metrics'].get('doors'),
                      'window_reference_overlap':result.get('window_reference_overlap')},ensure_ascii=False))


if __name__=='__main__':main()
