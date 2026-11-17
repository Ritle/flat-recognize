import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate_door_leaf_diagnostics import evaluate_diagnostic_reports


class DoorDiagnosticEvaluationTest(unittest.TestCase):
    def test_flags_only_deduplicate_and_respect_partial_coverage(self):
        gt=dict(version=1,image=dict(width=100,height=100),coverage=dict(doors='geometry'),
                annotation_regions=dict(doors=[[0,0,60,60]]),doors=[dict(id='d1',span=[[10,20],[30,20]])])
        reports=[[dict(accepted=True,span=[[10,20],[30,20]]),dict(accepted=False,span=[[15,30],[35,30]])],
                 [dict(accepted=True,span=[[30,20],[10,20]]),dict(accepted=True,span=[[70,80],[90,80]])]]
        before=copy.deepcopy((gt,reports))
        metrics=evaluate_diagnostic_reports(gt,reports)['metrics']['doors']
        self.assertEqual((metrics['true_positive'],metrics['false_positive'],metrics['false_negative']),(1,0,0))
        self.assertEqual(metrics['scope'],'annotation_regions')
        self.assertEqual((gt,reports),before)

    def test_missing_geometry_or_malformed_report_fail(self):
        with self.assertRaises(ValueError):evaluate_diagnostic_reports(dict(coverage={}),[])
        with self.assertRaises(ValueError):
            evaluate_diagnostic_reports(dict(coverage=dict(doors='geometry')),['invalid'])

    def test_span_source_is_explicit_and_falls_back_without_verified_width(self):
        gt=dict(version=1,image=dict(width=100,height=100),coverage=dict(doors='geometry'),
                doors=[dict(id='d',span=[[20,20],[40,20]])])
        row=dict(accepted=True,span=[[10,20],[50,20]],model_clear_span=[[21,20],[39,20]],
                 raster_verified_clear_span=[[20,20],[40,20]])
        reports=[[row]]
        carrier=evaluate_diagnostic_reports(gt,reports)
        measured=evaluate_diagnostic_reports(gt,reports,span_source='verified-clear')
        model=evaluate_diagnostic_reports(gt,reports,span_source='model-clear')
        self.assertGreater(carrier['metrics']['doors']['matches'][0]['length_error_px'],
                           measured['metrics']['doors']['matches'][0]['length_error_px'])
        self.assertEqual(measured['span_selection']['used']['verified-clear'],1)
        self.assertEqual(model['span_selection']['used']['model-clear'],1)
        row.pop('raster_verified_clear_span')
        fallback=evaluate_diagnostic_reports(gt,reports,span_source='verified-clear')
        self.assertEqual(fallback['span_selection']['used']['carrier'],1)

    def test_window_overlap_respects_partial_coverage_and_does_not_claim_door_ground_truth(self):
        gt = dict(version=1, image=dict(width=100,height=100), coverage=dict(windows='geometry'),
                  annotation_regions=dict(windows=[[0,0,60,60]]),
                  windows=[dict(id='w1', span=[[10,20],[30,20]])])
        rows = [[dict(accepted=True,span=[[10,20],[30,20]]),
                 dict(accepted=True,span=[[70,80],[90,80]])]]
        result = evaluate_diagnostic_reports(gt, rows)
        self.assertNotIn('doors', result['metrics'])
        overlap = result['window_reference_overlap']
        self.assertEqual(overlap['matched_flagged_spans'],1)
        self.assertEqual(overlap['flagged_spans_in_scope'],1)
        self.assertEqual(overlap['excluded_flagged_spans'],1)
        self.assertEqual(overlap['matches'][0]['truth'],'w1')

    def test_window_overlap_is_one_to_one_not_a_count_of_all_nearby_flags(self):
        gt = dict(version=1, image=dict(width=100,height=100), coverage=dict(windows='geometry'),
                  windows=[dict(id='w1',span=[[10,20],[30,20]])])
        rows = [[dict(accepted=True,span=[[10,20],[30,20]]),
                 dict(accepted=True,span=[[11,20],[31,20]])]]
        overlap=evaluate_diagnostic_reports(gt,rows)['window_reference_overlap']
        self.assertEqual(overlap['flagged_spans_in_scope'],2)
        self.assertEqual(overlap['matched_flagged_spans'],1)


if __name__=='__main__':unittest.main()
