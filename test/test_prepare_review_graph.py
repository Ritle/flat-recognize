import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import prepare_review_graph as module


class PrepareReviewGraphTest(unittest.TestCase):
    def test_invalid_optional_combinations_fail_before_processing(self):
        rgb=np.zeros((10,10,3),dtype=np.uint8)
        with patch.object(module,'build_graph') as build:
            for kw in (dict(reconsider_rejected=True),dict(measure_door_widths=True),dict(door_review={'answers':[]})):
                with self.subTest(kw=kw),self.assertRaises(ValueError):
                    module.prepare_review_graph({},rgb,1,{}, {},**kw)
            build.assert_not_called()

    def test_orchestrates_existing_stages_in_order_without_gt(self):
        source=dict(meta=dict(width=10,height=10));before=copy.deepcopy(source)
        rgb=np.zeros((10,10,3),dtype=np.uint8)
        graph=dict(nodes=[],edges=[],diagnostics=dict(opening_bridges=[]))
        trace=[]
        def cleanup(g,s,r,indices,**kw):
            trace.append(('cleanup',list(indices),kw))
            return g,[dict(accepted=True)]
        def review(g,s,r,d,h):trace.append(('review',h));return g
        def classify(g,s,r):trace.append(('classify',));return g,[]
        def measure(g,s,r):trace.append(('measure',));return g,[]
        decision=dict(answers=[])
        with patch.object(module,'build_graph',return_value=graph) as build, \
             patch.object(module,'build_graph_project',return_value=({},dict(unbound_openings=[dict(bridge_index=2)]))), \
             patch.object(module,'cleanup_opening_graph',side_effect=cleanup), \
             patch.object(module,'apply_graph_door_decisions',side_effect=review), \
             patch.object(module,'reclassify_graph_doors',side_effect=classify), \
             patch.object(module,'measure_graph_doors',side_effect=measure), \
             patch.object(module,'recover_graph_paired_doors',return_value=(graph,[])), \
             patch.object(module,'apply_passage_reviews',side_effect=review):
            result,stages=module.prepare_review_graph(source,rgb,2,{}, {},prediction={'meta':{}},
                recover_missing=True,reconsider_rejected=True,door_review=decision,refine_door_types=True,
                measure_door_widths=True,passage_review=decision,digest='hash')
        self.assertEqual(result,graph)
        self.assertEqual([s['stage'] for s in stages],['graph','failed_binding_cleanup','missing_windows',
                         'rejected_frames','door_review','door_types','door_widths','paired_branch_doors',
                         'interior_gap_doors','window_jamb_widths','passage_review'])
        self.assertEqual(trace[0][1],[2])
        self.assertEqual(trace[2][2]['prediction'],{'meta':{}})
        self.assertEqual(trace[3:], [('review','hash'),('classify',),('measure',),('review','hash')])
        self.assertEqual(source,before)
        self.assertEqual(build.call_args.kwargs['prediction'],{'meta':{}})


if __name__=='__main__':unittest.main()
