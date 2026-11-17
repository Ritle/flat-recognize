import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from replay_graph_reviews import replay_reviews
from test.test_review_graph_doors import fixture as door_fixture
from test.test_review_window_as_door import fixture as window_fixture


class ReviewReplayTest(unittest.TestCase):
    def test_gap_review_replays_without_mutating_inputs(self):
        graph,source,rgb,decision=door_fixture()
        before=copy.deepcopy((graph,source,decision))
        steps=[dict(kind='gap_door',decision=decision)]
        result,report=replay_reviews(graph,source,rgb,'a'*64,steps)
        self.assertEqual(result['diagnostics']['opening_bridges'][0]['kind'],'doors')
        self.assertEqual(report,[dict(index=0,kind='gap_door')])
        self.assertEqual((graph,source,decision),before)
        again,_=replay_reviews(result,source,rgb,'a'*64,steps)
        self.assertEqual(again,result)

    def test_unrecognized_or_bad_hash_does_not_modify_input(self):
        graph,source,rgb,decision=door_fixture();before=copy.deepcopy(graph)
        for steps in ([dict(kind='unknown',decision=decision)],
                      [dict(kind='gap_door',decision=dict(decision,image_sha256='b'*64))],
                      [dict(kind='gap_door')]):
            with self.subTest(steps=steps),self.assertRaises(ValueError):
                replay_reviews(graph,source,rgb,'a'*64,steps)
            self.assertEqual(graph,before)

    def test_window_review_uses_image_dimensions(self):
        graph,source,decision=window_fixture()
        import numpy as np
        rgb=np.zeros((100,100,3),dtype=np.uint8)
        result,_=replay_reviews(graph,source,rgb,'a'*64,[dict(kind='window_door',decision=decision)])
        self.assertEqual(result['diagnostics']['opening_bridges'][0]['kind'],'doors')
        with self.assertRaises(ValueError):replay_reviews(graph,source,rgb[:90], 'a'*64,[dict(kind='window_door',decision=decision)])


if __name__=='__main__':unittest.main()
