import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_graph_passages import passage_fingerprint
from review_window_as_door import apply_window_door_reviews


def fixture():
    bridge = dict(kind='windows', opening_id=7, span=[[10,20],[40,20]])
    graph = dict(coordinate_space='source_raster', diagnostics=dict(opening_bridges=[bridge]))
    source = dict(meta=dict(width=100,height=100), windows=[dict(id=7)])
    review = dict(version=1, image_sha256='a'*64, answers=[dict(fingerprint=passage_fingerprint(bridge),
                  choice='door', user_statement='diagonal is a door')])
    return graph,source,review


class WindowDoorReviewTest(unittest.TestCase):
    def test_replay_preserves_matching_automatic_door_evidence(self):
        graph, source, review = fixture()
        bridge = graph['diagnostics']['opening_bridges'][0]
        bridge.update(kind='doors', type_review=dict(source='raster_leaf_reclassification',
                                                     previous_kind='windows'))
        changed = apply_window_door_reviews(graph, source, (100,100), review, 'a'*64)
        result = changed['diagnostics']['opening_bridges'][0]
        self.assertEqual(result['type_review'], bridge['type_review'])
        self.assertEqual(result['door_review']['choice'], 'door')
        del bridge['type_review']
        with self.assertRaises(ValueError):
            apply_window_door_reviews(graph, source, (100,100), review, 'a'*64)
    def test_review_preserves_geometry_and_is_idempotent(self):
        graph,source,review=fixture();before=copy.deepcopy(graph)
        changed=apply_window_door_reviews(graph,source,(100,100),review,'a'*64)
        bridge=changed['diagnostics']['opening_bridges'][0]
        self.assertEqual(bridge['kind'],'doors')
        self.assertEqual(bridge['span'],before['diagnostics']['opening_bridges'][0]['span'])
        self.assertEqual(bridge['door_review']['previous_kind'],'windows')
        self.assertEqual(graph,before)
        self.assertEqual(apply_window_door_reviews(changed,source,(100,100),review,'a'*64),changed)

    def test_stale_ambiguous_unattributed_and_wrong_image_fail(self):
        for mode in ('hash','changed','duplicate','source','statement','size','type'):
            graph,source,review=fixture()
            if mode=='hash':review['image_sha256']='b'*64
            elif mode=='changed':graph['diagnostics']['opening_bridges'][0]['span'][0][0]+=1
            elif mode=='duplicate':review['answers']*=2
            elif mode=='source':source['windows']*=2
            elif mode=='statement':review['answers'][0].pop('user_statement')
            elif mode=='size':source['meta']['width']=1
            else:graph['diagnostics']['opening_bridges'][0]['kind']='passages'
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                apply_window_door_reviews(graph,source,(100,100),review,'a'*64)

    def test_assistant_visual_review_requires_attribution_and_basis(self):
        graph,source,review=fixture()
        answer=review['answers'][0]
        answer.pop('user_statement')
        answer.update(reviewer='assistant_visual_review',basis='Visible diagonal attached to jamb')
        changed=apply_window_door_reviews(graph,source,(100,100),review,'a'*64)
        provenance=changed['diagnostics']['opening_bridges'][0]['door_review']
        self.assertEqual(provenance['reviewer'],'assistant_visual_review')
        self.assertNotIn('user_statement',provenance)
        self.assertEqual(provenance['basis'],'Visible diagonal attached to jamb')
        answer.pop('basis')
        with self.assertRaises(ValueError):apply_window_door_reviews(graph,source,(100,100),review,'a'*64)


if __name__=='__main__':unittest.main()
