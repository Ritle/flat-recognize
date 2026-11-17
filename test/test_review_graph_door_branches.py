import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_graph_door_branches import apply_branch_door_review
from test.test_review_graph_doors import fixture as base_fixture


def fixture():
    graph,source,rgb,_=base_fixture()
    graph['diagnostics']['topology_gaps']=[]
    graph['nodes'].extend([dict(id=8,point=[54,64],adjacency=[6]),dict(id=9,point=[66,64],adjacency=[7])])
    graph['nodes'][6]['adjacency'].append(8);graph['nodes'][7]['adjacency'].append(9)
    graph['edges'].extend([dict(start=6,end=8,thickness_px=3),dict(start=7,end=9,thickness_px=3)])
    review=dict(version=1,image_sha256='a'*64,reviewer='assistant_visual_review',basis='Two visible leaves',
                span=[[50,60],[70,60]],local_width_px=6,visible_leaf_count=2,
                remove_dangling_branches=[[[50,60],[54,64]],[[70,60],[66,64]]])
    return graph,source,rgb,review


class ReviewedBranchDoorTest(unittest.TestCase):
    def apply(self,g,s,r,d):return apply_branch_door_review(g,s,r,d,'a'*64)

    def test_review_preserves_input_and_creates_bound_door(self):
        graph,source,rgb,review=fixture();before=copy.deepcopy(graph)
        result=self.apply(graph,source,rgb,review)
        self.assertEqual(graph,before)
        bridge=result['diagnostics']['opening_bridges'][0]
        self.assertEqual(bridge['source'],'assistant_visual_review')
        self.assertNotIn('type_confirmed_by_user',bridge['evidence'])
        self.assertEqual(bridge['type_review']['evidence']['leaf_count'],2)
        self.assertEqual(len(result['faces']),2)
        self.assertEqual(len(result['nodes']),len(graph['nodes'])-2)
        self.assertEqual(self.apply(result,source,rgb,review),result)

    def test_invalid_or_unsafe_reviews_fail(self):
        for mode in ('hash','basis','stale','long','junction','excluded','solid','unbounded','other_opening'):
            graph,source,rgb,review=fixture()
            if mode=='hash':review['image_sha256']='b'*64
            elif mode=='basis':review.pop('basis')
            elif mode=='stale':review['span'][0][0]+=1
            elif mode=='long':graph['nodes'][8]['point']=[60,70];review['remove_dangling_branches'][0][1]=[60,70]
            elif mode=='junction':
                graph['nodes'].append(dict(id=10,point=[55,70],adjacency=[8]))
                graph['edges'].append(dict(start=8,end=10,thickness_px=6))
            elif mode=='excluded':graph['diagnostics']['excluded_regions']=[[[55,50],[65,50],[65,70],[55,70]]]
            elif mode=='solid':rgb[57:64,53:68]=0
            elif mode=='unbounded':graph['edges'].pop(0)
            else:graph['diagnostics']['opening_bridges'].append(dict(kind='windows',span=[[50,60],[54,64]]))
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.apply(graph,source,rgb,review)


if __name__=='__main__':unittest.main()
