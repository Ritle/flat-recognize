import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_graph_furniture import apply_furniture_review
from test.test_review_graph_doors import fixture as base_fixture


def fixture():
    g,s,r,_=base_fixture()
    g['edges']=g['edges'][:6];g['nodes']=g['nodes'][:6]
    g['nodes'][5]['adjacency'].remove(6);g['nodes'][2]['adjacency'].remove(7)
    g['diagnostics']['topology_gaps']=[]
    g['nodes'].append(dict(id=8,point=[20,65],adjacency=[5]));g['nodes'][5]['adjacency'].append(8)
    g['edges'].append(dict(start=5,end=8,thickness_px=3))
    d=dict(version=1,image_sha256='a'*64,candidates=[dict(terminal_point=[20,65],
           reviewer='assistant_visual_review',basis='Furniture stroke visible in source')])
    return g,s,r,d


class FurnitureReviewTest(unittest.TestCase):
    def apply(self,g,s,r,d):return apply_furniture_review(g,s,r,d,'a'*64)

    def test_remove_reviewed_dangling_branch_preserves_room_and_input(self):
        g,s,r,d=fixture();before=copy.deepcopy(g);result=self.apply(g,s,r,d)
        self.assertEqual(g,before)
        self.assertEqual(len(result['edges']),len(g['edges'])-1)
        self.assertEqual(len(result['faces']),1)
        self.assertEqual(result['diagnostics']['reviewed_furniture_removals'][0]['reviewer'],'assistant_visual_review')
        self.assertEqual(self.apply(result,s,r,d),result)

    def test_empty_review_is_noop(self):
        g,s,r,d=fixture();d['candidates']=[]
        self.assertEqual(self.apply(g,s,r,d),g)

    def test_window_jamb_duplicate_needs_a_separate_full_carrier(self):
        g,s,r,d=fixture()
        tip=[10.1,65]
        g['nodes'][-1]['point']=tip
        d['candidates'][0].update(terminal_point=tip, expected_path=[tip,[10,60]],
                                  coincident_window_jamb=True)
        bridge=dict(kind='windows',span=[[10,60],[10,110]])
        g['diagnostics']['opening_bridges']=[bridge]
        self.assertEqual(len(self.apply(g,s,r,d)['edges']),len(g['edges'])-1)
        bridge['span']=[[10,60],tip]
        with self.assertRaisesRegex(ValueError,'overlaps an opening'):
            self.apply(g,s,r,d)

    def test_pinned_path_checked_on_first_application_and_replay(self):
        g,s,r,d=fixture()
        choice=d['candidates'][0]
        choice['expected_path']=[choice['terminal_point'],g['nodes'][5]['point']]
        reviewed=self.apply(g,s,r,d)
        self.assertEqual(self.apply(reviewed,s,r,d),reviewed)
        choice['expected_path']=[choice['terminal_point'],[99,99]]
        for source_graph in (g,reviewed):
            with self.assertRaisesRegex(ValueError,'branch path has changed'):
                self.apply(source_graph,s,r,d)

    def test_unsafe_inputs_fail(self):
        for mode in ('hash','attribution','missing','duplicate','junction','long','opening','dimensions'):
            g,s,r,d=fixture()
            if mode=='hash':d['image_sha256']='b'*64
            elif mode=='attribution':d['candidates'][0].pop('basis')
            elif mode=='missing':d['candidates'][0]['terminal_point']=[21,65]
            elif mode=='duplicate':d['candidates']*=2
            elif mode=='junction':d['candidates'][0]['terminal_point']=[10,60]
            elif mode=='long':g['nodes'][-1]['point']=[60,65];d['candidates'][0]['terminal_point']=[60,65]
            elif mode=='opening':g['diagnostics']['opening_bridges']=[dict(kind='doors',span=[[10,60],[20,65]])]
            else:s['meta']['width']=1
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.apply(g,s,r,d)


if __name__=='__main__':unittest.main()
