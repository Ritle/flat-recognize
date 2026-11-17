import copy
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from review_graph_doors import apply_graph_door_decisions, gap_fingerprint


def fixture():
    points = [(10,10),(110,10),(110,60),(110,110),(10,110),(10,60),(50,60),(70,60)]
    pairs = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(5,6),(7,2)]
    graph = dict(coordinate_space='source_raster', scale_cm_per_px=5,
                 nodes=[dict(id=i,point=list(p),adjacency=[]) for i,p in enumerate(points)],
                 edges=[dict(start=a,end=b,thickness_px=6) for a,b in pairs], faces=[],
                 diagnostics=dict(topology_gaps=[dict(span=[[50,60],[70,60]],local_width_px=6)],opening_bridges=[]))
    rgb = np.full((120,120,3),255,np.uint8)
    walls=[]
    for a,b in pairs:
        graph['nodes'][a]['adjacency'].append(b);graph['nodes'][b]['adjacency'].append(a)
        polygon=LineString([points[a],points[b]]).buffer(3,cap_style=3)
        outer=list(polygon.exterior.coords)
        cv2.fillPoly(rgb,[np.rint(outer).astype(np.int32)],(0,0,0))
        walls.append(dict(outer=outer))
    source=dict(meta=dict(width=120,height=120),walls=walls)
    decisions=dict(version=1,image_sha256='a'*64,answers=[dict(
        gap_fingerprint=gap_fingerprint(graph['diagnostics']['topology_gaps'][0]),choice='door',user_statement='door')])
    return graph,source,rgb,decisions


class ReviewedGraphDoorTest(unittest.TestCase):
    def apply(self,g,s,r,d):
        return apply_graph_door_decisions(g,s,r,d,'a'*64)

    def test_confirmed_type_creates_real_door_and_preserves_provenance(self):
        g,s,r,d=fixture();before=copy.deepcopy(g)
        result=self.apply(g,s,r,d)
        self.assertEqual(g,before)
        self.assertEqual(result['diagnostics']['topology_gaps'],[])
        self.assertEqual(len(result['faces']),2)
        bridge=result['diagnostics']['opening_bridges'][0]
        self.assertEqual(bridge['kind'],'doors')
        self.assertEqual(bridge['source'],'explicit_user_review')
        self.assertTrue(bridge['requires_review'])
        self.assertEqual(bridge['review']['choice'],'door')
        self.assertEqual(bridge['review']['hinge_and_swing'],'unconfirmed_template_defaults')
        self.assertLess(LineString(bridge['clear_span']).length,LineString(bridge['span']).length)
        self.assertEqual(self.apply(result,s,r,d),result)

    def test_no_answer_does_not_promote_topology_boundary(self):
        g,s,r,d=fixture();d['answers']=[]
        self.assertEqual(self.apply(g,s,r,d),g)

    def test_wrong_hash_stale_gap_duplicate_and_non_door_choice_fail(self):
        for mode in ('hash','geometry','duplicate','open'):
            g,s,r,d=fixture()
            if mode=='hash':d['image_sha256']='b'*64
            elif mode=='geometry':g['diagnostics']['topology_gaps'][0]['span'][0][0]+=1
            elif mode=='duplicate':d['answers']*=2
            else:d['answers'][0]['choice']='open'
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.apply(g,s,r,d)

    def test_user_type_does_not_override_thick_structure_or_missing_jambs(self):
        for mode in ('solid','no_jambs','model_solid'):
            g,s,r,d=fixture()
            if mode=='solid':r[57:64,53:68]=0
            elif mode=='no_jambs':r[:]=255
            else:s['walls'].append(dict(outer=[[48,57],[72,57],[72,63],[48,63]]))
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.apply(g,s,r,d)

    def test_exclusion_existing_opening_and_crossing_remain_blocking(self):
        for mode in ('excluded','opening','crossing'):
            g,s,r,d=fixture()
            if mode=='excluded':g['diagnostics']['excluded_regions']=[[[55,50],[65,50],[65,70],[55,70]]]
            elif mode=='opening':g['diagnostics']['opening_bridges']=[dict(kind='windows',span=[[53,60],[67,60]])]
            else:
                g['nodes'].extend([dict(id=8,point=[60,50],adjacency=[]),dict(id=9,point=[60,70],adjacency=[])])
                g['edges'].append(dict(start=8,end=9,thickness_px=6))
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.apply(g,s,r,d)

    def test_unbounded_area_cannot_be_partitioned_by_this_review_mode(self):
        g,s,r,d=fixture();g['edges'].pop(0)
        with self.assertRaises(ValueError):self.apply(g,s,r,d)

    def test_bad_source_dimensions_are_rejected(self):
        g,s,r,d=fixture();s['meta']['width']=99
        with self.assertRaises(ValueError):self.apply(g,s,r,d)


if __name__=='__main__':unittest.main()
