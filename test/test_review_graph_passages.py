import copy
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from review_graph_passages import apply_passage_reviews, passage_fingerprint


def fixture():
    points=[(20,20),(210,20),(210,80),(210,150),(20,150),(20,80)]
    pairs=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(5,2)]
    graph=dict(coordinate_space='source_raster',scale_cm_per_px=2,
               nodes=[dict(id=i,point=list(p),adjacency=[]) for i,p in enumerate(points)],
               edges=[dict(start=a,end=b,thickness_px=8) for a,b in pairs],faces=[],
               diagnostics=dict(opening_bridges=[dict(kind='windows',opening_id=1,span=[[80,80],[120,80]])]))
    rgb=np.full((180,240,3),255,np.uint8)
    walls=[]
    spans=[(points[a],points[b]) for a,b in pairs[:-1]]+[((20,80),(80,80)),((120,80),(210,80))]
    for a,b in spans:
        poly=LineString([a,b]).buffer(4,cap_style=2)
        cv2.fillPoly(rgb,[np.rint(poly.exterior.coords).astype(np.int32)],(0,0,0))
        walls.append(dict(outer=list(poly.exterior.coords)))
    source=dict(meta=dict(width=240,height=180),walls=walls,
                windows=[dict(id=1,outer=[[80,76],[120,76],[120,84],[80,84]])])
    review=dict(version=1,image_sha256='a'*64,answers=[dict(
        fingerprint=passage_fingerprint(graph['diagnostics']['opening_bridges'][0]),choice='open_passage',
        reviewer='assistant_visual_review',basis='Explicit visual review of this raster')])
    return graph,source,rgb,review


class PassageReviewTest(unittest.TestCase):
    def apply(self,g,s,r,d):
        return apply_passage_reviews(g,s,r,d,'a'*64)

    def test_review_changes_only_type_and_keeps_audit_trail(self):
        g,s,r,d=fixture();before=copy.deepcopy(g)
        result=self.apply(g,s,r,d)
        b=result['diagnostics']['opening_bridges'][0]
        self.assertEqual(b['kind'],'passages')
        self.assertEqual(b['passage_review']['reviewer'],'assistant_visual_review')
        self.assertEqual(b['passage_review']['previous_kind'],'windows')
        for key in ('nodes','edges','faces'):
            self.assertEqual(result[key],g[key])
        self.assertEqual(g,before)
        self.assertEqual(self.apply(result,s,r,d),result)

    def test_no_choice_does_not_convert_interior_window(self):
        g,s,r,d=fixture();d['answers']=[]
        self.assertEqual(self.apply(g,s,r,d),g)

    def test_wrong_image_stale_geometry_duplicate_or_unattributed_review_rejected(self):
        for mode in ('hash','geometry','duplicate','reviewer','choice'):
            g,s,r,d=fixture()
            if mode=='hash':d['image_sha256']='b'*64
            elif mode=='geometry':g['diagnostics']['opening_bridges'][0]['span'][0][0]+=1
            elif mode=='duplicate':d['answers']*=2
            elif mode=='reviewer':d['answers'][0].pop('reviewer')
            else:d['answers'][0]['choice']='wall'
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.apply(g,s,r,d)

    def test_solid_wall_window_frame_and_door_leaf_contradict_passage(self):
        for mode in ('solid','frame','leaf'):
            g,s,r,d=fixture()
            if mode=='solid':cv2.line(r,(80,80),(120,80),(0,0,0),8)
            elif mode=='frame':
                cv2.line(r,(80,76),(120,76),(0,0,0),1)
                cv2.line(r,(80,84),(120,84),(0,0,0),1)
            else:cv2.line(r,(80,81),(112,102),(0,0,0),2)
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.apply(g,s,r,d)

    def test_exterior_excluded_and_ambiguous_source_rejected(self):
        for mode in ('exterior','excluded','source','dimensions'):
            g,s,r,d=fixture()
            if mode=='exterior':
                g['diagnostics']['opening_bridges'][0]['span']=[[20,35],[20,65]]
                d['answers'][0]['fingerprint']=passage_fingerprint(g['diagnostics']['opening_bridges'][0])
            elif mode=='excluded':g['diagnostics']['excluded_regions']=[[[75,75],[125,75],[125,85],[75,85]]]
            elif mode=='source':s['windows']*=2
            else:s['meta']['width']=1
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.apply(g,s,r,d)


if __name__=='__main__':unittest.main()
