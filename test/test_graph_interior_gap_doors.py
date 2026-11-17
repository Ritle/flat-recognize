import copy
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recover_interior_gap_doors import recover_interior_gap_doors
from review_graph_doors import apply_graph_door_decisions, gap_fingerprint


def fixture(mode):
    points = [[20,20],[180,20],[180,80],[180,140],[20,140],[20,80],[75,80],[110,80]]
    pairs = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(5,6),(7,2)]
    nodes = [dict(id=i,point=p,adjacency=[]) for i,p in enumerate(points)]
    rgb = np.full((160,200,3),255,np.uint8)
    walls, edges = [], []
    for a,b in pairs:
        nodes[a]['adjacency'].append(b)
        nodes[b]['adjacency'].append(a)
        edges.append(dict(start=a,end=b,thickness_px=8))
        shape = LineString([points[a],points[b]]).buffer(4,cap_style=3)
        walls.append(dict(outer=list(shape.exterior.coords)))
        cv2.fillPoly(rgb,[np.rint(shape.exterior.coords).astype(np.int32)],(0,0,0))
    gap = dict(span=[points[6],points[7]],local_width_px=8,reason='interior_raster_line_without_confirmed_type')
    diagnostics = dict(topology_gaps=[gap],opening_bridges=[])
    cv2.line(rgb,(79,83),(106,83),(80,80,80),1)
    start, tip = (79,84), (101,98)
    if mode=='floating': start,tip=(85,91),(107,105)
    if mode=='backwards': tip=(55,98)
    if mode!='blank': cv2.line(rgb,start,tip,(0,0,0),2)
    if mode=='furniture': cv2.line(rgb,tip,(86,118),(0,0,0),2)
    if mode=='neighbour':
        diagnostics['opening_bridges'].append(dict(kind='doors',span=[[79,84],[79,110]]))
    if mode=='other_gap':
        diagnostics['topology_gaps'].append(dict(span=[[79,84],[79,110]],local_width_px=8))
    if mode=='solid': cv2.line(rgb,(75,80),(110,80),(0,0,0),8)
    if mode=='frame': cv2.line(rgb,(79,77),(106,77),(0,0,0),1)
    if mode=='exclusion': diagnostics['excluded_regions']=[[[89,72],[95,72],[95,85],[89,85]]]
    if mode=='open_contour': edges.pop(0)
    if mode=='rotated':
        rgb = np.transpose(rgb,(1,0,2)).copy()
        for p in points: p.reverse()
        for wall in walls: wall['outer']=[p[::-1] for p in wall['outer']]
    graph = dict(nodes=nodes,edges=edges,faces=[],scale_cm_per_px=3.9,
                 coordinate_space='source_raster',diagnostics=diagnostics)
    source = dict(meta=dict(width=rgb.shape[1],height=rgb.shape[0]),walls=walls)
    return graph,source,rgb


class InteriorGapDoorTest(unittest.TestCase):
    def test_attached_leaf_resolves_only_its_own_bounded_gap(self):
        for mode in ('normal','rotated','blank','floating','backwards','furniture','neighbour',
                     'other_gap','solid','frame','exclusion','open_contour'):
            graph,source,rgb = fixture(mode)
            before = copy.deepcopy(graph)
            updated,rows = recover_interior_gap_doors(graph,source,rgb)
            with self.subTest(mode=mode):
                self.assertEqual(graph,before)
                if mode not in ('normal','rotated'):
                    self.assertEqual(updated,graph)
                    continue
                self.assertTrue(rows[0]['accepted'],rows)
                bridge, = updated['diagnostics']['opening_bridges']
                self.assertEqual(bridge['source'],'interior_jamb_face_leaf')
                self.assertNotIn('review',bridge)
                self.assertEqual(len(updated['faces']),2)
                self.assertEqual(updated['diagnostics']['topology_gaps'],[])
                self.assertEqual(recover_interior_gap_doors(updated,source,rgb)[0],updated)

    def test_previous_choice_can_be_replayed_only_on_the_same_door(self):
        graph,source,rgb = fixture('normal')
        key = gap_fingerprint(graph['diagnostics']['topology_gaps'][0])
        answers = dict(version=1,image_sha256='a'*64,answers=[dict(gap_fingerprint=key,choice='door')])
        updated,_ = recover_interior_gap_doors(graph,source,rgb)
        reviewed = apply_graph_door_decisions(updated,source,rgb,answers,'a'*64)
        bridge, = reviewed['diagnostics']['opening_bridges']
        self.assertEqual(bridge['source'],'interior_jamb_face_leaf')
        self.assertEqual(bridge['review']['gap_fingerprint'],key)
        self.assertEqual(reviewed['edges'],updated['edges'])
        self.assertEqual(apply_graph_door_decisions(reviewed,source,rgb,answers,'a'*64),reviewed)
        bridge['span'] = [[74,80],[110,80]]
        with self.assertRaises(ValueError):
            apply_graph_door_decisions(reviewed,source,rgb,answers,'a'*64)
