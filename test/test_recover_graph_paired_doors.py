import copy
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from recover_graph_paired_doors import recover_graph_paired_doors
from review_graph_door_branches import apply_branch_door_review


def fixture(mode):
    points = [[15,15],[87,15],[175,15],[175,145],[87,145],[15,145],
              [87,55],[87,101],[92,58],[91,96]]
    pairs = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(1,6),(4,7)]
    rgb = np.full((160,190,3),255,np.uint8)
    walls = []
    for a,b in pairs:
        shape = LineString([points[a],points[b]]).buffer(3,cap_style=3)
        walls.append(dict(outer=list(shape.exterior.coords)))
        cv2.fillPoly(rgb,[np.rint(shape.exterior.coords).astype(np.int32)],(0,0,0))
    upper = (100,75) if mode=='close_tips' else (102,73)
    lower = (100,81) if mode=='close_tips' else (102,83)
    cv2.line(rgb,(90,58),upper,(0,0,0),2)
    if mode!='single':
        cv2.line(rgb,(90,98),lower,(0,0,0),2)
    if mode=='furniture':
        cv2.line(rgb,upper,(115,66),(0,0,0),2)
    if mode=='solid':
        cv2.line(rgb,(87,55),(87,101),(0,0,0),6)
    if mode=='opposite_branch':
        points[8][0] = 82
    if mode=='open_contour':
        pairs.remove((2,3))
    pairs.extend([(6,8),(7,9)])
    if mode=='rotated':
        rgb = np.transpose(rgb,(1,0,2)).copy()
        points = [p[::-1] for p in points]
        for wall in walls:
            wall['outer'] = [p[::-1] for p in wall['outer']]
    nodes = [dict(id=i,point=p,adjacency=[]) for i,p in enumerate(points)]
    edges = []
    for a,b in pairs:
        nodes[a]['adjacency'].append(b)
        nodes[b]['adjacency'].append(a)
        edges.append(dict(start=a,end=b,thickness_px=6))
    diagnostics = dict(topology_gaps=[],opening_bridges=[])
    if mode=='other_opening':
        diagnostics['opening_bridges'].append(dict(kind='windows',span=[points[6],points[8]]))
    if mode=='excluded':
        diagnostics['excluded_regions'] = [[[85,67],[94,67],[94,74],[85,74]]]
    if mode=='closure':
        diagnostics['topology_gaps'].append(dict(span=[points[8],[110,59]],local_width_px=6))
    return (dict(coordinate_space='source_raster',scale_cm_per_px=3,nodes=nodes,edges=edges,
                 diagnostics=diagnostics,faces=[]),
            dict(meta=dict(width=rgb.shape[1],height=rgb.shape[0]),walls=walls),rgb)


class PairedBranchDoorTest(unittest.TestCase):
    def test_previous_visual_review_attaches_to_the_same_recovered_door(self):
        graph,source,rgb = fixture('paired')
        result,rows = recover_graph_paired_doors(graph,source,rgb)
        review = dict(version=1,image_sha256='a'*64,reviewer='assistant_visual_review',
                      basis='Visible paired leaves',span=rows[0]['span'][::-1],local_width_px=6,
                      visible_leaf_count=2,remove_dangling_branches=rows[0]['branches'][::-1])
        reviewed = apply_branch_door_review(result,source,rgb,review,'a'*64)
        bridge, = reviewed['diagnostics']['opening_bridges']
        self.assertEqual(bridge['review']['basis'],review['basis'])
        self.assertEqual(bridge['source'],'paired_raster_leaf_branches')
        self.assertEqual(reviewed['edges'],result['edges'])
        self.assertEqual(apply_branch_door_review(reviewed,source,rgb,review,'a'*64),reviewed)
        review['remove_dangling_branches'] = [rows[0]['branches'][0]]
        with self.assertRaises(ValueError):
            apply_branch_door_review(result,source,rgb,review,'a'*64)

    def test_recovery_and_protected_branches(self):
        for mode in ('paired','rotated','single','furniture','solid','opposite_branch',
                     'open_contour','other_opening','excluded','closure','close_tips'):
            graph,source,rgb = fixture(mode)
            before = copy.deepcopy(graph)
            result,rows = recover_graph_paired_doors(graph,source,rgb)
            with self.subTest(mode=mode):
                self.assertEqual(graph,before)
                if mode in ('paired','rotated'):
                    self.assertEqual(sum(row['accepted'] for row in rows),1,rows)
                    bridge, = result['diagnostics']['opening_bridges']
                    self.assertEqual(bridge['source'],'paired_raster_leaf_branches')
                    self.assertEqual(bridge['type_review']['evidence']['leaf_count'],2)
                    self.assertNotIn('review',bridge)
                    self.assertEqual(len(result['faces']),2)
                    self.assertEqual(len(result['nodes']),len(graph['nodes'])-2)
                    self.assertEqual(recover_graph_paired_doors(result,source,rgb)[0],result)
                else:
                    self.assertEqual(result,before)
