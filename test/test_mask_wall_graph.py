import sys
import unittest
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import LineString
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'scripts'))
from mask_wall_graph import build_graph,node_lines

class MaskGraphTests(unittest.TestCase):
    def test_x_intersection_is_noded(self):
        edges=node_lines([LineString([(10,50),(90,50)]),LineString([(50,10),(50,90)])],2)
        self.assertEqual(len(edges),4)
        self.assertTrue(all((50,50) in list(e.coords) for e in edges))

    def test_diagonal_retains_angle(self):
        edges=node_lines([LineString([(10,10),(90,70)])],2)
        self.assertEqual(list(edges[0].coords),[(10,10),(90,70)])

    def test_t_attachment_and_endpoint_snap_share_a_node(self):
        edges=node_lines([LineString([(10,50),(90,50)]),LineString([(50,10),(50,48)])],3)
        self.assertEqual(len(edges),3)
        self.assertTrue(all((50,50) in list(e.coords) for e in edges))

    def test_snapping_does_not_swap_close_endpoints(self):
        edges=node_lines([LineString([(10,20),(49,20)]),LineString([(51,20),(90,20)])],3)
        endpoints=[tuple(p) for edge in edges for p in edge.coords]
        self.assertEqual(endpoints.count((50,20)),2)

    def test_room_independent_and_explicit_exclusion(self):
        rgb=np.full((120,120,3),255,np.uint8)
        cv2.rectangle(rgb,(20,20),(100,100),(0,0,0),5)
        source=dict(meta=dict(width=120,height=120),rooms=[],walls=[dict(outer=[[18,18],[102,18],[102,102],[18,102]],holes=[[[23,23],[97,23],[97,97],[23,97]]])])
        graph=build_graph(source,rgb)
        self.assertGreater(len(graph['edges']),0)
        self.assertEqual(len(graph['faces']),1)
        empty=build_graph(source,rgb,excluded_regions=[[[0,0],[119,0],[119,119],[0,119]]])
        self.assertEqual(empty['edges'],[])

    def test_segmentation_without_raster_support_is_rejected(self):
        rgb=np.full((100,100,3),255,np.uint8)
        source=dict(meta=dict(width=100,height=100),walls=[dict(outer=[[10,10],[90,10],[90,20],[10,20]])])
        graph=build_graph(source,rgb)
        self.assertEqual(graph['edges'],[])
        self.assertTrue(graph['diagnostics']['rejected'])

if __name__=='__main__':unittest.main()
