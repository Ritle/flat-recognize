import copy
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reclassify_graph_doors import connected_leaf_ink, reclassify_graph_doors


def fixture(double=False, mode='valid'):
    rgb=np.full((180,240,3),255,np.uint8)
    a,b=(80,80),(140 if double else 120,80)
    cv2.rectangle(rgb,(20,76),(80,84),(0,0,0),-1)
    cv2.rectangle(rgb,(b[0],76),(210,84),(0,0,0),-1)
    color=(190,190,190) if mode=='pale' else (0,0,0)
    shift=15 if mode=='floating' else 0
    if double:
        cv2.line(rgb,(80,84+shift),(104,101+shift),color,2)
        if mode!='half':
            cv2.line(rgb,(140,84+shift),(116,101+shift if mode!='opposite' else 59),color,2)
    elif mode=='backwards':
        cv2.line(rgb,(80,84),(50,105),color,2)
    elif mode=='frame':
        cv2.line(rgb,(80,77),(120,77),color,1)
        cv2.line(rgb,(80,83),(120,83),color,1)
    else:
        cv2.line(rgb,(80,84+shift),(110,105+shift),color,8 if mode=='broad' else 2)
    if mode=='solid':
        cv2.line(rgb,a,b,(0,0,0),8)
    if mode=='no_jamb':
        rgb[70:91,:83]=255
    source=dict(meta=dict(width=240,height=180),windows=[dict(id=1,source_class='door',
                outer=[[80,76],[b[0],76],[b[0],84],[80,84]])])
    graph=dict(coordinate_space='source_raster',nodes=[],edges=[],faces=[],
               diagnostics=dict(opening_bridges=[dict(kind='windows',opening_id=1,span=[list(a),list(b)])]))
    return graph,source,rgb


class GraphDoorTypesTest(unittest.TestCase):
    def test_leaf_ink_cannot_connect_via_a_neighbouring_stroke(self):
        gray = np.full((40,50),255,np.uint8)
        cv2.line(gray,(5,15),(35,15),0,1)
        self.assertTrue(connected_leaf_ink(gray,(5,15),(35,15)))
        gray[15,14:17] = 255
        cv2.polylines(gray,[np.array([(5,15),(5,25),(35,25),(35,15)],np.int32)],False,0,1)
        self.assertFalse(connected_leaf_ink(gray,(5,15),(35,15)))

    def test_exterior_jamb_beyond_frame_rail_and_negative_controls(self):
        for mode in ('outward', 'reverse', 'rotated', 'inward', 'attached_furniture',
                     'detached', 'open_contour', 'confirmed_window', 'solid'):
            graph, source, rgb = fixture()
            bridge = graph['diagnostics']['opening_bridges'][0]
            points = [[10,10],[200,10],[200,80],[10,80]]
            graph.update(nodes=[dict(id=i,point=p) for i,p in enumerate(points)],
                         edges=[dict(start=i,end=(i+1)%4) for i in range(4)], scale_cm_per_px=2)
            rgb[:] = 255
            source['walls'] = []
            for x1,x2 in ((10,80),(120,200)):
                cv2.rectangle(rgb,(x1,75),(x2,85),(0,0,0),-1)
                source['walls'].append(dict(outer=[[x1,75],[x2,75],[x2,85],[x1,85]]))
            source['windows'][0]['outer'] = [[80,75],[120,75],[120,85],[80,85]]
            cv2.line(rgb,(80,80),(120,80),(0,0,0),10 if mode=='solid' else 2)
            hinge,tip = (80,85),(112,105)
            if mode=='inward': hinge,tip=(80,75),(112,55)
            if mode=='detached': hinge=(88,90)
            cv2.line(rgb,hinge,tip,(0,0,0),2)
            if mode=='attached_furniture':cv2.line(rgb,tip,(100,125),(0,0,0),2)
            if mode=='confirmed_window':bridge['window_review']=dict(choice='window')
            if mode=='open_contour':graph['edges'].pop()
            if mode=='reverse':bridge['span'].reverse()
            if mode=='rotated':
                rgb=np.transpose(rgb,(1,0,2)).copy()
                source['meta']=dict(width=rgb.shape[1],height=rgb.shape[0])
                for node in graph['nodes']:node['point'].reverse()
                for point in bridge['span']:point.reverse()
                for shape in source['windows']+source['walls']:
                    for point in shape['outer']:point.reverse()
            result,_=reclassify_graph_doors(graph,source,rgb)
            updated=result['diagnostics']['opening_bridges'][0]
            with self.subTest(mode=mode):
                if mode in ('outward','reverse','rotated'):
                    self.assertEqual(updated['type_review']['evidence']['method'],'outward_leaf_at_exterior_jamb')
                    self.assertEqual(updated['span'],bridge['span'])
                    self.assertEqual(updated['kind'],'doors')
                else:self.assertEqual(result,graph)

    def test_single_and_paired_leaves_change_only_type_with_provenance(self):
        for double in (False,True):
            with self.subTest(double=double):
                graph,source,rgb=fixture(double)
                before=copy.deepcopy(graph);source_before=copy.deepcopy(source)
                result,report=reclassify_graph_doors(graph,source,rgb)
                self.assertEqual(sum(r['accepted'] for r in report),1)
                bridge=result['diagnostics']['opening_bridges'][0]
                self.assertEqual(bridge['kind'],'doors')
                self.assertEqual(bridge['type_review']['evidence']['leaf_count'],2 if double else 1)
                self.assertEqual(bridge['span'],before['diagnostics']['opening_bridges'][0]['span'])
                for key in ('nodes','edges','faces'):
                    self.assertEqual(result[key],before[key])
                self.assertEqual(graph,before);self.assertEqual(source,source_before)
                self.assertEqual(reclassify_graph_doors(result,source,rgb)[0],result)

    def test_vertical_door(self):
        graph,source,rgb=fixture()
        rgb=np.transpose(rgb,(1,0,2)).copy()
        source['meta']=dict(width=180,height=240)
        source['windows'][0]['outer']=[p[::-1] for p in source['windows'][0]['outer']]
        graph['diagnostics']['opening_bridges'][0]['span']=[p[::-1] for p in graph['diagnostics']['opening_bridges'][0]['span']]
        self.assertTrue(reclassify_graph_doors(graph,source,rgb)[1][0]['accepted'])

    def test_invalid_single_leaf_evidence_does_not_promote_window(self):
        for mode in ('pale','floating','solid','no_jamb','backwards','frame','broad'):
            with self.subTest(mode=mode):
                graph,source,rgb=fixture(mode=mode)
                self.assertEqual(reclassify_graph_doors(graph,source,rgb)[0],graph)

    def test_double_requires_two_inward_leaves_on_same_side(self):
        for mode in ('half','opposite','floating','pale'):
            with self.subTest(mode=mode):
                graph,source,rgb=fixture(double=True,mode=mode)
                self.assertEqual(reclassify_graph_doors(graph,source,rgb)[0],graph)

    def test_unique_model_door_and_nonexcluded_gap_required(self):
        for mode in ('window','duplicate','missing','excluded'):
            with self.subTest(mode=mode):
                graph,source,rgb=fixture()
                if mode=='window':source['windows'][0]['source_class']='window'
                elif mode=='duplicate':source['windows']*=2
                elif mode=='missing':source['windows']=[]
                else:graph['diagnostics']['excluded_regions']=[[[70,70],[150,70],[150,90],[70,90]]]
                self.assertEqual(reclassify_graph_doors(graph,source,rgb)[0],graph)

    def test_dimensions_and_coordinate_space_checked(self):
        for mode in ('dimensions','source','graph'):
            graph,source,rgb=fixture()
            if mode=='dimensions':source['meta']['width']=10
            elif mode=='source':source['meta']['coordinate_space']='model'
            else:graph['coordinate_space']='model'
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                reclassify_graph_doors(graph,source,rgb)


if __name__=='__main__':unittest.main()
