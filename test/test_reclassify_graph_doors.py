import copy
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from reclassify_graph_doors import reclassify_graph_doors


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
