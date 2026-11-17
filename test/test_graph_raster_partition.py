import json
from pathlib import Path
import sys
import unittest

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from graph_review_questions import apply_graph_answers, make_graph_questions
from test.test_review_solid_wall_door import fixture


def panel_fixture(mode):
    graph,source,rgb,_ = fixture()
    graph['scale_cm_per_px'] = 3.9
    rgb[:] = 255
    cv2.rectangle(rgb,(10,10),(110,110),(0,0,0),6)
    for x in (47,74):
        cv2.rectangle(rgb,(x-3,56),(x+3,63),(0,0,0),-1)
    for y in (57,62):
        cv2.line(rgb,(10,y),(47,y),(0,0,0))
        if mode!='one_flank':
            cv2.line(rgb,(74,y),(110,y),(0,0,0))
        if mode=='continuous_frame':
            cv2.line(rgb,(47,y),(74,y),(0,0,0))
    if mode!='no_leaf':
        cv2.line(rgb,(71,63),(53,74),(0,0,0),2)
    if mode=='excluded':
        graph['diagnostics']['excluded_regions']=[[[28,52],[35,52],[35,66],[28,66]]]
    if mode=='rotated':
        rgb=np.transpose(rgb,(1,0,2)).copy()
        for node in graph['nodes']:node['point'].reverse()
    return graph,source,rgb


class RasterPartitionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.templates=[json.loads((ROOT/'test'/f).read_text()) for f in ('template.json','template_window.json')]

    def test_measured_wall_requires_matching_explicit_choice(self):
        for mode in ('normal','rotated','one_flank','continuous_frame','no_leaf','excluded'):
            g,s,r = panel_fixture(mode)
            inputs = (g,s,r,'a'*64,*self.templates)
            questions = make_graph_questions(*inputs)['questions']
            with self.subTest(mode=mode):
                if mode not in ('normal','rotated'):
                    self.assertEqual(questions,[])
                    continue
                question, = questions
                self.assertEqual(question['kind'],'partition')
                answers=dict(version=1,image_sha256='a'*64,answers=[dict(id=question['id'],choice='undecided')])
                self.assertEqual(apply_graph_answers(*inputs,answers),g)
                answers['answers'][0]['choice']='solid_wall_with_door'
                updated=apply_graph_answers(*inputs,answers)
                bridge, = updated['diagnostics']['opening_bridges']
                self.assertEqual(bridge['review']['width_source'],'raster_jambs')
                self.assertEqual(len(updated['faces']),2)
                self.assertEqual(make_graph_questions(updated,*inputs[1:])['questions'],[])
                with self.assertRaises(ValueError):
                    apply_graph_answers(updated,*inputs[1:],answers)
