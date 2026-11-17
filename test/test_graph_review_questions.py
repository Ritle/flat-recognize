import copy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from graph_review_questions import make_graph_questions, apply_graph_answers, draw_graph_questions
from test.test_review_graph_doors import fixture as gap_fixture
from test.test_review_graph_passages import fixture as passage_fixture


class GraphReviewQuestionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template=json.loads((ROOT/'test/template.json').read_text())
        cls.windows=json.loads((ROOT/'test/template_window.json').read_text())

    def questions(self,g,s,r):return make_graph_questions(g,s,r,'a'*64,self.template,self.windows)
    def apply(self,g,s,r,answers):return apply_graph_answers(g,s,r,'a'*64,self.template,self.windows,answers)

    def test_uncertain_gap_question_and_explicit_door(self):
        g,s,r,_=gap_fixture();g['diagnostics']['topology_gaps'][0]['reason']='interior_raster_line_without_confirmed_type'
        before=copy.deepcopy((g,s));questions=self.questions(g,s,r)['questions']
        self.assertEqual(len(questions),1)
        self.assertEqual(questions[0]['choices'],['door','undecided'])
        from build_graph_project import build_graph_project
        _,report=build_graph_project(g,self.template,self.windows)
        self.assertIn('graph_unresolved_internal_gap',[i['code'] for i in report['quality']['issues']])
        self.assertEqual((g,s),before)
        unanswered=dict(version=1,image_sha256='a'*64,answers=[dict(id=questions[0]['id'],choice='undecided')])
        self.assertEqual(self.apply(g,s,r,unanswered),g)
        answered=dict(version=1,image_sha256='a'*64,answers=[dict(id=questions[0]['id'],choice='door',user_statement='door')])
        result=self.apply(g,s,r,answered)
        self.assertEqual(result['diagnostics']['opening_bridges'][0]['kind'],'doors')
        self.assertEqual(self.questions(result,s,r)['questions'],[])
        _,report=build_graph_project(result,self.template,self.windows)
        self.assertNotIn('graph_unresolved_internal_gap',[i['code'] for i in report['quality']['issues']])

    def test_interior_window_question_can_be_passage_window_or_door(self):
        for choice in ('open_passage','window','door','undecided'):
            g,s,r,_=passage_fixture();q=self.questions(g,s,r)['questions']
            self.assertEqual(len(q),1)
            self.assertIn(choice,q[0]['choices'])
            answer=dict(version=1,image_sha256='a'*64,answers=[dict(id=q[0]['id'],choice=choice)])
            result=self.apply(g,s,r,answer);bridge=result['diagnostics']['opening_bridges'][0]
            with self.subTest(choice=choice):
                self.assertEqual(bridge['kind'],{'door':'doors','open_passage':'passages','window':'windows','undecided':'windows'}[choice])
                if choice=='window':self.assertEqual(bridge['window_review']['choice'],'window')
                if choice=='undecided':self.assertEqual(result,g)

    def test_stale_duplicate_wrong_hash_unsupported_and_invalid_source_rejected(self):
        for mode in ('hash','stale','duplicate','choice','dimensions'):
            g,s,r,_=gap_fixture();g['diagnostics']['topology_gaps'][0]['reason']='interior_raster_line_without_confirmed_type'
            q=self.questions(g,s,r)['questions'][0]
            answers=dict(version=1,image_sha256='a'*64,answers=[dict(id=q['id'],choice='door')])
            if mode=='hash':answers['image_sha256']='b'*64
            elif mode=='stale':g['diagnostics']['topology_gaps'][0]['span'][0][0]+=1
            elif mode=='duplicate':answers['answers']*=2
            elif mode=='choice':answers['answers'][0]['choice']='open_passage'
            else:s['meta']['width']=99
            with self.subTest(mode=mode),self.assertRaises(ValueError):self.apply(g,s,r,answers)

    def test_unreviewable_gaps_not_presented_as_actionable(self):
        g,s,r,_=gap_fixture();g['diagnostics']['topology_gaps'][0]['reason']='interior_raster_line_without_confirmed_type'
        r[57:64,53:68]=0
        self.assertEqual(self.questions(g,s,r)['questions'],[])

    def test_exterior_window_with_leaf_is_a_question_not_an_automatic_door(self):
        for choice in ('door', 'window', 'undecided'):
            graph, source, rgb, _ = passage_fixture()
            cv2.line(rgb, (80,84), (110,105), (0,0,0), 2)
            report = {'bindings': {'windows': [{'bridge_index': 0, 'exterior_context': {'status': 'exterior'}}]}}
            before = copy.deepcopy(graph)
            with patch('graph_review_questions.build_graph_project', return_value=({}, report)):
                questions = self.questions(graph, source, rgb)['questions']
                self.assertEqual(len(questions), 1)
                self.assertEqual(questions[0]['reason'], 'possible_door_leaf_near_window')
                self.assertEqual(questions[0]['choices'], ['door', 'window', 'undecided'])
                self.assertEqual(graph, before)
                payload = {'version': 1, 'image_sha256': 'a'*64,
                           'answers': [{'id': questions[0]['id'], 'choice': choice}]}
                result = self.apply(graph, source, rgb, payload)
                bridge = result['diagnostics']['opening_bridges'][0]
                self.assertEqual(bridge['kind'], 'doors' if choice == 'door' else 'windows')
                if choice == 'window': self.assertEqual(self.questions(result, source, rgb)['questions'], [])
                if choice == 'undecided': self.assertEqual(result, before)

    def test_exterior_furniture_like_stroke_remains_a_question(self):
        graph, source, rgb, _ = passage_fixture()
        for a, b in zip([(80,84),(110,105),(100,120),(70,99)],
                        [(110,105),(100,120),(70,99),(80,84)]):
            cv2.line(rgb, a, b, (0,0,0), 2)
        report = {'bindings': {'windows': [{'bridge_index': 0, 'exterior_context': {'status': 'exterior'}}]}}
        before = copy.deepcopy(graph)
        with patch('graph_review_questions.build_graph_project', return_value=({}, report)):
            questions = self.questions(graph, source, rgb)['questions']
        self.assertEqual(len(questions), 1)
        self.assertTrue(questions[0]['leaf_context']['near_tip_cross_strokes'])
        self.assertEqual(graph, before)

    def test_fragmented_near_jamb_stroke_is_flagged_with_review_provenance(self):
        graph, source, rgb, _ = passage_fixture()
        cv2.line(rgb, (87,84), (115,104), (0,0,0), 2)
        report = {'bindings': {'windows': [{'bridge_index': 0, 'exterior_context': {'status': 'exterior'}}]}}
        before = copy.deepcopy(graph)
        with patch('graph_review_questions.build_graph_project', return_value=({}, report)):
            questions = self.questions(graph, source, rgb)['questions']
        self.assertEqual(len(questions), 1)
        self.assertEqual(questions[0]['leaf_method'], 'model_jambs_and_nearby_diagonal_review_hint')
        self.assertEqual(questions[0]['choices'], ['door', 'window', 'undecided'])
        self.assertEqual(graph, before)

    def test_diagnostic_leaf_is_visible_on_questions_overlay(self):
        rgb = np.full((60,60,3), 255, dtype=np.uint8)
        q = {'questions': [{'kind':'window', 'span':[[10,10],[40,10]],
                            'leaf_evidence':{'hinge':[10,10], 'tip':[40,40]}}]}
        overlay = draw_graph_questions(rgb, q)
        self.assertEqual(overlay.getpixel((25,25)), (0,0,255))


if __name__=='__main__':unittest.main()
