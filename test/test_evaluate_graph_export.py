import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from build_graph_project import build_graph_project
from evaluate_graph_export import evaluate_graph_export
from test.test_build_graph_project import graph


class GraphExportEvaluationTest(unittest.TestCase):
    def test_evaluates_actual_hole_and_area_without_mutation(self):
        data=graph(openings=[dict(kind='doors',span=[[10,0],[90,0]],clear_span=[[30,0],[60,0]])])
        template=json.loads((ROOT/'test/template.json').read_text());windows=json.loads((ROOT/'test/template_window.json').read_text())
        project,report=build_graph_project(data,template,windows,origin=(100,-50))
        truth=dict(version=1,image=dict(width=120,height=120),coverage=dict(rooms='geometry',doors='geometry'),
                   rooms=[dict(id='r',polygon=[[0,0],[100,0],[100,100],[0,100]])],
                   doors=[dict(id='d',span=[[30,0],[60,0]])])
        before=copy.deepcopy((project,report,truth));result=evaluate_graph_export(truth,project,report)
        self.assertEqual(result['metrics']['rooms']['true_positive'],1)
        self.assertEqual(result['metrics']['doors']['true_positive'],1)
        self.assertAlmostEqual(result['metrics']['doors']['matches'][0]['length_error_px'],0,places=2)
        self.assertEqual((project,report,truth),before)

    def test_invalid_hole_or_adapter_rejected(self):
        data=graph(openings=[dict(kind='doors',span=[[20,0],[60,0]])])
        template=json.loads((ROOT/'test/template.json').read_text());windows=json.loads((ROOT/'test/template_window.json').read_text())
        project,report=build_graph_project(data,template,windows)
        truth=dict(version=1,image=dict(width=120,height=120),coverage=dict(doors='geometry'),doors=[])
        bad=copy.deepcopy(project);bad['levels'][0]['elements'][0]['wall']['position']=0
        with self.assertRaises(ValueError):evaluate_graph_export(truth,bad,report)
        report['adapter']='bbox'
        with self.assertRaises(ValueError):evaluate_graph_export(truth,project,report)


if __name__=='__main__':unittest.main()
