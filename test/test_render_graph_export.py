import copy
import json
from pathlib import Path
import sys
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from build_graph_project import build_graph_project
from render_graph_export import render_graph_export
from test.test_build_graph_project import graph


class RenderGraphExportTest(unittest.TestCase):
    def test_renders_serialized_door_and_passage_holes_in_source_pixels(self):
        load = lambda name: json.loads((ROOT / 'test' / name).read_text(encoding='utf8'))
        source = graph(openings=[dict(kind='doors', span=[[20,0],[50,0]]),
                                 dict(kind='passages', span=[[100,30],[100,70]])])
        project, report = build_graph_project(source, load('template.json'), load('template_window.json'),
            origin=(100,-50), passage_template=load('template_passage.json'))
        image = Image.new('RGB', (120,120), 'white')
        before = copy.deepcopy((project, report))
        overlay, geometry = render_graph_export(image, project, report)
        self.assertEqual(geometry['doors'][0]['span'], [[20,0],[50,0]])
        self.assertEqual(geometry['passages'][0]['span'], [[100,30],[100,70]])
        self.assertEqual(overlay.getpixel((35,0)), (0,224,112))
        self.assertEqual(overlay.getpixel((100,50)), (255,0,207))
        self.assertEqual(image.getpixel((35,0)), (255,255,255))
        self.assertEqual((project, report), before)
        invalid = copy.deepcopy(project)
        invalid['levels'][0]['elements'][0]['wall']['position'] = 0
        with self.assertRaises(ValueError):
            render_graph_export(image, invalid, report)


if __name__ == '__main__':
    unittest.main()
