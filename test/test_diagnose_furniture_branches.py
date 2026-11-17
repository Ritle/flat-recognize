import copy
from pathlib import Path
import sys
import unittest

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_furniture_branches import measure_branches


def fixture():
    points = [(10, 10), (90, 10), (90, 90), (10, 90), (40, 50), (50, 50)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5)]
    graph = dict(coordinate_space='source_raster', diagnostics={},
                 nodes=[dict(id=i, point=p) for i, p in enumerate(points)],
                 edges=[dict(start=a, end=b, thickness_px=6) for a, b in edges])
    raster = Image.new('RGB', (101, 101), 'white')
    draw = ImageDraw.Draw(raster)
    draw.rectangle((10, 10, 90, 90), outline='black', width=6)
    draw.rectangle((40, 50, 50, 60), outline='black')
    return graph, raster


class FurnitureDiagnosisTest(unittest.TestCase):
    def test_isolated_outline_counted_once_and_graph_preserved(self):
        graph, raster = fixture()
        original = copy.deepcopy(graph)
        row, = measure_branches(graph, np.asarray(raster))
        self.assertTrue(row['review_candidate'])
        self.assertTrue(row['isolated'])
        self.assertTrue(row['nearby_closed_details'])
        self.assertEqual(graph, original)

    def test_door_swing_wall_margin_open_room_and_blank(self):
        for case, reason in [('door', 'within_possible_door_swing'), ('corner', 'alongside_solid_ink'),
                             ('open', 'outside_or_incomplete_room'), ('blank', 'missing_raster_stroke')]:
            graph, raster = fixture()
            if case == 'door':
                graph['diagnostics']['opening_bridges'] = [dict(kind='doors', span=[[40,35],[60,35]])]
            elif case == 'corner':
                ImageDraw.Draw(raster).rectangle((30, 25, 60, 49), fill='black')
            elif case == 'open':
                del graph['edges'][1]
            else:
                ImageDraw.Draw(raster).rectangle((30, 40, 60, 65), fill='white')
            rows = measure_branches(graph, np.asarray(raster))
            row = next(r for r in rows if r['path'][0] == (40, 50))
            with self.subTest(case=case):
                self.assertFalse(row['review_candidate'])
                self.assertEqual(row['reason'], reason)
