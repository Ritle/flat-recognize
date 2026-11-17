import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from review_export_contacts import outside_regions, render_contacts, review_items


class ReviewExportContactsTest(unittest.TestCase):
    def test_partial_and_full_coverage_select_only_unscored_objects(self):
        def item(uid, x):
            return {'id': uid, 'span': [[x, 10], [x+10, 10]]}
        geometry = {'doors': [item('d1', 5), item('d2', 35)],
                    'windows': [item('w1', 5)], 'passages': [item('p1', 5)]}
        gt = {'coverage': {'doors': 'geometry', 'windows': 'geometry'},
              'annotation_regions': {'doors': [[0, 0, 20, 20]]}}
        rows = review_items(geometry, gt)
        self.assertEqual([(r['uuid'], r['number'], r['coverage']) for r in rows],
                         [('d2', 2, 'outside_regions'), ('p1', 1, 'unannotated')])
        self.assertFalse(outside_regions([[0,0],[20,0]], [[10,0,20,10]]))

    def test_sheet_contains_original_and_actual_span_and_never_overwrites(self):
        rows = [{'kind': 'doors', 'number': 2, 'uuid': 'd2', 'span': [[10,10],[20,10]],
                 'coverage': 'unannotated'}]
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'contacts'
            render_contacts(Image.new('RGB', (80,80), 'white'), rows, target)
            self.assertTrue((target/'contacts-01.png').is_file())
            self.assertEqual(json.loads((target/'contacts.json').read_text())[0]['uuid'], 'd2')
            with self.assertRaises(ValueError):
                render_contacts(Image.new('RGB', (80,80), 'white'), rows, target)


if __name__ == '__main__': unittest.main()
