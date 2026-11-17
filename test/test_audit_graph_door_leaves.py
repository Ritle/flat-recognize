import copy
from pathlib import Path
import sys
import unittest

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from audit_graph_door_leaves import audit_existing_graph_doors
from test.test_review_graph_passages import fixture


class ExistingDoorAuditTest(unittest.TestCase):
    def test_uses_existing_doors_as_candidates_without_mutation(self):
        graph,source,rgb,_=fixture()
        graph['diagnostics']['opening_bridges'][0]['kind']='doors'
        cv2.line(rgb,(80,84),(110,105),(0,0,0),2)
        before=copy.deepcopy((graph,source))
        rows=audit_existing_graph_doors(graph,source,rgb)
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]['original_bridge_index'],0)
        self.assertTrue(rows[0]['accepted'])
        self.assertEqual((graph,source),before)

    def test_audit_does_not_treat_windows_as_confirmed_doors(self):
        graph,source,rgb,_=fixture()
        self.assertEqual(audit_existing_graph_doors(graph,source,rgb),[])

    def test_no_edges_rejected(self):
        graph,source,rgb,_=fixture();graph['edges']=[]
        with self.assertRaises(ValueError):audit_existing_graph_doors(graph,source,rgb)

    def test_cleanup_requires_current_free_raster_and_jambs(self):
        for mode in ('valid','solid','no_jamb','no_provenance'):
            graph,source,rgb,_=fixture()
            bridge=graph['diagnostics']['opening_bridges'][0]
            bridge.update(kind='doors',clear_span=[[80,80],[120,80]],carrier_cleanup='raster_confirmed_blank_aperture')
            source['walls'].append(dict(outer=[[80,76],[120,76],[120,84],[80,84]]))
            if mode=='solid':cv2.line(rgb,(80,80),(120,80),(0,0,0),8)
            elif mode=='no_jamb':rgb[72:89,70:80]=255
            elif mode=='no_provenance':bridge.pop('carrier_cleanup')
            rows=audit_existing_graph_doors(graph,source,rgb)
            with self.subTest(mode=mode):
                self.assertEqual(rows[0]['wall_evidence'],
                                 'raster_rechecked_previously_cleaned_carrier' if mode=='valid' else 'original_model_walls')
                self.assertEqual('raster_verified_clear_span' in rows[0],mode=='valid')


if __name__=='__main__':unittest.main()
