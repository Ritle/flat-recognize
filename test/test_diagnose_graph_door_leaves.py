import copy
from pathlib import Path
import sys
import unittest

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from diagnose_graph_door_leaves import diagnose_graph_door_leaves
from reclassify_graph_doors import anchored_leaves
from test.test_review_graph_passages import fixture


class DoorLeafDiagnosticTest(unittest.TestCase):
    def test_independent_dark_leaf_and_model_jambs(self):
        graph,source,rgb,_=fixture()
        cv2.line(rgb,(80,84),(110,105),(0,0,0),2)
        before=copy.deepcopy((graph,source))
        rows=diagnose_graph_door_leaves(graph,source,rgb)
        self.assertTrue(rows[0]['accepted'])
        self.assertEqual(rows[0]['candidate_kind'],'opening')
        self.assertTrue(rows[0]['requires_review'])
        self.assertEqual((graph,source),before)

    def test_no_leaf_floating_leaf_solid_wall_and_exclusion(self):
        for mode in ('blank','floating','solid','excluded','missing_source'):
            graph,source,rgb,_=fixture()
            if mode=='floating':cv2.line(rgb,(80,100),(110,121),(0,0,0),2)
            elif mode=='solid':source['walls'].append(dict(outer=[[79,76],[121,76],[121,84],[79,84]]))
            elif mode=='excluded':graph['diagnostics']['excluded_regions']=[[[70,70],[130,70],[130,90],[70,90]]]
            elif mode=='missing_source':source['windows']=[]
            with self.subTest(mode=mode):self.assertFalse(diagnose_graph_door_leaves(graph,source,rgb)[0]['accepted'])

    def test_fragmented_leaf_near_jamb_is_review_only(self):
        graph, source, rgb, _ = fixture()
        cv2.line(rgb, (87,84), (115,104), (0,0,0), 2)
        before = copy.deepcopy((graph, source))
        row = diagnose_graph_door_leaves(graph, source, rgb)[0]
        self.assertTrue(row['accepted'])
        self.assertEqual(row['reason'], 'model_jambs_and_nearby_diagonal_review_hint')
        self.assertTrue(row['requires_review'])
        self.assertIn('candidate_is_not_a_confirmed_door', row['limitations'])
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        strict = anchored_leaves(gray, [80,80], [120,80], 8, 1)
        self.assertEqual(strict, [[], []])
        self.assertEqual((graph, source), before)

    def test_fragmented_line_too_far_from_jamb_is_not_flagged(self):
        graph, source, rgb, _ = fixture()
        cv2.line(rgb, (94,84), (122,104), (0,0,0), 2)
        self.assertFalse(diagnose_graph_door_leaves(graph, source, rgb)[0]['accepted'])

    def test_fragmented_leaf_review_hint_survives_rotations_and_reflection(self):
        for mirror in (False, True):
            graph, source, rgb, _ = fixture()
            cv2.line(rgb, (87,84), (115,104), (0,0,0), 2)
            for turns in range(4):
                if turns or mirror:
                    width, height = source['meta']['width'], source['meta']['height']
                    if turns == 0:
                        transform = lambda p: [width-1-p[0], p[1]]
                        rgb = cv2.flip(rgb, 1)
                    else:
                        transform = lambda p: [height-1-p[1], p[0]]
                        rgb = cv2.rotate(rgb, cv2.ROTATE_90_CLOCKWISE)
                        source['meta']['width'], source['meta']['height'] = height, width
                    for node in graph['nodes']: node['point'] = transform(node['point'])
                    for bridge in graph['diagnostics']['opening_bridges']:
                        bridge['span'] = [transform(p) for p in bridge['span']]
                    for polygon in source['walls'] + source['windows']:
                        polygon['outer'] = [transform(p) for p in polygon['outer']]
                with self.subTest(mirror=mirror, turns=turns):
                    row = diagnose_graph_door_leaves(graph, source, rgb)[0]
                    self.assertTrue(row['accepted'])
                    self.assertTrue(row['requires_review'])

    def test_gap_uses_local_width_and_rejects_invalid_metadata(self):
        graph,source,rgb,_=fixture()
        graph['diagnostics']['opening_bridges']=[]
        graph['diagnostics']['topology_gaps']=[dict(span=[[80,80],[120,80]],local_width_px=8)]
        cv2.line(rgb,(80,84),(110,105),(0,0,0),2)
        self.assertTrue(diagnose_graph_door_leaves(graph,source,rgb)[0]['accepted'])
        source['meta']['width']=99
        with self.assertRaises(ValueError):diagnose_graph_door_leaves(graph,source,rgb)

    def test_provisional_scale_does_not_suppress_raster_diagnostic(self):
        graph,source,rgb,_=fixture()
        graph['scale_cm_per_px']=1
        cv2.line(rgb,(80,84),(110,105),(0,0,0),2)
        row=diagnose_graph_door_leaves(graph,source,rgb)[0]
        self.assertTrue(row['accepted'])
        self.assertTrue(row['physical_minimum_not_applied'])

    def test_saved_graph_skips_resolved_gap_but_keeps_partial_overlap_and_index(self):
        graph, source, rgb, _ = fixture()
        bridge, = graph['diagnostics']['opening_bridges']
        bridge.update(kind='recovered_door', clear_span=[[84,80], [116,80]])
        graph['diagnostics']['topology_gaps'] = [
            dict(span=bridge['span'][::-1], local_width_px=8),
            dict(span=[[80,80], [119,80]], local_width_px=8)]
        rows = diagnose_graph_door_leaves(graph, source, rgb)
        self.assertEqual([(row['candidate_kind'], row['index']) for row in rows], [('gap', 1)])

    def test_neighbouring_backwards_leaf_is_not_attributed_to_opening(self):
        for start,end in (((80,84),(50,105)),((120,84),(150,105))):
            graph,source,rgb,_=fixture()
            cv2.line(rgb,start,end,(0,0,0),2)
            with self.subTest(start=start):
                self.assertFalse(diagnose_graph_door_leaves(graph,source,rgb)[0]['accepted'])

    def test_furniture_and_dimension_can_imitate_leaf_and_need_review(self):
        # These are deliberately indistinguishable by the current local evidence.
        # Guard against accidentally treating a diagnostic flag as a confirmed door.
        for mode in ('furniture','dimension'):
            graph,source,rgb,_=fixture()
            before=copy.deepcopy(graph)
            if mode=='furniture':
                points=[(80,84),(110,105),(100,120),(70,99),(80,84)]
                for a,b in zip(points,points[1:]):cv2.line(rgb,a,b,(0,0,0),2)
            else:
                cv2.line(rgb,(80,84),(110,105),(0,0,0),1)
                cv2.line(rgb,(107,109),(113,101),(0,0,0),1)
            with self.subTest(mode=mode):
                row=diagnose_graph_door_leaves(graph,source,rgb)[0]
                self.assertTrue(row['accepted'])
                self.assertTrue(row['requires_review'])
                self.assertIn('furniture_or_dimension_line_can_imitate_leaf',row['limitations'])
                self.assertTrue(row['context']['near_tip_cross_strokes'])
                self.assertEqual(graph,before)

    def test_cross_stroke_is_diagnostic_not_automatic_rejection(self):
        graph,source,rgb,_=fixture()
        cv2.line(rgb,(80,84),(110,105),(0,0,0),2)
        baseline=diagnose_graph_door_leaves(graph,source,rgb)[0]
        self.assertTrue(baseline['accepted'])
        self.assertEqual(baseline['context']['near_tip_cross_strokes'],[])
        self.assertTrue(baseline['context']['near_closed_contours'])
        self.assertEqual(baseline['context']['interpretation'],'ambiguous_visual_context_requires_review')

    def test_perpendicular_leaf_requires_matching_dark_swing_arc(self):
        for mode in ('arc','leaf_only','arc_only','furniture','pale'):
            graph,source,rgb,_=fixture()
            color=(190,190,190) if mode=='pale' else (0,0,0)
            if mode in ('arc','leaf_only','pale'):
                cv2.line(rgb,(80,80),(80,120),color,1)
            if mode in ('arc','arc_only','pale'):
                cv2.ellipse(rgb,(80,80),(40,40),0,0,90,color,1)
            if mode=='furniture':
                cv2.rectangle(rgb,(80,80),(120,120),(0,0,0),1)
            before=copy.deepcopy(graph)
            row=diagnose_graph_door_leaves(graph,source,rgb)[0]
            with self.subTest(mode=mode):
                self.assertEqual(row['accepted'],mode=='arc')
                self.assertEqual(graph,before)
                if mode=='arc':
                    self.assertEqual(row['reason'],'model_jambs_leaf_and_swing_arc')
                    self.assertGreaterEqual(row['arc_evidence']['arc_hits'],7)
                    self.assertTrue(row['requires_review'])

    def test_similar_furniture_or_dimension_arc_remains_ambiguous(self):
        for mode in ('furniture_circle','dimension_arc','neighbour_arc'):
            graph,source,rgb,_=fixture()
            cv2.line(rgb,(80,80),(80,120),(0,0,0),1)
            if mode=='furniture_circle':cv2.circle(rgb,(80,80),40,(0,0,0),1)
            elif mode=='neighbour_arc':cv2.ellipse(rgb,(80,80),(40,40),0,90,180,(0,0,0),1)
            else:
                cv2.ellipse(rgb,(80,80),(40,40),0,0,90,(0,0,0),1)
                cv2.line(rgb,(76,119),(84,121),(0,0,0),1)
            row=diagnose_graph_door_leaves(graph,source,rgb)[0]
            with self.subTest(mode=mode):
                self.assertEqual(row['accepted'],mode!='neighbour_arc')
                if row['accepted']:
                    self.assertTrue(row['requires_review'])
                    self.assertIn('furniture_or_dimension_arc_can_imitate_swing_arc',row['limitations'])


if __name__=='__main__':unittest.main()
