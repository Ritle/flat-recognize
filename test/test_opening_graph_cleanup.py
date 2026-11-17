import copy
import sys
from pathlib import Path
import unittest

import cv2
import numpy as np
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from opening_graph_cleanup import blank_aperture, cleanup_opening_graph, rejected_frame_candidates


def fixture(kind='windows'):
    rgb = np.full((100, 150, 3), 255, np.uint8)
    rgb[44:57, 5:145] = 0
    rgb[46:55, 50:101] = 255
    opening = dict(id=3, outer=[[50, 46], [100, 46], [100, 54], [50, 54]])
    source = dict(meta=dict(width=150, height=100, coordinate_space='source_raster'), doors=[], windows=[])
    source[kind].append(opening)
    points = [[5, 50], [45, 50], [55, 47], [95, 47], [105, 50], [145, 50], [55, 53], [95, 53]]
    pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (1, 6), (6, 7), (7, 4)]
    graph = dict(coordinate_space='source_raster', scale_cm_per_px=2,
                 nodes=[dict(id=i, point=p, adjacency=[]) for i, p in enumerate(points)],
                 edges=[dict(start=a, end=b, thickness_px=12) for a, b in pairs], faces=[],
                 diagnostics=dict(opening_bridges=[dict(kind=kind, opening_id=3, span=[[50, 50], [100, 50]])],
                                  topology_gaps=[]))
    return graph, source, rgb


def missing_fixture(paired=False):
    graph, source, rgb = fixture()
    points = [[5, 50], [45, 50], [85 if paired else 105, 50], [145, 50]]
    graph['nodes'] = [dict(id=i, point=p, adjacency=[]) for i, p in enumerate(points)]
    graph['edges'] = [dict(start=a, end=b, thickness_px=12) for a, b in [(0, 1), (2, 3)]]
    graph['diagnostics']['opening_bridges'] = []
    if paired:
        rgb[44:57, 5:145] = 0
        rgb[46:55, 50:69] = 255
        rgb[46:55, 80:101] = 255
        source['windows'] = [dict(id=3, outer=[[50,46],[68,46],[68,54],[50,54]]),
                             dict(id=4, outer=[[80,46],[100,46],[100,54],[80,54]])]
        graph['diagnostics']['opening_bridges'] = [dict(kind='windows', opening_id=4, span=[[80,50],[100,50]])]
    return graph, source, rgb


class OpeningCleanupTest(unittest.TestCase):
    def raw_fixture(self):
        graph,source,rgb=missing_fixture()
        source['windows']=[]
        raw=dict(meta=copy.deepcopy(source['meta']),diagnostics=dict(rejected=[dict(
            id=3,source_class='door',reject_reason=['not_elongated'],
            outer=[[50,35],[100,35],[100,65],[50,65]],bbox=[50,35,100,65],thickness=30)]))
        return graph,source,rgb,raw

    def test_rejected_geometry_candidate_uses_frame_width_and_keeps_raw_provenance(self):
        graph,source,rgb,raw=self.raw_fixture();before=copy.deepcopy(raw)
        prepared,_=rejected_frame_candidates(raw,rgb[:,:,0])
        self.assertEqual(len(prepared),1)
        self.assertNotIn('bbox',prepared[0])
        self.assertLess(prepared[0]['raw_recovery']['measured_frame_width_px'],30)
        result,report=cleanup_opening_graph(graph,source,rgb,[],recover_missing=True,prediction=raw)
        self.assertEqual(len(result['diagnostics']['opening_bridges']),1)
        bridge=result['diagnostics']['opening_bridges'][0]
        self.assertEqual(bridge['opening_id'],'raw-window-3')
        self.assertEqual(bridge['raw_recovery']['candidate']['source_class'],'door')
        self.assertEqual(bridge['raw_recovery']['candidate']['reject_reason'],['not_elongated'])
        self.assertEqual(raw,before)
        self.assertEqual(source['windows'],[])

    def test_raw_recovery_does_not_relax_thickness_rejection_or_duplicate_carriers(self):
        for mode in ('thickness','duplicate_id','existing','no_frame'):
            g,s,r,raw=self.raw_fixture()
            if mode=='thickness':raw['diagnostics']['rejected'][0]['reject_reason'].append('too_thick')
            elif mode=='duplicate_id':raw['diagnostics']['rejected']*=2
            elif mode=='existing':g['diagnostics']['opening_bridges']=[dict(kind='recovered_window',span=[[50,50],[100,50]])]
            else:r[:]=255
            result,report=cleanup_opening_graph(g,s,r,[],recover_missing=True,prediction=raw)
            self.assertEqual(result,g)
            self.assertFalse(any(x['accepted'] for x in report))

    def test_raw_prediction_requires_matching_source_and_explicit_mode(self):
        for mode in ('implicit','size','coordinates','name'):
            g,s,r,raw=self.raw_fixture()
            if mode=='size':raw['meta']['width']=99
            elif mode=='coordinates':raw['meta']['coordinate_space']='model'
            elif mode=='name':raw['meta']['source']='other.png';s['meta']['source']='this.png'
            with self.subTest(mode=mode),self.assertRaises(ValueError):
                cleanup_opening_graph(g,s,r,[],recover_missing=mode!='implicit',prediction=raw)

    def test_missing_carrier_requires_opt_in_and_real_raster_jambs(self):
        graph, source, rgb = missing_fixture()
        before = copy.deepcopy(graph)
        unchanged, _ = cleanup_opening_graph(graph, source, rgb, [])
        self.assertEqual(unchanged, graph)
        restored, decisions = cleanup_opening_graph(graph, source, rgb, [], recover_missing=True)
        self.assertTrue(decisions[0]['accepted'])
        self.assertEqual(len(restored['diagnostics']['opening_bridges']), 1)
        self.assertEqual(restored['diagnostics']['opening_bridges'][0]['opening_id'], 3)
        self.assertEqual(graph, before)
        self.assertEqual(cleanup_opening_graph(restored, source, rgb, [], recover_missing=True)[0], restored)
        rgb[:] = 255
        rejected, decisions = cleanup_opening_graph(graph, source, rgb, [], recover_missing=True)
        self.assertEqual(rejected, graph)
        self.assertFalse(any(r['accepted'] for r in decisions))

    def test_raster_jamb_alone_cannot_replace_missing_structural_port(self):
        graph, source, rgb = missing_fixture()
        graph['edges'] = graph['edges'][:1]
        result, decisions = cleanup_opening_graph(graph, source, rgb, [], recover_missing=True)
        self.assertEqual(result, graph)
        self.assertFalse(any(r['accepted'] for r in decisions))

    def test_existing_recovered_opening_prevents_duplicate_and_type_override(self):
        for kind in ('recovered_window', 'recovered_door'):
            graph, source, rgb = missing_fixture()
            graph['diagnostics']['opening_bridges'] = [dict(kind=kind, span=[[50,50],[100,50]])]
            result, decisions = cleanup_opening_graph(graph, source, rgb, [], recover_missing=True)
            self.assertEqual(result, graph)
            self.assertEqual(decisions[0]['reason'], 'already_represented_by_existing_carrier')

    def test_only_parallel_fully_replaced_topology_gaps_are_resolved(self):
        graph, source, rgb = missing_fixture()
        resolved = dict(span=[[45,50],[105,50]])
        partial = dict(span=[[5,50],[70,50]])
        transverse = dict(span=[[75,45],[75,55]])
        graph['diagnostics']['topology_gaps'] = [resolved, partial, transverse]
        result, decisions = cleanup_opening_graph(graph, source, rgb, [], recover_missing=True)
        self.assertTrue(decisions[0]['accepted'])
        self.assertEqual(result['diagnostics']['topology_gaps'], [partial, transverse])
        self.assertEqual(decisions[0]['resolved_topology_gaps'], [resolved])

    def test_pair_recovers_missing_pier_and_preserves_two_apertures(self):
        graph, source, rgb = missing_fixture(paired=True)
        result, decisions = cleanup_opening_graph(graph, source, rgb, [], recover_missing=True)
        accepted = [r for r in decisions if r['accepted']]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(accepted[0]['neighbour_opening_id'], 4)
        bridges = result['diagnostics']['opening_bridges']
        self.assertEqual(len(bridges), 2)
        self.assertEqual(bridges[0]['span'], bridges[1]['span'])
        first, second = sorted((LineString(b['clear_span']) for b in bridges), key=lambda l:l.bounds[0])
        self.assertGreater(first.distance(second), 10)
        self.assertEqual(cleanup_opening_graph(result, source, rgb, [], recover_missing=True)[0], result)

    def test_pair_requires_solid_pier_and_already_bound_unmeasured_neighbour(self):
        for mode in ('white_pier', 'no_neighbour', 'measured_neighbour'):
            graph, source, rgb = missing_fixture(paired=True)
            if mode == 'white_pier':
                rgb[46:55,66:80] = 255
            elif mode == 'no_neighbour':
                graph['diagnostics']['opening_bridges'] = []
            else:
                graph['diagnostics']['opening_bridges'][0]['clear_span'] = [[80,50],[100,50]]
            result, decisions = cleanup_opening_graph(graph, source, rgb, [], recover_missing=True)
            with self.subTest(mode=mode):
                self.assertEqual(result, graph)
                self.assertFalse(any(r['accepted'] for r in decisions))

    def test_pair_cannot_remove_a_real_branch_from_the_pier(self):
        graph, source, rgb = missing_fixture(paired=True)
        rgb[15:50, 69:76] = 0
        graph['nodes'].extend([dict(id=4, point=[72,50], adjacency=[]), dict(id=5, point=[72,15], adjacency=[])])
        graph['edges'].append(dict(start=4,end=5,thickness_px=7))
        result, decisions = cleanup_opening_graph(graph, source, rgb, [], recover_missing=True)
        self.assertEqual(result, graph)
        self.assertFalse(any(r['accepted'] for r in decisions))

    def test_missing_carrier_still_obeys_exclusions_and_unique_source_ids(self):
        for mode in ('excluded', 'duplicate'):
            graph, source, rgb = missing_fixture()
            if mode == 'excluded':
                graph['diagnostics']['excluded_regions'] = [[[60,40],[90,40],[90,60],[60,60]]]
            else:
                source['windows'] *= 2
            result, decisions = cleanup_opening_graph(graph, source, rgb, [], recover_missing=True)
            self.assertEqual(result, graph)
            self.assertFalse(any(r['accepted'] for r in decisions))

    def test_dashed_passage_cannot_be_restored_as_window(self):
        graph, source, rgb = missing_fixture()
        rgb[44:46,50:101] = 255
        rgb[55:57,50:101] = 255
        for x in range(50,101,12):
            rgb[44:46,x:x+4] = 0
            rgb[55:57,x:x+4] = 0
        # White aperture and solid jambs alone are insufficient to call it a window.
        self.assertIsNotNone(blank_aperture(rgb[:,:,0], source['windows'][0]))
        result, decisions = cleanup_opening_graph(graph, source, rgb, [], recover_missing=True)
        self.assertEqual(result, graph)
        self.assertEqual(decisions[0]['reason'], 'missing_continuous_frame_rails')

    def test_paired_neighbour_also_requires_frame_evidence(self):
        graph, source, rgb = missing_fixture(paired=True)
        rgb[44:46,80:101] = 255
        rgb[55:57,80:101] = 255
        self.assertIsNotNone(blank_aperture(rgb[:,:,0], source['windows'][1]))
        result, decisions = cleanup_opening_graph(graph, source, rgb, [], recover_missing=True)
        self.assertEqual(result, graph)
        self.assertFalse(any(r['accepted'] for r in decisions))

    def test_duplicate_frame_axes_are_replaced_by_measured_clear_aperture(self):
        graph, source, rgb = fixture()
        before, original_source = copy.deepcopy(graph), copy.deepcopy(source)
        result, decisions = cleanup_opening_graph(graph, source, rgb, [0])
        self.assertTrue(decisions[0]['accepted'])
        self.assertLess(len(result['edges']), len(graph['edges']))
        bridge = result['diagnostics']['opening_bridges'][0]
        self.assertEqual(bridge['carrier_cleanup'], 'raster_confirmed_blank_aperture')
        self.assertTrue(bridge['requires_review'])
        self.assertAlmostEqual(LineString(bridge['clear_span']).length, 51, delta=1)
        self.assertGreater(LineString(bridge['span']).length, LineString(bridge['clear_span']).length)
        self.assertEqual(graph, before)
        self.assertEqual(source, original_source)
        again, report = cleanup_opening_graph(result, source, rgb, [0])
        self.assertEqual(again, result)
        self.assertFalse(report[0]['accepted'])

    def test_solid_wall_unbounded_white_and_missing_jamb_do_not_pass(self):
        for mode in ('solid', 'all_white', 'no_right_jamb', 'cross_wall'):
            graph, source, rgb = fixture()
            if mode == 'solid':
                rgb[44:57, 5:145] = 0
            elif mode == 'all_white':
                rgb[:] = 255
            elif mode == 'no_right_jamb':
                rgb[:, 100:] = 255
            else:
                rgb[30:70, 73:78] = 0
            result, report = cleanup_opening_graph(graph, source, rgb, [0])
            with self.subTest(mode=mode):
                self.assertFalse(report[0]['accepted'])
                self.assertEqual(result, graph)

    def test_real_junction_beside_aperture_is_preserved(self):
        graph, source, rgb = fixture('doors')
        rgb[10:50, 42:49] = 0
        graph['nodes'].append(dict(id=8, point=[45, 10], adjacency=[]))
        graph['edges'].append(dict(start=1, end=8, thickness_px=7))
        result, report = cleanup_opening_graph(graph, source, rgb, [0])
        self.assertFalse(report[0]['accepted'])
        self.assertEqual(result, graph)

    def test_thin_isolated_door_leaf_can_be_removed_but_connected_branch_cannot(self):
        graph, source, rgb = fixture('doors')
        cv2.line(rgb, (49, 50), (49, 20), (0, 0, 0), 1)
        graph['nodes'].extend([dict(id=8, point=[49, 50], adjacency=[]),
                               dict(id=9, point=[49, 20], adjacency=[])])
        graph['edges'].append(dict(start=8, end=9, thickness_px=2))
        result, report = cleanup_opening_graph(graph, source, rgb, [0])
        self.assertTrue(report[0]['accepted'])
        self.assertEqual(len(report[0]['removed_side_branches']), 1)
        graph['nodes'].extend([dict(id=10, point=[30, 20], adjacency=[]),
                               dict(id=11, point=[70, 20], adjacency=[])])
        graph['edges'].extend([dict(start=9, end=10, thickness_px=2), dict(start=9, end=11, thickness_px=2)])
        result, report = cleanup_opening_graph(graph, source, rgb, [0])
        self.assertFalse(report[0]['accepted'])
        self.assertEqual(result, graph)

    def test_parallel_separate_wall_is_not_mistaken_for_frame(self):
        graph, source, rgb = fixture()
        rgb[57:61, :] = 0
        graph['nodes'].extend([dict(id=8, point=[5, 59], adjacency=[]), dict(id=9, point=[145, 59], adjacency=[])])
        graph['edges'].append(dict(start=8, end=9, thickness_px=4))
        result, report = cleanup_opening_graph(graph, source, rgb, [0])
        # A wall outside the cleanup domain must remain exactly present.
        if report[0]['accepted']:
            points = {n['id']: n['point'] for n in result['nodes']}
            self.assertTrue(any(points[e['start']] == [5, 59] and points[e['end']] == [145, 59] for e in result['edges']))
        else:
            self.assertEqual(result, graph)

    def test_explicit_selection_and_exclusions_are_honored(self):
        graph, source, rgb = fixture()
        result, decisions = cleanup_opening_graph(graph, source, rgb, [])
        self.assertEqual(result, graph)
        self.assertFalse(decisions)
        graph['diagnostics']['excluded_regions'] = [[[60, 40], [90, 40], [90, 60], [60, 60]]]
        result, decisions = cleanup_opening_graph(graph, source, rgb, [0])
        self.assertEqual(result, graph)
        self.assertEqual(decisions[0]['reason'], 'excluded_region')

    def test_missing_duplicate_or_recovered_classification_does_not_authorize_repair(self):
        for mode in ('missing', 'duplicate', 'recovered'):
            graph, source, rgb = fixture()
            if mode == 'missing':
                source['windows'] = []
            elif mode == 'duplicate':
                source['windows'] *= 2
            else:
                graph['diagnostics']['opening_bridges'][0]['kind'] = 'recovered_window'
            result, report = cleanup_opening_graph(graph, source, rgb, [0])
            self.assertFalse(report[0]['accepted'])
            self.assertEqual(result, graph)

    def test_wrong_dimensions_or_coordinate_space_are_rejected(self):
        graph, source, rgb = fixture()
        with self.assertRaises(ValueError):
            cleanup_opening_graph(graph, source, rgb[:50], [0])
        graph['coordinate_space'] = 'model'
        with self.assertRaises(ValueError):
            cleanup_opening_graph(graph, source, rgb, [0])

    def test_vertical_raster_and_graph_have_the_same_result(self):
        graph, source, rgb = fixture()
        source['meta'].update(width=100, height=150)
        source['windows'][0]['outer'] = [[y, x] for x, y in source['windows'][0]['outer']]
        for node in graph['nodes']:
            node['point'].reverse()
        graph['diagnostics']['opening_bridges'][0]['span'] = [[y, x] for x, y in graph['diagnostics']['opening_bridges'][0]['span']]
        result, report = cleanup_opening_graph(graph, source, rgb.transpose(1, 0, 2).copy(), [0])
        self.assertTrue(report[0]['accepted'])
        span = result['diagnostics']['opening_bridges'][0]['clear_span']
        self.assertAlmostEqual(span[0][0], span[1][0])


if __name__ == '__main__':
    unittest.main()
