import sys
import unittest
from unittest.mock import patch
from pathlib import Path
import cv2
import numpy as np
from shapely.geometry import LineString, box
from shapely.ops import unary_union, polygonize
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'scripts'))
from mask_wall_graph import (build_graph,node_lines,t_jamb_gaps,supported_wall_runs,
                             terminal_jamb_gaps,clear_jamb_span,topology_gap_support,
                             projected_jamb_gaps,raster_pier_runs,rejected_window_bridges,
                             continued_jamb_gaps,interior_line_closures,
                             bounded_window_attachment,raster_frame_gaps,candidate_door_jamb_repairs)
from door_leaf_evidence import diagonal_leaf_with_jambs, dark_leaf_with_jambs

class MaskGraphTests(unittest.TestCase):
    def candidate_jamb_fixture(self):
        gray = np.full((110,140),255,np.uint8)
        for bounds in ((10,36,34,44),(51,36,61,44),(81,36,120,44)):
            cv2.rectangle(gray,bounds[:2],bounds[2:],0,-1)
        cv2.line(gray,(81,40),(65,29),0,2)
        edges = [LineString([(10,40),(45,40)]),LineString([(85,40),(120,40)]),
                 LineString([(85,40),(85,100)])]
        widths = [dict(start=list(e.coords[0]),end=list(e.coords[-1]),thickness_px=8) for e in edges]
        walls = unary_union([box(10,36,34,44),box(81,36,120,44),box(81,40,89,100)])
        windows = [dict(id=2,outer=list(box(35,37,54,43).exterior.coords))]
        prediction = dict(diagnostics=dict(rejected=[dict(id=3,source_class='door',
                            reject_reason=['not_elongated'],outer=list(box(61,35,81,44).exterior.coords))]))
        distance = cv2.distanceTransform((gray<140).astype(np.uint8),cv2.DIST_L2,5)
        leaf = dict(hinge=[81,40],tip=[65,29],jamb_support=[3,3],tip_clearance=10)
        return gray,edges,widths,walls,windows,prediction,distance,leaf

    def test_candidate_jamb_requires_model_window_raw_pier_and_dark_leaf_together(self):
        gray,edges,widths,walls,windows,prediction,distance,leaf = self.candidate_jamb_fixture()
        with patch('mask_wall_graph.diagonal_leaf_with_jambs',return_value=leaf):
            repairs,decisions = candidate_door_jamb_repairs(
                edges,widths,windows,prediction,walls,gray,distance,3,unary_union([]))
            self.assertEqual(len(repairs),1)
            self.assertTrue(decisions[0]['accepted'])
            self.assertEqual(repairs[0]['bridge']['source_id'],3)
            self.assertEqual(repairs[0]['bridge']['window_ids'],[2])
            self.assertEqual(repairs[0]['bridge']['evidence']['dark_raster_support'],1)
            self.assertTrue(repairs[0]['jamb']['requires_review'])
            for context,model,raw,excluded in (([],prediction,distance,unary_union([])),
                                              (windows,None,distance,unary_union([])),
                                              (windows,prediction,np.zeros_like(distance),unary_union([])),
                                              (windows,prediction,distance,box(55,35,70,45))):
                self.assertEqual(candidate_door_jamb_repairs(
                    edges,widths,context,model,walls,gray,raw,3,excluded)[0],[])

    def test_candidate_jamb_rejects_pale_leaf_and_does_not_repeat_a_repair(self):
        gray,edges,widths,walls,windows,prediction,distance,leaf = self.candidate_jamb_fixture()
        with patch('mask_wall_graph.diagonal_leaf_with_jambs',return_value=leaf):
            repairs,_ = candidate_door_jamb_repairs(
                edges,widths,windows,prediction,walls,gray,distance,3,unary_union([]))
            repair = repairs[0]
            updated = node_lines(edges+[LineString([repair['jamb']['start'],repair['jamb']['end']]),
                                        LineString(repair['bridge']['span'])],.01,straighten=False)
            self.assertEqual(candidate_door_jamb_repairs(
                updated,widths,windows,prediction,walls,gray,distance,3,unary_union([]))[0],[])
            cv2.line(gray,(81,40),(65,29),210,2)
            self.assertEqual(candidate_door_jamb_repairs(
                edges,widths,windows,prediction,walls,gray,distance,3,unary_union([]))[0],[])

    def bounded_gap_fixture(self):
        gray = np.full((70,100),255,np.uint8)
        cv2.rectangle(gray,(10,26),(39,34),0,-1)
        cv2.rectangle(gray,(61,26),(90,34),0,-1)
        for y in (27,33):cv2.line(gray,(39,y),(61,y),0,1)
        edges = [LineString([(10,30),(34,30)]),LineString([(66,30),(90,30)])]
        widths = [dict(start=list(e.coords[0]),end=list(e.coords[-1]),thickness_px=10) for e in edges]
        walls = unary_union([box(10,26,39,34),box(61,26,90,34)])
        opening = dict(id=7,outer=list(box(40,27,60,33).exterior.coords))
        distance = cv2.distanceTransform((gray<140).astype(np.uint8),cv2.DIST_L2,5)
        return gray,edges,widths,walls,opening,distance

    def test_known_bay_keeps_aperture_shorter_than_carrier_and_rejects_crossing(self):
        gray,edges,widths,walls,opening,_ = self.bounded_gap_fixture()
        bridge,record = bounded_window_attachment(opening,edges,widths,walls,gray,unary_union([]))
        self.assertTrue(record['accepted'])
        self.assertEqual(bridge['span'],[(34,30),(66,30)])
        self.assertEqual(bridge['clear_span'],[[39.5,30],[60.5,30]])
        self.assertTrue(bridge['requires_review'])
        for graph,image,excluded in ((edges,np.full_like(gray,255),unary_union([])),
                                     (edges+[LineString([(50,20),(50,40)])],gray,unary_union([])),
                                     (edges,gray,box(45,25,55,35))):
            self.assertIsNone(bounded_window_attachment(opening,graph,widths,walls,image,excluded)[0])

    def test_unclassified_frame_requires_model_and_thick_raster_jambs(self):
        gray,edges,_,walls,opening,distance = self.bounded_gap_fixture()
        bridges,decisions = raster_frame_gaps(edges,walls,[],gray,distance,8,3,unary_union([]))
        self.assertEqual(len(bridges),1)
        self.assertEqual(bridges[0]['source'],'bounded_raster_frame_gap')
        self.assertEqual(bridges[0]['clear_span'],[[60.5,30],[39.5,30]])
        for shapes,known,raw,excluded in ((box(0,0,1,1),[],distance,unary_union([])),
                                        (walls,[opening],distance,unary_union([])),
                                        (walls,[],np.ones_like(distance),unary_union([])),
                                        (walls,[],distance,box(45,25,55,35))):
            self.assertEqual(raster_frame_gaps(edges,shapes,known,gray,raw,8,3,excluded)[0],[])

    def test_unclassified_frame_cannot_cross_graph_or_follow_backwards_tips(self):
        gray,edges,_,walls,_,distance = self.bounded_gap_fixture()
        for graph in (edges+[LineString([(50,20),(50,40)])],
                      [LineString([(34,30),(45,31)]),edges[1]]):
            self.assertEqual(raster_frame_gaps(graph,walls,[],gray,distance,8,3,unary_union([]))[0],[])

    def interior_passage_fixture(self):
        edges = node_lines([LineString([(10,10),(210,10)]),LineString([(210,10),(210,130)]),
                            LineString([(210,130),(10,130)]),LineString([(10,130),(10,10)]),
                            LineString([(115,10),(115,130)]),LineString([(10,60),(90,60)])],.01)
        widths = [dict(start=list(e.coords[0]),end=list(e.coords[-1]),thickness_px=6) for e in edges]
        walls = unary_union([e.buffer(3,cap_style=2) for e in edges])
        gray = np.full((140,220),255,np.uint8)
        cv2.line(gray,(90,60),(115,60),0,1)
        candidate = dict(span=[[90,60],[115,60]],clear_span=[[90,60],[112,60]],accepted=False,
                         reason='missing_inward_leaf_evidence')
        return edges,widths,walls,gray,candidate

    def test_interior_passage_splits_topology_without_creating_physical_edge(self):
        edges,widths,walls,gray,candidate = self.interior_passage_fixture()
        before = [list(e.coords) for e in edges]
        closures,decisions = interior_line_closures(edges,[],[candidate,candidate],widths,walls,gray,1)
        self.assertEqual(len(closures),1)
        self.assertEqual(closures[0]['opening_type'],'uncertain')
        self.assertTrue(closures[0]['requires_review'])
        self.assertEqual(before,[list(e.coords) for e in edges])
        self.assertEqual(len(list(polygonize(unary_union(edges)))),2)
        self.assertEqual(len(list(polygonize(unary_union(edges+[LineString(closures[0]['span'])])))),3)

    def test_interior_passage_requires_raster_jambs_and_existing_interior(self):
        edges,widths,walls,gray,candidate = self.interior_passage_fixture()
        open_edges = [e for e in edges if not all(x==10 for x,y in e.coords)]
        for graph,wall_shapes,image,reason in (
                (edges,walls,np.full_like(gray,255),'missing_continuous_raster_line'),
                (edges,box(0,0,1,1),gray,'missing_structural_jambs'),
                (open_edges,walls,gray,'not_inside_existing_face')):
            closures,decisions = interior_line_closures(graph,[],[candidate],widths,wall_shapes,image,1)
            self.assertEqual(closures,[])
            self.assertEqual(decisions[0]['reason'],reason)

    def test_interior_passage_rejects_long_thin_gap_and_tiny_partition(self):
        edges,widths,walls,gray,candidate = self.interior_passage_fixture()
        thin = [dict(r,thickness_px=2) for r in widths]
        for evidence,scale,reason in ((thin,1,'gap_too_long_for_local_wall_width'),
                                      (widths,.5,'no_two_valid_interior_faces')):
            closures,decisions = interior_line_closures(edges,[],[candidate],evidence,walls,gray,scale)
            self.assertEqual(closures,[])
            self.assertEqual(decisions[0]['reason'],reason)

    def continued_gap_fixture(self):
        edges = [LineString([(50,30),(50,50)]),LineString([(50,50),(51,54)]),
                 LineString([(50,50),(80,50)]),LineString([(20,74),(90,74)])]
        widths = [dict(start=list(e.coords[0]),end=list(e.coords[-1]),thickness_px=6) for e in edges]
        return edges,widths

    def test_continued_axis_uses_long_wall_through_t_to_correct_short_tail(self):
        edges,widths = self.continued_gap_fixture()
        gaps = list(continued_jamb_gaps(edges,3,widths))
        self.assertEqual(len(gaps),1)
        gap,support = gaps[0]
        self.assertEqual(gap.coords[0],(51,54))
        self.assertAlmostEqual(gap.coords[-1][0],51+20/24,places=5)
        self.assertEqual(gap.coords[-1][1],74)
        self.assertGreaterEqual(support['support_length_px'],gap.length)
        self.assertEqual(support['support_path'],[[51,54],[50,50],[50,30]])
        self.assertEqual(list(projected_jamb_gaps(edges,3)),[])

    def test_continued_axis_rejects_isolated_spur_crossing_and_ambiguous_branch(self):
        edges,widths = self.continued_gap_fixture()
        for graph in (edges[1:],edges+[LineString([(20,64),(90,64)])],
                      edges+[LineString([(50,50),(49,30)])]):
            self.assertEqual(list(continued_jamb_gaps(graph,3,widths)),[])

    def test_confirmed_bridge_removes_old_tip_from_continued_candidates(self):
        edges,widths = self.continued_gap_fixture()
        gap,_ = next(continued_jamb_gaps(edges,3,widths))
        updated = node_lines(edges+[gap],.01,straighten=False)
        self.assertEqual(list(continued_jamb_gaps(updated,3,widths)),[])

    def test_continued_leaf_requires_dark_original_stroke(self):
        evidence = dict(hinge=[20,20],tip=[30,40],jamb_support=[3,3])
        for value,expected in ((0,True),(210,False)):
            gray = np.full((60,60),255,np.uint8)
            cv2.line(gray,(20,20),(30,40),value,1)
            with patch('door_leaf_evidence.diagonal_leaf_with_jambs',return_value=evidence):
                result = dark_leaf_with_jambs(gray,[20,20],[20,45],6,box(0,0,1,1))
            self.assertEqual(result is not None,expected)
            if result:
                self.assertEqual(result['dark_raster_support'],1)

    def rejected_frame_fixture(self):
        gray = np.full((80,120),255,np.uint8)
        for y in (27,33):
            cv2.line(gray,(40,y),(70,y),0,1)
        candidate = dict(id=42,source_class='door',reject_reason=['too_far_from_wall'],
                         outer=list(box(40,27,70,33).exterior.coords))
        edges = [LineString([(10,30),(40,30)]),LineString([(70,30),(100,30)])]
        return gray,candidate,edges

    def test_known_window_attaches_to_retreated_jamb_only_with_visible_frame(self):
        rgb = np.full((130,210,3),255,np.uint8)
        cv2.rectangle(rgb,(10,10),(190,110),(0,0,0),8)
        cv2.rectangle(rgb,(80,5),(110,15),(255,255,255),-1)
        wall = box(6,6,194,114).difference(box(14,14,186,106)).difference(box(78,5,110,15))
        source = dict(meta=dict(width=210,height=130),
                      walls=[dict(outer=list(wall.exterior.coords),holes=[list(r.coords) for r in wall.interiors])],
                      windows=[dict(id=1,outer=list(box(80,8,110,12).exterior.coords))])
        # Isolate the graph stage from skeletonization: its wall tips have the
        # observed medial-axis retreat, 6 px from one model opening endpoint.
        paths = [[(10,10),(74,10)],[(114,10),(190,10)],[(190,10),(190,110)],
                 [(190,110),(10,110)],[(10,110),(10,10)]]
        with patch('mask_wall_graph.skeleton_paths',return_value=(paths,None)):
            no_frame = build_graph(source,rgb)
            for y in (8,12):cv2.line(rgb,(80,y),(110,y),(0,0,0),1)
            with_frame = build_graph(source,rgb)
        self.assertEqual(no_frame['faces'],[])
        self.assertEqual(len(with_frame['faces']),1)
        bridge = with_frame['diagnostics']['opening_bridges'][0]
        self.assertEqual(bridge['attachment_method'],'local_jamb_width_and_raster_frame')
        self.assertTrue(bridge['requires_review'])

    def test_short_rejected_window_needs_measured_bay_and_preserves_aperture(self):
        gray = np.full((60,80),255,np.uint8)
        cv2.rectangle(gray,(10,26),(29,34),0,-1)
        cv2.rectangle(gray,(41,26),(60,34),0,-1)
        for y in (27,33):cv2.line(gray,(29,y),(41,y),0,1)
        edges = [LineString([(10,30),(24,30)]),LineString([(46,30),(60,30)])]
        candidate = dict(id=1,source_class='door',reject_reason=['too_short','not_elongated'],
                         outer=list(box(29,27,41,33).exterior.coords))
        prediction = dict(diagnostics=dict(rejected=[candidate]))
        bridges,decisions = rejected_window_bridges(edges,prediction,gray,unary_union([]))
        self.assertEqual(len(bridges),1)
        self.assertEqual(bridges[0]['span'],[[24,30],[46,30]])
        self.assertEqual(bridges[0]['clear_span'],[[29.5,30],[40.5,30]])
        self.assertEqual(decisions[0]['reason'],'short_candidate_with_bounded_raster_bay')
        gray[28:33,:] = 255
        self.assertEqual(rejected_window_bridges(edges,prediction,gray,unary_union([]))[0],[])

    def test_rejected_frame_requires_two_jambs_and_does_not_duplicate_bridge(self):
        gray,candidate,edges = self.rejected_frame_fixture()
        prediction = dict(diagnostics=dict(rejected=[candidate,candidate]))
        bridges,decisions = rejected_window_bridges(edges,prediction,gray,unary_union([]))
        self.assertEqual(len(bridges),1)
        self.assertEqual(bridges[0]['kind'],'recovered_window')
        self.assertEqual(bridges[0]['source_id'],42)
        self.assertTrue(bridges[0]['requires_review'])
        self.assertEqual(decisions[1]['reason'],'occupied_graph_span')
        self.assertEqual(rejected_window_bridges(edges[:1],prediction,gray,unary_union([]))[0],[])

    def test_rejected_frame_respects_exclusion_crossing_and_raster_evidence(self):
        gray,candidate,edges = self.rejected_frame_fixture()
        prediction = dict(diagnostics=dict(rejected=[candidate]))
        for graph,image,excluded in ((edges,gray,box(50,20,60,40)),
                                     (edges+[LineString([(55,20),(55,40)])],gray,unary_union([])),
                                     (edges,np.full_like(gray,255),unary_union([]))):
            self.assertEqual(rejected_window_bridges(graph,prediction,image,excluded)[0],[])
        candidate['reject_reason'].append('too_thick')
        self.assertEqual(rejected_window_bridges(edges,prediction,gray,unary_union([]))[0],[])

    def test_frame_recovery_preserves_explicit_prediction_provenance(self):
        rgb = np.full((120,170,3),255,np.uint8)
        cv2.rectangle(rgb,(10,10),(150,100),(0,0,0),6)
        opening = box(60,6,95,14)
        cv2.rectangle(rgb,(60,6),(95,14),(255,255,255),-1)
        for y in (8,12):
            cv2.line(rgb,(60,y),(95,y),(0,0,0),1)
        wall = box(7,7,153,103).difference(box(13,13,147,97)).difference(opening)
        source = dict(meta=dict(width=170,height=120,source='test.png'),
                      walls=[dict(outer=list(wall.exterior.coords),
                                  holes=[list(r.coords) for r in wall.interiors])])
        prediction = dict(meta=dict(source['meta']),diagnostics=dict(rejected=[
            dict(id=5,source_class='window',reject_reason=['too_far_from_wall'],
                 outer=list(opening.exterior.coords))]))
        automatic = build_graph(source,rgb)
        self.assertEqual(len(automatic['faces']),1)
        self.assertEqual(automatic['diagnostics']['opening_bridges'][0]['source'],
                         'bounded_raster_frame_gap')
        recovered = build_graph(source,rgb,prediction=prediction)
        self.assertEqual(len(recovered['faces']),1)
        self.assertTrue(recovered['faces'][0]['requires_review'])
        self.assertEqual(recovered['diagnostics']['opening_bridges'][0]['kind'],'recovered_window')
        self.assertEqual(len(recovered['diagnostics']['opening_bridges']),1)
        self.assertEqual(recovered['diagnostics']['opening_bridges'][0]['source_id'],5)
        blank = rgb.copy()
        cv2.rectangle(blank,(60,6),(95,14),(255,255,255),-1)
        self.assertEqual(build_graph(source,blank)['faces'],[])
        self.assertNotIn('windows',source)
        for change in (dict(width=85),dict(source='other.png'),dict(coordinate_space='crop')):
            with self.assertRaises(ValueError):
                build_graph(source,rgb,prediction={**prediction,'meta':{**prediction['meta'],**change}})

    def pier_fixture(self):
        edges = [LineString([(10,30),(30,30)]), LineString([(130,30),(150,30)])]
        windows = [dict(id=i,outer=list(shape.exterior.coords)) for i,shape in enumerate(
            (box(30,27,70,33), box(90,27,130,33)))]
        evidence = [dict(start=list(e.coords[0]),end=list(e.coords[-1]),thickness_px=6)
                    for e in edges]
        raster = np.zeros((60,160),np.float32)
        raster[30,70:91] = 3
        return edges, windows, raster, evidence

    def test_missing_pier_restored_without_turning_windows_into_walls(self):
        edges,windows,raster,evidence = self.pier_fixture()
        # Even a dark frame inside the supplied window must remain an opening.
        raster[30,30:131] = 3
        runs = list(raster_pier_runs(edges,windows,raster,evidence))
        self.assertEqual(len(runs),1)
        self.assertTrue(runs[0][0].equals(LineString([(70,30),(90,30)])))
        self.assertTrue(runs[0][1]['requires_review'])

    def test_pier_requires_separate_aligned_windows_and_clear_carrier(self):
        edges,windows,raster,evidence = self.pier_fixture()
        for unsupported in ([],windows[:1],[windows[0],windows[0]],
                            [windows[0],dict(outer=list(box(90,42,130,48).exterior.coords))]):
            self.assertEqual(list(raster_pier_runs(edges,unsupported,raster,evidence)),[])
        crossed = edges + [LineString([(80,10),(80,50)])]
        self.assertEqual(list(raster_pier_runs(crossed,windows,raster,evidence)),[])
        self.assertEqual(list(raster_pier_runs(edges,windows,np.zeros_like(raster),evidence)),[])

    def test_piers_follow_vertical_carrier(self):
        edges,windows,raster,evidence = self.pier_fixture()
        # Rotate all input evidence by 90 degrees; no horizontal-only recovery.
        rotated = [LineString([(y,x) for x,y in e.coords]) for e in edges]
        openings = [dict(outer=[(y,x) for x,y in op['outer']]) for op in windows]
        widths = [dict(start=list(e.coords[0]),end=list(e.coords[-1]),thickness_px=6)
                  for e in rotated]
        runs = list(raster_pier_runs(rotated,openings,raster.T,widths))
        self.assertEqual(len(runs),1)
        self.assertTrue(runs[0][0].equals(LineString([(30,70),(30,90)])))

    def test_raster_only_piers_cannot_invent_doors_from_furniture(self):
        rgb = np.full((80,180,3),255,np.uint8)
        walls = [box(10,27,30,33),box(150,27,170,33)]
        for shape in walls + [box(65,27,75,33),box(105,27,115,33)]:
            x1,y1,x2,y2 = map(int,shape.bounds)
            cv2.rectangle(rgb,(x1,y1),(x2,y2),(0,0,0),-1)
        source = dict(meta=dict(width=180,height=80),
                      walls=[dict(outer=list(s.exterior.coords)) for s in walls],
                      windows=[dict(outer=list(s.exterior.coords)) for s in
                               [box(30,28,65,32),box(115,28,150,32)]])
        # A nearby furniture stroke can satisfy the legacy leaf detector.
        # Its response must never be used on newly inferred raster jambs.
        with patch('mask_wall_graph.leaf_and_arc',return_value={'false_leaf': True}), \
             patch('mask_wall_graph.diagonal_leaf_with_jambs',return_value={'false_leaf': True}):
            graph = build_graph(source,rgb,scale=3)
        self.assertTrue(graph['diagnostics']['recovered_raster_piers'])
        self.assertTrue(graph['diagnostics']['unconfirmed_pier_gaps'])
        self.assertFalse(any(r['kind']=='recovered_door' for r in graph['diagnostics']['opening_bridges']))
        physical = [LineString([graph['nodes'][e[k]]['point'] for k in ('start','end')])
                    for e in graph['edges']]
        self.assertGreater(unary_union(physical).distance(LineString([(85,30),(95,30)])),1)

    def test_projected_jamb_splits_transverse_edge_without_preexisting_node(self):
        lines = [LineString([(10,50),(60,50)]),LineString([(90,10),(90,90)])]
        gaps = list(projected_jamb_gaps(lines,3))
        self.assertEqual([list(g.coords) for g in gaps],[[(60,50),(90,50)]])
        graph = node_lines(lines+gaps, .01, straighten=False)
        self.assertEqual(sum((90,50) in list(edge.coords) for edge in graph),3)

    def test_projection_rejects_short_spur_and_intervening_wall(self):
        transverse = LineString([(90,10),(90,90)])
        self.assertEqual(list(projected_jamb_gaps(
            [LineString([(55,50),(60,50)]),transverse],3)),[])
        lines = [LineString([(10,50),(60,50)]),transverse,
                 LineString([(75,40),(75,60)])]
        self.assertEqual(list(projected_jamb_gaps(lines,3)),[])

    def test_thin_enclosure_stays_open_without_a_confirmed_opening(self):
        rgb = np.full((100,120,3),255,np.uint8)
        cv2.rectangle(rgb,(15,15),(105,85),(0,0,0),2)
        cv2.rectangle(rgb,(40,10),(75,20),(255,255,255),-1)
        opening = box(40,10,75,20)
        wall = box(14,14,106,86).difference(box(16,16,104,84)).difference(opening)
        def item(shape):
            return dict(outer=list(shape.exterior.coords),
                        holes=[list(r.coords) for r in shape.interiors])
        source = dict(meta=dict(width=120,height=100),walls=[item(wall)])
        graph = build_graph(source,rgb,scale=2)
        self.assertEqual(graph['faces'],[])
        self.assertTrue(graph['diagnostics']['rejected_topology_gaps'])
        with_opening = build_graph({**source,'windows':[item(opening)]},rgb,scale=2)
        self.assertEqual(len(with_opening['faces']),1)
        self.assertEqual(len(with_opening['diagnostics']['opening_bridges']),1)

    def test_local_closure_limit_scales_with_raster_resolution(self):
        for factor in (.5,1,3):
            result = topology_gap_support(LineString([(0,0),(30*factor,0)]),
                                          2*factor,12*factor)
            self.assertFalse(result['accepted'])
            self.assertEqual(result['max_gap_px'],12*factor)

    def test_transverse_tip_between_two_doors_can_support_gap(self):
        lines = [LineString([(20,50),(50,50)]),
                 LineString([(50,10),(50,20)]),
                 LineString([(50,80),(50,100)])]
        gaps = list(terminal_jamb_gaps(lines, 3))
        self.assertEqual({tuple(sorted(g.coords)) for g in gaps},
                         {((50,20),(50,50)), ((50,50),(50,80))})
        lines.append(LineString([(40,35),(60,35)]))
        gaps = list(terminal_jamb_gaps(lines, 3))
        self.assertNotIn(((50,20),(50,50)), {tuple(sorted(g.coords)) for g in gaps})

    def test_clear_span_removes_only_bounded_jamb_caps(self):
        gap = LineString([(0,0),(30,0)])
        walls = unary_union([box(-10,-5,4,5), box(26,-5,40,5)])
        clear = clear_jamb_span(gap, walls, 8, 3)
        self.assertEqual(list(clear.coords), [(4,0),(26,0)])
        self.assertIsNone(clear_jamb_span(gap, walls.union(box(14,-5,16,5)), 8, 3))
        self.assertIsNone(clear_jamb_span(gap, box(-10,-5,10,5), 8, 3))

    def test_jamb_occupancy_uses_clear_span_without_accepting_backwards_leaf(self):
        gray = np.full((80,80), 255, np.uint8)
        walls = unary_union([box(0,15,15,25), box(35,15,60,25)])
        candidate = dict(hinge=[15,20], tip=[26,34], jamb_index=0,
                         hinge_distance=0, swing_angle_degrees=45, leaf_length=18)
        with patch('door_leaf_evidence.leaf_candidates', return_value=([candidate],None,None)):
            self.assertIsNone(diagonal_leaf_with_jambs(gray,[10,20],[40,20],10,walls))
            self.assertIsNotNone(diagonal_leaf_with_jambs(
                gray,[10,20],[40,20],10,walls, clear_span=[[15,20],[35,20]], inward_only=True))
            candidate['swing_angle_degrees'] = 130
            self.assertIsNone(diagonal_leaf_with_jambs(
                gray,[10,20],[40,20],10,walls, clear_span=[[15,20],[35,20]], inward_only=True))

    def test_two_windows_keep_middle_pier_and_close_room_only_with_openings(self):
        rgb = np.full((150, 230, 3), 255, np.uint8)
        cv2.rectangle(rgb, (15,20), (215,130), (0,0,0), 6)
        openings = [box(40,16,95,24), box(112,16,170,24)]
        for shape in openings:
            x1,y1,x2,y2 = map(int, shape.bounds)
            cv2.rectangle(rgb, (x1,y1), (x2,y2), (255,255,255), -1)
        wall = box(12,17,218,133).difference(box(18,23,212,127))
        wall = wall.difference(unary_union(openings))
        def item(shape):
            return dict(outer=list(shape.exterior.coords),
                        holes=[list(ring.coords) for ring in shape.interiors])
        source = dict(meta=dict(width=230,height=150),
                      walls=[item(shape) for shape in wall.geoms],
                      windows=[item(shape) for shape in openings])
        graph = build_graph(source, rgb)
        self.assertEqual(len(graph['faces']), 1)
        self.assertEqual(len(graph['diagnostics']['opening_bridges']), 2)
        self.assertTrue(graph['diagnostics']['retained_wall_runs'])
        without_openings = build_graph({**source, 'windows': []}, rgb)
        self.assertEqual(without_openings['diagnostics']['opening_bridges'], [])
        physical = [LineString([without_openings['nodes'][edge[key]]['point']
                                for key in ('start', 'end')])
                    for edge in without_openings['edges']]
        self.assertEqual(list(polygonize(physical)), [])
        self.assertTrue(all(face['requires_review'] for face in without_openings['faces']))

    def test_mixed_wall_window_axis_keeps_pier_without_filling_windows(self):
        distance = np.zeros((30, 120), np.float32)
        distance[10, 10:25] = 3
        distance[10, 60:70] = 3
        distance[10, 100:111] = 3
        runs = list(supported_wall_runs(LineString([(10,10),(110,10)]),
                                        distance, radius=2, thickness=6))
        self.assertEqual(len(runs), 3)
        self.assertTrue(any(run.distance(LineString([(62,10),(68,10)])) == 0
                            for run in runs))
        self.assertTrue(all(not run.covers(LineString([(35,10),(50,10)])) for run in runs))
        self.assertTrue(all(run.length >= 6 for run in runs))

    def test_thin_graphics_and_short_speckles_do_not_restore_wall(self):
        distance = np.ones((30, 120), np.float32)
        distance[10, 60:62] = 3
        self.assertEqual(list(supported_wall_runs(LineString([(10,10),(110,10)]),
                                                distance, radius=1, thickness=6)), [])

    def t_gap_edges(self):
        return [LineString([(20, 50), (40, 50)]),
                LineString([(70, 20), (70, 50)]),
                LineString([(70, 50), (70, 80)])]

    def test_door_gap_can_end_on_transverse_junction(self):
        gaps = list(t_jamb_gaps(self.t_gap_edges(), 3))
        self.assertEqual([list(g.coords) for g in gaps], [[(40, 50), (70, 50)]])
        self.assertEqual(list(t_jamb_gaps(self.t_gap_edges(), 1)), [])

    def test_t_gap_does_not_cross_an_existing_wall(self):
        edges = self.t_gap_edges() + [LineString([(55, 40), (55, 60)])]
        self.assertEqual(list(t_jamb_gaps(edges, 3)), [])

    def test_t_gap_rejects_wall_tip_pointing_away(self):
        edges = self.t_gap_edges()
        edges[0] = LineString([(60, 50), (40, 50)])
        self.assertEqual(list(t_jamb_gaps(edges, 3)), [])

    def test_t_gap_requires_transverse_branches(self):
        edges = self.t_gap_edges()
        edges[2] = LineString([(70, 50), (90, 70)])
        self.assertEqual(list(t_jamb_gaps(edges, 3)), [])

    def test_t_gap_without_leaf_stays_open(self):
        rgb = np.full((120, 120, 3), 255, np.uint8)
        walls = []
        for x1, y1, x2, y2 in [(10, 47, 40, 53), (70, 10, 76, 100)]:
            cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 0, 0), -1)
            walls.append(dict(outer=[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]))
        with patch('mask_wall_graph.t_jamb_gaps',
                   return_value=[LineString([(40,50),(73,50)])]):
            graph = build_graph(dict(meta=dict(width=120, height=120), walls=walls),
                                rgb, scale=3)
        self.assertEqual(graph['diagnostics']['opening_bridges'], [])
        self.assertEqual(graph['diagnostics']['t_jamb_candidates'][0]['reason'],
                         'missing_door_leaf_evidence')
        self.assertEqual(graph['faces'], [])

    def test_x_intersection_is_noded(self):
        edges=node_lines([LineString([(10,50),(90,50)]),LineString([(50,10),(50,90)])],2)
        self.assertEqual(len(edges),4)
        self.assertTrue(all((50,50) in list(e.coords) for e in edges))

    def test_diagonal_retains_angle(self):
        edges=node_lines([LineString([(10,10),(90,70)])],2)
        self.assertEqual(list(edges[0].coords),[(10,10),(90,70)])

    def test_t_attachment_and_endpoint_snap_share_a_node(self):
        edges=node_lines([LineString([(10,50),(90,50)]),LineString([(50,10),(50,48)])],3)
        self.assertEqual(len(edges),3)
        self.assertTrue(all((50,50) in list(e.coords) for e in edges))

    def test_snapping_does_not_swap_close_endpoints(self):
        edges=node_lines([LineString([(10,20),(49,20)]),LineString([(51,20),(90,20)])],3)
        endpoints=[tuple(p) for edge in edges for p in edge.coords]
        self.assertEqual(endpoints.count((50,20)),2)

    def test_room_independent_and_explicit_exclusion(self):
        rgb=np.full((120,120,3),255,np.uint8)
        cv2.rectangle(rgb,(20,20),(100,100),(0,0,0),5)
        source=dict(meta=dict(width=120,height=120),rooms=[],walls=[dict(outer=[[18,18],[102,18],[102,102],[18,102]],holes=[[[23,23],[97,23],[97,97],[23,97]]])])
        graph=build_graph(source,rgb)
        self.assertGreater(len(graph['edges']),0)
        self.assertEqual(len(graph['faces']),1)
        empty=build_graph(source,rgb,excluded_regions=[[[0,0],[119,0],[119,119],[0,119]]])
        self.assertEqual(empty['edges'],[])

    def test_segmentation_without_raster_support_is_rejected(self):
        rgb=np.full((100,100,3),255,np.uint8)
        source=dict(meta=dict(width=100,height=100),walls=[dict(outer=[[10,10],[90,10],[90,20],[10,20]])])
        graph=build_graph(source,rgb)
        self.assertEqual(graph['edges'],[])
        self.assertTrue(graph['diagnostics']['rejected'])

if __name__=='__main__':unittest.main()
