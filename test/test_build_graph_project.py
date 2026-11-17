import copy
import json
import math
from pathlib import Path
import sys
import unittest

from shapely.geometry import LineString, Point

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from build_graph_project import build_graph_project, opening_binding
from graph_export_geometry import merge_straight_chains, regularize_opening_carriers
from validate_recognition import validate_recognition


def graph(points=None, edges=None, openings=()):
    points = points or [(0, 0), (100, 0), (100, 100), (0, 100)]
    edges = edges if edges is not None else [(0, 1), (1, 2), (2, 3), (3, 0)]
    return dict(coordinate_space="source_raster", scale_cm_per_px=2,
                nodes=[dict(id=i, point=list(p), adjacency=[]) for i, p in enumerate(points)],
                edges=[dict(start=a, end=b, thickness_px=10) for a, b in edges], faces=[],
                diagnostics=dict(opening_bridges=list(openings), topology_gaps=[]))


class GraphExportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = json.loads((ROOT / "test/template.json").read_text())
        cls.windows = json.loads((ROOT / "test/template_window.json").read_text())

    def build(self, data, **kw):
        return build_graph_project(data, self.template, self.windows, **kw)

    def test_physical_faces_transform_area_and_template_immutability(self):
        data = graph()
        before, template = copy.deepcopy(data), copy.deepcopy(self.template)
        project, report = self.build(data, origin=(1000, -50))
        level = project["levels"][0]
        self.assertEqual((len(level["areas"]), len(level["segments"])), (1, 4))
        self.assertEqual((level["connectors"][1]["x"], level["connectors"][1]["y"]), (1200, -50))
        self.assertAlmostEqual(level["areas"][0]["area_value"], 90 * 90 * 4)
        self.assertEqual(report["quality"]["status"], "review")
        self.assertEqual(data, before)
        self.assertEqual(self.template, template)
        for segment in level["segments"]:
            self.assertEqual(segment["thickness"], 20)
            self.assertEqual(segment["left"], int(segment["start"][1:]))

    def test_topology_faces_cannot_create_physical_walls_or_areas(self):
        data = graph(edges=[(0, 1), (1, 2), (2, 3)])
        data["diagnostics"]["topology_gaps"] = [{"span": [[0, 100], [0, 0]]}]
        data["faces"] = [{"polygon": [n["point"] for n in data["nodes"]]}]
        project, report = self.build(data)
        self.assertEqual(len(project["levels"][0]["segments"]), 3)
        self.assertEqual(project["levels"][0]["areas"], [])
        self.assertTrue(report["quality"]["export_allowed"])
        self.assertEqual(report["quality"]["status"], "review")
        self.assertEqual(len(report["topology_gaps_omitted"]), 1)

    def test_interior_window_is_explicit_review_without_guessing_replacement(self):
        data=graph(points=[(0,0),(50,0),(100,0),(100,100),(50,100),(0,100)],
                   edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(1,4)],
                   openings=[dict(kind='windows',span=[[50,30],[50,70]]),
                             dict(kind='windows',span=[[0,30],[0,70]])])
        project,report=self.build(data)
        windows=report['bindings']['windows']
        self.assertEqual([w['exterior_context']['status'] for w in windows],['interior','exterior'])
        issues=[i for i in report['quality']['issues'] if i['code']=='interior_window_candidate']
        self.assertEqual(len(issues),1)
        self.assertEqual(issues[0]['element_uuid'],windows[0]['element_uuid'])
        self.assertEqual(len(project['levels'][0]['elements']),2)
        self.assertEqual(len(project['levels'][0]['areas']),2)
        self.assertEqual(report['quality']['status'],'review')

    def test_saved_graph_bound_door_is_not_reported_as_an_omitted_gap(self):
        data = graph(openings=[dict(kind='recovered_door', span=[[10,0], [90,0]],
                                    clear_span=[[15,0], [85,0]])])
        adjacent = dict(span=[[90,0], [100,0]])
        data['diagnostics']['topology_gaps'] = [dict(span=[[90,0], [10,0]]), adjacent]
        project, report = self.build(data)
        self.assertEqual(report['topology_gaps_omitted'], [adjacent])
        self.assertEqual([element['type'] for element in project['levels'][0]['elements']], ['doorItem'])

    def test_nominal_wall_joins_jamb_steps_and_rebinds_two_apertures(self):
        data = graph(points=[(0,0),(30,0),(30,.6),(70,.6),(70,0),(100,0),(100,100),(0,100)],
                     edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,0)],
                     openings=[dict(kind='doors', span=[[5,0],[25,0]]),
                               dict(kind='windows', span=[[35,.6],[65,.6]])])
        for i, edge in enumerate(data['edges']):
            edge['thickness_px'] = 4 + i
        old, _ = self.build(data)
        project, report = self.build(data, wall_thickness_cm=20)
        level = project['levels'][0]
        self.assertGreater(len(old['levels'][0]['connectors']), 4)
        self.assertEqual(len(level['connectors']), 4)
        self.assertEqual(len(level['areas']), 1)
        self.assertEqual(report['unbound_openings'], [])
        self.assertEqual({s['thickness'] for s in level['segments']}, {20})
        self.assertEqual({s['half_width_left'] for s in level['segments']}, {10})
        self.assertEqual(len({i['wall']['uuid'] for i in level['elements']}), 1)
        self.assertEqual(sorted(round(b['width_cm']) for group in report['bindings'].values() for b in group),
                         [40,60])
        self.assertEqual(level['elements'][0]['size']['z'], 20)

    def test_reviewed_axes_keep_host_and_correct_jamb_at_corner(self):
        data = graph(points=[(0,1),(51,-2),(100,0),(100,100),(0,100)],
                     edges=[(0,1),(1,2),(2,3),(3,4),(4,0)],
                     openings=[dict(kind='doors',span=[[0,1],[22,-.3]])])
        before = copy.deepcopy(data)
        review = dict(image_sha256='fixture', horizontal=[[0,0,100]], vertical=[],
                      jambs=[dict(kind='doors',near=[11,0],limits=[6,23])])
        project, report = self.build(data, wall_axis_review=review, wall_thickness_cm=20)
        level = project['levels'][0]
        self.assertEqual((len(level['connectors']), len(level['areas'])), (4,1))
        self.assertEqual(level['elements'][0]['size']['x'], 34)
        change = report['reviewed_wall_axes']['openings'][0]
        self.assertEqual(change['clear_span'], [[6,0],[23,0]])
        self.assertEqual(change['corner_clearance_px'], 1)
        self.assertEqual(data, before)
        review['jambs'][0]['limits'] = [2,23]
        with self.assertRaisesRegex(ValueError, 'perpendicular wall'):
            self.build(data, wall_axis_review=review, wall_thickness_cm=20)

    def test_scale_check_uses_both_reviewed_dimension_lines(self):
        review = dict(image_sha256="fixture", horizontal=[], vertical=[], dimension_lines=[
            dict(span=[[0, 110], [100, 110]], length_cm=200),
            dict(span=[[110, 0], [110, 60]], length_cm=121)])
        _, report = self.build(graph(), wall_thickness_cm=20, wall_axis_review=review)
        issues = {i["code"]: i for i in report["quality"]["issues"]}
        self.assertEqual(issues["scale_checked_on_dimensions"]["max_error_px"], .5)
        self.assertNotIn("approximate_scale", issues)
        review["dimension_lines"][1]["length_cm"] = 130
        _, report = self.build(graph(), wall_thickness_cm=20, wall_axis_review=review)
        codes = [i["code"] for i in report["quality"]["issues"]]
        self.assertIn("scale_reference_mismatch", codes)
        self.assertNotIn("scale_checked_on_dimensions", codes)

    def test_offset_junction_keeps_faces_and_openings_on_their_own_runs(self):
        for vertical in (False, True):
            def pt(x, y):
                return [y, x] if vertical else [x, y]
            points = [pt(0,0), pt(50,0), pt(100,0), pt(100,100), pt(50,100), pt(0,100)]
            data = graph(points=points, edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(1,4)],
                         openings=[dict(kind="doors", span=[pt(10,0), pt(30,0)]),
                                   dict(kind="doors", span=[pt(60,0), pt(80,0)])])
            step = dict(ends=[points[0], points[2]], junction=points[1], normals=[1,-1])
            review = dict(image_sha256="fixture", horizontal=[], vertical=[], offset_junctions=[step])
            original = copy.deepcopy(data)
            project, report = self.build(data, wall_axis_review=review, wall_thickness_cm=20)
            level = project["levels"][0]
            self.assertEqual((len(level["areas"]), len(level["segments"]), len(level["connectors"])), (2,8,7))
            self.assertTrue(report["quality"]["export_allowed"])
            self.assertEqual(report["reviewed_wall_axes"]["offset_junctions"], [step])
            self.assertEqual([b["pixel_center"] for b in report["bindings"]["doors"]], [pt(20,1), pt(70,-1)])
            self.assertEqual(data, original)
            step["normals"] = [1,20]
            with self.assertRaisesRegex(ValueError, "Invalid offset junction"):
                self.build(data, wall_axis_review=review, wall_thickness_cm=20)

    def test_single_door_substitution_for_pair_is_explicit_and_survives_simplification(self):
        data = graph(points=[(0,0),(50,0),(100,0),(100,100),(0,100)],
                     edges=[(0,1),(1,2),(2,3),(3,4),(4,0)],
                     openings=[dict(kind='doors',span=[[20,0],[80,0]],
                                    type_review=dict(evidence=dict(leaf_count=2)))])
        _, pending = self.build(data)
        project, approved = self.build(data, allow_single_door_for_paired_leaves=True)
        self.assertEqual(pending['bindings']['doors'][0]['model_selection'],'single_template_requires_review')
        self.assertEqual(approved['bindings']['doors'][0]['model_selection'],'single_template_user_approved')
        self.assertTrue(any('paired leaves' in w for w in pending['warnings']))
        self.assertFalse(any('paired leaves' in w for w in approved['warnings']))
        self.assertEqual(project['levels'][0]['elements'][0]['type'],'doorItem')
        self.assertEqual(len(approved['unbound_openings']),0)
        self.assertTrue(approved['merged_chains'] or approved['regularized_carriers'])

    def test_measured_openings_holes_models_materials_and_bindings(self):
        data = graph(openings=[dict(kind="recovered_door", span=[[10, 0], [90, 0]],
                                    clear_span=[[30, 0], [60, 0]]),
                               dict(kind="windows", span=[[100, 20], [100, 70]])])
        project, report = self.build(data)
        level = project["levels"][0]
        door, window = level["elements"]
        self.assertTrue(report["quality"]["export_allowed"])
        self.assertEqual(door["size"]["x"], 60)  # Neither 160 cm carrier nor x1.12 correction.
        self.assertEqual(door["wall"]["position"], 45)
        self.assertLess(window["size"]["x"], 100)  # Template installation allowance is inside the hole.
        for item in (door, window):
            self.assertAlmostEqual(item["scale"]["x"] * item["model"]["size"]["x"], item["size"]["x"], places=4)
            self.assertTrue(all(m["itemUuid"] == item["uuid"] for m in item["materials"]))
            xs = [p["x"] for p in item["wall"]["holeShape"]]
            self.assertAlmostEqual((max(xs) - min(xs)) * 100,
                                   60 if item["type"] == "doorItem" else 100, places=2)
            self.assertAlmostEqual(max(xs) + min(xs), 0, places=5)
        self.assertEqual(report["bindings"]["doors"][0]["width_source"], "clear_span")

    def test_open_passage_uses_real_template_and_floor_level_hole(self):
        template=json.loads((ROOT/'test/template_passage.json').read_text())
        before=copy.deepcopy(template)
        data=graph(openings=[dict(kind='passages',span=[[20,0],[80,0]])])
        project,report=self.build(data,passage_template=template)
        item=project['levels'][0]['elements'][0]
        self.assertEqual(item['type'],'wallOpening')
        self.assertEqual(item['size'],dict(x=120,y=180))
        self.assertEqual(item['position'],dict(z=0))
        self.assertNotIn('model',item);self.assertNotIn('sill',item);self.assertNotIn('scale',item)
        self.assertEqual(report['quality']['metrics']['exported_passages'],1)
        self.assertEqual(len(report['bindings']['passages']),1)
        self.assertEqual(len(report['bindings']['windows']),0)
        self.assertTrue(report['quality']['export_allowed'])
        self.assertEqual(template,before)
        recognition=dict(rooms=report['rooms'],doors=[],windows=[],passages=[{}])
        for mode in ('hole','wall','floor','binding'):
            changed=copy.deepcopy(project);bindings=copy.deepcopy(report['bindings'])
            e=changed['levels'][0]['elements'][0]
            if mode=='hole':e['wall']['holeShape'][0]['y']+=1
            elif mode=='wall':e['wall']['uuid']='missing'
            elif mode=='floor':e['position']['z']=50
            else:bindings['passages'][0]['wall_uuid']='missing'
            with self.subTest(mode=mode):
                self.assertFalse(validate_recognition(recognition,changed,bindings,2)['export_allowed'])

    def test_passage_requires_template_and_preserves_height_limit(self):
        data=graph(openings=[dict(kind='passages',span=[[20,0],[80,0]])])
        with self.assertRaises(ValueError):self.build(data)
        template=json.loads((ROOT/'test/template_passage.json').read_text())
        with self.assertRaises(ValueError):self.build(data,passage_template=template,height=150)

    def test_straight_split_carrier_is_merged_before_binding(self):
        data = graph(points=[(0, 0), (50, 0), (100, 0)], edges=[(0, 1), (1, 2)],
                     openings=[dict(kind="doors", span=[[20, 0], [80, 0]])])
        project, report = self.build(data)
        self.assertEqual(len(project["levels"][0]["elements"]), 1)
        self.assertTrue(report["quality"]["export_allowed"])
        self.assertAlmostEqual(project["levels"][0]["elements"][0]["size"]["x"], 120)

    def test_template_installation_allowance_cannot_enlarge_measured_aperture(self):
        template = copy.deepcopy(self.windows)
        original = next(e for e in template["levels"][0]["elements"] if e["type"] == "windowItem")
        for p in original["wall"]["holeShape"]:
            p["x"] = p["x"] * 1.2 + .1  # Non-centered template, substantial allowance.
        data = graph(openings=[dict(kind="windows", span=[[0, 0], [100, 0]])])
        project, report = build_graph_project(data, self.template, template)
        self.assertTrue(report["quality"]["export_allowed"])
        window = project["levels"][0]["elements"][0]
        self.assertLess(window["size"]["x"], 200)
        xs = [p["x"] for p in window["wall"]["holeShape"]]
        self.assertAlmostEqual(min(xs), -1, places=5)
        self.assertAlmostEqual(max(xs), 1, places=5)
        self.assertEqual(report["bindings"]["windows"][0]["width_cm"], 200)
        self.assertEqual(report["bindings"]["windows"][0]["element_width_cm"], window["size"]["x"])

    def test_t_junction_is_not_erased_to_fit_opening(self):
        data = graph(points=[(0, 0), (50, 0), (100, 0), (50, 30)], edges=[(0, 1), (1, 2), (1, 3)],
                     openings=[dict(kind="doors", span=[[20, 0], [80, 0]])])
        project, report = self.build(data)
        self.assertEqual(len(project["levels"][0]["segments"]), 3)
        self.assertEqual(project["levels"][0]["elements"], [])
        self.assertFalse(report["quality"]["export_allowed"])
        self.assertEqual(report["unbound_openings"][0]["reason"], "no_single_segment_contains_clear_span")

    def test_duplicate_apertures_block_draft(self):
        bridge = dict(kind="windows", span=[[20, 0], [80, 0]])
        project, report = self.build(graph(openings=[bridge, bridge]))
        self.assertEqual(len(project["levels"][0]["elements"]), 1)
        self.assertEqual(report["unbound_openings"][0]["reason"], "overlapping_openings")
        self.assertFalse(report["quality"]["export_allowed"])

    def test_bad_graph_references_and_geometry_fail_before_simplification(self):
        cases = []
        bad = graph(); bad["edges"][0]["end"] = 99; cases.append(bad)
        bad = graph(); bad["nodes"][0]["point"][0] = float("nan"); cases.append(bad)
        bad = graph(); bad["edges"].append(bad["edges"][0]); cases.append(bad)
        bad = graph(); bad["edges"][0]["thickness_px"] = -1; cases.append(bad)
        bad = graph(); bad["coordinate_space"] = "model"; cases.append(bad)
        cases.append(graph(points=[(0, 0), (10, 10), (0, 10), (10, 0)], edges=[(0, 1), (2, 3)]))
        cases.append(graph(edges=[]))
        for bad in cases:
            with self.subTest(data=bad), self.assertRaises(ValueError):
                self.build(bad)
        for scale in (0, -1, float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                self.build(graph(), scale=scale)

    def test_diagonal_binding_and_no_clamping_at_end(self):
        line = LineString([(0, 0), (100, 100)])
        binding, reason = opening_binding(dict(span=[[20, 20], [80, 80]]), [line])
        self.assertIsNone(reason)
        self.assertAlmostEqual(binding["width_px"], math.sqrt(2) * 60)
        self.assertAlmostEqual(binding["position"], 50)
        self.assertIsNone(opening_binding(dict(span=[[80, 80], [100.5, 100.5]]), [line])[0])
        self.assertIsNone(opening_binding(dict(span=[[20, 20], [80, 80]], clear_span=[[20, 50], [80, 90]]), [line])[0])

    def test_parallel_wall_is_not_used_as_fallback(self):
        binding, _ = opening_binding(dict(span=[[20, 0], [80, 0]]), [LineString([(0, 10), (100, 10)])])
        self.assertIsNone(binding)

    def test_area_omits_holes_instead_of_filling_them(self):
        data = graph(points=[(0, 0), (100, 0), (100, 100), (0, 100), (40, 40), (60, 40), (60, 60), (40, 60)],
                     edges=[(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4)])
        _, report = self.build(data)
        self.assertTrue(report["omitted_faces"])
        self.assertFalse(report["rooms"])

    def test_empty_rooms_exception_is_opt_in_and_does_not_relax_graph_checks(self):
        project, report = self.build(graph(edges=[(0, 1)]))
        source = dict(rooms=[], doors=[], windows=[])
        self.assertFalse(validate_recognition(source, project)["export_allowed"])
        self.assertTrue(validate_recognition(source, project, allow_empty_rooms=True)["export_allowed"])
        project["levels"][0]["segments"][0]["end"] = "missing"
        self.assertFalse(validate_recognition(source, project, allow_empty_rooms=True)["export_allowed"])
        self.assertFalse(validate_recognition(source, {"levels": []}, allow_empty_rooms=True)["export_allowed"])
        self.assertFalse(validate_recognition(source, {"levels": None}, allow_empty_rooms=True)["export_allowed"])


class ChainGeometryTest(unittest.TestCase):
    def regularize_fixture(self):
        return graph([(0, 0), (40, 0), (40.4, .1), (100, .1)], [(0, 1), (1, 2), (2, 3)],
                     [dict(kind="windows", span=[[10, 0], [40.2, 0]])])

    def test_opening_carrier_can_cross_sampling_kink_without_narrowing_hole(self):
        data = self.regularize_fixture()
        before = copy.deepcopy(data)
        result, decisions = regularize_opening_carriers(data, [0])
        self.assertTrue(decisions[0]["accepted"])
        self.assertLessEqual(decisions[0]["max_deviation_px"], .5)
        points = {n["id"]: n["point"] for n in result["nodes"]}
        lines = [LineString([points[e[k]] for k in ("start", "end")]) for e in result["edges"]]
        binding, reason = opening_binding(data["diagnostics"]["opening_bridges"][0], lines)
        self.assertIsNone(reason)
        self.assertAlmostEqual(binding["width_px"], 30.2, places=3)
        self.assertEqual(data, before)
        self.assertIn([0, 0], points.values())
        self.assertIn([100, .1], points.values())
        self.assertEqual(sum(len(n["adjacency"]) == 1 for n in result["nodes"]), 2)

    def test_junction_and_unsupported_gap_cannot_be_replaced(self):
        data = self.regularize_fixture()
        data["nodes"].append(dict(id=4, point=[40, 10], adjacency=[]))
        data["edges"].append(dict(start=1, end=4, thickness_px=10))
        result, decisions = regularize_opening_carriers(data, [0])
        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(result["edges"], data["edges"])
        data = self.regularize_fixture()
        data["edges"].pop(1)
        result, decisions = regularize_opening_carriers(data, [0])
        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(result["edges"], data["edges"])

    def test_corner_exceeding_error_budget_cannot_be_straightened(self):
        data = graph([(0, 0), (20, 1.5), (40, 0)], [(0, 1), (1, 2)],
                     [dict(kind="windows", span=[[5, 0], [35, 0]])])
        result, decisions = regularize_opening_carriers(data, [0])
        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(decisions[0]["reason"], "axis_deviation_exceeds_budget")
        self.assertEqual(result["edges"], data["edges"])

    def test_refined_chord_cannot_create_intersection(self):
        data = graph([(0, 0), (20, .4), (40, 0), (20, -.5), (20, .1)],
                     [(0, 1), (1, 2), (3, 4)], [dict(kind="windows", span=[[5, 0], [35, 0]])])
        result, decisions = regularize_opening_carriers(data, [0])
        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(decisions[0]["reason"], "new_intersection")
        self.assertEqual(result["edges"], data["edges"])

    def test_equally_near_parallel_carriers_are_ambiguous(self):
        data = graph([(0, -1), (50, -1), (0, 1), (50, 1)], [(0, 1), (2, 3)],
                     [dict(kind="windows", span=[[5, 0], [35, 0]])])
        result, decisions = regularize_opening_carriers(data, [0])
        self.assertFalse(decisions[0]["accepted"])
        self.assertEqual(decisions[0]["reason"], "missing_unique_carrier_anchors")
        self.assertEqual(result["edges"], data["edges"])

    def test_later_simplification_keeps_carrier_endpoints_and_error_budget(self):
        data = self.regularize_fixture()
        refined, decisions = regularize_opening_carriers(data, [0])
        protected = set(decisions[0]["protected_nodes"])
        result, _ = merge_straight_chains(refined, protected_nodes=protected)
        points = {n["id"]: n["point"] for n in result["nodes"]}
        self.assertTrue(protected.issubset(points))
        self.assertTrue(any({e["start"], e["end"]} == protected for e in result["edges"]))

    def test_template_margin_and_true_diagonal_are_supported(self):
        data = graph([(0, 0), (30, 30), (100, 100)], [(0, 1), (1, 2)],
                     [dict(kind="windows", span=[[10, 10], [60, 60]])])
        result, decisions = regularize_opening_carriers(data, [0], end_margins={0: 2})
        self.assertTrue(decisions[0]["accepted"])
        chord = LineString(decisions[0]["span"])
        self.assertAlmostEqual(chord.length, math.sqrt(2) * 50 + 4)
        binding, _ = opening_binding(data["diagnostics"]["opening_bridges"][0], [chord])
        self.assertAlmostEqual(binding["width_px"], math.sqrt(2) * 50)

    def test_corner_and_junction_are_preserved(self):
        for points, edges in [([(0, 0), (10, 0), (10, 10)], [(0, 1), (1, 2)]),
                              ([(0, 0), (10, 0), (20, 0), (10, 10)], [(0, 1), (1, 2), (1, 3)])]:
            data = graph(points, edges)
            result, merged = merge_straight_chains(data)
            self.assertEqual(result["edges"], data["edges"])
            self.assertFalse(merged)

    def test_cumulative_deviation_is_bounded_against_all_original_samples(self):
        points = [(i * 10, .07 * i * i) for i in range(12)]
        data = graph(points, [(i, i + 1) for i in range(len(points) - 1)])
        before = copy.deepcopy(data)
        result, merged = merge_straight_chains(data)
        self.assertTrue(merged)
        self.assertGreater(len(result["edges"]), 1)
        nodes = {n["id"]: n["point"] for n in result["nodes"]}
        for item in merged:
            edge = result["edges"][item["edge_index"]]
            line = LineString([nodes[edge["start"]], nodes[edge["end"]]])
            for i in item["source_edges"]:
                for uid in (data["edges"][i]["start"], data["edges"][i]["end"]):
                    self.assertLessEqual(line.distance(Point(points[uid])), .5)
        self.assertEqual(data, before)

    def test_merge_cannot_cross_nearby_partition(self):
        data = graph([(0, 0), (50, .4), (100, 0), (50, -.5), (50, .2)], [(0, 1), (1, 2), (3, 4)])
        result, merged = merge_straight_chains(data)
        self.assertEqual(len(result["edges"]), 3)
        self.assertFalse(merged)


if __name__ == "__main__":
    unittest.main()
