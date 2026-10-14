import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from filter_interior_windows_v2 import (duplicate_parallel_window_layers,
                                        filter_interior_windows,
                                        one_sided_window_endcaps,
                                        one_sided_unconfirmed_windows,
                                        parallel_weak_window_pairs)
from recheck_reviewed_rooms import recheck_reviewed_rooms
from review_decisions import (apply_review_decisions, make_review_questions,
                              opening_fingerprint)
from validate_recognition import validate_recognition


def window(uid, x1, y1, x2, y2):
    return {"id": uid, "outer": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
            "bbox": [x1, y1, x2, y2], "center": [(x1+x2)/2, (y1+y2)/2],
            "type": "window"}


class InteriorWindowFilterTests(unittest.TestCase):
    def test_exterior_one_sided_window_is_kept_for_incomplete_facade_jamb(self):
        facade = window(1, 70, 10, 110, 16)
        interior = window(2, 70, 95, 110, 101)
        for item in (facade, interior):
            item["source_class"] = "window"
            item["classification"] = {"method": "jamb_leaf_and_arc"}
        with patch("filter_interior_windows_v2.wall_jamb_support", return_value=[3, 0]):
            candidates = one_sided_unconfirmed_windows(
                [facade, interior], Polygon([(10, 10), (190, 10),
                                             (190, 190), (10, 190)]))
        self.assertEqual([item["id"] for item in candidates], [2])

    def test_one_sided_door_sourced_window_waits_for_review(self):
        candidate = window(2, 70, 20, 110, 26)
        candidate["source_class"] = "door"
        candidate["classification"] = {"method": "jamb_leaf_and_arc",
                                        "confirmed": False, "requires_review": False}
        self.classified["windows"] = [candidate]
        self.classified["openings"] = copy.deepcopy([candidate])
        self.rooms["windows"] = [candidate]
        with patch("filter_interior_windows_v2.wall_jamb_support", return_value=[3, 0]), \
             patch("filter_interior_windows_v2.extract_rooms") as extract:
            classified, rooms, report = filter_interior_windows(
                self.classified, self.rooms, self.rgb)
        extract.assert_not_called()
        self.assertEqual(classified["windows"], [])
        self.assertEqual(rooms["windows"], [])
        self.assertEqual([item["id"] for item in classified["topology_only_openings"]], [2])
        self.assertEqual(report["one_sided_unconfirmed_windows"][0]["jamb_support"], [3, 0])
        self.assertEqual(rooms["topology_only_openings"][0]["classification"]["state"],
                         "uncertain")
        questions = make_review_questions(classified, {"issues": []}, "a" * 64)
        self.assertEqual(questions[0]["reasons"], ["one_sided_unconfirmed_window"])
        self.assertEqual(questions[0]["current"], "topology_only")
        reviewed = apply_review_decisions(classified, {
            "version": 1, "image_sha256": "a" * 64,
            "answers": [{"opening_id": 2, "choice": "window",
                         "fingerprint": opening_fingerprint(candidate)}]}, "a" * 64)
        self.assertEqual([item["id"] for item in reviewed["windows"]], [2])

    def test_two_jamb_candidate_stays_exportable(self):
        candidate = window(2, 70, 20, 110, 26)
        candidate["source_class"] = "window"
        candidate["classification"] = {"method": "jamb_leaf_and_arc"}
        self.classified["windows"] = [candidate]
        self.classified["openings"] = copy.deepcopy([candidate])
        self.rooms["windows"] = [candidate]
        with patch("filter_interior_windows_v2.wall_jamb_support", return_value=[3, 3]):
            classified, _, _ = filter_interior_windows(self.classified, self.rooms, self.rgb)
        self.assertEqual([item["id"] for item in classified["windows"]], [2])

    def test_parallel_window_layers_collapse_only_near_identical_thin_spans(self):
        first = window(1, 20, 30, 100, 42)
        second = window(2, 23, 39, 97, 59)
        distant = window(3, 20, 75, 100, 87)
        duplicates = duplicate_parallel_window_layers([first, second, distant])
        self.assertEqual([(item["id"], item["paired_with"]) for item in duplicates],
                         [(2, 1)])

    def test_duplicate_layer_stays_in_room_mask_but_not_exported_windows(self):
        first = window(1, 20, 30, 100, 42)
        second = window(2, 23, 39, 97, 59)
        self.classified["windows"] = [first, second]
        self.classified["openings"] = copy.deepcopy([first, second])
        self.rooms["windows"] = copy.deepcopy([first, second])
        with patch("filter_interior_windows_v2.wall_jamb_support", return_value=[3, 3]):
            classified, rooms, report = filter_interior_windows(
                self.classified, self.rooms, self.rgb)
        self.assertEqual([item["id"] for item in classified["windows"]], [1])
        self.assertEqual([item["id"] for item in rooms["windows"]], [1])
        self.assertEqual([item["id"] for item in classified["topology_only_openings"]], [2])
        self.assertEqual([item["id"] for item in rooms["topology_only_openings"]], [2])
        self.assertEqual(report["rejected_windows"], 1)
        self.assertEqual(report["duplicate_parallel_layers"][0]["paired_with"], 1)
        self.assertEqual(validate_recognition(rooms)["status"], "review")

    def test_short_one_sided_endcap_is_topology_only(self):
        parent = window(4, 20, 30, 120, 50)
        fragment = window(3, 119, 44, 124, 72)
        self.classified["windows"] = [parent, fragment]
        self.classified["openings"] = copy.deepcopy([parent, fragment])
        self.rooms["windows"] = copy.deepcopy([parent, fragment])
        self.rooms["rooms"][0]["polygon"] = [[20, 50], [180, 50],
                                             [180, 180], [20, 180]]
        def support(start, end, width, walls):
            return [3, 3] if abs(end[0] - start[0]) > abs(end[1] - start[1]) else [3, 0]
        with patch("filter_interior_windows_v2.wall_jamb_support", side_effect=support):
            classified, rooms, report = filter_interior_windows(
                self.classified, self.rooms, self.rgb)
        self.assertEqual([item["id"] for item in report["one_sided_window_endcaps"]], [3])
        self.assertEqual([item["id"] for item in classified["windows"]], [4])
        self.assertEqual([item["id"] for item in rooms["topology_only_openings"]], [3])
        self.assertEqual(classified["topology_only_openings"][0]["classification"]["method"],
                         "one_sided_window_endcap")
        self.assertEqual(validate_recognition(rooms)["status"], "review")

    def test_real_corner_window_is_not_rejected_as_short_endcap(self):
        parent = window(4, 20, 30, 120, 50)
        corner = window(3, 119, 44, 129, 104)
        def support(start, end, width, walls):
            return [3, 3] if abs(end[0] - start[0]) > abs(end[1] - start[1]) else [3, 0]
        with patch("filter_interior_windows_v2.wall_jamb_support", side_effect=support):
            self.assertEqual(one_sided_window_endcaps([parent, corner], Polygon()), [])

    def test_parallel_candidates_with_weak_jambs_are_review_only(self):
        first = window(1, 20, 30, 100, 42)
        second = window(2, 25, 38, 95, 50)
        separate = window(3, 120, 30, 180, 42)
        with patch("filter_interior_windows_v2.wall_jamb_support",
                   side_effect=[[2, 0], [2, 1], [0, 0]]):
            candidates = parallel_weak_window_pairs(
                [first, second, separate], Polygon())
        self.assertEqual([(item["id"], item["paired_with"]) for item in candidates],
                         [(1, [2]), (2, [1])])

    def test_parallel_pair_with_two_supported_jambs_stays_unflagged(self):
        first = window(1, 20, 30, 100, 42)
        second = window(2, 25, 38, 95, 50)
        with patch("filter_interior_windows_v2.wall_jamb_support",
                   side_effect=[[2, 2], [2, 3]]):
            self.assertEqual(parallel_weak_window_pairs([first, second], Polygon()), [])

    def setUp(self):
        self.rgb = np.full((200, 200, 3), 255, np.uint8)
        interior = window(1, 70, 95, 110, 101)
        boundary = window(2, 70, 20, 110, 26)
        self.classified = {"meta": {"width": 200, "height": 200}, "walls": [],
                           "doors": [], "windows": [interior, boundary],
                           "openings": copy.deepcopy([interior, boundary])}
        self.rooms = {"meta": {"width": 200, "height": 200}, "windows": [interior, boundary],
                      "doors": [], "rooms": [{"id": 0, "polygon": [[20, 20], [180, 20],
                                                               [180, 180], [20, 180]]}],
                      "diagnostics": {"warnings": []}}
    def test_rejects_only_deep_interior_and_rebuilds_rooms(self):
        original = copy.deepcopy((self.classified, self.rooms))
        rebuilt = copy.deepcopy(self.rooms)
        rebuilt["windows"] = rebuilt["windows"][1:]
        with patch("filter_interior_windows_v2.extract_rooms", return_value=(rebuilt, None, None)):
            classified, rooms, report = filter_interior_windows(
                self.classified, self.rooms, self.rgb)
        self.assertEqual(report["rejected_windows"], 1)
        self.assertEqual([w["id"] for w in classified["windows"]], [2])
        self.assertEqual([w["id"] for w in rooms["windows"]], [2])
        self.assertEqual(len(rooms["rooms"]), 1)
        self.assertEqual(classified["rejected_openings"][0]["id"], 1)
        self.assertTrue(report["topology_stable"])
        self.assertEqual((self.classified, self.rooms), original)

    def test_no_candidate_needs_no_recheck(self):
        self.classified["windows"] = self.classified["windows"][1:]
        self.classified["openings"] = self.classified["openings"][1:]
        self.rooms["windows"] = self.rooms["windows"][1:]
        with patch("filter_interior_windows_v2.extract_rooms") as extract:
            _, _, report = filter_interior_windows(self.classified, self.rooms, self.rgb)
        self.assertEqual(report["rejected_windows"], 0)
        extract.assert_not_called()

    def test_same_room_on_both_sides_is_diagnostic_only(self):
        # A narrow indentation can be furniture, or a merged real opening.
        self.rooms["rooms"][0]["polygon"] = [[20, 20], [180, 20], [180, 180],
                                             [20, 180], [20, 101], [120, 101],
                                             [120, 95], [20, 95]]
        self.classified["windows"][0]["source_class"] = "window"
        with patch("filter_interior_windows_v2.extract_rooms") as extract:
            classified, rooms, report = filter_interior_windows(self.classified, self.rooms, self.rgb)
        self.assertEqual([candidate["id"] for candidate in report["ambiguous_candidates"]], [1])
        self.assertEqual(report["ambiguous_candidates"][0]["reason"],
                         "same_room_both_sides_weak_jamb_window")
        self.assertEqual([window["id"] for window in classified["windows"]], [1, 2])
        self.assertEqual(report["rejected_windows"], 0)
        extract.assert_not_called()
        quality = validate_recognition(rooms)
        self.assertTrue(any(issue["code"] == "same_room_window_ambiguous"
                            and issue["severity"] == "warning"
                            and issue["opening_id"] == 1 for issue in quality["issues"]))

    def test_review_recomputes_uncertainty_for_remaining_windows(self):
        self.rooms["rooms"][0]["polygon"] = [[20, 20], [180, 20], [180, 180],
                                             [20, 180], [20, 101], [120, 101],
                                             [120, 95], [20, 95]]
        previous = copy.deepcopy(self.rooms)
        previous["diagnostics"]["interior_window_filter"] = {
            "ambiguous_windows": 1, "ambiguous_candidates": [{"id": 999}]}
        previous["diagnostics"]["opening_recovery"] = {"candidates": 2}
        previous["reclassified_wall_support"] = [{"source_opening_id": 9,
                                                   "outer": [[0, 0], [1, 0], [1, 1]]}]
        reviewed = copy.deepcopy(self.rooms)
        result = recheck_reviewed_rooms(self.classified, previous, reviewed, self.rgb)
        report = result["diagnostics"]["interior_window_filter"]
        self.assertEqual([candidate["id"] for candidate in report["ambiguous_candidates"]], [1])
        self.assertEqual(result["diagnostics"]["opening_recovery"], {"candidates": 2})
        self.assertEqual(result["reclassified_wall_support"],
                         previous["reclassified_wall_support"])
        quality = validate_recognition(result)
        self.assertEqual(quality["status"], "review")
        self.assertIn(1, [question["opening_id"] for question in
                          make_review_questions(self.classified, quality, "a" * 64)])
        self.assertNotIn(999, [question["opening_id"] for question in
                               make_review_questions(self.classified, quality, "a" * 64)])

    def test_review_surfaces_new_deep_interior_window(self):
        reviewed = copy.deepcopy(self.rooms)
        result = recheck_reviewed_rooms(self.classified, self.rooms, reviewed, self.rgb)
        report = result["diagnostics"]["interior_window_filter"]
        self.assertEqual([candidate["id"] for candidate in report["deep_interior_candidates"]], [1])
        quality = validate_recognition(result)
        self.assertTrue(any(issue["code"] == "reviewed_window_unsupported"
                            and issue["opening_id"] == 1 for issue in quality["issues"]))

    def test_merged_room_does_not_remove_door_class_opening(self):
        self.rooms["rooms"][0]["polygon"] = [[20, 20], [180, 20], [180, 180],
                                             [20, 180], [20, 101], [120, 101],
                                             [120, 95], [20, 95]]
        self.classified["windows"][0]["source_class"] = "door"
        with patch("filter_interior_windows_v2.extract_rooms") as extract:
            classified, _, report = filter_interior_windows(self.classified, self.rooms, self.rgb)
        self.assertEqual([window["id"] for window in classified["windows"]], [1, 2])
        self.assertEqual(report["ambiguous_windows"], 1)
        extract.assert_not_called()

    def test_changed_topology_keeps_provisional_rooms_and_blocks_export(self):
        rebuilt = copy.deepcopy(self.rooms)
        rebuilt["rooms"] = []
        with patch("filter_interior_windows_v2.extract_rooms", return_value=(rebuilt, None, None)):
            _, rooms, report = filter_interior_windows(self.classified, self.rooms, self.rgb)
        self.assertFalse(report["topology_stable"])
        self.assertEqual(len(rooms["rooms"]), 1)
        self.assertEqual([w["id"] for w in rooms["windows"]], [2])
        quality = validate_recognition(rooms)
        self.assertEqual(quality["status"], "invalid")
        self.assertTrue(any(issue["code"] == "unstable_room_topology"
                            for issue in quality["issues"]))

    def test_coordinate_mismatch(self):
        self.rooms["meta"]["width"] = 199
        with self.assertRaisesRegex(ValueError, "coordinates"):
            filter_interior_windows(self.classified, self.rooms, self.rgb)

    def test_unsupported_perpendicular_endcap_is_removed_without_room_change(self):
        parent = window(10, 20, 20, 120, 35)
        endcap = window(11, 100, 30, 113, 75)
        separate = window(12, 145, 30, 158, 75)
        self.classified["windows"] = [parent, endcap, separate]
        self.classified["openings"] = copy.deepcopy(self.classified["windows"])
        self.rooms["rooms"] = []
        self.rooms["windows"] = copy.deepcopy(self.classified["windows"])
        rebuilt = copy.deepcopy(self.rooms)
        rebuilt["windows"] = [parent, separate]
        def jamb_support(start, end, width, walls):
            return [3, 3] if abs(end[0] - start[0]) > abs(end[1] - start[1]) else [0, 0]
        with patch("filter_interior_windows_v2.extract_rooms", return_value=(rebuilt, None, None)):
            with patch("filter_interior_windows_v2.wall_jamb_support", side_effect=jamb_support):
                classified, rooms, report = filter_interior_windows(
                    self.classified, self.rooms, self.rgb)
        self.assertEqual([item["id"] for item in report["perpendicular_endcaps"]], [11])
        self.assertEqual([item["id"] for item in classified["windows"]], [10, 12])
        self.assertEqual([item["id"] for item in rooms["windows"]], [10, 12])
        self.assertEqual(rooms["reclassified_wall_support"][0]["source_opening_id"], 11)
        self.assertTrue(report["topology_stable"])


if __name__ == "__main__":
    unittest.main()
