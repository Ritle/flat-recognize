import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from reconcile_topology_windows_v2 import (adjacent_solid_wall, expanded_component,
                                           reconcile_topology_windows, verify_single_strip)


class TopologyWindowReconciliationTest(unittest.TestCase):
    def test_single_strip_requires_continuous_colored_wall_and_local_room_growth(self):
        rgb = np.full((70, 100, 3), 255, np.uint8)
        rgb[40:46, 10:90] = (190, 80, 0)
        candidate = {"outer": [[20, 34], [70, 34], [70, 39], [20, 39]]}
        data = {"walls": [{"outer": [[10, 40], [89, 40], [89, 45], [10, 45]]}]}
        trial = np.full(rgb.shape[:2], 255, np.uint8)
        trial[10:40, 10:90] = 0
        barrier = trial.copy()
        barrier[34:40, 20:71] = 255
        zeros = np.zeros_like(trial)
        room = {"id": 1, "polygon": [[10, 10], [89, 10], [89, 39], [10, 39]]}
        before = {"rooms": [dict(room, area_px=int(np.sum(barrier == 0)))]}
        after = {"rooms": [dict(room, area_px=int(np.sum(trial == 0)))]}
        with patch("reconcile_topology_windows_v2.extract_rooms", return_value=(before, barrier, zeros)):
            def verify():
                return verify_single_strip(data, before, after, rgb, trial, zeros, candidate)
            self.assertEqual(verify()["room_id"], 1)
            rgb[40:46, 50] = 255
            self.assertIsNone(verify())
            rgb[40:46, 50] = (190, 80, 0)
            rgb[35:39, 35:45] = 0
            self.assertIsNone(verify())
            rgb[35:39, 35:45] = 255
            barrier[20:24, 20:25] = 255
            self.assertIsNone(verify())

    def test_solid_band_must_span_edit_and_removed_ink_must_be_thin(self):
        walls = np.zeros((70, 100), dtype=np.uint8)
        walls[45:51, 10:90] = 255
        rgb = np.repeat((255 - walls)[:, :, None], 3, axis=2)
        removed = np.zeros_like(walls)
        removed[35:42, 20:75] = 255
        room = np.zeros_like(walls, dtype=bool)
        room[20:45, 10:90] = True
        self.assertIsNotNone(adjacent_solid_wall(rgb, walls, removed, room))
        self.assertIsNotNone(adjacent_solid_wall(rgb.transpose(1, 0, 2), walls.T, removed.T, room.T))
        rgb[36:40, 30:70] = 0
        self.assertIsNone(adjacent_solid_wall(rgb, walls, removed, room))
        rgb[36:40, 30:70] = 255
        rgb[45:51, 40] = 255
        self.assertIsNone(adjacent_solid_wall(rgb, walls, removed, room))

    def test_released_pixels_cannot_connect_to_another_free_region(self):
        barrier = np.full((20, 20), 255, dtype=np.uint8)
        barrier[2:10, 2:10] = 0
        removed = np.zeros_like(barrier)
        removed[2:10, 10] = 255
        old = [{"id": 1, "polygon": [[2, 2]], "area_px": 64}]
        new = [dict(old[0], area_px=72)]
        self.assertIsNotNone(expanded_component(old, new, barrier, barrier & ~removed, removed))
        barrier[3:7, 11:15] = 0
        self.assertIsNone(expanded_component(old, new, barrier, barrier & ~removed, removed))

    def setUp(self):
        self.opening = {"id": 2, "outer": [[40, 30], [50, 30], [50, 80], [40, 80]],
                        "classification": {"method": "one_sided_unconfirmed_window"}}
        self.classified = {"topology_only_openings": [copy.deepcopy(self.opening)]}
        self.rooms = {"rooms": [{"polygon": [[20, 20], [100, 20], [100, 100], [20, 100]]}],
                      "diagnostics": {"opening_recovery": {"recovered_doors": 1}}}
        self.rgb = np.full((120, 120, 3), 255, dtype=np.uint8)
        self.edge = {"start": [40, 35], "end": [40, 75]}

    def test_accepts_only_stable_rooms_with_fewer_unsupported_edges(self):
        def rebuilt(classified, _rgb):
            result = copy.deepcopy(self.rooms)
            result["topology_only_openings"] = copy.deepcopy(classified["topology_only_openings"])
            return result, None, None

        with (patch("reconcile_topology_windows_v2.requires_polygons", return_value=True),
              patch("reconcile_topology_windows_v2.extract_rooms", side_effect=rebuilt),
              patch("reconcile_topology_windows_v2.project_geometry",
                    side_effect=[{"unsupported_segments": [self.edge]},
                                 {"unsupported_segments": []}])):
            classified, rooms, report = reconcile_topology_windows(
                self.classified, self.rooms, self.rgb, {})
        self.assertEqual(report["removed_from_room_mask"], [2])
        self.assertFalse(classified["topology_only_openings"][0]["topology_mask_use"])
        self.assertEqual(rooms["diagnostics"]["opening_recovery"], {"recovered_doors": 1})
        self.assertNotIn("topology_mask_use", self.classified["topology_only_openings"][0])

        with (patch("reconcile_topology_windows_v2.requires_polygons", return_value=True),
              patch("reconcile_topology_windows_v2.extract_rooms", side_effect=rebuilt),
              patch("reconcile_topology_windows_v2.project_geometry",
                    side_effect=[{"unsupported_segments": [self.edge]},
                                 {"unsupported_segments": [self.edge]}])):
            classified, _, report = reconcile_topology_windows(
                self.classified, self.rooms, self.rgb, {})
        self.assertEqual(report["removed_from_room_mask"], [])
        self.assertNotIn("topology_mask_use", classified["topology_only_openings"][0])


if __name__ == "__main__":
    unittest.main()
