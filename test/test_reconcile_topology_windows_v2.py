import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from reconcile_topology_windows_v2 import reconcile_topology_windows


class TopologyWindowReconciliationTest(unittest.TestCase):
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
