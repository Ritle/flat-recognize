import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from add_openings import choose_segment_binding, door_width_cm


def segment(uid, ax, ay, bx, by):
    return {"segment": {"uuid": uid, "thickness": 20},
            "ax": ax, "ay": ay, "bx": bx, "by": by}


class OpeningBindingSelectionTests(unittest.TestCase):
    def test_door_minimum_width_is_used_for_selection_and_export(self):
        self.assertEqual(door_width_cm(20), 40)
        self.assertAlmostEqual(door_width_cm(80), 89.6)

    def test_nearby_full_span_beats_closer_short_segment(self):
        segments = [segment("short", 0, 0, 70, 0),
                    segment("full", -20, 2, 100, 2)]
        result = choose_segment_binding(segments, 35, 0, 90, "horizontal")
        self.assertEqual(result["segment"]["uuid"], "full")
        self.assertEqual(result["selection"], "span_aware")
        self.assertEqual(result["overhang"], 0)
        self.assertEqual(result["nearest_distance"], 0)

    def test_fitting_nearest_segment_is_unchanged(self):
        segments = [segment("nearest", 0, 0, 100, 0),
                    segment("other", -20, 2, 120, 2)]
        result = choose_segment_binding(segments, 50, 0, 80, "horizontal")
        self.assertEqual(result["segment"]["uuid"], "nearest")
        self.assertEqual(result["selection"], "nearest")

    def test_distant_or_perpendicular_alternative_is_not_used(self):
        segments = [segment("short", 0, 0, 70, 0),
                    segment("far", -20, 12, 100, 12),
                    segment("perpendicular", 35, -60, 35, 60)]
        result = choose_segment_binding(segments, 35, 0, 90, "horizontal")
        self.assertEqual(result["segment"]["uuid"], "short")
        self.assertEqual(result["selection"], "nearest")
        self.assertGreater(result["overhang"], 5)


if __name__ == "__main__":
    unittest.main()
