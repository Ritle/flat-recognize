import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'scripts'))
from window_frame_evidence import narrow_frame_rails, bounded_frame_bay


class WindowFrameEvidenceTests(unittest.TestCase):
    def short_bay(self):
        gray = np.full((60,80),255,np.uint8)
        cv2.rectangle(gray,(10,26),(29,34),0,-1)
        cv2.rectangle(gray,(41,26),(60,34),0,-1)
        for y in (27,33):
            cv2.line(gray,(29,y),(41,y),0,1)
        return gray

    def test_antialiased_rails_survive_fractional_axis_offset(self):
        gray = np.full((80,100),255,np.uint8)
        for y in (27,33):
            cv2.line(gray,(20,y),(70,y),120,1)
        self.assertIsNotNone(narrow_frame_rails(gray,[20,29.55],[70,29.55],6))

    def test_short_bay_measures_aperture_separately_from_graph_span(self):
        gray = self.short_bay()
        bay = bounded_frame_bay(gray,[29,30],[41,30],6,[[24,30],[46,30]])
        self.assertIsNotNone(bay)
        self.assertEqual(bay['clear_span'],[[29.5,30],[40.5,30]])
        self.assertEqual(bay['raster_jamb_support'],[1,1])
        vertical = bounded_frame_bay(gray.T,[30,29],[30,41],6,[[30,24],[30,46]])
        self.assertEqual(vertical['clear_span'],[[30,29.5],[30,40.5]])

    def test_short_bay_cannot_use_free_rails_or_solid_ink_as_jambs(self):
        rails = np.full((60,80),255,np.uint8)
        for y in (27,33):
            cv2.line(rails,(10,y),(60,y),0,1)
        solid = self.short_bay()
        solid[27:34,29:42] = 0
        for gray in (rails,solid):
            self.assertIsNone(bounded_frame_bay(gray,[29,30],[41,30],6,[[24,30],[46,30]]))

    def frame(self, rails=(27,33)):
        gray = np.full((80,100),255,np.uint8)
        for y in rails:
            cv2.line(gray,(20,y),(70,y),0,1)
        return gray

    def test_two_and_three_rails_have_clear_interior(self):
        for rails in ((27,33),(26,30,34)):
            evidence = narrow_frame_rails(self.frame(rails),[20,30],[70,30],8)
            self.assertIsNotNone(evidence)
            self.assertEqual(len(evidence['rail_offsets_px']),len(rails))
            self.assertLess(evidence['inner_dark_fraction'],.35)

    def test_vertical_and_diagonal_frames(self):
        self.assertIsNotNone(narrow_frame_rails(self.frame().T,[30,20],[30,70],6))
        gray = np.full((100,100),255,np.uint8)
        cv2.line(gray,(20,24),(70,74),0,1)
        cv2.line(gray,(24,20),(74,70),0,1)
        self.assertIsNotNone(narrow_frame_rails(gray,[22,22],[72,72],6))

    def test_single_line_solid_wall_and_missing_rail_are_not_frames(self):
        solid = np.full((80,100),255,np.uint8)
        solid[26:35,20:71] = 0
        broken = self.frame()
        broken[33,35:60] = 255
        for gray in (self.frame((30,)), solid, broken):
            self.assertIsNone(narrow_frame_rails(gray,[20,30],[70,30],8))

    def test_rails_on_one_side_do_not_support_an_offset_candidate(self):
        self.assertIsNone(narrow_frame_rails(self.frame((32,35)),[20,30],[70,30],8))

    def test_outside_image_or_degenerate_spans_are_rejected(self):
        gray = self.frame()
        for a,b,width in (([20,0],[70,0],8),([20,30],[20,30],8),
                          ([20,30],[70,30],float('nan')),([20,30],[70,30],0)):
            self.assertIsNone(narrow_frame_rails(gray,a,b,width))


if __name__ == '__main__':
    unittest.main()
