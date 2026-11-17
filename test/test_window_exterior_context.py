import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from window_exterior_context import window_exterior_context


class ExteriorContextTest(unittest.TestCase):
    def test_facade_faces_outside_regardless_of_image_margins(self):
        rooms=[dict(polygon=[[100,100],[200,100],[200,200],[100,200]])]
        result=window_exterior_context([[100,125],[100,175]],8,rooms)
        self.assertEqual(result['status'],'exterior')

    def test_between_rooms_and_inside_same_room_are_not_exterior(self):
        rooms=[dict(polygon=[[0,0],[50,0],[50,100],[0,100]]),
               dict(polygon=[[50,0],[100,0],[100,100],[50,100]])]
        for span in ([[50,30],[50,70]],[[20,30],[20,70]]):
            with self.subTest(span=span):
                self.assertEqual(window_exterior_context(span,8,rooms)['status'],'interior')

    def test_concave_facade_inside_bounding_box_is_exterior(self):
        rooms=[dict(polygon=[[0,0],[100,0],[100,40],[40,40],[40,100],[0,100]])]
        self.assertEqual(window_exterior_context([[40,55],[40,85]],6,rooms)['status'],'exterior')

    def test_diagonal_facade(self):
        rooms=[dict(polygon=[[0,100],[100,0],[200,100],[100,200]])]
        self.assertEqual(window_exterior_context([[30,70],[70,30]],8,rooms)['status'],'exterior')

    def test_missing_rooms_and_enclosed_void_stay_unknown(self):
        self.assertEqual(window_exterior_context([[0,20],[0,60]],8,[])['status'],'unknown')
        rooms=[dict(polygon=[[0,0],[200,0],[200,200],[0,200]],
                    holes=[[[60,60],[140,60],[140,140],[60,140]]])]
        self.assertEqual(window_exterior_context([[60,80],[60,120]],8,rooms)['status'],'unknown')

    def test_mixed_span_is_not_forced_to_window_or_wall(self):
        rooms=[dict(polygon=[[0,0],[100,0],[100,100],[0,100]])]
        self.assertEqual(window_exterior_context([[-20,0],[40,0]],8,rooms)['status'],'unknown')


if __name__=='__main__':unittest.main()
