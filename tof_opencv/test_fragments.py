"""python -m unittest discover -s tof_opencv -p test_fragments.py -v"""
import unittest
import numpy as np
from view_camera import nearest_fragment


class FragmentTests(unittest.TestCase):
    def detect(self, depth, confidence=None, area=4, tolerance=50):
        if confidence is None:
            confidence = np.full(depth.shape, 300, np.float32)
        return nearest_fragment(depth, confidence, 200, 300, 1200, tolerance, area)

    def test_rejects_nearest_speck_and_selects_near_fragment(self):
        depth = np.zeros((10, 15), np.float32)
        depth[0, 0] = 300
        depth[2:4, 2:5] = 500
        depth[6:9, 8:12] = 900
        result = self.detect(depth)
        self.assertEqual(result['area'], 6)
        self.assertEqual(result['distance'], 500)
        self.assertEqual(result['confidence'], 300)

    def test_adjacent_surfaces_separate_by_depth(self):
        depth = np.full((4, 6), 900, np.float32)
        depth[:, :3] = 500
        result = self.detect(depth)
        self.assertEqual(result['area'], 12)
        self.assertEqual(result['distance'], 500)

    def test_holes_excluded_and_local_tolerance_inclusive(self):
        depth = np.tile(np.array([500, 550, 600], np.float32), (3, 1))
        confidence = np.full(depth.shape, 300, np.float32)
        confidence[1, 1] = 199
        result = self.detect(depth, confidence)
        self.assertEqual(result['area'], 8)
        self.assertFalse(result['mask'][1, 1])
        self.assertEqual(result['distance'], 550)

    def test_diagonal_contact_does_not_merge(self):
        depth = np.array([[500, 0], [0, 500]], np.float32)
        self.assertIsNone(self.detect(depth, area=2))

    def test_empty_and_invalid(self):
        depth = np.array([[0, np.nan, np.inf, 299, 1201]], np.float32)
        self.assertIsNone(self.detect(depth, area=1))


if __name__ == '__main__':
    unittest.main()
