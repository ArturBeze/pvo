
"""Run: python3 -m unittest -v test_tracker.py (no camera/display needed)."""

import unittest
import numpy as np
from tof_tracker import Motion, Intrinsics, Tracker, Settings, valid_mask

class Checks(unittest.TestCase):
    def test_prediction_uses_seconds_and_depth(self):
        motion = Motion()
        for t in [0, .04, .09, .14, .21]:
            motion.add(t, np.array([.2 + .5*t, -.1, 2 - t]), 10)
        np.testing.assert_allclose(motion.velocity(), [.5, 0, -1], atol=1e-8)
        np.testing.assert_allclose(motion.predict(.31), [.355, -.1, 1.69], atol=1e-8)

    def test_projection_changes_as_object_approaches(self):
        k = Intrinsics(100, 100, 50, 50)
        np.testing.assert_allclose(k.project([.2, 0, 2]), [60, 50])
        np.testing.assert_allclose(k.project([.2, 0, 1]), [70, 50])
        np.testing.assert_allclose(k.unproject(60, 50, 2000), [.2, 0, 2])

    def test_invalid_depth_and_confidence(self):
        depth = np.array([[0, np.nan, np.inf, 500, 1000, 2000]], np.float32)
        conf = np.array([[100,100,100,100,1,100]], np.float32)
        np.testing.assert_array_equal(valid_mask(depth, conf, Settings(near=300, far=1500)),
                                      [[False,False,False,True,False,False]])

    def test_selection_follows_target_ignores_distractor_and_loses(self):
        k = Intrinsics(100,100,50,40)
        s = Settings(near=300, far=2000, min_area=9, gate_px=20, lost=2)
        tracker = Tracker()
        def scene(x, z=1000):
            d = np.full((80,100),3000,np.float32)
            d[30:40,x:x+10] = z
            d[50:70,70:90] = 1000  # larger distractor
            return d
        tracker.click = (24,34)
        for i in range(5):
            tracker.update(scene(20+i*2,1000-i*10), None, i*.05, s, k)
        self.assertTrue(tracker.active)
        self.assertGreater(tracker.motion.velocity()[0], 0)
        self.assertLess(tracker.motion.velocity()[2], 0)
        blank = np.full((80,100),3000,np.float32)
        for i in range(3):
            tracker.update(blank,None,.25+i*.05,s,k)
        self.assertFalse(tracker.active)
        self.assertEqual(len(tracker.motion.samples),0)
        tracker.update(scene(60),None,.5,s,k)
        self.assertFalse(tracker.active)  # no automatic identity switch

if __name__ == '__main__':
    unittest.main()
