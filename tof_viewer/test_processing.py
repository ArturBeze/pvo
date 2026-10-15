import unittest
import numpy as np
from processing import Processor, Settings


class ProcessingTests(unittest.TestCase):
    def run_frame(self, d, c, **kwargs):
        return Processor().process(d, c, Settings(**kwargs), 4000)

    def test_no_measurements_stays_unknown(self):
        p, valid, filled = self.run_frame(np.ones((20, 20))*1000, np.zeros((20, 20)))
        self.assertTrue(np.isnan(p).all())
        self.assertFalse(valid.any() or filled.any())

    def test_small_hole_filled_without_mutating_inputs(self):
        d = np.full((20, 20), 1000, np.float32)
        c = np.full_like(d, 100)
        d[10, 10], c[10, 10] = np.nan, 0
        before = d.copy()
        p, valid, filled = self.run_frame(d, c)
        self.assertTrue(filled[10, 10])
        self.assertFalse(valid[10, 10])
        self.assertAlmostEqual(float(p[10, 10]), 1000, places=2)
        np.testing.assert_array_equal(d, before)

    def test_hole_at_edge_not_invented(self):
        d = np.full((20, 20), 1000, np.float32)
        d[:, 10:] = 2000
        c = np.full_like(d, 100)
        c[10, 10] = 0
        p, _, filled = self.run_frame(d, c)
        self.assertFalse(filled[10, 10])
        self.assertTrue(np.isnan(p[10, 10]))
        self.assertLess(abs(float(p[5, 9])-1000), 1)

    def test_large_hole_center_unknown(self):
        d = np.full((30, 30), 1000, np.float32)
        c = np.full_like(d, 100)
        c[5:25, 5:25] = 0
        p, _, _ = self.run_frame(d, c)
        self.assertTrue(np.isnan(p[15, 15]))

    def test_temporal_does_not_keep_lost_or_moving_surface(self):
        processor = Processor()
        d = np.full((20, 20), 1000, np.float32)
        c = np.full_like(d, 100)
        settings = Settings(fill_radius=0, smooth_radius=0, alpha=0.1)
        processor.process(d, c, settings, 4000)
        p, _, _ = processor.process(d+1000, c, settings, 4000)
        np.testing.assert_allclose(p, 2000)
        p, _, _ = processor.process(d, c*0, settings, 4000)
        self.assertTrue(np.isnan(p).all())

    def test_smoothing_reduces_noise(self):
        rng = np.random.default_rng(2)
        d = (1000+rng.normal(0, 20, (60, 60))).astype(np.float32)
        p, _, _ = self.run_frame(d, np.full_like(d, 100))
        self.assertLess(p.std(), d.std()*0.6)

    def test_invalid_depths_rejected(self):
        d = np.array([[np.nan, np.inf, -1, 0, 4001, 1000]], np.float32)
        p, valid, _ = self.run_frame(d, np.full_like(d, 100), fill_radius=0, smooth_radius=0)
        self.assertEqual(valid.sum(), 1)
        self.assertTrue(np.isnan(p[0, :5]).all())


if __name__ == '__main__':
    unittest.main()
