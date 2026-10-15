import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from camera import ToFCamera, DepthFrame
from processing import FilterSettings, filter_depth, render


class CoreTests(unittest.TestCase):
    def test_demo_stops_and_publishes_readonly_data(self):
        from time import monotonic, sleep
        camera = ToFCamera(demo=True).start()
        try:
            deadline = monotonic() + 2
            while camera.latest() is None and monotonic() < deadline:
                sleep(.01)
            self.assertIsNotNone(camera.latest())
            self.assertFalse(camera.latest().depth.flags.writeable)
        finally:
            camera.stop()
            camera.join(2)
        self.assertFalse(camera.alive)
        self.assertIsNone(camera.error)

    def test_sdk_buffer_copied_and_released_on_error(self):
        class FakeFrame:
            depth_data = np.full((5, 5), 1000, np.float32)
            confidence_data = np.full((5, 5), 100, np.float32)

        class FakeCamera:
            released = stopped = closed = False
            def open(self, *_): return 0
            def start(self, *_): return 0
            def setControl(self, *_): return 0
            def getControl(self, *_): return 4000
            def requestFrame(self, *_):
                if self.released:
                    raise RuntimeError("Disconnected")
                return FakeFrame()
            def releaseFrame(self, frame):
                frame.depth_data[:] = -1
                frame.confidence_data[:] = -1
                self.released = True
            def stop(self): self.stopped = True
            def close(self): self.closed = True

        device = FakeCamera()
        sdk = SimpleNamespace(ArducamCamera=lambda: device, DepthData=FakeFrame,
                              Connection=SimpleNamespace(CSI=0), FrameType=SimpleNamespace(DEPTH=0),
                              Control=SimpleNamespace(RANGE=0))
        camera = ToFCamera(sdk=sdk).start()
        camera.join(2)
        self.assertFalse(camera.alive)
        self.assertTrue(device.released and device.stopped and device.closed)
        np.testing.assert_equal(camera.latest().depth, 1000)
        np.testing.assert_equal(camera.latest().confidence, 100)
        self.assertIn("Disconnected", camera.error)

    def test_filters_preserve_invalid_mask_and_source(self):
        depth = np.full((11, 11), 1000, np.float32)
        depth[5, 5] = np.nan
        confidence = np.full_like(depth, 100)
        confidence[3, 3] = 0
        original = depth.copy()
        frame = DepthFrame(1, 0, depth, confidence)
        for mode in ("Нет", "Гаусс", "Медианный", "Билатеральный"):
            settings = FilterSettings(smoothing=mode)
            result = filter_depth(frame, settings)
            self.assertTrue(np.isnan(result[5, 5]))
            self.assertTrue(np.isnan(result[3, 3]))
            np.testing.assert_allclose(result[np.isfinite(result)], 1000, atol=.01)
            rgb = render(frame, settings, "Глубина: фильтр", 4000)
            np.testing.assert_equal(rgb[5, 5], 0)
        np.testing.assert_equal(depth, original)

    def test_missing_confidence_supported(self):
        frame = DepthFrame(1, 0, np.full((5, 5), 1000, np.float32), None)
        np.testing.assert_equal(filter_depth(frame, FilterSettings()), 1000)
        np.testing.assert_equal(render(frame, FilterSettings(), "Confidence", 4000), 0)


if __name__ == "__main__":
    unittest.main()
