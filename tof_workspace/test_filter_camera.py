"""Проверки без дисплея и камеры: python -m unittest discover -s tof_workspace -v."""
from queue import Queue
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
import unittest

import numpy as np
from filter_camera import CameraStream, CaptureWorker, make_images


class CameraTests(unittest.TestCase):
    def test_spawn_delivers_frames_and_stops_with_full_queue(self):
        stream = CameraStream(demo=True)
        try:
            stream.start()
            deadline = monotonic() + 10
            frame = None
            while monotonic() < deadline:
                frame, error = stream.snapshot()
                self.assertIsNone(error)
                if frame is not None:
                    break
                sleep(.02)
            self.assertIsNotNone(frame)
            self.assertEqual(frame[2].shape, (180, 240))
            sleep(.2)  # GUI перестал читать: очередь заполнена.
            stream.stop_event.set()
            stream.process.join(3)
            self.assertFalse(stream.process.is_alive())
            self.assertEqual(stream.process.exitcode, 0)
        finally:
            stream.stop_event.set()
            if stream.started and stream.process.is_alive():
                stream.process.terminate()
                stream.process.join(3)
            stream.dispose()

    def test_sdk_copies_buffers_and_closes_after_disconnect(self):
        class Frame:
            def __init__(self):
                self.depth_data = np.full((3, 4), 1000, np.float32)
                self.confidence_data = np.full((3, 4), 100, np.float32)

        class Device:
            released = stopped = closed = False
            def open(self, *_): return 0
            def start(self, *_): return 0
            def setControl(self, *_): return 0
            def getControl(self, *_): return 4000
            def requestFrame(self, *_):
                if self.released:
                    raise RuntimeError("Disconnected")
                return Frame()
            def releaseFrame(self, frame):
                frame.depth_data[:] = -1
                frame.confidence_data[:] = -1
                self.released = True
            def stop(self): self.stopped = True
            def close(self): self.closed = True

        device = Device()
        sdk = SimpleNamespace(ArducamCamera=lambda: device, DepthData=Frame,
            Connection=SimpleNamespace(CSI=0), FrameType=SimpleNamespace(DEPTH=0),
            Control=SimpleNamespace(RANGE=0))
        frames, messages = Queue(2), Queue()
        worker = CaptureWorker(0, 4000, Event(), frames, messages, sdk)
        worker.run()
        frame = frames.get_nowait()
        np.testing.assert_array_equal(frame[2], 1000)
        np.testing.assert_array_equal(frame[3], 100)
        self.assertTrue(device.released and device.stopped and device.closed)
        self.assertTrue(any(kind == "error" and "Disconnected" in text
                            for kind, text in list(messages.queue)))

    def test_threshold_boundaries_and_invalid_data(self):
        depth = np.array([[100, 200, 300, 400, 0, np.nan]], np.float32)
        confidence = np.array([[30, 29, 30, 100, 100, 100]], np.float32)
        original, filtered, count = make_images(depth, confidence, 30, 100, 300, 4000)
        self.assertEqual(count, 2)
        np.testing.assert_array_equal(filtered[:, [0, 2]], original[:, [0, 2]])
        self.assertTrue((filtered[:, [1, 3, 4, 5]] == 0).all())


if __name__ == "__main__":
    unittest.main()
