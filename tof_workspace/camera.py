"""Acquisition only: no filtering, GUI or Arduino dependencies."""
from dataclasses import dataclass
from threading import Event, Lock, Thread
from time import monotonic
import numpy as np


@dataclass(frozen=True)
class DepthFrame:
    sequence: int
    timestamp: float
    depth: np.ndarray  # millimetres, owned copy, read-only
    confidence: np.ndarray | None  # native SDK units, not normalized


class ToFCamera:
    """Single-use threaded camera; latest() never blocks on hardware.

    Call start(), read latest(), then stop() and join(). SDK calls, including
    cleanup, run exclusively in the acquisition thread. Frames may be skipped.
    """

    def __init__(self, index=0, range_mm=4000, demo=False, sdk=None):
        if range_mm not in (2000, 4000):
            raise ValueError("range_mm must be 2000 or 4000")
        self.index, self.range_mm, self.demo = index, range_mm, demo
        self._sdk = sdk
        self._stop = Event()
        self._lock = Lock()
        self._frame = None
        self._error = None
        self._thread = Thread(target=self._run, name="tof-capture", daemon=True)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def join(self, timeout=None):
        self._thread.join(timeout)

    @property
    def alive(self):
        return self._thread.is_alive()

    @property
    def error(self):
        with self._lock:
            return self._error

    def latest(self):
        with self._lock:
            return self._frame

    def _publish(self, sequence, depth, confidence):
        depth = np.array(depth, dtype=np.float32, copy=True)
        if depth.ndim != 2 or not depth.size:
            raise ValueError("Camera returned an invalid depth image")
        if confidence is not None:
            confidence = np.array(confidence, dtype=np.float32, copy=True)
            if confidence.shape != depth.shape:
                raise ValueError("Depth and confidence shapes differ")
            confidence.setflags(write=False)
        depth.setflags(write=False)
        frame = DepthFrame(sequence, monotonic(), depth, confidence)
        with self._lock:
            self._frame = frame

    def _run(self):
        cam = None
        opened = started = False
        try:
            if self.demo:
                self._run_demo()
                return
            ac = self._sdk
            if ac is None:
                import ArducamDepthCamera as ac
            cam = ac.ArducamCamera()
            result = cam.open(ac.Connection.CSI, self.index)
            if result != 0:
                raise RuntimeError(f"Camera open failed: {result}")
            opened = True
            result = cam.start(ac.FrameType.DEPTH)
            if result != 0:
                raise RuntimeError(f"Camera start failed: {result}")
            started = True
            result = cam.setControl(ac.Control.RANGE, self.range_mm)
            if result not in (None, 0):
                raise RuntimeError(f"Setting camera range failed: {result}")
            actual_range = cam.getControl(ac.Control.RANGE)
            if actual_range != self.range_mm:
                raise RuntimeError(f"Requested range {self.range_mm}, received {actual_range}")
            sequence = 0
            last_received = monotonic()
            while not self._stop.is_set():
                frame = cam.requestFrame(200)
                if frame is None:
                    if monotonic() - last_received > 5:
                        raise TimeoutError("No camera frames for 5 seconds")
                    self._stop.wait(0.005)
                    continue
                try:
                    if not isinstance(frame, ac.DepthData):
                        raise TypeError("SDK returned a non-depth frame")
                    sequence += 1
                    self._publish(sequence, frame.depth_data,
                                  getattr(frame, "confidence_data", None))
                    last_received = monotonic()
                finally:
                    cam.releaseFrame(frame)
        except Exception as exc:
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            # Always attempt close even if stop raises.
            for action, needed in (("stop", started), ("close", opened)):
                if needed:
                    try:
                        getattr(cam, action)()
                    except Exception as exc:
                        with self._lock:
                            self._error = (self._error or "") + f" {action}: {exc}"

    def _run_demo(self):
        rng = np.random.default_rng(42)
        y, x = np.mgrid[:180, :240]
        sequence = 0
        while not self._stop.is_set():
            sequence += 1
            center = 120 + 60 * np.sin(sequence / 30)
            depth = np.full(x.shape, self.range_mm * 0.7)
            depth[(x-center)**2 + (y-90)**2 < 35**2] = self.range_mm * 0.25
            depth += rng.normal(0, 25, x.shape)
            confidence = np.full(x.shape, 200, dtype=np.float32)
            bad = rng.random(x.shape) < 0.04
            confidence[bad] = 10
            depth[bad] = 0
            self._publish(sequence, depth, confidence)
            self._stop.wait(1 / 30)
