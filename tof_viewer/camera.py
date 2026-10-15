"""SDK загружается только при выборе реальной камеры."""
import time
import numpy as np


class Camera:
    def __init__(self, max_depth=4000, index=0):
        import ArducamDepthCamera as ac
        self.cam = ac.ArducamCamera()
        self.started = False
        self.opened = False
        try:
            self.check(self.cam.open(ac.Connection.CSI, index), 'open')
            self.opened = True
            self.check(self.cam.start(ac.FrameType.DEPTH), 'start')
            self.started = True
            self.check(self.cam.setControl(ac.Control.RANGE, max_depth), 'RANGE')
            self.max_depth = int(self.cam.getControl(ac.Control.RANGE))
            if self.max_depth <= 0:
                raise RuntimeError('SDK вернул некорректный диапазон')
        except Exception:
            self.close()
            raise

    @staticmethod
    def check(code, action):
        if code != 0:
            raise RuntimeError(f'Arducam {action}: код {code}')

    def read(self):
        frame = self.cam.requestFrame(250)
        if frame is None:
            return None
        try:
            depth = np.array(frame.depth_data, dtype=np.float32, copy=True)
            confidence = np.array(frame.confidence_data, dtype=np.float32, copy=True)
            if depth.ndim != 2 or confidence.shape != depth.shape:
                raise RuntimeError('Камера не предоставила совместимые depth/confidence')
            return depth, confidence
        finally:
            self.cam.releaseFrame(frame)

    def close(self):
        try:
            if self.started:
                self.cam.stop()
        finally:
            self.started = False
            if self.opened:
                self.cam.close()
                self.opened = False


class DemoCamera:
    def __init__(self, max_depth=4000, index=0):
        self.max_depth = max_depth
        self.rng = np.random.default_rng(42)
        self.start = time.monotonic()

    def read(self):
        time.sleep(1/30)
        y, x = np.mgrid[:180, :240]
        center = 120 + 40*np.sin((time.monotonic()-self.start)*0.5)
        d = np.where((x-center)**2+(y-90)**2 < 35**2,
                     self.max_depth*0.3, self.max_depth*(0.65+x/1600))
        d = d + self.rng.normal(0, 25, d.shape)
        c = self.rng.uniform(60, 220, d.shape)
        c[self.rng.random(d.shape) < 0.08] = 5
        c[15:35, 15:55] = 0
        d[c < 30] = self.rng.uniform(0, self.max_depth, (c < 30).sum())
        return d.astype(np.float32), c.astype(np.float32)

    def close(self):
        pass
