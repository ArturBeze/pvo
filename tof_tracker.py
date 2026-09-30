#!/usr/bin/env python3

"""One-object ToF tracking. See README_RU.md. Depth units: millimetres."""

import argparse
from collections import deque
from dataclasses import dataclass
import math
import time

import cv2
import numpy as np


@dataclass
class Settings:
    near: int = 300
    far: int = 3500
    confidence: int = 30
    min_area: int = 30
    band: int = 200
    gate_px: int = 45
    horizon: int = 10
    history: int = 8
    lost: int = 8


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def unproject(self, u, v, depth_mm):
        z = depth_mm / 1000.0
        return np.array([(u-self.cx)*z/self.fx, (v-self.cy)*z/self.fy, z])

    def project(self, xyz):
        x, y, z = xyz
        if z <= .01 or not np.isfinite(xyz).all():
            return None
        return np.array([self.fx*x/z+self.cx, self.fy*y/z+self.cy])


class Motion:
    """Least-squares constant-velocity fit over real acquisition times."""
    def __init__(self):
        self.samples = deque()

    def add(self, t, xyz, history):
        self.samples.append((t, xyz.copy()))
        while len(self.samples) > max(2, history):
            self.samples.popleft()

    def velocity(self):
        if len(self.samples) < 3:
            return np.zeros(3)
        ts = np.array([t for t, _ in self.samples])
        ts -= ts.mean()
        denom = ts @ ts
        if denom < 1e-9:
            return np.zeros(3)
        positions = np.array([p for _, p in self.samples])
        return ts @ positions / denom

    def predict(self, t):
        if not self.samples:
            return None
        # Anchor at the fitted current position to reduce centroid jitter.
        ts = np.array([stamp for stamp, _ in self.samples])
        mean_pos = np.mean([p for _, p in self.samples], axis=0)
        return mean_pos + self.velocity() * (t-ts.mean())


def valid_mask(depth, confidence, s):
    mask = np.isfinite(depth) & (depth > 0) & (depth >= s.near) & (depth <= s.far)
    if confidence is not None:
        mask &= np.isfinite(confidence) & (confidence >= s.confidence)
    return mask


class Tracker:
    def __init__(self):
        self.motion = Motion()
        self.active = False
        self.click = None
        self.missed = 0
        self.box = None
        self.message = 'Click the object to start'

    def reset(self):
        self.motion = Motion()
        self.active = False
        self.missed = 0
        self.box = None
        self.click = None
        self.message = 'Click the object to start'

    def update(self, depth, confidence, t, s, k):
        mask = valid_mask(depth, confidence, s)
        click, self.click = self.click, None
        seed = None
        if click is not None:
            self.reset()
            x, y = click
            if not (0 <= x < depth.shape[1] and 0 <= y < depth.shape[0]) or not mask[y,x]:
                self.message = 'Click valid depth inside the selected range'
                return mask
            seed = float(depth[y,x])
        elif self.active:
            # Do not fit across a camera stall or reuse an obsolete identity.
            if t-self.motion.samples[-1][0] > 1.0:
                self.reset()
                self.message = 'Camera gap: select object again'
                return mask
            predicted = self.motion.predict(t)
            seed = predicted[2]*1000
        else:
            return mask

        segment = mask & (np.abs(depth-seed) <= s.band)
        count, labels, stats, centers = cv2.connectedComponentsWithStats(
            segment.astype(np.uint8), connectivity=8)
        candidates = []
        predicted_uv = k.project(self.motion.predict(t)) if self.active else None
        for label in range(1, count):
            x, y, w, h, area = stats[label]
            if area < s.min_area:
                continue
            if click is not None and labels[click[1],click[0]] != label:
                continue
            u, v = centers[label]
            if self.active:
                if predicted_uv is None:
                    continue
                distance = float(np.linalg.norm(np.array([u,v])-predicted_uv))
                if distance > s.gate_px:
                    continue
            else:
                distance = 0.0
            z = float(np.median(depth[y:y+h,x:x+w][labels[y:y+h,x:x+w] == label]))
            candidates.append((distance, u, v, z, (int(x),int(y),int(w),int(h))))
        if candidates:
            _, u, v, z, self.box = min(candidates, key=lambda item: item[0])
            self.motion.add(t, k.unproject(u,v,z), s.history)
            self.active = True
            self.missed = 0
            self.message = 'TRACKING' if len(self.motion.samples) >= 3 else 'Collecting motion samples'
        elif self.active:
            self.missed += 1
            self.box = None
            self.message = f'COASTING: {self.missed}/{s.lost}'
            if self.missed > s.lost:
                self.reset()
                self.message = 'LOST: click the object again'
        else:
            self.message = 'Object too small: reduce Min area'
        return mask


def synthetic(t):
    """A moving approaching rectangle and a stationary distractor."""
    depth = np.full((180,240),3900,np.float32)
    x = int(95 + 48*math.sin(t*.55))
    z = 1700 + 600*math.sin(t*.4)
    depth[65:100,x:x+28] = z
    depth[115:150,185:220] = 2100
    return depth, np.full(depth.shape,150,np.float32)


def draw(depth, mask, tracker, t, dt, s, k, scale, calibrated):
    h,w = depth.shape
    normalized = np.zeros_like(depth, dtype=np.uint8)
    normalized[mask] = np.clip((depth[mask]-s.near)*255/max(1,s.far-s.near),0,255)
    image = cv2.applyColorMap(normalized,cv2.COLORMAP_TURBO)
    image[~mask] = 0
    image = cv2.resize(image,(w*scale,h*scale),interpolation=cv2.INTER_NEAREST)
    if tracker.box is not None:
        x,y,bw,bh = tracker.box
        cv2.rectangle(image,(x*scale,y*scale),((x+bw)*scale,(y+bh)*scale),(0,255,0),2)
    lines = [tracker.message, f'Range: {s.near}..{s.far} mm | FPS {1/max(dt,.001):.1f}',
             '3D: calibrated' if calibrated else '3D: approximate FOV (calibrate for metric accuracy)']
    if tracker.active:
        current = tracker.motion.predict(t)
        future = tracker.motion.predict(t+s.horizon*dt)
        v = tracker.motion.velocity()
        lines += [f'XYZ: {current[0]:+.3f}, {current[1]:+.3f}, {current[2]:.3f} m',
                  f'V: {np.linalg.norm(v):.3f} m/s | Vz: {v[2]:+.3f} m/s',
                  f'+{s.horizon} frames ({s.horizon*dt:.3f}s): Z={future[2]:.3f} m']
        if len(tracker.motion.samples) >= 3:
            last = None
            for step in range(s.horizon+1):
                xyz = tracker.motion.predict(t+step*dt)
                uv = k.project(xyz)
                if uv is None or not (0 <= uv[0] < w and 0 <= uv[1] < h):
                    last = None
                    continue
                point = tuple(np.rint(uv*scale).astype(int))
                color = (0,255,255) if s.near <= xyz[2]*1000 <= s.far else (0,0,255)
                if last is not None:
                    cv2.line(image,last,point,color,2)
                cv2.circle(image,point,3,color,-1)
                last = point
            uv = k.project(future)
            if uv is not None and 0 <= uv[0] < w and 0 <= uv[1] < h:
                point = tuple(np.rint(uv*scale).astype(int))
                cv2.drawMarker(image,point,(255,0,255),cv2.MARKER_CROSS,18,2)
                lines.append(f'Future pixel: ({uv[0]:.1f}, {uv[1]:.1f})')
            else:
                lines.append('Future point outside image / behind camera')
        else:
            lines.append('Prediction pending: need 3 measurements')
    panel = np.zeros((185,w*scale,3),np.uint8)
    for i,line in enumerate(lines):
        cv2.putText(panel,line,(8,20+i*20),cv2.FONT_HERSHEY_SIMPLEX,.45,(240,240,240),1,cv2.LINE_AA)
    return np.vstack((image,panel))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',action='store_true',help='Synthetic camera')
    parser.add_argument('--cfg',help='SDK configuration file')
    parser.add_argument('--index',type=int,default=0)
    parser.add_argument('--sensor-range',type=int,choices=[2000,4000],default=4000,
                        help='Physical range in millimetres')
    parser.add_argument('--sdk-range-value',type=int,help='Override SDK RANGE value (e.g. 4 for a metres-based SDK)')
    parser.add_argument('--intrinsics',type=float,nargs=4,metavar=('FX','FY','CX','CY'))
    parser.add_argument('--scale',type=int,choices=[1,2,3,4],default=3)
    args = parser.parse_args()
    if args.intrinsics and (not np.isfinite(args.intrinsics).all() or min(args.intrinsics[:2]) <= 0):
        parser.error('fx/fy must be positive and all intrinsics finite')
    tracker = Tracker()
    cam = None
    started = False
    window, controls = 'ToF tracker', 'Settings'
    try:
        if not args.demo:
            import ArducamDepthCamera as ac
            cam = ac.ArducamCamera()
            ret = cam.openWithFile(args.cfg,args.index) if args.cfg else cam.open(ac.Connection.CSI,args.index)
            if ret != 0:
                raise RuntimeError(f'Camera open failed: {ret}. Check driver/CSI/index/config.')
            ret = cam.start(ac.FrameType.DEPTH)
            if ret != 0:
                raise RuntimeError(f'Camera start failed: {ret}')
            started = True
            value = args.sdk_range_value if args.sdk_range_value is not None else args.sensor_range
            # VGA range is determined by its working mode (Arducam SDK docs).
            if cam.getCameraInfo().device_type != ac.DeviceType.VGA:
                ret = cam.setControl(ac.Control.RANGE,value)
                if ret not in (None,0):
                    raise RuntimeError(f'RANGE rejected: {ret}; check --sdk-range-value for your SDK')
            print('SDK:',ac.__version__,'RANGE readback:',cam.getControl(ac.Control.RANGE))
        cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(controls,cv2.WINDOW_NORMAL)
        cv2.resizeWindow(controls,640,620)
        cv2.imshow(controls,np.zeros((30,640,3),np.uint8))
        definitions = [('Near mm',300,args.sensor_range),('Far mm',min(3500,args.sensor_range),args.sensor_range),
                       ('Confidence',30,255),('Min area px',30,2000),('Depth band mm',200,1000),
                       ('Gate px',45,300),('Future frames',10,120),('History frames',8,60),
                       ('Lost frames',8,60),('HFOV deg approx',60,120),('VFOV deg approx',45,120)]
        for name,value,maximum in definitions:
            cv2.createTrackbar(name,controls,value,maximum,lambda _: None)
        def mouse(event,x,y,flags,param):
            if event == cv2.EVENT_LBUTTONDOWN:
                tracker.click = (x//args.scale,y//args.scale)
        cv2.setMouseCallback(window,mouse)
        origin = time.monotonic()
        intervals = deque(maxlen=30)
        previous_t = None
        previous_config = None
        last_frame = origin
        while True:
            if args.demo:
                t = time.monotonic()
                depth, confidence = synthetic(t-origin)
            else:
                frame = cam.requestFrame(100)  # Keep GUI responsive even on missing frames.
                if frame is None:
                    if time.monotonic()-last_frame > 5:
                        raise RuntimeError('No camera frames for 5 seconds')
                    if cv2.waitKey(1) & 255 in (27,ord('q')):
                        break
                    if cv2.getWindowProperty(window,cv2.WND_PROP_VISIBLE) < 1:
                        break
                    continue
                try:
                    if not isinstance(frame,ac.DepthData):
                        raise RuntimeError('SDK returned a non-depth frame')
                    t = time.monotonic()
                    depth = np.array(frame.depth_data,dtype=np.float32,copy=True)
                    # As in the supplied example, confidence is used only on VGA.
                    confidence = None
                    if cam.getCameraInfo().device_type == ac.DeviceType.VGA:
                        confidence = np.array(frame.confidence_data,dtype=np.float32,copy=True)
                finally:
                    cam.releaseFrame(frame)
            last_frame = t
            if previous_t is not None and 0 < t-previous_t < 1:
                intervals.append(t-previous_t)
            previous_t = t
            dt = float(np.median(intervals)) if intervals else 1/30
            values = [cv2.getTrackbarPos(name,controls) for name,_,_ in definitions]
            near,far,conf,area,band,gate,horizon,history,lost,hfov,vfov = values
            s = Settings(near,far,conf,max(1,area),max(1,band),max(1,gate),horizon,max(3,history),lost)
            h,w = depth.shape
            k = Intrinsics(*args.intrinsics) if args.intrinsics else Intrinsics(
                w/(2*math.tan(math.radians(max(1,hfov))/2)),
                h/(2*math.tan(math.radians(max(1,vfov))/2)),(w-1)/2,(h-1)/2)
            config = (near,far,conf,area,band,gate,k.fx,k.fy,k.cx,k.cy,h,w)
            if previous_config is not None and config != previous_config:
                tracker.reset()
                tracker.message = 'Filter/calibration changed: select object again'
            previous_config = config
            if near >= far:
                tracker.reset()
                tracker.message = 'Invalid range: Near must be less than Far'
                mask = np.zeros(depth.shape,bool)
            else:
                mask = tracker.update(depth,confidence,t,s,k)
            cv2.imshow(window,draw(depth,mask,tracker,t,dt,s,k,args.scale,bool(args.intrinsics)))
            key = cv2.waitKey(30 if args.demo else 1) & 255
            if key in (27,ord('q')):
                break
            if key == ord('r'):
                tracker.reset()
            if any(cv2.getWindowProperty(name,cv2.WND_PROP_VISIBLE) < 1 for name in (window,controls)):
                break
    finally:
        if cam is not None:
            try:
                if started:
                    cam.stop()
            finally:
                cam.close()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError,ImportError,cv2.error) as error:
        raise SystemExit(str(error))
