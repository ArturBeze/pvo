#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arducam ToF + Arduino Uno camera gimbal, Python 3.10+.

Start: python3 tof_gimbal_tracker.py
Demo (no hardware): python3 tof_gimbal_tracker.py --demo
Algorithm tests (no display/hardware): python3 tof_gimbal_tracker.py --self-test

Dependencies: numpy, opencv-python, Pillow, pyFirmata2, pyserial, tkinter;
real camera additionally needs ArducamDepthCamera and the correct Pi 5 driver.
Arduino: StandardFirmata; X=D9, Y=D10, common GND, external servo supply.

Coordinates: camera X=right, Y=down, Z=forward; metres and seconds internally.
Base coordinates are approximate, relative to the stationary gimbal base.
No GPS/IMU/encoder input: velocities are estimates, NOT aviation navigation data.
Area is projected area normal to the optical axis, not total surface area.
Frame time is HOST receipt time, not a hardware exposure timestamp.
See the adjacent Russian README for installation, calibration and limitations.

SDK reference: https://github.com/ArduCAM/Arducam_tof_camera
Firmata reference: https://github.com/berndporr/pyFirmata2
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

LOG = logging.getLogger("tof")


@dataclass(frozen=True)
class Settings:
    near: float = 0.25
    far: float = 3.5
    area_min: float = 0.002
    area_max: float = 0.50
    pixels_min: int = 35
    frame_fraction: float = 0.65
    depth_gap: float = 0.10
    quality_min: float = 30.0
    quality: str = "auto"
    confirm: int = 5
    lost_time: float = 0.5
    association: float = 0.45
    switch_margin: float = 0.20
    switch_frames: int = 5
    horizon_frames: int = 10
    horizon_max: float = 0.8
    latency: float = 0.05
    measurement_noise: float = 0.04
    acceleration_noise: float = 2.0
    max_sigma: float = 0.40
    fov_x: float = 60.0
    fov_y: float = 45.0
    radial_depth: bool = False
    compensate: bool = True
    x_min: float = 20.0
    x_max: float = 160.0
    y_min: float = 45.0
    y_max: float = 135.0
    invert_x: bool = True
    invert_y: bool = False
    gain: float = 1.8
    deadband: float = 1.5
    max_speed: float = 40.0
    max_acceleration: float = 100.0
    manual_step: float = 3.0
    stale_time: float = 0.35
    sdk_range: int = 4000
    camera_index: int = 0
    connection: str = "CSI"
    port: str = "AUTO"

    def validate(self):
        for key, default in asdict(Settings()).items():
            value = getattr(self, key)
            if type(default) is bool:
                if type(value) is not bool:
                    raise ValueError(f"{key}: требуется checkbox / boolean")
            elif type(default) in (int, float):
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"{key}: требуется конечное число")
                if type(default) is int and (type(value) is not int):
                    raise ValueError(f"{key}: требуется целое число")
            elif not isinstance(value, str):
                raise ValueError(f"{key}: требуется строка")
        if not 0 < self.near < self.far <= self.sdk_range / 1000:
            raise ValueError("Дальность: 0 < минимум < максимум ≤ режим SDK / 1000")
        if not 0 < self.area_min < self.area_max <= 100:
            raise ValueError("Площадь: 0 < минимум < максимум ≤ 100 м²")
        bounds = {
            'pixels_min': (3, 1000000), 'frame_fraction': (.01, 1),
            'depth_gap': (.005, 1), 'quality_min': (0, 65535),
            'confirm': (1, 120), 'lost_time': (.05, 5), 'association': (.02, 3),
            'switch_margin': (0, 3), 'switch_frames': (1, 120),
            'horizon_frames': (0, 120), 'horizon_max': (0, 3), 'latency': (0, 1),
            'measurement_noise': (.001, 1), 'acceleration_noise': (.01, 30),
            'max_sigma': (.01, 3), 'fov_x': (5, 150), 'fov_y': (5, 150),
            'gain': (.01, 10), 'deadband': (0, 20), 'max_speed': (1, 180),
            'max_acceleration': (1, 1000), 'manual_step': (.1, 30),
            'stale_time': (.05, 2), 'camera_index': (0, 32),
        }
        for key, (low, high) in bounds.items():
            if not low <= getattr(self, key) <= high:
                raise ValueError(f"{key}: допустимо от {low} до {high}")
        for axis in ('x', 'y'):
            if not 0 <= getattr(self, axis + '_min') < 90 < getattr(self, axis + '_max') <= 180:
                raise ValueError(f"Ось {axis.upper()}: 0 ≤ min < 90 < max ≤ 180")
        if self.sdk_range not in (2000, 4000):
            raise ValueError("SDK RANGE: 2000 или 4000 мм")
        if self.quality not in ('auto', 'confidence', 'amplitude', 'off'):
            raise ValueError("Фильтр качества: auto / confidence / amplitude / off")
        if self.connection not in ('CSI', 'USB') or not self.port.strip():
            raise ValueError("Интерфейс: CSI/USB; порт: AUTO или имя устройства")
        return self


def intrinsics(shape, cfg):
    h, w = shape
    return (w / (2 * math.tan(math.radians(cfg.fov_x) / 2)),
            h / (2 * math.tan(math.radians(cfg.fov_y) / 2)), (w - 1) / 2, (h - 1) / 2)


def rotation(angles, cfg):
    """Approximate camera-to-base rotation from commanded angles (no encoders)."""
    if not cfg.compensate:
        return np.eye(3)
    yaw = math.radians((angles[0] - 90) * (-1 if cfg.invert_x else 1))
    pitch = math.radians(-(angles[1] - 90) * (-1 if cfg.invert_y else 1))
    cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
    return np.array([[cy, sy * sp, sy * cp], [0, cp, -sp], [-sy, cy * sp, cy * cp]])


def project(point, intr):
    fx, fy, cx, cy = intr
    if not np.all(np.isfinite(point)) or point[2] <= .05:
        return None
    return np.array([point[0] * fx / point[2] + cx, point[1] * fy / point[2] + cy])


@dataclass
class Detection:
    point: np.ndarray
    area: float
    pixels: int
    box: tuple
    uv: tuple
    distance: float


def detect(depth, quality, cfg):
    """Depth-continuous components; a sloping surface is not split into depth bins.

    floodFill uses a floating local depth tolerance. Connectivity cannot separate
    touching objects of similar depth; the GUI exposes this tolerance explicitly.
    Returns foreground mask plus detections. A processing cap limits noise storms.
    """
    h, w = depth.shape
    fx, fy, cx, cy = intrinsics(depth.shape, cfg)
    valid = np.isfinite(depth) & (depth >= cfg.near) & (depth <= cfg.far)
    if quality is not None and cfg.quality != 'off':
        valid &= np.isfinite(quality) & (quality >= cfg.quality_min)
    valid = cv2.morphologyEx(valid.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    clean = np.where(valid, depth, 0).astype(np.float32)
    smooth = cv2.medianBlur(clean, 3)
    mask = np.ones((h + 2, w + 2), np.uint8)
    mask[1:-1, 1:-1] = (~valid).astype(np.uint8)
    inside = mask[1:-1, 1:-1]
    indices = np.flatnonzero(valid)
    # Near-first processing makes the cap deterministic and relevant to selection.
    indices = indices[np.argsort(clean.ravel()[indices])]
    detections = []
    components = 0
    for index in indices:
        v, u = divmod(int(index), w)
        if inside[v, u]:
            continue
        count, _, _, rect = cv2.floodFill(smooth, mask, (u, v), 0,
            loDiff=cfg.depth_gap, upDiff=cfg.depth_gap,
            flags=4 | cv2.FLOODFILL_MASK_ONLY | (2 << 8))
        components += 1
        x, y, rw, rh = rect
        sub = inside[y:y + rh, x:x + rw]
        part = sub == 2
        sub[part] = 1
        if cfg.pixels_min <= count <= h * w * cfg.frame_fraction:
            yy, xx = np.nonzero(part)
            xx, yy = xx + x, yy + y
            z = depth[yy, xx].astype(np.float64)
            xn, yn = (xx - cx) / fx, (yy - cy) / fy
            if cfg.radial_depth:
                z = z / np.sqrt(1 + xn * xn + yn * yn)
            area = float(np.sum(z * z) / (fx * fy))
            if cfg.area_min <= area <= cfg.area_max:
                point = np.array([np.median(xn * z), np.median(yn * z), np.median(z)])
                uv = project(point, (fx, fy, cx, cy))
                detections.append(Detection(point, area, count, rect, tuple(uv), float(np.linalg.norm(point))))
        if components >= 512:
            break
    return detections, valid, components >= 512


class Filter:
    """6-state constant-velocity Kalman filter in metres and seconds."""
    def __init__(self, point, cfg):
        self.x = np.r_[point, np.zeros(3)]
        self.P = np.diag([cfg.measurement_noise ** 2] * 3 + [1.] * 3)

    def predict(self, dt, cfg):
        F = np.eye(6)
        F[:3, 3:] = np.eye(3) * dt
        G = np.vstack((np.eye(3) * dt * dt / 2, np.eye(3) * dt))
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * cfg.acceleration_noise ** 2

    def correct(self, point, cfg):
        R = np.eye(3) * cfg.measurement_noise ** 2
        K = np.linalg.solve(self.P[:3, :3] + R, self.P[:3, :]).T
        self.x += K @ (point - self.x[:3])
        A = np.eye(6)
        A[:, :3] -= K
        self.P = A @ self.P @ A.T + K @ R @ K.T  # Joseph form
        self.P = (self.P + self.P.T) / 2

    def future(self, horizon, cfg):
        H = np.hstack((np.eye(3), np.eye(3) * horizon))
        covariance = H @ self.P @ H.T + np.eye(3) * cfg.acceleration_noise ** 2 * horizon ** 4 / 4
        return H @ self.x, math.sqrt(max(0., float(np.linalg.eigvalsh(covariance)[-1])))


@dataclass
class Track:
    id: int
    kf: Filter
    detection: Detection
    last_seen: float
    hits: int = 1
    streak: int = 1
    visible: bool = True
    confirmed: bool = False


class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1
        self.selected = None
        self.challenger = None
        self.challenger_count = 0
        self.last_time = None
        self.period = 1 / 30

    def update(self, detections, timestamp, R, cfg):
        dt = 0 if self.last_time is None else timestamp - self.last_time
        if dt < 0 or dt > cfg.lost_time:
            self.tracks.clear()
            self.selected = None
        if 0 < dt <= cfg.lost_time:
            self.period = .9 * self.period + .1 * dt
        self.last_time = timestamp
        self.tracks = [t for t in self.tracks if timestamp - t.last_seen <= cfg.lost_time]
        for t in self.tracks:
            t.kf.predict(max(0, dt), cfg)
            t.visible = False
        points = [R @ d.point for d in detections]
        pairs = []
        for i, t in enumerate(self.tracks):
            for j, p in enumerate(points):
                residual = p - t.kf.x[:3]
                distance = float(np.linalg.norm(residual))
                ratio = detections[j].area / max(t.detection.area, 1e-9)
                S = t.kf.P[:3, :3] + np.eye(3) * cfg.measurement_noise ** 2
                mahal = float(residual @ np.linalg.solve(S, residual))
                if distance <= cfg.association and .25 <= ratio <= 4 and mahal <= 16.27:
                    pairs.append((mahal + abs(math.log(ratio)), i, j))
        used_t, used_d = set(), set()
        for _, i, j in sorted(pairs):
            if i in used_t or j in used_d:
                continue
            used_t.add(i)
            used_d.add(j)
            t = self.tracks[i]
            t.kf.correct(points[j], cfg)
            t.detection, t.last_seen, t.visible = detections[j], timestamp, True
            t.hits += 1
            t.streak += 1
            t.confirmed |= t.streak >= cfg.confirm
        for i, t in enumerate(self.tracks):
            if i not in used_t:
                t.streak = 0
        for j, d in enumerate(detections):
            if j not in used_d:
                self.tracks.append(Track(self.next_id, Filter(points[j], cfg), d, timestamp,
                                         confirmed=cfg.confirm == 1))
                self.next_id += 1
        eligible = [t for t in self.tracks if t.confirmed and t.visible]
        current = next((t for t in self.tracks if t.id == self.selected), None)
        nearest = min(eligible, key=lambda t: t.detection.distance, default=None)
        # Keep identity during short occlusions; NEVER drive on unobserved tracks.
        if current is None:
            current = nearest
            self.selected = current.id if current else None
            self.challenger, self.challenger_count = None, 0
        elif (nearest and current.visible and nearest.id != current.id and
              nearest.detection.distance + cfg.switch_margin < current.detection.distance):
            self.challenger_count = self.challenger_count + 1 if self.challenger == nearest.id else 1
            self.challenger = nearest.id
            if self.challenger_count >= cfg.switch_frames:
                current, self.selected = nearest, nearest.id
                self.challenger, self.challenger_count = None, 0
        else:
            self.challenger, self.challenger_count = None, 0
        return current


class Gimbal:
    def __init__(self, demo=False):
        self.demo, self.board = demo, None
        self.angles = np.array([90., 90.])
        self.velocity = np.zeros(2)
        self.sent = np.array([np.nan, np.nan])
        self.last_write = 0.
        self.port = 'DEMO' if demo else 'не подключено'

    @property
    def connected(self):
        return self.demo or self.board is not None

    def connect(self, port, cancelled=lambda: False):
        if cancelled():
            raise RuntimeError('Подключение отменено')
        if self.demo:
            self.angles[:] = 90
            self.velocity[:] = 0
            return
        import pyfirmata2
        from serial.tools import list_ports
        if port.upper() == 'AUTO':
            candidates = [p for p in list_ports.comports()
                          if p.vid in (0x2341, 0x2A03, 0x1A86, 0x0403, 0x10C4)
                          or 'arduino' in (p.description or '').lower()]
            if len(candidates) != 1:
                ports = ', '.join(p.device for p in candidates) or 'нет подходящих USB-портов'
                raise RuntimeError(f"AUTO: {ports}. Укажите порт вручную во вкладке «Связь».")
            port = candidates[0].device
        self.close()
        board = None
        try:
            board = pyfirmata2.Arduino(port, timeout=.1)
            board.sp.write_timeout = .2
            # Read protocol version before attaching servos. Avoid arbitrary USB devices.
            deadline = time.monotonic() + 2
            board.sp.write(bytes([0xF9]))
            while not board.get_firmata_version() and time.monotonic() < deadline:
                if board.bytes_available():
                    board.iterate()
                else:
                    time.sleep(.01)
            if not board.get_firmata_version():
                raise RuntimeError("Нет ответа Firmata. Загрузите StandardFirmata в Arduino Uno.")
            if cancelled():
                raise RuntimeError("Подключение отменено до включения сервоприводов")
            # servo_config attaches at 90, avoiding get_pin's default attach-at-zero.
            board.servo_config(9, angle=90)
            board.servo_config(10, angle=90)
            self.pins = [board.digital[9], board.digital[10]]
            board.sp.write_timeout = .2
            self.board, self.port = board, port
            self.angles[:] = 90
            self.velocity[:] = 0
            self.sent[:] = 90
        except Exception:
            if board is not None:
                try:
                    board.exit()
                except Exception:
                    if getattr(board, 'sp', None):
                        board.sp.close()
            raise

    def move(self, desired, dt, cfg):
        limits_low = np.array([cfg.x_min, cfg.y_min])
        limits_high = np.array([cfg.x_max, cfg.y_max])
        desired = np.clip(desired, limits_low, limits_high)
        dt = min(max(dt, 0), .1)
        if dt <= 0 or not self.connected:
            self.velocity[:] = 0
            return
        delta = desired - self.angles
        requested = np.clip(delta / dt, -cfg.max_speed, cfg.max_speed)
        self.velocity += np.clip(requested - self.velocity,
                                 -cfg.max_acceleration * dt, cfg.max_acceleration * dt)
        step = self.velocity * dt
        # Do not coast away from a newly requested hold/reversal point.
        step = np.where(step * delta > 0, np.sign(step) * np.minimum(abs(step), abs(delta)), 0)
        self.angles = np.clip(self.angles + step, limits_low, limits_high)
        now = time.monotonic()
        rounded = np.rint(self.angles).astype(int)
        # Limits may be fractional; integer Firmata commands must still stay inside.
        rounded = np.clip(rounded, np.ceil(limits_low), np.floor(limits_high)).astype(int)
        if now - self.last_write >= .02 and np.any(rounded != self.sent):
            if self.board:
                for pin, angle in zip(self.pins, rounded):
                    pin.write(int(angle))
            self.sent = rounded.astype(float)
            self.last_write = now

    def hold(self):
        self.velocity[:] = 0

    def close(self):
        board, self.board = self.board, None
        if board:
            try:
                board.exit()
            finally:
                if getattr(board, 'sp', None):
                    board.sp.close()
        if not self.demo:
            self.port = 'не подключено'


class Camera:
    def __init__(self, cfg, demo=False):
        self.demo, self.cam, self.started = demo, None, False
        self.cfg, self.quality_name = cfg, 'demo'
        self.rng = np.random.default_rng(7)
        self.begin = time.monotonic()
        if demo:
            return
        import ArducamDepthCamera as ac
        self.ac = ac
        self.cam = ac.ArducamCamera()
        try:
            self.check(self.cam.open(getattr(ac.Connection, cfg.connection), cfg.camera_index), 'open')
            self.check(self.cam.start(ac.FrameType.DEPTH), 'start')
            self.started = True
            self.check(self.cam.setControl(ac.Control.RANGE, cfg.sdk_range), 'RANGE')
            actual = self.cam.getControl(ac.Control.RANGE)
            if int(actual) != cfg.sdk_range:
                raise RuntimeError(f"SDK вернул RANGE={actual}, ожидался {cfg.sdk_range}")
            self.info = self.cam.getCameraInfo()
            self.quality_name = cfg.quality
            if cfg.quality == 'auto':
                self.quality_name = 'confidence' if self.info.device_type == ac.DeviceType.VGA else 'amplitude'
        except Exception:
            self.close()
            raise

    @staticmethod
    def check(result, name):
        value = getattr(result, 'value', result)
        if value is not None and value != 0:
            raise RuntimeError(f"Arducam {name}: {result}")

    def read(self, angles, cfg):
        if self.demo:
            time.sleep(1 / 30)
            h, w = 180, 240
            depth = np.full((h, w), 3.8, np.float32)
            t = time.monotonic() - self.begin
            R = rotation(angles, cfg)
            intr = intrinsics(depth.shape, cfg)
            # A metric moving object plus a farther stationary object.
            for point, size in [(np.array([.48 * math.sin(t * .5), .15 * math.cos(t * .7), 1.65 + .25 * math.sin(t * .35)]), .12),
                                (np.array([-.65, .15, 2.8]), .16)]:
                p = R.T @ point
                uv = project(p, intr)
                if uv is not None and max(abs(uv)) < 10000:
                    rx, ry = max(2, int(size * intr[0] / p[2])), max(2, int(size * intr[1] / p[2]))
                    cv2.ellipse(depth, tuple(np.rint(uv).astype(int)), (rx, ry), 0, 0, 360, float(p[2]), -1)
            depth += self.rng.normal(0, .006, depth.shape).astype(np.float32)
            return depth, np.full(depth.shape, 100, np.float32), time.monotonic(), 0.
        start = time.monotonic()
        frame = self.cam.requestFrame(150)
        received = time.monotonic()
        if frame is None:
            return None
        try:
            if not isinstance(frame, self.ac.DepthData):
                return None
            # SDK owns buffers: copies MUST happen before releaseFrame.
            depth = np.array(frame.depth_data, dtype=np.float32, copy=True) / 1000.
            quality = None
            if self.quality_name != 'off':
                name = self.quality_name + '_data'
                data = getattr(frame, name, None)
                if data is None or np.size(data) != depth.size:
                    raise RuntimeError(f"SDK не предоставляет {name}; выберите другой фильтр качества")
                quality = np.array(data, dtype=np.float32, copy=True).reshape(depth.shape)
            if depth.ndim != 2 or not depth.size:
                raise RuntimeError("SDK вернул некорректную карту глубины")
            return depth, quality, received, received - start
        finally:
            self.cam.releaseFrame(frame)

    def close(self):
        cam, self.cam = self.cam, None
        if cam:
            try:
                if self.started:
                    cam.stop()
            finally:
                cam.close()


class Shared:
    def __init__(self, cfg):
        self.lock = threading.Lock()
        self.cfg, self.revision = cfg, 0
        self.mode = 'manual'
        self.heartbeat = time.monotonic()
        self.emergency = threading.Event()
        self.stop = threading.Event()
        self.actions = queue.Queue(maxsize=32)
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()

    def snapshot(self):
        with self.lock:
            return self.cfg, self.revision, self.mode, self.heartbeat

    def publish(self, packet):
        try:
            self.frames.get_nowait()
        except queue.Empty:
            pass
        self.frames.put_nowait(packet)

    def event(self, message):
        self.events.put(message)
        LOG.info(message)


class Worker(threading.Thread):
    def __init__(self, shared, demo=False):
        super().__init__(name='ToF acquisition and control', daemon=True)
        self.s, self.demo = shared, demo
        self.gimbal, self.camera = Gimbal(demo), None
        self.csv_file = self.csv_writer = None

    def open_csv(self, path):
        if self.csv_file:
            self.csv_file.close()
        self.csv_file = self.csv_writer = None
        if path:
            self.csv_file = open(path, 'w', newline='', encoding='utf-8')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow(['utc_host', 'monotonic_host_s', 'mode', 'state', 'track_id',
                'range_m', 'projected_area_m2', 'base_vx_m_s', 'base_vy_m_s', 'base_vz_m_s',
                'range_rate_m_s', 'prediction_s', 'prediction_sigma_m', 'x_command_deg',
                'y_command_deg', 'processing_age_s', 'fps', 'config_json'])
            self.csv_file.flush()

    def run(self):
        s = self.s
        cfg, revision, _, _ = s.snapshot()
        tracker = Tracker()
        desired = np.array([90., 90.])
        last_control, last_frame, last_flush = time.monotonic(), time.monotonic(), 0.
        previous_mode = 'manual'
        try:
            self.camera = Camera(cfg, self.demo)
            s.event(f"Камера готова; качество: {self.camera.quality_name}")
            if s.stop.is_set():
                return
            try:
                self.gimbal.connect(cfg.port, lambda: s.stop.is_set() or s.emergency.is_set())
                s.event(f"Arduino: {self.gimbal.port}; центр 90° / 90°")
            except Exception as exc:
                s.event(f"Arduino: {exc}. Просмотр работает; доступно переподключение.")
            while not s.stop.is_set():
                cfg, new_revision, mode, heartbeat = s.snapshot()
                if new_revision != revision:
                    tracker = Tracker()
                    desired = self.gimbal.angles.copy()
                    revision = new_revision
                if mode != previous_mode:
                    desired = self.gimbal.angles.copy()
                    self.gimbal.hold()
                    previous_mode = mode
                frozen = s.emergency.is_set() or time.monotonic() - heartbeat > .6
                if frozen:
                    desired = self.gimbal.angles.copy()
                    self.gimbal.hold()
                while True:
                    try:
                        action, value = s.actions.get_nowait()
                    except queue.Empty:
                        break
                    if action == 'csv':
                        try:
                            self.open_csv(value)
                            s.event('Запись CSV включена' if value else 'Запись CSV выключена')
                        except OSError as exc:
                            try:
                                self.open_csv(None)
                            except OSError:
                                pass
                            s.event(f"CSV: {exc}")
                    elif action == 'connect':
                        with s.lock:
                            s.mode = 'manual'
                        mode = 'manual'
                        try:
                            self.gimbal.connect(cfg.port, lambda: s.stop.is_set() or s.emergency.is_set())
                            desired = self.gimbal.angles.copy()
                            tracker = Tracker()
                            s.event(f"Arduino: {self.gimbal.port}")
                        except Exception as exc:
                            s.event(f"Arduino: {exc}")
                    elif not frozen and mode == 'manual':
                        if action == 'center':
                            desired[:] = 90
                        elif action == 'step':
                            desired += np.array(value) * cfg.manual_step * np.array([
                                -1 if cfg.invert_x else 1, -1 if cfg.invert_y else 1])
                desired = np.clip(desired, [cfg.x_min, cfg.y_min], [cfg.x_max, cfg.y_max])
                pose = self.gimbal.angles.copy()
                packet = self.camera.read(pose, cfg)
                now = time.monotonic()
                dt_control, last_control = min(now - last_control, .1), now
                # Re-read stop/heartbeat after any blocking SDK/serial operation.
                _, _, live_mode, heartbeat = s.snapshot()
                frozen = s.emergency.is_set() or s.stop.is_set() or now - heartbeat > .6
                if live_mode != mode:
                    mode = live_mode
                    desired = self.gimbal.angles.copy()
                if packet is None:
                    self.gimbal.hold()
                    if mode == 'manual' and not frozen:
                        self.gimbal.move(desired, dt_control, cfg)
                    if now - last_frame > cfg.stale_time:
                        s.publish((None, {'state': 'НЕТ КАДРОВ • автоматическое движение остановлено',
                                          'port': self.gimbal.port, 'angles': self.gimbal.angles.copy()}))
                    continue
                depth, quality, stamp, request_delay = packet
                last_frame = now
                detections, valid, capped = detect(depth, quality, cfg)
                R = rotation(pose, cfg)
                target = tracker.update(detections, stamp, R, cfg)
                age = time.monotonic() - stamp
                horizon = min(cfg.horizon_frames * tracker.period + cfg.latency + age, cfg.horizon_max)
                future_uv = None
                sigma = None
                future_camera = None
                state = 'ПОИСК • подтверждение объекта'
                if target:
                    future, sigma = target.kf.future(horizon, cfg)
                    future_camera = R.T @ future
                    future_uv = project(future_camera, intrinsics(depth.shape, cfg))
                    state = 'СОПРОВОЖДЕНИЕ' if target.visible else 'ПОТЕРЯ • ожидание объекта'
                    if sigma > cfg.max_sigma:
                        state = 'ПРОГНОЗ НЕУВЕРЕННЫЙ • удержание'
                fresh = age + request_delay <= cfg.stale_time
                if not fresh:
                    state = 'КАДР УСТАРЕЛ • удержание'
                if mode == 'auto':
                    desired = self.gimbal.angles.copy()
                    if (target and target.visible and fresh and not frozen and future_uv is not None
                            and sigma <= cfg.max_sigma and
                            cfg.near <= (np.linalg.norm(future_camera) if cfg.radial_depth else future_camera[2]) <= cfg.far):
                        errors = np.degrees(np.arctan2(future_camera[:2], future_camera[2]))
                        errors = np.where(abs(errors) > cfg.deadband, errors - np.sign(errors) * cfg.deadband, 0)
                        signs = np.array([-1 if cfg.invert_x else 1, -1 if cfg.invert_y else 1])
                        desired += errors * cfg.gain * dt_control * signs
                _, _, latest_mode, heartbeat = s.snapshot()
                frozen |= (s.emergency.is_set() or s.stop.is_set() or
                           time.monotonic() - heartbeat > .6 or latest_mode != mode)
                if frozen:
                    self.gimbal.hold()
                    state = 'СТОП • команды движения заблокированы'
                else:
                    try:
                        self.gimbal.move(desired, dt_control, cfg)
                    except Exception as exc:
                        s.emergency.set()
                        s.event(f"Ошибка связи Arduino: {exc}. Движение заблокировано.")
                        try:
                            self.gimbal.close()
                        except Exception:
                            LOG.exception('Arduino close')
                image = self.render(depth, valid, detections, target, future_uv, cfg)
                telemetry = dict(state=state, port=self.gimbal.port, angles=self.gimbal.angles.copy(),
                    fps=1 / tracker.period, age=age, horizon=horizon, sigma=sigma,
                    count=len(detections), quality=self.camera.quality_name, capped=capped,
                    range=None, area=None, speed=None, radial=None, id=None)
                velocity = [None] * 3
                if target:
                    velocity = target.kf.x[3:]
                    distance = float(np.linalg.norm(target.kf.x[:3]))
                    telemetry.update(id=target.id, range=distance, area=target.detection.area,
                                     speed=float(np.linalg.norm(velocity)),
                                     radial=float(target.kf.x[:3] @ velocity / max(distance, .001)))
                s.publish((image, telemetry))
                if self.csv_writer:
                    try:
                        self.csv_writer.writerow([datetime.now(timezone.utc).isoformat(), stamp, mode, state,
                            telemetry['id'], telemetry['range'], telemetry['area'], *velocity,
                            telemetry['radial'], horizon, sigma, *self.gimbal.angles, age,
                            telemetry['fps'], json.dumps(asdict(cfg), ensure_ascii=False)])
                        if now - last_flush >= 1:
                            self.csv_file.flush()
                            last_flush = now
                    except OSError as exc:
                        s.event(f"Запись CSV остановлена: {exc}")
                        self.open_csv(None)
        except Exception as exc:
            LOG.exception('Worker stopped')
            s.event(f"Остановлено: {exc}")
            s.publish((None, {'state': f'ОШИБКА: {exc}', 'port': self.gimbal.port,
                              'angles': self.gimbal.angles.copy()}))
        finally:
            for close in (lambda: self.open_csv(None), self.gimbal.close,
                          lambda: self.camera.close() if self.camera else None):
                try:
                    close()
                except Exception:
                    LOG.exception('Resource cleanup')
            s.event('Поток оборудования завершён')

    @staticmethod
    def render(depth, valid, detections, target, future_uv, cfg):
        value = np.nan_to_num((depth - cfg.near) / (cfg.far - cfg.near), nan=1, posinf=1, neginf=0)
        image = cv2.applyColorMap((255 * (1 - np.clip(value, 0, 1))).astype(np.uint8), cv2.COLORMAP_TURBO)
        image[~valid] = (23, 19, 15)
        h, w = depth.shape
        for d in detections:
            x, y, bw, bh = d.box
            cv2.rectangle(image, (x, y), (x + bw - 1, y + bh - 1), (125, 125, 125), 1)
        cv2.drawMarker(image, (w // 2, h // 2), (230, 230, 230), cv2.MARKER_CROSS, 14, 1)
        if target and target.visible:
            x, y, bw, bh = target.detection.box
            cv2.rectangle(image, (x, y), (x + bw - 1, y + bh - 1), (100, 255, 110), 2)
            p = tuple(np.rint(target.detection.uv).astype(int))
            cv2.circle(image, p, 3, (255, 255, 255), -1)
            if future_uv is not None:
                q = tuple(np.rint(np.clip(future_uv, [-w, -h], [2*w, 2*h])).astype(int))
                cv2.arrowedLine(image, p, q, (255, 225, 30), 1, tipLength=.2)
                if 0 <= q[0] < w and 0 <= q[1] < h:
                    cv2.drawMarker(image, q, (255, 225, 30), cv2.MARKER_DIAMOND, 10, 1)
            cv2.putText(image, f'ID {target.id}  {target.detection.distance:.2f} m',
                        (max(0, x), max(12, y - 4)), cv2.FONT_HERSHEY_SIMPLEX, .35, (255, 255, 255), 1)
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# All widget access belongs exclusively to the Tk thread.
GROUPS = {
    'Объект': [
        ('near', 'Ближняя граница, м'), ('far', 'Дальняя граница, м'),
        ('area_min', 'Мин. проекция, м²'), ('area_max', 'Макс. проекция, м²'),
        ('pixels_min', 'Мин. площадь, пикс.'), ('frame_fraction', 'Макс. доля кадра, 0–1'),
        ('depth_gap', 'Связность по глубине, м'), ('quality_min', 'Порог качества, ед. SDK'),
        ('confirm', 'Подтверждение, кадров'), ('lost_time', 'Память потери, с'),
        ('association', 'Ворота сопоставления, м'), ('switch_margin', 'Ближе текущего на, м'),
        ('switch_frames', 'Подтверждение смены, кадров')],
    'Прогноз': [
        ('horizon_frames', 'Прогноз на N кадров'), ('horizon_max', 'Макс. горизонт, с'),
        ('latency', 'Доп. задержка системы, с'), ('measurement_noise', 'Шум измерения σ, м'),
        ('acceleration_noise', 'Шум ускорения σ, м/с²'), ('max_sigma', 'Допуск σ прогноза, м'),
        ('fov_x', 'Угол обзора X, °'), ('fov_y', 'Угол обзора Y, °'),
        ('stale_time', 'Порог устаревания, с'),
        ('radial_depth', 'SDK выдаёт дальность по лучу'),
        ('compensate', 'Компенсация командных углов')],
    'Приводы': [
        ('x_min', 'X минимум, ° · D9'), ('x_max', 'X максимум, °'),
        ('y_min', 'Y минимум, ° · D10'), ('y_max', 'Y максимум, °'),
        ('invert_x', 'Инверсия X'), ('invert_y', 'Инверсия Y'),
        ('gain', 'Коэффициент P, 1/с'), ('deadband', 'Мёртвая зона, °'),
        ('max_speed', 'Макс. скорость, °/с'), ('max_acceleration', 'Макс. ускорение, °/с²'),
        ('manual_step', 'Шаг ручного режима, °')],
    'Связь': [
        ('port', 'Arduino: AUTO или порт'), ('connection', 'Подключение камеры'),
        ('camera_index', 'Индекс камеры'), ('sdk_range', 'Режим SDK, мм'),
        ('quality', 'Источник качества')],
}

HELP = """ПОРЯДОК НАСТРОЙКИ
1. Проверьте механические пределы и питание. Arduino: StandardFirmata.
2. Нажмите «Запуск». Начальный режим — ручной, центр — 90° / 90°.
3. Проверьте направления кнопками; при необходимости измените инверсию.
4. Задайте дальность, площадь, угол обзора камеры. Нажмите «Применить».
5. Включите «Авто». Зеленая рамка — выбранный объект; голубой ромб — прогноз.

ПЛОЩАДЬ И ДАЛЬНОСТЬ
Проекция в м² ≈ сумма Z²/(fx·fy) по пикселям компоненты.
Это оценка видимой площади поперёк оптической оси, а не площадь поверхности.
Она зависит от калибровки FOV, ракурса и перекрытия объекта.
Дальность отбора пикселей — величина SDK (обычно Z). В телеметрии —
евклидово расстояние до оценённого центра объекта.
Объекты с соприкасающимися контурами и похожей глубиной могут сливаться.
«Связность» задаёт допустимый локальный перепад между соседними пикселями.

УСТОЙЧИВОСТЬ И ПРОГНОЗ
Объект подтверждается N последовательными наблюдениями.
Выбирается ближайший подтверждённый объект. Для смены нужны отрыв
по расстоянию и несколько кадров подряд. При краткой потере идентификатор
сохраняется, движение в авто останавливается. Затем выполняется новый поиск.
Kalman XYZ+VXYZ учитывает реальное dt. Горизонт = N×период кадров +
дополнительная задержка + время обработки, с ограничением в секундах.
Период измеряется по полученным/обработанным кадрам, не по паспортному FPS.
Прогноз при слишком большой неопределённости не управляет приводами.
Параметры шума измерения/ускорения требуют настройки по реальным данным.

ТЕЛЕМЕТРИЯ
Система координат: X вправо, Y вниз, Z вперёд от нейтрального положения.
Скорость приближения отрицательна, удаления положительна.
Углы — команды, а не показания датчиков. Компенсация вращения приближённая:
люфт, задержка, наклон осей и поступательное движение основания неизвестны.
Без энкодеров/IMU это не абсолютная скорость объекта и не навигационное решение.
Нет высоты, географических координат или синхронизации с авиационной шиной.
CSV содержит UTC компьютера и монотонное время получения кадров.

УПРАВЛЕНИЕ И ОШИБКИ
Ручные стрелки задают шаг направления изображения с учётом инверсии.
Escape / «СТОП» отменяет дальнейшие команды; сервоприводы остаются запитаны
и могут закончить уже полученное движение. Это не аппаратная аварийная кнопка.
«Продолжить вручную» снимает блокировку. «Отключить» освобождает ресурсы;
StandardFirmata снимает управляющие импульсы при закрытии соединения.
При неоднозначном AUTO выберите порт вручную. Повторное подключение
возвращает 90/90; автоматический режим надо включать самостоятельно.
Параметры подключения камеры/качества меняются после «Отключить».

ДЕМОНСТРАЦИЯ
--demo не обращается к камере и Arduino. Два синтетических объекта,
шум глубины и геометрическая реакция изображения на команды подвеса.
Это проверка логики и интерфейса, а не модель механики сервоприводов.
"""


class App:
    def __init__(self, args):
        import tkinter as tk
        from tkinter import ttk, messagebox, filedialog
        from PIL import Image, ImageTk
        self.tk, self.ttk, self.messagebox, self.filedialog = tk, ttk, messagebox, filedialog
        self.Image, self.ImageTk = Image, ImageTk
        self.root = tk.Tk()
        self.root.title('ToF Gimbal Tracker · камера глубины')
        self.root.geometry('1280x850')
        self.root.minsize(1000, 720)
        self.root.configure(bg='#101823')
        self.args, self.worker, self.closing = args, None, False
        cfg = Settings()
        if args.config:
            cfg = self.read_config(args.config)
        self.shared = Shared(cfg)
        self.photo = None
        self.csv_active = False
        self.variables = {}
        self.style()
        top = ttk.Frame(self.root, padding=(20, 16))
        top.pack(fill='x')
        ttk.Label(top, text='ToF / GIMBAL', style='Title.TLabel').pack(side='left')
        ttk.Label(top, text='  ГЛУБИНА · СОПРОВОЖДЕНИЕ · ПРОГНОЗ', style='Muted.TLabel').pack(side='left', padx=12)
        self.badge = ttk.Label(top, text='ДЕМОНСТРАЦИЯ' if args.demo else 'Raspberry Pi 5  →  Arduino Uno', style='Accent.TLabel')
        self.badge.pack(side='right')
        toolbar = ttk.Frame(self.root, padding=(20, 0, 20, 12))
        toolbar.pack(fill='x')
        self.start_button = ttk.Button(toolbar, text='Запуск', command=self.start, style='Accent.TButton')
        self.start_button.pack(side='left', padx=(0, 6))
        ttk.Button(toolbar, text='Отключить', command=self.disconnect).pack(side='left', padx=6)
        self.mode = tk.StringVar(value='manual')
        for value, title in [('manual', 'Ручной'), ('auto', 'Авто')]:
            ttk.Radiobutton(toolbar, text=title, value=value, variable=self.mode, command=self.change_mode).pack(side='left', padx=8)
        ttk.Button(toolbar, text='СТОП  [Esc]', style='Stop.TButton', command=self.emergency).pack(side='right')
        ttk.Button(toolbar, text='Продолжить вручную', command=self.resume).pack(side='right', padx=8)
        body = ttk.Panedwindow(self.root, orient='horizontal')
        body.pack(fill='both', expand=True, padx=20)
        left = ttk.Frame(body)
        right = ttk.Frame(body, width=430)
        body.add(left, weight=3)
        body.add(right, weight=2)
        self.status = tk.StringVar(value='ГОТОВО К ЗАПУСКУ • ручной режим')
        ttk.Label(left, textvariable=self.status, style='Accent.TLabel', wraplength=650).pack(anchor='w', pady=(0, 8))
        self.video = tk.Canvas(left, bg='#0a1019', highlightthickness=1, highlightbackground='#26384b', height=340)
        self.video.pack(fill='both', expand=True, padx=(0, 12))
        self.video.create_text(300, 160, text='Карта глубины появится после запуска', fill='#8293a7', font=('Helvetica', 13), tags='placeholder')
        ttk.Label(left, text='Ближе: красный  →  жёлтый  →  синий: дальше   |   тёмный: нет данных', style='Muted.TLabel').pack(anchor='w', pady=8)
        cards = ttk.Frame(left)
        cards.pack(fill='x', pady=8)
        self.metrics = {}
        for i, (key, label) in enumerate([('range', 'ДАЛЬНОСТЬ'), ('speed', 'СКОРОСТЬ*'), ('radial', 'УДАЛЕНИЕ / СБЛИЖЕНИЕ'), ('area', 'ПРОЕКЦИЯ')]):
            box = ttk.Frame(cards, padding=9, style='Card.TFrame')
            box.grid(row=i // 2, column=i % 2, sticky='nsew', padx=(0, 8), pady=(0, 8))
            cards.columnconfigure(i % 2, weight=1)
            ttk.Label(box, text=label, style='CardCaption.TLabel').pack(anchor='w')
            var = tk.StringVar(value='—')
            self.metrics[key] = var
            ttk.Label(box, textvariable=var, style='CardValue.TLabel').pack(anchor='w')
        self.detail = tk.StringVar(value='X 90°  /  Y 90° · команды, без обратной связи')
        ttk.Label(left, textvariable=self.detail, wraplength=640, style='Muted.TLabel').pack(anchor='w', pady=5)
        ttk.Label(left, text='* Оценка относительно основания; компенсация поворота по командам.', style='Muted.TLabel', wraplength=640).pack(anchor='w')
        manual = ttk.Frame(left, padding=(0, 12))
        manual.pack(fill='x')
        for text, vector in [('← Влево', (-1, 0)), ('↑ Вверх', (0, -1)), ('↓ Вниз', (0, 1)), ('Вправо →', (1, 0))]:
            ttk.Button(manual, text=text, command=lambda v=vector: self.manual(v)).pack(side='left', padx=(0, 4))
        ttk.Button(manual, text='Центр 90/90', command=lambda: self.manual(None)).pack(side='left', padx=4)
        notebook = ttk.Notebook(right)
        notebook.pack(fill='both', expand=True)
        for name, fields in GROUPS.items():
            tab = ttk.Frame(notebook, padding=12)
            notebook.add(tab, text=name)
            tab.columnconfigure(0, weight=1)
            for row, (key, label) in enumerate(fields):
                default = getattr(cfg, key)
                if isinstance(default, bool):
                    var = tk.BooleanVar(value=default)
                    ttk.Checkbutton(tab, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky='w', pady=7)
                else:
                    var = tk.StringVar(value=str(default))
                    ttk.Label(tab, text=label).grid(row=row, column=0, sticky='w', pady=6)
                    options = {'quality': ['auto', 'confidence', 'amplitude', 'off'],
                               'connection': ['CSI', 'USB'], 'sdk_range': ['2000', '4000']}
                    if key in options:
                        widget = ttk.Combobox(tab, textvariable=var, values=options[key], state='readonly', width=13)
                    else:
                        widget = ttk.Entry(tab, textvariable=var, width=14)
                    widget.grid(row=row, column=1, sticky='e', padx=(8, 0), pady=6)
                self.variables[key] = var
            if name == 'Связь':
                ttk.Button(tab, text='Показать доступные порты', command=self.ports).grid(row=6, column=0, columnspan=2, sticky='ew', pady=8)
                ttk.Button(tab, text='Переподключить Arduino', command=lambda: self.action('connect', None)).grid(row=7, column=0, columnspan=2, sticky='ew', pady=8)
                ttk.Label(tab, text='AUTO выбирает единственный USB-кандидат и проверяет Firmata.\n\nИзменения полей вступают в силу после «Применить».\n\nПовторное подключение центрирует подвес.\n\nFOV 60°/45° — начальная оценка: укажите углы обзора своей модели камеры.', wraplength=345, style='Muted.TLabel').grid(row=8, column=0, columnspan=2, sticky='w', pady=12)
        help_tab = ttk.Frame(notebook)
        notebook.add(help_tab, text='Помощь')
        help_text = tk.Text(help_tab, wrap='word', bg='#172333', fg='#dce6f1', relief='flat', font=('Helvetica', 11), padx=12, pady=12, width=38)
        help_scroll = ttk.Scrollbar(help_tab, command=help_text.yview)
        help_scroll.pack(side='right', fill='y')
        help_text.configure(yscrollcommand=help_scroll.set)
        help_text.pack(fill='both', expand=True)
        help_text.insert('1.0', HELP)
        help_text.configure(state='disabled')
        settings_bar = ttk.Frame(right, padding=(0, 12))
        settings_bar.pack(fill='x')
        ttk.Button(settings_bar, text='Применить', style='Accent.TButton', command=self.apply).pack(side='left')
        ttk.Button(settings_bar, text='Сохранить', command=self.save).pack(side='left', padx=6)
        ttk.Button(settings_bar, text='Загрузить', command=self.load).pack(side='left')
        footer = ttk.Frame(self.root, padding=(20, 8, 20, 16))
        footer.pack(fill='x')
        self.log_line = tk.StringVar(value='Измените настройки и нажмите «Применить». Подробности — во вкладке «Помощь».')
        ttk.Label(footer, textvariable=self.log_line, wraplength=920, style='Muted.TLabel').pack(side='left', fill='x', expand=True)
        self.csv_button = ttk.Button(footer, text='Запись CSV', command=self.toggle_csv)
        self.csv_button.pack(side='right')
        self.root.bind('<Escape>', lambda event: self.emergency())
        self.root.protocol('WM_DELETE_WINDOW', self.close)
        self.root.after(40, self.poll)

    def style(self):
        style = self.ttk.Style(self.root)
        style.theme_use('clam')
        style.configure('.', background='#101823', foreground='#dce6f1', font=('Helvetica', 11))
        style.configure('TFrame', background='#101823')
        style.configure('TLabel', background='#101823')
        style.configure('Title.TLabel', font=('Helvetica', 22, 'bold'), foreground='#f4f8fc')
        style.configure('Muted.TLabel', foreground='#94a8bd', font=('Helvetica', 10))
        style.configure('Accent.TLabel', foreground='#57d8cc', font=('Helvetica', 11, 'bold'))
        style.configure('TButton', background='#24364a', padding=(10, 8), borderwidth=0)
        style.map('TButton', background=[('active', '#36506b'), ('disabled', '#1b2735')])
        style.configure('Accent.TButton', background='#17665f', foreground='#ffffff')
        style.map('Accent.TButton', background=[('active', '#20897f')])
        style.configure('Stop.TButton', background='#923e4c', foreground='white')
        style.map('Stop.TButton', background=[('active', '#b64e60')])
        style.configure('TEntry', fieldbackground='#1b293a', foreground='#f2f6fa', insertcolor='white')
        style.configure('TCombobox', fieldbackground='#1b293a', foreground='#f2f6fa', arrowcolor='#57d8cc')
        style.map('TCombobox', fieldbackground=[('readonly', '#1b293a')], foreground=[('readonly', '#f2f6fa')])
        style.configure('TNotebook', borderwidth=0)
        style.configure('TNotebook.Tab', background='#1b293a', padding=(8, 8))
        style.map('TNotebook.Tab', background=[('selected', '#264252')], foreground=[('selected', '#65e1d4')])
        style.configure('TCheckbutton', background='#101823')
        style.configure('TRadiobutton', background='#101823')
        style.configure('Card.TFrame', background='#1a2839')
        style.configure('CardCaption.TLabel', background='#1a2839', foreground='#95a9bf', font=('Helvetica', 9))
        style.configure('CardValue.TLabel', background='#1a2839', foreground='#eef6ff', font=('Helvetica', 20, 'bold'))

    @staticmethod
    def read_config(path):
        with open(path, encoding='utf-8') as stream:
            data = json.load(stream)
        if not isinstance(data, dict):
            raise ValueError('Настройки должны быть JSON-объектом')
        if set(data) - set(asdict(Settings())):
            raise ValueError('Неизвестные поля в настройках')
        return Settings(**data).validate()

    def collect(self):
        values = {}
        for key, default in asdict(Settings()).items():
            raw = self.variables[key].get()
            try:
                values[key] = raw if isinstance(default, bool) else type(default)(str(raw).strip().replace(',', '.') if not isinstance(default, str) else str(raw).strip())
            except (TypeError, ValueError):
                raise ValueError(f'{key}: некорректное значение {raw!r}') from None
        return Settings(**values).validate()

    def apply(self):
        try:
            cfg = self.collect()
            old, _, _, _ = self.shared.snapshot()
            if self.worker and self.worker.is_alive():
                for key in ('connection', 'camera_index', 'sdk_range', 'quality'):
                    if getattr(cfg, key) != getattr(old, key):
                        raise ValueError('Сначала нажмите «Отключить» для изменения подключения камеры или источника качества')
            with self.shared.lock:
                self.shared.cfg = cfg
                self.shared.revision += 1
            self.log_line.set('Настройки применены; объекты будут подтверждены заново.')
            return True
        except ValueError as exc:
            self.messagebox.showerror('Проверьте настройки', str(exc))
            return False

    def start(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.apply():
            return
        cfg, _, _, _ = self.shared.snapshot()
        self.shared = Shared(cfg)
        self.mode.set('manual')
        self.csv_active = False
        self.csv_button.configure(text='Запись CSV')
        self.worker = Worker(self.shared, self.args.demo)
        self.worker.start()
        self.start_button.configure(state='disabled')
        self.status.set('ПОДКЛЮЧЕНИЕ • Arduino может перезагружаться несколько секунд')

    def disconnect(self):
        self.shared.emergency.set()
        self.shared.stop.set()
        self.mode.set('manual')
        with self.shared.lock:
            self.shared.mode = 'manual'
        self.status.set('ОТКЛЮЧЕНИЕ • освобождение камеры и Arduino')

    def action(self, key, value):
        if not self.worker or not self.worker.is_alive() or self.shared.stop.is_set():
            self.log_line.set('Сначала запустите подключение.')
            return
        if key == 'connect':
            self.mode.set('manual')
            self.change_mode()
        try:
            self.shared.actions.put_nowait((key, value))
        except queue.Full:
            self.log_line.set('Очередь команд заполнена; дождитесь выполнения.')

    def manual(self, vector):
        if self.mode.get() != 'manual' or self.shared.emergency.is_set():
            self.log_line.set('Ручное движение доступно в ручном режиме после снятия СТОП.')
            return
        self.action('center' if vector is None else 'step', vector)

    def change_mode(self):
        if self.shared.emergency.is_set():
            self.mode.set('manual')
        with self.shared.lock:
            self.shared.mode = self.mode.get()

    def emergency(self):
        self.shared.emergency.set()
        self.mode.set('manual')
        with self.shared.lock:
            self.shared.mode = 'manual'
        self.status.set('СТОП • дальнейшие команды заблокированы')

    def resume(self):
        if self.shared.stop.is_set():
            self.log_line.set('Дождитесь отключения, затем нажмите «Запуск».')
            return
        # Purge old directional actions before clearing the stop latch.
        retained = []
        while True:
            try:
                item = self.shared.actions.get_nowait()
                if item[0] not in ('step', 'center'):
                    retained.append(item)
            except queue.Empty:
                break
        for item in retained:
            self.shared.actions.put_nowait(item)
        self.mode.set('manual')
        with self.shared.lock:
            self.shared.mode = 'manual'
        self.shared.emergency.clear()
        self.log_line.set('Блокировка снята. Ручной режим.')

    def save(self):
        try:
            cfg = self.collect()
            path = self.filedialog.asksaveasfilename(title='Сохранить настройки', defaultextension='.json', initialfile='tof_settings.json', filetypes=[('JSON', '*.json')])
            if path:
                target = Path(path)
                temporary = target.with_suffix(target.suffix + '.tmp')
                temporary.write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2), encoding='utf-8')
                temporary.replace(target)
                self.log_line.set(f'Настройки сохранены: {path}')
        except (OSError, ValueError) as exc:
            self.messagebox.showerror('Сохранение', str(exc))

    def load(self):
        path = self.filedialog.askopenfilename(title='Настройки', filetypes=[('JSON', '*.json')])
        if not path:
            return
        try:
            cfg = self.read_config(path)
            for key, value in asdict(cfg).items():
                self.variables[key].set(value)
            self.log_line.set('Настройки загружены в поля. Нажмите «Применить».')
        except (OSError, ValueError, TypeError) as exc:
            self.messagebox.showerror('Загрузка', str(exc))

    def ports(self):
        try:
            from serial.tools import list_ports
            ports = list(list_ports.comports())
            self.messagebox.showinfo('Последовательные порты', '\n'.join(f'{p.device} — {p.description}' for p in ports) or 'Порты не найдены')
        except ImportError:
            self.messagebox.showerror('Зависимости', 'Установите pyserial: python3 -m pip install pyserial')

    def toggle_csv(self):
        if not self.worker or not self.worker.is_alive():
            self.log_line.set('Запись доступна после запуска.')
            return
        if self.csv_active:
            self.action('csv', None)
        else:
            path = self.filedialog.asksaveasfilename(title='Запись телеметрии', defaultextension='.csv', initialfile='tof_telemetry.csv', filetypes=[('CSV', '*.csv')])
            if not path:
                return
            self.action('csv', path)

    def poll(self):
        with self.shared.lock:
            self.shared.heartbeat = time.monotonic()
        if self.worker and not self.worker.is_alive():
            self.start_button.configure(state='normal')
            self.csv_active = False
            self.csv_button.configure(text='Запись CSV')
        while True:
            try:
                message = self.shared.events.get_nowait()
                self.log_line.set(message)
                if message == 'Запись CSV включена':
                    self.csv_active = True
                    self.csv_button.configure(text='Остановить CSV')
                elif message.startswith(('CSV:', 'Запись CSV выключена', 'Запись CSV остановлена')):
                    self.csv_active = False
                    self.csv_button.configure(text='Запись CSV')
            except queue.Empty:
                break
        try:
            image, data = self.shared.frames.get_nowait()
        except queue.Empty:
            image, data = None, None
        if data:
            self.status.set(data['state'])
            for key, units, digits in [('range', 'м', 2), ('speed', 'м/с', 2), ('radial', 'м/с', 2), ('area', 'м²', 4)]:
                value = data.get(key)
                self.metrics[key].set(f'{value:+.{digits}f} {units}' if key == 'radial' and value is not None else f'{value:.{digits}f} {units}' if value is not None else '—')
            x, y = data['angles']
            detail = f"X {x:.1f}° / Y {y:.1f}° · {data['port']}"
            if 'fps' in data:
                sigma = '—' if data['sigma'] is None else f"{data['sigma']:.3f} м"
                detail += f"\n{data['fps']:.1f} FPS · обработка {data['age']*1000:.0f} мс · прогноз {data['horizon']:.3f} с · σ {sigma}"
                detail += f"\nОбъектов: {data['count']} · качество: {data['quality']}"
                if data['capped']:
                    detail += ' · лимит компонент: увеличьте фильтрацию'
            self.detail.set(detail)
            if image is None:
                self.video.delete('all')
                self.video.create_text(max(1, self.video.winfo_width()) // 2, max(1, self.video.winfo_height()) // 2,
                    text='НЕТ АКТУАЛЬНОГО ИЗОБРАЖЕНИЯ', fill='#c4a3a8', font=('Helvetica', 14))
        if image is not None:
            picture = self.Image.fromarray(image)
            scale = min(max(1, self.video.winfo_width() - 4) / picture.width,
                        max(1, self.video.winfo_height() - 4) / picture.height)
            picture = picture.resize((max(1, round(picture.width * scale)),
                                      max(1, round(picture.height * scale))), self.Image.Resampling.NEAREST)
            self.photo = self.ImageTk.PhotoImage(picture)
            self.video.delete('all')
            self.video.create_image(self.video.winfo_width() // 2, self.video.winfo_height() // 2, image=self.photo)
        if self.closing and (not self.worker or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(40, self.poll)

    def close(self):
        self.closing = True
        self.disconnect()
        self.log_line.set('Завершение: ожидается возврат SDK и закрытие устройств…')

    def run(self):
        self.root.mainloop()


def self_test():
    """Deterministic regression checks; require numpy/OpenCV, not Tk/hardware."""
    import unittest

    class Tests(unittest.TestCase):
        def setUp(self):
            self.cfg = replace(Settings(), area_min=.0001, area_max=1., confirm=3)

        def detection(self, z=1., x=0., area=.01):
            return Detection(np.array([x, 0., z]), area, 100, (10, 10, 10, 10), (15, 15), math.hypot(x, z))

        def test_validation(self):
            Settings().validate()
            for bad in [dict(near=5), dict(x_min=100), dict(horizon_frames=-1),
                        dict(fov_x=float('nan')), dict(confirm=2.5), dict(invert_x=1), dict(area_min=2)]:
                with self.assertRaises(ValueError):
                    replace(Settings(), **bad).validate()

        def test_invalid_depth_quality_and_size(self):
            d = np.full((100, 100), np.nan, np.float32)
            d[20:40, 20:40] = 1
            d[60:85, 60:85] = 2
            quality = np.full_like(d, 100)
            quality[60:85, 60:85] = 0
            found, _, _ = detect(d, quality, self.cfg)
            self.assertEqual(len(found), 1)
            self.assertAlmostEqual(found[0].point[2], 1)
            found, _, _ = detect(d, quality, replace(self.cfg, area_min=.5))
            self.assertFalse(found)

        def test_depth_adjacency(self):
            d = np.full((100, 100), np.nan, np.float32)
            d[20:60, 20:40], d[20:60, 40:60] = 1., 2.
            found, _, _ = detect(d, None, self.cfg)
            self.assertEqual(len(found), 2)

        def test_area_scales_with_distance_squared(self):
            d = np.full((100, 100), np.nan, np.float32)
            d[30:50, 30:50] = 1
            a = detect(d, None, self.cfg)[0][0].area
            d[30:50, 30:50] = 2
            b = detect(d, None, self.cfg)[0][0].area
            self.assertAlmostEqual(b / a, 4)

        def test_nearest_stable_and_occlusion(self):
            tracker = Tracker()
            for n in range(3):
                target = tracker.update([self.detection(1), self.detection(2)], n * .04, np.eye(3), self.cfg)
                if n < 2:
                    self.assertIsNone(target)
            self.assertAlmostEqual(target.detection.distance, 1.)
            identity = target.id
            target = tracker.update([], .16, np.eye(3), self.cfg)
            self.assertFalse(target.visible)
            target = tracker.update([self.detection(1)], .20, np.eye(3), self.cfg)
            self.assertEqual(identity, target.id)
            self.assertIsNone(tracker.update([], 1., np.eye(3), self.cfg))

        def test_switch_hysteresis(self):
            cfg = replace(self.cfg, confirm=1, switch_frames=3)
            tr = Tracker()
            original = tr.update([self.detection(2)], 0, np.eye(3), cfg).id
            for n in (1, 2):
                t = tr.update([self.detection(2), self.detection(1)], n * .04, np.eye(3), cfg)
                self.assertEqual(t.id, original)
            t = tr.update([self.detection(2), self.detection(1)], .12, np.eye(3), cfg)
            self.assertNotEqual(t.id, original)

        def test_velocity_and_approach_prediction(self):
            k = Filter(np.array([0., 0., 2.]), self.cfg)
            for i in range(1, 101):
                k.predict(.02, self.cfg)
                k.correct(np.array([.1 * i * .02, 0., 2 - .2 * i * .02]), self.cfg)
            np.testing.assert_allclose(k.x[3:], [.1, 0., -.2], atol=.015)
            point, sigma = k.future(.3, self.cfg)
            self.assertLess(point[2], k.x[2])
            self.assertGreater(sigma, 0)
            self.assertTrue(np.all(np.linalg.eigvalsh(k.P) >= -1e-9))

        def test_rotation_reprojection(self):
            for inv in (False, True):
                cfg = replace(self.cfg, invert_x=inv)
                R = rotation([120, 105], cfg)
                np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
                p = np.array([.1, .2, 1.])
                np.testing.assert_allclose(R.T @ (R @ p), p, atol=1e-12)

        def test_stationary_object_rotating_camera(self):
            tr = Tracker()
            base = np.array([.1, .1, 2.])
            for i in range(60):
                R = rotation([90 + i * .2, 90], self.cfg)
                d = self.detection()
                d.point = R.T @ base
                t = tr.update([d], i * .04, R, self.cfg)
            np.testing.assert_allclose(t.kf.x[3:], np.zeros(3), atol=1e-8)

        def test_servo_limits_speed_acceleration(self):
            g = Gimbal(True)
            previous = g.angles.copy()
            previous_v = g.velocity.copy()
            for _ in range(200):
                g.move([500, -500], .02, self.cfg)
                self.assertTrue(np.all(abs(g.angles - previous) <= self.cfg.max_speed * .02 + 1e-8))
                self.assertTrue(np.all(abs(g.velocity - previous_v) <= self.cfg.max_acceleration * .02 + 1e-8))
                previous, previous_v = g.angles.copy(), g.velocity.copy()
            self.assertLessEqual(g.angles[0], 160)
            self.assertGreaterEqual(g.angles[1], 45)
            g.hold()
            np.testing.assert_equal(g.velocity, [0, 0])

        def test_demo_pipeline(self):
            camera = Camera(self.cfg, True)
            packet = camera.read([90, 90], self.cfg)
            depth, quality, timestamp, _ = packet
            detections, valid, _ = detect(depth, quality, self.cfg)
            self.assertGreaterEqual(len(detections), 1)
            image = Worker.render(depth, valid, detections, None, None, self.cfg)
            self.assertEqual(image.shape, (180, 240, 3))

        def test_sdk_buffer_copy_and_release(self):
            from types import SimpleNamespace
            class Frame:
                depth_data = np.full((4, 5), 1500., np.float32)
                confidence_data = np.full((4, 5), 100., np.float32)
            frame = Frame()
            released = []
            def release(item):
                released.append(item)
                item.depth_data[:] = -1
                item.confidence_data[:] = -1
            camera = Camera.__new__(Camera)
            camera.demo = False
            camera.quality_name = 'confidence'
            camera.ac = SimpleNamespace(DepthData=Frame)
            camera.cam = SimpleNamespace(requestFrame=lambda timeout: frame, releaseFrame=release)
            depth, quality, _, _ = camera.read([90, 90], self.cfg)
            self.assertEqual(len(released), 1)
            np.testing.assert_allclose(depth, 1.5)
            np.testing.assert_allclose(quality, 100)

        def test_sdk_bad_frame_released(self):
            from types import SimpleNamespace
            class Frame:
                depth_data = np.ones((4, 5), np.float32)
            frame = Frame()
            released = []
            camera = Camera.__new__(Camera)
            camera.demo = False
            camera.quality_name = 'confidence'
            camera.ac = SimpleNamespace(DepthData=Frame)
            camera.cam = SimpleNamespace(requestFrame=lambda timeout: frame,
                                         releaseFrame=lambda item: released.append(item))
            with self.assertRaises(RuntimeError):
                camera.read([90, 90], self.cfg)
            self.assertEqual(len(released), 1)

        def test_outlier_not_associated(self):
            tracker = Tracker()
            cfg = replace(self.cfg, confirm=1)
            original = tracker.update([self.detection(1)], 0, np.eye(3), cfg).id
            selected = tracker.update([self.detection(3)], .04, np.eye(3), cfg)
            self.assertEqual(selected.id, original)
            self.assertFalse(selected.visible)
            self.assertEqual(len(tracker.tracks), 2)

        def test_newest_frame_queue(self):
            shared = Shared(self.cfg)
            shared.publish((1, {}))
            shared.publish((2, {}))
            self.assertEqual(shared.frames.get_nowait()[0], 2)
            self.assertTrue(shared.frames.empty())

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    return unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful()


def main():
    parser = argparse.ArgumentParser(description='Arducam ToF camera gimbal · Raspberry Pi 5 + Arduino Uno')
    parser.add_argument('--demo', action='store_true', help='синтетическая камера и виртуальные сервоприводы')
    parser.add_argument('--self-test', action='store_true', help='проверка алгоритмов без оборудования и дисплея')
    parser.add_argument('--config', type=Path, help='JSON с настройками')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    if args.self_test:
        return 0 if self_test() else 1
    try:
        App(args).run()
    except (ImportError, ValueError, OSError) as exc:
        LOG.error('%s', exc)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
