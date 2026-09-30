#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arducam ToF + Arduino Uno 2-axis gimbal tracker for Raspberry Pi 5.

Hardware:
  - Arducam ToF camera connected over CSI
  - Arduino Uno running StandardFirmata
  - X servo on D9
  - Y servo on D10

GUI:
  - tkinter/ttk
Image handling:
  - OpenCV + Pillow
Arduino link:
  - pyFirmata2

Design goals:
  - Tkinter runs only in the main thread.
  - Camera acquisition/tracking/servo control runs in a worker thread.
  - Monotonic timestamps and sequence numbers are used for telemetry-like state.
  - Tracking has confirmation, dropout tolerance, outlier gates, filtering,
    command dead-band, rate limiting and servo angle saturation.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk
from PIL import Image, ImageTk

import pyfirmata2
import ArducamDepthCamera as ac


# -----------------------------------------------------------------------------
# Hardware constants
# -----------------------------------------------------------------------------
ARDUINO_DEFAULT_PORT = "/dev/ttyACM0"
SERVO_X_PIN = 9
SERVO_Y_PIN = 10
SERVO_CENTER_X = 90.0
SERVO_CENTER_Y = 90.0
SENSOR_RANGE_MM = 4000  # the example camera API uses 2000 or 4000


# -----------------------------------------------------------------------------
# Helpers / data structures
# -----------------------------------------------------------------------------
def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def finite_or(value: float, fallback: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else fallback


@dataclass
class RuntimeConfig:
    # Depth / segmentation
    min_distance_mm: float = 300.0
    max_distance_mm: float = 3500.0
    confidence_threshold: float = 30.0
    min_object_area_px: int = 70
    morphology_kernel: int = 3

    # Target stability / reacquisition
    stable_frames: int = 5
    lost_tolerance_frames: int = 5
    reacquire_radius_px: float = 45.0
    max_depth_jump_mm: float = 500.0

    # Alpha-beta filter + prediction
    filter_alpha: float = 0.65
    filter_beta: float = 0.12
    prediction_frames: int = 10
    max_pixel_speed_px_s: float = 1800.0
    max_depth_speed_m_s: float = 8.0

    # Arduino / servos
    arduino_port: str = ARDUINO_DEFAULT_PORT
    servo_enabled: bool = True
    servo_x_min: float = 20.0
    servo_x_max: float = 160.0
    servo_y_min: float = 45.0
    servo_y_max: float = 135.0
    servo_gain_x_deg_s: float = 55.0
    servo_gain_y_deg_s: float = 55.0
    servo_max_rate_deg_s: float = 70.0
    servo_update_hz: float = 20.0
    deadband_px: float = 6.0
    invert_x: bool = False
    invert_y: bool = False
    return_center_on_loss: bool = False
    return_center_delay_s: float = 1.5

    # Control mode / manual gimbal control
    control_mode: str = "AUTO"  # AUTO or MANUAL
    manual_step_deg: float = 2.0
    manual_rate_deg_s: float = 45.0


@dataclass
class Candidate:
    cx: float
    cy: float
    depth_mm: float
    area: int
    bbox: Tuple[int, int, int, int]
    mean_confidence: float


@dataclass
class TelemetrySnapshot:
    seq: int = 0
    monotonic_ns: int = 0
    fps: float = 0.0
    frame_age_ms: float = 0.0
    state: str = "STOPPED"
    target_valid: bool = False
    stable_count: int = 0
    lost_count: int = 0
    x_px: float = 0.0
    y_px: float = 0.0
    z_m: float = 0.0
    vx_px_s: float = 0.0
    vy_px_s: float = 0.0
    vz_m_s: float = 0.0
    pred_x_px: float = 0.0
    pred_y_px: float = 0.0
    pred_z_m: float = 0.0
    horizon_s: float = 0.0
    servo_x_deg: float = SERVO_CENTER_X
    servo_y_deg: float = SERVO_CENTER_Y
    target_area_px: int = 0
    confidence: float = 0.0
    control_mode: str = "AUTO"
    manual_x: int = 0
    manual_y: int = 0
    error: str = ""


class SharedState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._config = RuntimeConfig()
        self._image_bgr: Optional[np.ndarray] = None
        self._telemetry = TelemetrySnapshot()

        # Manual input is kept separate from RuntimeConfig. Direction is a
        # level state (-1/0/+1), while nudges and CENTER are edge-triggered.
        # The worker thread is the only code that actually writes to Arduino.
        self._manual_x = 0
        self._manual_y = 0
        self._manual_nudge_x_deg = 0.0
        self._manual_nudge_y_deg = 0.0
        self._manual_center_requested = False

    def get_config(self) -> RuntimeConfig:
        with self._lock:
            return replace(self._config)

    def set_config(self, cfg: RuntimeConfig) -> None:
        with self._lock:
            self._config = replace(cfg)

    def set_frame(self, image_bgr: np.ndarray, telemetry: TelemetrySnapshot) -> None:
        with self._lock:
            self._image_bgr = image_bgr.copy()
            self._telemetry = replace(telemetry)

    def set_telemetry(self, telemetry: TelemetrySnapshot) -> None:
        with self._lock:
            self._telemetry = replace(telemetry)

    def snapshot(self) -> Tuple[Optional[np.ndarray], TelemetrySnapshot]:
        with self._lock:
            image = None if self._image_bgr is None else self._image_bgr.copy()
            return image, replace(self._telemetry)

    def set_manual_direction(self, x: int, y: int) -> None:
        with self._lock:
            self._manual_x = int(clamp(x, -1, 1))
            self._manual_y = int(clamp(y, -1, 1))

    def queue_manual_nudge(self, dx_deg: float, dy_deg: float) -> None:
        with self._lock:
            self._manual_nudge_x_deg += float(dx_deg)
            self._manual_nudge_y_deg += float(dy_deg)

    def request_manual_center(self) -> None:
        with self._lock:
            self._manual_center_requested = True

    def consume_manual_input(self) -> Tuple[int, int, float, float, bool]:
        """Return current held direction and consume one-shot commands."""
        with self._lock:
            result = (
                self._manual_x,
                self._manual_y,
                self._manual_nudge_x_deg,
                self._manual_nudge_y_deg,
                self._manual_center_requested,
            )
            self._manual_nudge_x_deg = 0.0
            self._manual_nudge_y_deg = 0.0
            self._manual_center_requested = False
            return result

    def clear_manual_input(self) -> None:
        with self._lock:
            self._manual_x = 0
            self._manual_y = 0
            self._manual_nudge_x_deg = 0.0
            self._manual_nudge_y_deg = 0.0
            self._manual_center_requested = False


# -----------------------------------------------------------------------------
# Alpha-beta filter (image X, image Y, depth Z)
# -----------------------------------------------------------------------------
class AlphaBetaFilter3D:
    def __init__(self) -> None:
        self.initialized = False
        self.x = self.y = self.z = 0.0
        self.vx = self.vy = self.vz = 0.0
        self.last_t: Optional[float] = None

    def reset(self) -> None:
        self.__init__()

    def initialize(self, x: float, y: float, z_m: float, now: float) -> None:
        self.x, self.y, self.z = x, y, z_m
        self.vx = self.vy = self.vz = 0.0
        self.last_t = now
        self.initialized = True

    def update(
        self,
        mx: float,
        my: float,
        mz_m: float,
        now: float,
        alpha: float,
        beta: float,
        max_pixel_speed: float,
        max_depth_speed: float,
    ) -> Tuple[float, float, float, float, float, float, float]:
        if not self.initialized or self.last_t is None:
            self.initialize(mx, my, mz_m, now)
            return self.x, self.y, self.z, self.vx, self.vy, self.vz, 0.0

        raw_dt = now - self.last_t
        self.last_t = now

        # Very large dt means the telemetry stream was effectively interrupted.
        # Keep position but discard stale velocity to prevent a prediction jump.
        if raw_dt > 0.5 or raw_dt <= 0.0:
            self.vx = self.vy = self.vz = 0.0
            self.x, self.y, self.z = mx, my, mz_m
            return self.x, self.y, self.z, self.vx, self.vy, self.vz, raw_dt

        dt = clamp(raw_dt, 1.0 / 200.0, 0.20)

        px = self.x + self.vx * dt
        py = self.y + self.vy * dt
        pz = self.z + self.vz * dt

        rx = mx - px
        ry = my - py
        rz = mz_m - pz

        self.x = px + alpha * rx
        self.y = py + alpha * ry
        self.z = pz + alpha * rz

        self.vx += beta * rx / dt
        self.vy += beta * ry / dt
        self.vz += beta * rz / dt

        self.vx = clamp(self.vx, -max_pixel_speed, max_pixel_speed)
        self.vy = clamp(self.vy, -max_pixel_speed, max_pixel_speed)
        self.vz = clamp(self.vz, -max_depth_speed, max_depth_speed)

        return self.x, self.y, self.z, self.vx, self.vy, self.vz, raw_dt

    def predict(self, horizon_s: float) -> Tuple[float, float, float]:
        return (
            self.x + self.vx * horizon_s,
            self.y + self.vy * horizon_s,
            max(0.0, self.z + self.vz * horizon_s),
        )


# -----------------------------------------------------------------------------
# Stable target selector
# -----------------------------------------------------------------------------
class StableTargetSelector:
    """Acquire the nearest object only after it persists for stable_frames.

    Once locked, keep the same connected component using image-space and depth
    gates. This deliberately avoids target hopping caused by a one-frame closer
    blob/noise return.
    """

    def __init__(self) -> None:
        self.locked = False
        self.stable_count = 0
        self.lost_count = 0
        self.tentative: Optional[Candidate] = None
        self.last_target: Optional[Candidate] = None
        self.last_seen_t: Optional[float] = None

    def reset(self) -> None:
        self.__init__()

    @staticmethod
    def _match(a: Candidate, b: Candidate, cfg: RuntimeConfig) -> bool:
        dxy = math.hypot(a.cx - b.cx, a.cy - b.cy)
        dz = abs(a.depth_mm - b.depth_mm)
        return dxy <= cfg.reacquire_radius_px and dz <= cfg.max_depth_jump_mm

    def update(
        self,
        candidates: List[Candidate],
        cfg: RuntimeConfig,
        now: float,
    ) -> Optional[Candidate]:
        if not candidates:
            if self.locked:
                self.lost_count += 1
                if self.lost_count > cfg.lost_tolerance_frames:
                    self.locked = False
                    self.stable_count = 0
                    self.last_target = None
            else:
                self.tentative = None
                self.stable_count = 0
            return None

        # Locked mode: first try to continue the same physical blob.
        if self.locked and self.last_target is not None:
            matches = [c for c in candidates if self._match(c, self.last_target, cfg)]
            if matches:
                target = min(
                    matches,
                    key=lambda c: (
                        math.hypot(c.cx - self.last_target.cx, c.cy - self.last_target.cy)
                        + 0.03 * abs(c.depth_mm - self.last_target.depth_mm)
                    ),
                )
                self.last_target = target
                self.last_seen_t = now
                self.lost_count = 0
                return target

            self.lost_count += 1
            if self.lost_count <= cfg.lost_tolerance_frames:
                return None

            # Drop lock and begin a fresh nearest-target acquisition.
            self.locked = False
            self.last_target = None
            self.stable_count = 0
            self.tentative = None

        # Acquisition mode: nearest object wins, but only after persistence.
        nearest = min(candidates, key=lambda c: c.depth_mm)
        if self.tentative is not None and self._match(nearest, self.tentative, cfg):
            self.stable_count += 1
        else:
            self.tentative = nearest
            self.stable_count = 1

        self.tentative = nearest

        if self.stable_count >= cfg.stable_frames:
            self.locked = True
            self.last_target = nearest
            self.last_seen_t = now
            self.lost_count = 0
            return nearest

        return None


# -----------------------------------------------------------------------------
# Arduino / servo controller
# -----------------------------------------------------------------------------
class ServoController:
    def __init__(self, port: str) -> None:
        self.port = port
        self.board = None
        self.servo_x = None
        self.servo_y = None
        self.x_angle = SERVO_CENTER_X
        self.y_angle = SERVO_CENTER_Y
        self.last_cmd_t = time.monotonic()

    def connect(self) -> None:
        self.board = pyfirmata2.Arduino(self.port)
        self.servo_x = self.board.get_pin(f"d:{SERVO_X_PIN}:s")
        self.servo_y = self.board.get_pin(f"d:{SERVO_Y_PIN}:s")
        self.write_angles(SERVO_CENTER_X, SERVO_CENTER_Y, force=True)
        time.sleep(0.4)

    @property
    def connected(self) -> bool:
        return self.board is not None and self.servo_x is not None and self.servo_y is not None

    def write_angles(self, x_deg: float, y_deg: float, force: bool = False) -> None:
        if not self.connected:
            return
        if force or abs(x_deg - self.x_angle) >= 0.05:
            self.servo_x.write(float(x_deg))
            self.x_angle = float(x_deg)
        if force or abs(y_deg - self.y_angle) >= 0.05:
            self.servo_y.write(float(y_deg))
            self.y_angle = float(y_deg)

    def center(self, cfg: RuntimeConfig) -> None:
        x = clamp(SERVO_CENTER_X, cfg.servo_x_min, cfg.servo_x_max)
        y = clamp(SERVO_CENTER_Y, cfg.servo_y_min, cfg.servo_y_max)
        self.write_angles(x, y, force=True)

    def update_tracking(
        self,
        target_x: float,
        target_y: float,
        width: int,
        height: int,
        cfg: RuntimeConfig,
        now: float,
    ) -> None:
        if not self.connected or not cfg.servo_enabled:
            return

        min_period = 1.0 / max(1.0, cfg.servo_update_hz)
        dt = now - self.last_cmd_t
        if dt < min_period:
            return
        self.last_cmd_t = now
        dt = clamp(dt, min_period, 0.20)

        cx = width * 0.5
        cy = height * 0.5
        ex_px = target_x - cx
        ey_px = target_y - cy

        if abs(ex_px) <= cfg.deadband_px:
            ex_px = 0.0
        if abs(ey_px) <= cfg.deadband_px:
            ey_px = 0.0

        ex_norm = clamp(ex_px / max(1.0, cx), -1.0, 1.0)
        ey_norm = clamp(ey_px / max(1.0, cy), -1.0, 1.0)

        if cfg.invert_x:
            ex_norm *= -1.0
        if cfg.invert_y:
            ey_norm *= -1.0

        x_rate = clamp(
            cfg.servo_gain_x_deg_s * ex_norm,
            -cfg.servo_max_rate_deg_s,
            cfg.servo_max_rate_deg_s,
        )
        y_rate = clamp(
            cfg.servo_gain_y_deg_s * ey_norm,
            -cfg.servo_max_rate_deg_s,
            cfg.servo_max_rate_deg_s,
        )

        new_x = clamp(self.x_angle + x_rate * dt, cfg.servo_x_min, cfg.servo_x_max)
        new_y = clamp(self.y_angle + y_rate * dt, cfg.servo_y_min, cfg.servo_y_max)
        self.write_angles(new_x, new_y)

    def update_manual(
        self,
        held_x: int,
        held_y: int,
        nudge_x_deg: float,
        nudge_y_deg: float,
        center_requested: bool,
        cfg: RuntimeConfig,
        now: float,
    ) -> None:
        """Apply MANUAL commands while preserving limits and slew behavior.

        held_x / held_y are semantic directions:
          X: -1 = left,  +1 = right
          Y: -1 = up,    +1 = down

        Nudge values use the same semantic sign and guarantee that a very short
        click still changes the position even if it falls between worker ticks.
        """
        if not self.connected or not cfg.servo_enabled:
            return

        if center_requested:
            self.center(cfg)
            self.last_cmd_t = now
            return

        # Physical installation may reverse one or both servo directions.
        x_sign = -1.0 if cfg.invert_x else 1.0
        y_sign = -1.0 if cfg.invert_y else 1.0

        # Edge-triggered nudge: useful for precise single clicks / key taps.
        if abs(nudge_x_deg) > 1e-9 or abs(nudge_y_deg) > 1e-9:
            new_x = clamp(
                self.x_angle + x_sign * nudge_x_deg,
                cfg.servo_x_min,
                cfg.servo_x_max,
            )
            new_y = clamp(
                self.y_angle + y_sign * nudge_y_deg,
                cfg.servo_y_min,
                cfg.servo_y_max,
            )
            self.write_angles(new_x, new_y)
            self.last_cmd_t = now
            return

        # Level-triggered command: smooth movement while a key/button is held.
        if held_x == 0 and held_y == 0:
            return

        min_period = 1.0 / max(1.0, cfg.servo_update_hz)
        dt = now - self.last_cmd_t
        if dt < min_period:
            return
        self.last_cmd_t = now
        dt = clamp(dt, min_period, 0.20)

        max_rate = min(cfg.manual_rate_deg_s, cfg.servo_max_rate_deg_s)
        x_rate = x_sign * float(held_x) * max_rate
        y_rate = y_sign * float(held_y) * max_rate

        new_x = clamp(self.x_angle + x_rate * dt, cfg.servo_x_min, cfg.servo_x_max)
        new_y = clamp(self.y_angle + y_rate * dt, cfg.servo_y_min, cfg.servo_y_max)
        self.write_angles(new_x, new_y)

    def close(self) -> None:
        if self.board is not None:
            try:
                self.board.exit()
            except Exception:
                pass
        self.board = None
        self.servo_x = None
        self.servo_y = None


# -----------------------------------------------------------------------------
# Vision / tracking utilities
# -----------------------------------------------------------------------------
def build_candidates(
    depth_mm: np.ndarray,
    confidence: Optional[np.ndarray],
    cfg: RuntimeConfig,
) -> Tuple[List[Candidate], np.ndarray]:
    depth = np.asarray(depth_mm, dtype=np.float32)
    valid = np.isfinite(depth)
    valid &= depth >= cfg.min_distance_mm
    valid &= depth <= cfg.max_distance_mm

    conf_arr: Optional[np.ndarray]
    if confidence is not None:
        conf_arr = np.asarray(confidence, dtype=np.float32)
        if conf_arr.shape == depth.shape:
            valid &= np.isfinite(conf_arr)
            valid &= conf_arr >= cfg.confidence_threshold
        else:
            conf_arr = None
    else:
        conf_arr = None

    mask = (valid.astype(np.uint8) * 255)

    k = int(cfg.morphology_kernel)
    if k >= 3:
        if k % 2 == 0:
            k += 1
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: List[Candidate] = []

    for label_id in range(1, n):
        area = int(stats[label_id, cv2.CC_STAT_AREA])
        if area < cfg.min_object_area_px:
            continue

        x = int(stats[label_id, cv2.CC_STAT_LEFT])
        y = int(stats[label_id, cv2.CC_STAT_TOP])
        w = int(stats[label_id, cv2.CC_STAT_WIDTH])
        h = int(stats[label_id, cv2.CC_STAT_HEIGHT])
        region = labels == label_id
        values = depth[region]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue

        depth_med = float(np.median(values))
        cx, cy = map(float, centroids[label_id])

        mean_conf = 0.0
        if conf_arr is not None:
            cvals = conf_arr[region]
            cvals = cvals[np.isfinite(cvals)]
            if cvals.size:
                mean_conf = float(np.mean(cvals))

        candidates.append(
            Candidate(
                cx=cx,
                cy=cy,
                depth_mm=depth_med,
                area=area,
                bbox=(x, y, w, h),
                mean_confidence=mean_conf,
            )
        )

    return candidates, mask


def render_depth(
    depth_mm: np.ndarray,
    confidence: Optional[np.ndarray],
    cfg: RuntimeConfig,
    candidates: List[Candidate],
    target: Optional[Candidate],
    filt: AlphaBetaFilter3D,
    pred: Optional[Tuple[float, float, float]],
    telemetry: TelemetrySnapshot,
) -> np.ndarray:
    depth = np.asarray(depth_mm, dtype=np.float32)
    finite = np.isfinite(depth)
    valid = finite & (depth >= cfg.min_distance_mm) & (depth <= cfg.max_distance_mm)

    if confidence is not None:
        conf_arr = np.asarray(confidence, dtype=np.float32)
        if conf_arr.shape == depth.shape:
            valid &= np.isfinite(conf_arr) & (conf_arr >= cfg.confidence_threshold)

    span = max(1.0, cfg.max_distance_mm - cfg.min_distance_mm)
    normalized = (depth - cfg.min_distance_mm) / span
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0)
    normalized = np.clip(normalized, 0.0, 1.0)
    # Nearer = warmer/brighter side of the selected colormap.
    gray = ((1.0 - normalized) * 255.0).astype(np.uint8)
    gray[~valid] = 0
    image = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    image[~valid] = (0, 0, 0)

    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    cv2.drawMarker(image, center, (255, 255, 255), cv2.MARKER_CROSS, 12, 1)

    for c in candidates:
        x, y, bw, bh = c.bbox
        cv2.rectangle(image, (x, y), (x + bw, y + bh), (120, 120, 120), 1)

    if target is not None and filt.initialized:
        x, y, bw, bh = target.bbox
        cv2.rectangle(image, (x, y), (x + bw, y + bh), (255, 255, 255), 2)
        current = (int(round(filt.x)), int(round(filt.y)))
        cv2.circle(image, current, 4, (255, 255, 255), -1)

        if pred is not None:
            px = int(round(clamp(pred[0], 0, w - 1)))
            py = int(round(clamp(pred[1], 0, h - 1)))
            cv2.arrowedLine(image, current, (px, py), (255, 255, 255), 1, tipLength=0.20)
            cv2.circle(image, (px, py), 5, (0, 255, 255), 1)

    # Minimal in-frame telemetry. Detailed values are also shown in the GUI.
    approach = "APPROACH" if telemetry.vz_m_s < -0.03 else ("RECEDE" if telemetry.vz_m_s > 0.03 else "STEADY")
    lines = [
        f"{telemetry.state}  {telemetry.control_mode}  FPS {telemetry.fps:.1f}",
        f"Z {telemetry.z_m:.2f} m  Vz {telemetry.vz_m_s:+.2f} m/s  {approach}",
        f"Servo X/Y {telemetry.servo_x_deg:.1f}/{telemetry.servo_y_deg:.1f}",
    ]
    y0 = 14
    for i, text in enumerate(lines):
        cv2.putText(image, text, (5, y0 + i * 14), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

    return image


# -----------------------------------------------------------------------------
# Worker thread
# -----------------------------------------------------------------------------
class TrackerWorker(threading.Thread):
    def __init__(self, shared: SharedState) -> None:
        super().__init__(daemon=True)
        self.shared = shared
        self.stop_event = threading.Event()
        self.camera = None
        self.servo: Optional[ServoController] = None
        self.selector = StableTargetSelector()
        self.filter = AlphaBetaFilter3D()
        self.seq = 0
        self.fps_ema = 0.0
        self.prev_frame_t: Optional[float] = None
        self.last_valid_target_t: Optional[float] = None

    def stop(self) -> None:
        self.stop_event.set()

    def _set_error(self, message: str) -> None:
        _, old = self.shared.snapshot()
        old.state = "ERROR"
        old.error = message
        old.monotonic_ns = time.monotonic_ns()
        self.shared.set_telemetry(old)

    def _open_camera(self) -> None:
        self.camera = ac.ArducamCamera()
        ret = self.camera.open(ac.Connection.CSI, 0)
        if ret != 0:
            raise RuntimeError(f"Arducam open(CSI, 0) failed, code={ret}")

        ret = self.camera.start(ac.FrameType.DEPTH)
        if ret != 0:
            try:
                self.camera.close()
            finally:
                self.camera = None
            raise RuntimeError(f"Arducam start(DEPTH) failed, code={ret}")

        self.camera.setControl(ac.Control.RANGE, SENSOR_RANGE_MM)

    def _close_camera(self) -> None:
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
            try:
                self.camera.close()
            except Exception:
                pass
            self.camera = None

    def _service_manual_servo(self, cfg: RuntimeConfig, now: float) -> Tuple[int, int]:
        if cfg.control_mode != "MANUAL":
            self.shared.clear_manual_input()
            return 0, 0

        held_x, held_y, nudge_x, nudge_y, center = self.shared.consume_manual_input()
        if self.servo is not None:
            self.servo.update_manual(
                held_x, held_y, nudge_x, nudge_y, center, cfg, now
            )
        return held_x, held_y

    def run(self) -> None:
        telemetry = TelemetrySnapshot(state="STARTING", monotonic_ns=time.monotonic_ns())
        self.shared.set_telemetry(telemetry)

        try:
            cfg = self.shared.get_config()
            telemetry.control_mode = cfg.control_mode

            # Arduino failure is non-fatal: vision can still be debugged.
            try:
                self.servo = ServoController(cfg.arduino_port)
                self.servo.connect()
            except Exception as exc:
                self.servo = None
                telemetry.error = f"Arduino unavailable: {exc}"

            self._open_camera()
            telemetry.state = "SEARCH"
            self.shared.set_telemetry(telemetry)

            while not self.stop_event.is_set():
                cfg = self.shared.get_config()

                # Manual control is serviced independently of successful camera
                # frames. A shorter request timeout keeps MANUAL responsive even
                # if the ToF stream temporarily stalls.
                input_t = time.monotonic()
                manual_x, manual_y = self._service_manual_servo(cfg, input_t)

                frame = self.camera.requestFrame(100)
                now = time.monotonic()
                stamp_ns = time.monotonic_ns()

                if frame is None:
                    telemetry.state = "NO_FRAME"
                    telemetry.control_mode = cfg.control_mode
                    telemetry.manual_x = manual_x
                    telemetry.manual_y = manual_y
                    telemetry.monotonic_ns = stamp_ns
                    if self.servo is not None:
                        telemetry.servo_x_deg = self.servo.x_angle
                        telemetry.servo_y_deg = self.servo.y_angle
                    self.shared.set_telemetry(telemetry)
                    continue

                try:
                    if not isinstance(frame, ac.DepthData):
                        continue

                    depth_buf = np.array(frame.depth_data, dtype=np.float32, copy=True)
                    confidence_src = getattr(frame, "confidence_data", None)
                    confidence_buf = None
                    if confidence_src is not None:
                        confidence_buf = np.array(confidence_src, dtype=np.float32, copy=True)
                finally:
                    # Release SDK-owned frame memory as early as possible.
                    self.camera.releaseFrame(frame)

                self.seq += 1
                if self.prev_frame_t is not None:
                    dt_frame = now - self.prev_frame_t
                    if dt_frame > 0:
                        inst_fps = 1.0 / dt_frame
                        self.fps_ema = inst_fps if self.fps_ema <= 0 else (0.90 * self.fps_ema + 0.10 * inst_fps)
                self.prev_frame_t = now

                candidates, _ = build_candidates(depth_buf, confidence_buf, cfg)
                target = self.selector.update(candidates, cfg, now)

                telemetry = TelemetrySnapshot(
                    seq=self.seq,
                    monotonic_ns=stamp_ns,
                    fps=self.fps_ema,
                    frame_age_ms=0.0,
                    state="ACQUIRE" if not self.selector.locked else "TRACK",
                    target_valid=False,
                    stable_count=self.selector.stable_count,
                    lost_count=self.selector.lost_count,
                    servo_x_deg=self.servo.x_angle if self.servo else SERVO_CENTER_X,
                    servo_y_deg=self.servo.y_angle if self.servo else SERVO_CENTER_Y,
                    control_mode=cfg.control_mode,
                    manual_x=manual_x,
                    manual_y=manual_y,
                    error=telemetry.error,
                )

                prediction: Optional[Tuple[float, float, float]] = None

                if target is not None:
                    self.last_valid_target_t = now
                    z_m = target.depth_mm / 1000.0
                    x, y, z, vx, vy, vz, _ = self.filter.update(
                        target.cx,
                        target.cy,
                        z_m,
                        now,
                        cfg.filter_alpha,
                        cfg.filter_beta,
                        cfg.max_pixel_speed_px_s,
                        cfg.max_depth_speed_m_s,
                    )

                    # "N frames into the future" is converted to time using EMA FPS.
                    effective_fps = self.fps_ema if self.fps_ema >= 1.0 else 30.0
                    horizon_s = cfg.prediction_frames / effective_fps
                    prediction = self.filter.predict(horizon_s)

                    telemetry.target_valid = True
                    telemetry.x_px = x
                    telemetry.y_px = y
                    telemetry.z_m = z
                    telemetry.vx_px_s = vx
                    telemetry.vy_px_s = vy
                    telemetry.vz_m_s = vz
                    telemetry.pred_x_px = prediction[0]
                    telemetry.pred_y_px = prediction[1]
                    telemetry.pred_z_m = prediction[2]
                    telemetry.horizon_s = horizon_s
                    telemetry.target_area_px = target.area
                    telemetry.confidence = target.mean_confidence

                    if self.servo is not None and cfg.control_mode == "AUTO":
                        # Point at predicted screen position to compensate for lag.
                        self.servo.update_tracking(
                            prediction[0],
                            prediction[1],
                            depth_buf.shape[1],
                            depth_buf.shape[0],
                            cfg,
                            now,
                        )
                        telemetry.servo_x_deg = self.servo.x_angle
                        telemetry.servo_y_deg = self.servo.y_angle
                    elif self.servo is not None:
                        # In MANUAL the vision pipeline continues to track and
                        # predict, but it is read-only with respect to the gimbal.
                        telemetry.servo_x_deg = self.servo.x_angle
                        telemetry.servo_y_deg = self.servo.y_angle
                else:
                    # Avoid retaining stale velocity after lock is truly lost.
                    if not self.selector.locked:
                        self.filter.reset()
                        telemetry.state = "SEARCH" if self.selector.stable_count == 0 else "ACQUIRE"

                    if (
                        cfg.control_mode == "AUTO"
                        and cfg.return_center_on_loss
                        and self.servo is not None
                        and self.last_valid_target_t is not None
                        and now - self.last_valid_target_t >= cfg.return_center_delay_s
                    ):
                        self.servo.center(cfg)
                        telemetry.servo_x_deg = self.servo.x_angle
                        telemetry.servo_y_deg = self.servo.y_angle

                image = render_depth(
                    depth_buf,
                    confidence_buf,
                    cfg,
                    candidates,
                    target,
                    self.filter,
                    prediction,
                    telemetry,
                )
                self.shared.set_frame(image, telemetry)

        except Exception as exc:
            self._set_error(str(exc))
        finally:
            self._close_camera()
            if self.servo is not None:
                try:
                    # Defined neutral/park position for an orderly shutdown.
                    self.servo.center(self.shared.get_config())
                    time.sleep(0.15)
                except Exception:
                    pass
                self.servo.close()
            _, final_t = self.shared.snapshot()
            if final_t.state != "ERROR":
                final_t.state = "STOPPED"
                final_t.monotonic_ns = time.monotonic_ns()
                self.shared.set_telemetry(final_t)


# -----------------------------------------------------------------------------
# Tkinter GUI
# -----------------------------------------------------------------------------
class TrackerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Arducam ToF + Arduino 2-axis tracker")
        self.root.minsize(1120, 720)

        style = ttk.Style(self.root)
        # 'clam' is generally more predictable on Raspberry Pi/X11 than some
        # platform themes and avoids several redraw/visibility glitches.
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        self.shared = SharedState()
        self.worker: Optional[TrackerWorker] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None

        self.vars = {}
        self.status_var = tk.StringVar(value="STOPPED")
        self.telemetry_var = tk.StringVar(value="Система не запущена")
        self.error_var = tk.StringVar(value="")
        self.control_mode_var = tk.StringVar(value="AUTO")
        self.manual_buttons: List[ttk.Button] = []
        self._pressed_manual_keys = set()

        self._build_ui()
        self._load_config_to_ui(self.shared.get_config())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._bind_manual_keyboard()
        self.root.after(50, self._ui_tick)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=4)
        outer.columnconfigure(1, weight=2)
        outer.rowconfigure(0, weight=1)

        # Video panel ----------------------------------------------------------
        video_box = ttk.LabelFrame(outer, text="ToF / tracking", padding=6)
        video_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        video_box.rowconfigure(0, weight=1)
        video_box.columnconfigure(0, weight=1)

        self.video_label = ttk.Label(video_box, anchor="center")
        self.video_label.grid(row=0, column=0, sticky="nsew")

        ttk.Label(video_box, textvariable=self.telemetry_var, justify="left").grid(
            row=1, column=0, sticky="ew", pady=(6, 0)
        )

        # Control panel --------------------------------------------------------
        right = ttk.Frame(outer)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        action = ttk.LabelFrame(right, text="Управление", padding=8)
        action.grid(row=0, column=0, sticky="ew")
        action.columnconfigure((0, 1, 2), weight=1)

        self.start_button = ttk.Button(action, text="Старт", command=self.start_system)
        self.start_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
        self.stop_button = ttk.Button(action, text="Стоп", command=self.stop_system)
        self.stop_button.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(action, text="Применить", command=self.apply_settings).grid(
            row=0, column=2, sticky="ew", padx=(4, 0)
        )
        ttk.Label(action, text="Состояние:").grid(row=1, column=0, sticky="w", pady=(7, 0))
        ttk.Label(action, textvariable=self.status_var).grid(row=1, column=1, columnspan=2, sticky="w", pady=(7, 0))

        ttk.Separator(action, orient="horizontal").grid(
            row=2, column=0, columnspan=3, sticky="ew", pady=7
        )
        ttk.Label(action, text="Режим:").grid(row=3, column=0, sticky="w")
        ttk.Radiobutton(
            action, text="AUTO", value="AUTO", variable=self.control_mode_var,
            command=self._on_mode_change
        ).grid(row=3, column=1, sticky="w")
        ttk.Radiobutton(
            action, text="MANUAL", value="MANUAL", variable=self.control_mode_var,
            command=self._on_mode_change
        ).grid(row=3, column=2, sticky="w")

        up = ttk.Button(action, text="▲ Вверх")
        left = ttk.Button(action, text="◀ Влево")
        center = ttk.Button(action, text="Центр 90/90", command=self.manual_center)
        right_btn = ttk.Button(action, text="Вправо ▶")
        down = ttk.Button(action, text="▼ Вниз")

        up.grid(row=4, column=1, sticky="ew", padx=3, pady=(7, 2))
        left.grid(row=5, column=0, sticky="ew", padx=(0, 3), pady=2)
        center.grid(row=5, column=1, sticky="ew", padx=3, pady=2)
        right_btn.grid(row=5, column=2, sticky="ew", padx=(3, 0), pady=2)
        down.grid(row=6, column=1, sticky="ew", padx=3, pady=2)

        self._bind_manual_button(up, 0, -1)
        self._bind_manual_button(down, 0, +1)
        self._bind_manual_button(left, -1, 0)
        self._bind_manual_button(right_btn, +1, 0)
        self.manual_buttons = [up, down, left, right_btn, center]

        ttk.Label(
            action,
            text="MANUAL: кнопки или стрелки клавиатуры; удержание = плавное движение",
            wraplength=340,
            justify="left",
        ).grid(row=7, column=0, columnspan=3, sticky="w", pady=(5, 0))

        settings_tabs = ttk.Notebook(right)
        settings_tabs.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        depth = ttk.Frame(settings_tabs, padding=8)
        pred = ttk.Frame(settings_tabs, padding=8)
        servo = ttk.Frame(settings_tabs, padding=8)
        settings_tabs.add(depth, text="ToF / объект")
        settings_tabs.add(pred, text="Фильтр / прогноз")
        settings_tabs.add(servo, text="Сервоприводы")
        self._entry(depth, 0, "Мин. дальность, мм", "min_distance_mm")
        self._entry(depth, 1, "Макс. дальность, мм", "max_distance_mm")
        self._entry(depth, 2, "Порог confidence", "confidence_threshold")
        self._entry(depth, 3, "Мин. площадь, px", "min_object_area_px")
        self._entry(depth, 4, "Morph kernel (odd)", "morphology_kernel")
        self._entry(depth, 5, "Кадров до LOCK", "stable_frames")
        self._entry(depth, 6, "Допуск потерь, кадров", "lost_tolerance_frames")
        self._entry(depth, 7, "Радиус reacquire, px", "reacquire_radius_px")
        self._entry(depth, 8, "Макс. скачок Z, мм", "max_depth_jump_mm")

        self._entry(pred, 0, "Alpha", "filter_alpha")
        self._entry(pred, 1, "Beta", "filter_beta")
        self._entry(pred, 2, "Прогноз, кадров", "prediction_frames")
        self._entry(pred, 3, "Max XY speed, px/s", "max_pixel_speed_px_s")
        self._entry(pred, 4, "Max Z speed, m/s", "max_depth_speed_m_s")

        self._entry(servo, 0, "Arduino port", "arduino_port")
        self._entry(servo, 1, "X min", "servo_x_min")
        self._entry(servo, 2, "X max", "servo_x_max")
        self._entry(servo, 3, "Y min", "servo_y_min")
        self._entry(servo, 4, "Y max", "servo_y_max")
        self._entry(servo, 5, "Gain X, deg/s", "servo_gain_x_deg_s")
        self._entry(servo, 6, "Gain Y, deg/s", "servo_gain_y_deg_s")
        self._entry(servo, 7, "Max slew, deg/s", "servo_max_rate_deg_s")
        self._entry(servo, 8, "Servo update, Hz", "servo_update_hz")
        self._entry(servo, 9, "Deadband, px", "deadband_px")
        self._entry(servo, 10, "Center delay, s", "return_center_delay_s")
        self._entry(servo, 11, "Manual step, deg", "manual_step_deg")
        self._entry(servo, 12, "Manual speed, deg/s", "manual_rate_deg_s")

        self._check(servo, 13, "Servo control", "servo_enabled")
        self._check(servo, 14, "Invert X", "invert_x")
        self._check(servo, 15, "Invert Y", "invert_y")
        self._check(servo, 16, "Центрировать при потере", "return_center_on_loss")

        error = ttk.LabelFrame(right, text="Диагностика", padding=8)
        error.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(error, textvariable=self.error_var, wraplength=350, justify="left").pack(fill="x")

    def _entry(self, parent, row: int, label: str, key: str) -> None:
        parent.columnconfigure(1, weight=1)
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2, padx=(0, 8))
        var = tk.StringVar()
        self.vars[key] = var
        ttk.Entry(parent, textvariable=var, width=16).grid(row=row, column=1, sticky="ew", pady=2)

    def _check(self, parent, row: int, label: str, key: str) -> None:
        var = tk.BooleanVar()
        self.vars[key] = var
        ttk.Checkbutton(parent, text=label, variable=var).grid(row=row, column=0, columnspan=2, sticky="w", pady=2)

    def _load_config_to_ui(self, cfg: RuntimeConfig) -> None:
        for key, var in self.vars.items():
            value = getattr(cfg, key)
            var.set(value)
        self.control_mode_var.set(cfg.control_mode)
        self._update_manual_controls_state()

    def _read_config_from_ui(self) -> RuntimeConfig:
        def f(key: str) -> float:
            return float(self.vars[key].get())

        def i(key: str) -> int:
            return int(round(float(self.vars[key].get())))

        cfg = RuntimeConfig(
            min_distance_mm=f("min_distance_mm"),
            max_distance_mm=f("max_distance_mm"),
            confidence_threshold=f("confidence_threshold"),
            min_object_area_px=i("min_object_area_px"),
            morphology_kernel=i("morphology_kernel"),
            stable_frames=i("stable_frames"),
            lost_tolerance_frames=i("lost_tolerance_frames"),
            reacquire_radius_px=f("reacquire_radius_px"),
            max_depth_jump_mm=f("max_depth_jump_mm"),
            filter_alpha=f("filter_alpha"),
            filter_beta=f("filter_beta"),
            prediction_frames=i("prediction_frames"),
            max_pixel_speed_px_s=f("max_pixel_speed_px_s"),
            max_depth_speed_m_s=f("max_depth_speed_m_s"),
            # Port is consumed when the worker starts; restart after changing it.
            arduino_port=str(self.vars["arduino_port"].get()).strip(),
            servo_enabled=bool(self.vars["servo_enabled"].get()),
            servo_x_min=f("servo_x_min"),
            servo_x_max=f("servo_x_max"),
            servo_y_min=f("servo_y_min"),
            servo_y_max=f("servo_y_max"),
            servo_gain_x_deg_s=f("servo_gain_x_deg_s"),
            servo_gain_y_deg_s=f("servo_gain_y_deg_s"),
            servo_max_rate_deg_s=f("servo_max_rate_deg_s"),
            servo_update_hz=f("servo_update_hz"),
            deadband_px=f("deadband_px"),
            invert_x=bool(self.vars["invert_x"].get()),
            invert_y=bool(self.vars["invert_y"].get()),
            return_center_on_loss=bool(self.vars["return_center_on_loss"].get()),
            return_center_delay_s=f("return_center_delay_s"),
            control_mode=str(self.control_mode_var.get()).strip().upper(),
            manual_step_deg=f("manual_step_deg"),
            manual_rate_deg_s=f("manual_rate_deg_s"),
        )
        self._validate_config(cfg)
        return cfg

    @staticmethod
    def _validate_config(cfg: RuntimeConfig) -> None:
        if not (0 <= cfg.min_distance_mm < cfg.max_distance_mm <= SENSOR_RANGE_MM):
            raise ValueError(f"Дальность: 0 <= min < max <= {SENSOR_RANGE_MM} мм")
        if not (0 <= cfg.confidence_threshold <= 255):
            raise ValueError("confidence должен быть в диапазоне 0..255")
        if cfg.min_object_area_px < 1:
            raise ValueError("Минимальная площадь должна быть >= 1")
        if cfg.morphology_kernel < 1 or cfg.morphology_kernel > 15:
            raise ValueError("Morph kernel должен быть 1..15")
        if cfg.stable_frames < 1 or cfg.lost_tolerance_frames < 0:
            raise ValueError("Некорректные параметры устойчивости")
        if cfg.reacquire_radius_px <= 0 or cfg.max_depth_jump_mm <= 0:
            raise ValueError("Reacquire/depth gate должны быть > 0")
        if not (0.0 < cfg.filter_alpha <= 1.0):
            raise ValueError("Alpha должен быть (0, 1]")
        if not (0.0 <= cfg.filter_beta <= 1.0):
            raise ValueError("Beta должен быть [0, 1]")
        if cfg.prediction_frames < 0:
            raise ValueError("Prediction frames должен быть >= 0")
        if not (0 <= cfg.servo_x_min <= SERVO_CENTER_X <= cfg.servo_x_max <= 180):
            raise ValueError("X limits должны содержать 90° и лежать в 0..180")
        if not (0 <= cfg.servo_y_min <= SERVO_CENTER_Y <= cfg.servo_y_max <= 180):
            raise ValueError("Y limits должны содержать 90° и лежать в 0..180")
        if cfg.servo_max_rate_deg_s <= 0 or cfg.servo_update_hz <= 0:
            raise ValueError("Servo rate/update должны быть > 0")
        if cfg.control_mode not in ("AUTO", "MANUAL"):
            raise ValueError("Режим управления должен быть AUTO или MANUAL")
        if not (0.05 <= cfg.manual_step_deg <= 30.0):
            raise ValueError("Manual step должен быть в диапазоне 0.05..30 градусов")
        if not (0.1 <= cfg.manual_rate_deg_s <= 180.0):
            raise ValueError("Manual speed должен быть в диапазоне 0.1..180 deg/s")
        if not cfg.arduino_port:
            raise ValueError("Укажите Arduino port")

    def _update_manual_controls_state(self) -> None:
        running = self.worker is not None and self.worker.is_alive()
        state = "normal" if self.control_mode_var.get() == "MANUAL" and running else "disabled"
        for button in self.manual_buttons:
            try:
                button.configure(state=state)
            except tk.TclError:
                pass

    def _on_mode_change(self) -> None:
        mode = self.control_mode_var.get().strip().upper()
        if mode not in ("AUTO", "MANUAL"):
            return
        cfg = self.shared.get_config()
        cfg.control_mode = mode
        self.shared.set_config(cfg)
        self.shared.clear_manual_input()
        self._pressed_manual_keys.clear()
        self._update_manual_controls_state()
        self.error_var.set(
            "MANUAL: автотрекинг продолжает считать цель, но не двигает подвес"
            if mode == "MANUAL"
            else "AUTO: управление сервоприводами передано автотрекеру"
        )

    def _manual_available(self) -> bool:
        return (
            self.control_mode_var.get() == "MANUAL"
            and self.worker is not None
            and self.worker.is_alive()
        )

    def _manual_press(self, x: int, y: int) -> None:
        if not self._manual_available():
            return
        cfg = self.shared.get_config()
        # A one-shot nudge guarantees that a short click/tap is not lost.
        self.shared.queue_manual_nudge(x * cfg.manual_step_deg, y * cfg.manual_step_deg)
        self.shared.set_manual_direction(x, y)

    def _manual_release(self) -> None:
        self.shared.set_manual_direction(0, 0)

    def _bind_manual_button(self, button: ttk.Button, x: int, y: int) -> None:
        button.bind("<ButtonPress-1>", lambda _e, dx=x, dy=y: self._manual_press(dx, dy))
        button.bind("<ButtonRelease-1>", lambda _e: self._manual_release())
        button.bind("<Leave>", lambda _e: self._manual_release())

    def manual_center(self) -> None:
        if not self._manual_available():
            return
        self.shared.set_manual_direction(0, 0)
        self.shared.request_manual_center()

    def _bind_manual_keyboard(self) -> None:
        mapping = {
            "Left": (-1, 0),
            "Right": (+1, 0),
            "Up": (0, -1),
            "Down": (0, +1),
        }
        for key, (x, y) in mapping.items():
            self.root.bind_all(
                f"<KeyPress-{key}>",
                lambda event, k=key, dx=x, dy=y: self._manual_key_press(event, k, dx, dy),
                add="+",
            )
            self.root.bind_all(
                f"<KeyRelease-{key}>",
                lambda event, k=key: self._manual_key_release(event, k),
                add="+",
            )
        self.root.bind_all("<KeyPress-Home>", self._manual_home_key, add="+")

    def _keyboard_control_allowed(self) -> bool:
        if not self._manual_available():
            return False
        focus = self.root.focus_get()
        # Do not hijack cursor movement while the user edits an Entry/Text.
        if isinstance(focus, (tk.Entry, ttk.Entry, tk.Text)):
            return False
        return True

    def _sync_manual_keyboard_direction(self) -> None:
        vectors = {
            "Left": (-1, 0),
            "Right": (+1, 0),
            "Up": (0, -1),
            "Down": (0, +1),
        }
        x = sum(vectors[k][0] for k in self._pressed_manual_keys if k in vectors)
        y = sum(vectors[k][1] for k in self._pressed_manual_keys if k in vectors)
        self.shared.set_manual_direction(int(clamp(x, -1, 1)), int(clamp(y, -1, 1)))

    def _manual_key_press(self, event, key: str, x: int, y: int):
        if not self._keyboard_control_allowed():
            return None
        # Ignore OS key-repeat for the nudge. The held direction still remains set.
        if key not in self._pressed_manual_keys:
            self._pressed_manual_keys.add(key)
            cfg = self.shared.get_config()
            self.shared.queue_manual_nudge(x * cfg.manual_step_deg, y * cfg.manual_step_deg)
        self._sync_manual_keyboard_direction()
        return "break"

    def _manual_key_release(self, event, key: str):
        if key in self._pressed_manual_keys:
            self._pressed_manual_keys.discard(key)
            self._sync_manual_keyboard_direction()
            return "break"
        return None

    def _manual_home_key(self, event):
        if not self._keyboard_control_allowed():
            return None
        self.manual_center()
        return "break"

    def apply_settings(self) -> None:
        try:
            cfg = self._read_config_from_ui()
            self.shared.set_config(cfg)
            self.shared.clear_manual_input()
            self._pressed_manual_keys.clear()
            self._update_manual_controls_state()
            self.error_var.set("Настройки применены")
        except Exception as exc:
            self.error_var.set(f"Ошибка настроек: {exc}")
            messagebox.showerror("Настройки", str(exc))

    def start_system(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            cfg = self._read_config_from_ui()
            self.shared.set_config(cfg)
        except Exception as exc:
            messagebox.showerror("Настройки", str(exc))
            return

        self.worker = TrackerWorker(self.shared)
        self.worker.start()
        self.status_var.set("STARTING")
        self.error_var.set("")
        self._update_manual_controls_state()

    def stop_system(self) -> None:
        self.shared.clear_manual_input()
        self._pressed_manual_keys.clear()
        if self.worker is not None and self.worker.is_alive():
            self.worker.stop()
            self.status_var.set("STOPPING")

    def _ui_tick(self) -> None:
        image_bgr, t = self.shared.snapshot()
        self.status_var.set(t.state)
        self.error_var.set(t.error)
        self._update_manual_controls_state()

        approach = "приближается" if t.vz_m_s < -0.03 else ("удаляется" if t.vz_m_s > 0.03 else "стабильно по Z")
        self.telemetry_var.set(
            f"seq={t.seq}  FPS={t.fps:.1f}  state={t.state}  mode={t.control_mode}  valid={int(t.target_valid)}\n"
            f"XYZ(image/depth): x={t.x_px:.1f}px  y={t.y_px:.1f}px  z={t.z_m:.3f}m\n"
            f"V: vx={t.vx_px_s:+.1f}px/s  vy={t.vy_px_s:+.1f}px/s  vz={t.vz_m_s:+.3f}m/s ({approach})\n"
            f"Prediction: +{t.horizon_s:.3f}s -> ({t.pred_x_px:.1f}px, {t.pred_y_px:.1f}px, {t.pred_z_m:.3f}m)\n"
            f"Servo: X={t.servo_x_deg:.1f}°  Y={t.servo_y_deg:.1f}°  "
            f"manual=({t.manual_x:+d},{t.manual_y:+d})  stable={t.stable_count}  lost={t.lost_count}"
        )

        if image_bgr is not None:
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)

            # Fit into a large preview without changing tracking coordinates.
            max_w = max(480, self.video_label.winfo_width() - 8)
            max_h = max(360, self.video_label.winfo_height() - 8)
            if pil.width > 0 and pil.height > 0:
                scale = min(max_w / pil.width, max_h / pil.height)
                scale = max(1.0, scale)
                new_size = (int(pil.width * scale), int(pil.height * scale))
                pil = pil.resize(new_size, Image.Resampling.NEAREST)

            self.tk_image = ImageTk.PhotoImage(pil)
            self.video_label.configure(image=self.tk_image)

        if self.worker is not None and not self.worker.is_alive() and t.state == "STOPPING":
            self.status_var.set("STOPPED")

        self.root.after(50, self._ui_tick)

    def on_close(self) -> None:
        self.shared.clear_manual_input()
        self._pressed_manual_keys.clear()
        if self.worker is not None and self.worker.is_alive():
            self.worker.stop()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
