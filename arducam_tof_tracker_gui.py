#!/usr/bin/env python3
"""
Arducam ToF nearest stable object tracker for Raspberry Pi 5.

Based on the ArducamDepthCamera API style used by the supplied
`time-of-flight.py` example:
  - ArducamCamera()
  - open(Connection.CSI, 0)
  - start(FrameType.DEPTH)
  - requestFrame(...)
  - DepthData.depth_data / confidence_data
  - releaseFrame(frame)
  - setControl(Control.RANGE, ...)

GUI: tkinter + Pillow
Tracking: robust nearest-object segmentation + constant-velocity Kalman filter
Prediction: image position (pixels) + depth (mm) after N future frames
"""

from __future__ import annotations

import math
import queue
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk

import ArducamDepthCamera as ac


CAMERA_RANGE_MM = 4000  # Arducam example says the range control is 2000 or 4000.
FRAME_TIMEOUT_MS = 250
DISPLAY_SCALE = 3


@dataclass(frozen=True)
class TrackerConfig:
    min_distance_mm: float = 250.0
    max_distance_mm: float = 3000.0
    confidence_threshold: float = 30.0
    min_area_px: int = 80
    depth_band_mm: float = 140.0
    confirm_frames: int = 4
    max_missed_frames: int = 8
    association_radius_px: float = 45.0
    association_depth_mm: float = 280.0
    switch_margin_mm: float = 120.0
    prediction_frames: int = 10


@dataclass
class Detection:
    x: float
    y: float
    z_mm: float
    bbox: Tuple[int, int, int, int]
    area: int


@dataclass
class TrackingResult:
    status: str
    active: bool
    detection: Optional[Detection]
    x: Optional[float] = None
    y: Optional[float] = None
    z_mm: Optional[float] = None
    vx_px_s: Optional[float] = None
    vy_px_s: Optional[float] = None
    vz_mm_s: Optional[float] = None
    future_x: Optional[float] = None
    future_y: Optional[float] = None
    future_z_mm: Optional[float] = None
    missed_frames: int = 0


class SharedSettings:
    def __init__(self, initial: TrackerConfig):
        self._lock = threading.Lock()
        self._config = initial

    def get(self) -> TrackerConfig:
        with self._lock:
            return self._config

    def set(self, config: TrackerConfig) -> None:
        with self._lock:
            self._config = config


class KalmanCV3D:
    """Constant-velocity Kalman filter for [x_px, y_px, z_mm, vx, vy, vz]."""

    def __init__(self) -> None:
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64)
        self.initialized = False

        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # Measurement noise: image centroid is usually much steadier than raw ToF depth.
        self.R = np.diag([4.0**2, 4.0**2, 35.0**2]).astype(np.float64)

    def reset(self) -> None:
        self.initialized = False
        self.x.fill(0.0)
        self.P = np.eye(6, dtype=np.float64)

    def initialize(self, x_px: float, y_px: float, z_mm: float) -> None:
        self.x[:, 0] = [x_px, y_px, z_mm, 0.0, 0.0, 0.0]
        self.P = np.diag([
            16.0**2,
            16.0**2,
            80.0**2,
            150.0**2,
            150.0**2,
            700.0**2,
        ]).astype(np.float64)
        self.initialized = True

    @staticmethod
    def _transition(dt: float) -> np.ndarray:
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        return F

    @staticmethod
    def _process_noise(dt: float) -> np.ndarray:
        # Separate acceleration uncertainty for image-plane motion and radial depth motion.
        sigma_a_xy = 260.0   # px/s^2
        sigma_a_z = 1800.0   # mm/s^2

        Q = np.zeros((6, 6), dtype=np.float64)
        for p, v, sigma_a in ((0, 3, sigma_a_xy), (1, 4, sigma_a_xy), (2, 5, sigma_a_z)):
            q = sigma_a**2
            Q[p, p] = 0.25 * dt**4 * q
            Q[p, v] = 0.5 * dt**3 * q
            Q[v, p] = 0.5 * dt**3 * q
            Q[v, v] = dt**2 * q
        return Q

    def predict(self, dt: float) -> np.ndarray:
        if not self.initialized:
            return self.x.copy()
        F = self._transition(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self._process_noise(dt)
        return self.x.copy()

    def update(self, x_px: float, y_px: float, z_mm: float) -> np.ndarray:
        if not self.initialized:
            self.initialize(x_px, y_px, z_mm)
            return self.x.copy()

        z = np.array([[x_px], [y_px], [z_mm]], dtype=np.float64)
        innovation = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        return self.x.copy()

    def future_position(self, seconds: float) -> Tuple[float, float, float]:
        return (
            float(self.x[0, 0] + self.x[3, 0] * seconds),
            float(self.x[1, 0] + self.x[4, 0] * seconds),
            float(self.x[2, 0] + self.x[5, 0] * seconds),
        )


class StableNearestTracker:
    """
    Finds the closest spatially coherent object and tracks it stably.

    Strategy:
      1) Valid depth + confidence mask.
      2) For acquisition/switching: use a low depth percentile instead of the absolute
         minimum, then expand a small depth band and find connected components.
      3) Require a candidate to persist for several frames before lock-on.
      4) Once locked, search locally around the Kalman prediction so a new foreground
         object does not instantly destroy the current track.
      5) Switch only when a distinctly closer candidate remains stable.
    """

    def __init__(self) -> None:
        self.kf = KalmanCV3D()
        self.active = False
        self.pending: Optional[Detection] = None
        self.pending_count = 0
        self.switch_pending: Optional[Detection] = None
        self.switch_count = 0
        self.missed = 0
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None

    def reset(self) -> None:
        self.kf.reset()
        self.active = False
        self.pending = None
        self.pending_count = 0
        self.switch_pending = None
        self.switch_count = 0
        self.missed = 0
        self.last_bbox = None

    @staticmethod
    def _valid_mask(
        depth: np.ndarray,
        confidence: Optional[np.ndarray],
        cfg: TrackerConfig,
    ) -> np.ndarray:
        valid = np.isfinite(depth)
        valid &= depth >= cfg.min_distance_mm
        valid &= depth <= cfg.max_distance_mm
        if confidence is not None and confidence.shape == depth.shape:
            valid &= np.isfinite(confidence)
            valid &= confidence >= cfg.confidence_threshold
        return valid

    @staticmethod
    def _clean_mask(mask: np.ndarray) -> np.ndarray:
        u8 = (mask.astype(np.uint8) * 255)
        kernel3 = np.ones((3, 3), np.uint8)
        kernel5 = np.ones((5, 5), np.uint8)
        u8 = cv2.morphologyEx(u8, cv2.MORPH_OPEN, kernel3, iterations=1)
        u8 = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel5, iterations=1)
        return u8

    @staticmethod
    def _components(
        depth: np.ndarray,
        mask_u8: np.ndarray,
        min_area: int,
        offset_x: int = 0,
        offset_y: int = 0,
    ) -> list[Detection]:
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        detections: list[Detection] = []
        for label in range(1, n_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])

            local_labels = labels[y:y + h, x:x + w]
            component_mask = local_labels == label
            local_depth = depth[y:y + h, x:x + w]
            z_values = local_depth[component_mask]
            z_values = z_values[np.isfinite(z_values)]
            if z_values.size == 0:
                continue

            z_mm = float(np.median(z_values))
            cx = float(centroids[label, 0]) + offset_x
            cy = float(centroids[label, 1]) + offset_y
            detections.append(
                Detection(
                    x=cx,
                    y=cy,
                    z_mm=z_mm,
                    bbox=(x + offset_x, y + offset_y, w, h),
                    area=area,
                )
            )
        return detections

    def _nearest_candidate(
        self,
        depth: np.ndarray,
        confidence: Optional[np.ndarray],
        cfg: TrackerConfig,
    ) -> Optional[Detection]:
        valid = self._valid_mask(depth, confidence, cfg)
        values = depth[valid]
        if values.size < cfg.min_area_px:
            return None

        # Choose a robust low order statistic. Tying it to min_area_px means a handful
        # of abnormally-close ToF pixels cannot define the "nearest object", while an
        # actual object large enough to be accepted still contributes enough pixels.
        adaptive_percentile = float(
            np.clip(50.0 * cfg.min_area_px / max(values.size, 1), 0.02, 5.0)
        )
        nearest_seed = float(np.percentile(values, adaptive_percentile))

        # Start narrow to avoid merging with the background; expand if the front surface
        # is incomplete/noisy.
        for factor in (1.0, 1.7, 2.5):
            limit = min(cfg.max_distance_mm, nearest_seed + cfg.depth_band_mm * factor)
            near_mask = valid & (depth <= limit)
            cleaned = self._clean_mask(near_mask)
            candidates = self._components(depth, cleaned, cfg.min_area_px)
            if candidates:
                # Median depth is much less sensitive to ToF speckle than min(depth).
                return min(candidates, key=lambda d: (d.z_mm, -d.area))
        return None

    def _tracked_candidate(
        self,
        depth: np.ndarray,
        confidence: Optional[np.ndarray],
        cfg: TrackerConfig,
        predicted_xyz: Tuple[float, float, float],
    ) -> Optional[Detection]:
        h_img, w_img = depth.shape[:2]
        px, py, pz = predicted_xyz
        radius = max(8, int(round(cfg.association_radius_px)))

        x0 = max(0, int(math.floor(px)) - radius)
        y0 = max(0, int(math.floor(py)) - radius)
        x1 = min(w_img, int(math.floor(px)) + radius + 1)
        y1 = min(h_img, int(math.floor(py)) + radius + 1)
        if x1 <= x0 or y1 <= y0:
            return None

        d_roi = depth[y0:y1, x0:x1]
        c_roi = confidence[y0:y1, x0:x1] if confidence is not None else None

        valid = self._valid_mask(d_roi, c_roi, cfg)
        valid &= np.abs(d_roi - pz) <= cfg.association_depth_mm
        cleaned = self._clean_mask(valid)

        local_min_area = max(12, int(cfg.min_area_px * 0.35))
        candidates = self._components(
            d_roi,
            cleaned,
            local_min_area,
            offset_x=x0,
            offset_y=y0,
        )
        if not candidates:
            return None

        def score(det: Detection) -> float:
            dxy = math.hypot(det.x - px, det.y - py) / max(cfg.association_radius_px, 1.0)
            dz = abs(det.z_mm - pz) / max(cfg.association_depth_mm, 1.0)
            return dxy + 0.8 * dz

        best = min(candidates, key=score)
        if math.hypot(best.x - px, best.y - py) > cfg.association_radius_px:
            return None
        if abs(best.z_mm - pz) > cfg.association_depth_mm:
            return None
        return best

    @staticmethod
    def _same_candidate(
        a: Detection,
        b: Detection,
        radius_px: float,
        depth_mm: float,
    ) -> bool:
        return (
            math.hypot(a.x - b.x, a.y - b.y) <= radius_px
            and abs(a.z_mm - b.z_mm) <= depth_mm
        )

    def _confirm_pending(
        self,
        candidate: Optional[Detection],
        cfg: TrackerConfig,
    ) -> bool:
        if candidate is None:
            self.pending = None
            self.pending_count = 0
            return False

        if self.pending is not None and self._same_candidate(
            candidate,
            self.pending,
            max(12.0, cfg.association_radius_px * 0.7),
            max(60.0, cfg.association_depth_mm * 0.7),
        ):
            self.pending_count += 1
        else:
            self.pending = candidate
            self.pending_count = 1

        self.pending = candidate
        return self.pending_count >= cfg.confirm_frames

    def _consider_switch(
        self,
        nearest: Optional[Detection],
        current_xyz: Tuple[float, float, float],
        cfg: TrackerConfig,
    ) -> Optional[Detection]:
        if nearest is None:
            self.switch_pending = None
            self.switch_count = 0
            return None

        cx, cy, cz = current_xyz
        same_as_current = (
            math.hypot(nearest.x - cx, nearest.y - cy) <= cfg.association_radius_px * 0.65
            and abs(nearest.z_mm - cz) <= cfg.association_depth_mm * 0.65
        )
        sufficiently_closer = nearest.z_mm < (cz - cfg.switch_margin_mm)

        if same_as_current or not sufficiently_closer:
            self.switch_pending = None
            self.switch_count = 0
            return None

        if self.switch_pending is not None and self._same_candidate(
            nearest,
            self.switch_pending,
            max(12.0, cfg.association_radius_px * 0.7),
            max(60.0, cfg.association_depth_mm * 0.7),
        ):
            self.switch_count += 1
        else:
            self.switch_pending = nearest
            self.switch_count = 1

        self.switch_pending = nearest
        if self.switch_count >= cfg.confirm_frames:
            confirmed = self.switch_pending
            self.switch_pending = None
            self.switch_count = 0
            return confirmed
        return None

    def step(
        self,
        depth: np.ndarray,
        confidence: Optional[np.ndarray],
        cfg: TrackerConfig,
        dt: float,
        avg_frame_period: float,
    ) -> TrackingResult:
        dt = float(np.clip(dt, 1.0 / 120.0, 0.25))
        avg_frame_period = float(np.clip(avg_frame_period, 1.0 / 120.0, 0.25))

        nearest = self._nearest_candidate(depth, confidence, cfg)

        if not self.active:
            if self._confirm_pending(nearest, cfg):
                assert self.pending is not None
                self.kf.initialize(self.pending.x, self.pending.y, self.pending.z_mm)
                self.last_bbox = self.pending.bbox
                self.active = True
                self.missed = 0
                self.pending = None
                self.pending_count = 0
            else:
                status = "ПОИСК"
                if self.pending_count > 0:
                    status = f"ПОДТВЕРЖДЕНИЕ {self.pending_count}/{cfg.confirm_frames}"
                return TrackingResult(status=status, active=False, detection=nearest)

        predicted = self.kf.predict(dt)
        predicted_xyz = (
            float(predicted[0, 0]),
            float(predicted[1, 0]),
            float(predicted[2, 0]),
        )
        tracked = self._tracked_candidate(depth, confidence, cfg, predicted_xyz)

        if tracked is not None:
            state = self.kf.update(tracked.x, tracked.y, tracked.z_mm)
            self.last_bbox = tracked.bbox
            self.missed = 0
            status = "СОПРОВОЖДЕНИЕ"
        else:
            state = self.kf.x.copy()
            self.missed += 1
            status = f"ПРОГНОЗ БЕЗ ИЗМЕРЕНИЯ {self.missed}/{cfg.max_missed_frames}"
            if self.missed > cfg.max_missed_frames:
                self.reset()
                return TrackingResult(status="ПОТЕРЯН — ПОВТОРНЫЙ ПОИСК", active=False, detection=nearest)

        current_xyz = (
            float(state[0, 0]),
            float(state[1, 0]),
            float(state[2, 0]),
        )

        # A closer object must itself be stable before we switch to it.
        switch_to = self._consider_switch(nearest, current_xyz, cfg)
        if switch_to is not None:
            self.kf.initialize(switch_to.x, switch_to.y, switch_to.z_mm)
            self.last_bbox = switch_to.bbox
            self.missed = 0
            tracked = switch_to
            state = self.kf.x.copy()
            status = "ПЕРЕКЛЮЧЕНИЕ НА БЛИЖАЙШИЙ УСТОЙЧИВЫЙ ОБЪЕКТ"

        horizon_s = cfg.prediction_frames * avg_frame_period
        fx, fy, fz = self.kf.future_position(horizon_s)

        return TrackingResult(
            status=status,
            active=True,
            detection=tracked,
            x=float(self.kf.x[0, 0]),
            y=float(self.kf.x[1, 0]),
            z_mm=float(self.kf.x[2, 0]),
            vx_px_s=float(self.kf.x[3, 0]),
            vy_px_s=float(self.kf.x[4, 0]),
            vz_mm_s=float(self.kf.x[5, 0]),
            future_x=fx,
            future_y=fy,
            future_z_mm=fz,
            missed_frames=self.missed,
        )


class CameraWorker(threading.Thread):
    def __init__(
        self,
        settings: SharedSettings,
        output_queue: queue.Queue,
        error_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True)
        self.settings = settings
        self.output_queue = output_queue
        self.error_queue = error_queue
        self.stop_event = threading.Event()
        self.reset_tracker_event = threading.Event()
        self.tracker = StableNearestTracker()
        self.fps_ema = 0.0
        self.period_ema = 1.0 / 30.0

    def stop(self) -> None:
        self.stop_event.set()

    def reset_tracker(self) -> None:
        self.reset_tracker_event.set()

    @staticmethod
    def _put_latest(q: queue.Queue, item) -> None:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            q.get_nowait()
        except queue.Empty:
            pass
        try:
            q.put_nowait(item)
        except queue.Full:
            pass

    @staticmethod
    def _quality_mask(
        depth: np.ndarray,
        confidence: Optional[np.ndarray],
        confidence_threshold: float,
        camera_range_mm: float,
    ) -> np.ndarray:
        valid = np.isfinite(depth) & (depth > 0) & (depth <= camera_range_mm)
        if confidence is not None and confidence.shape == depth.shape:
            valid &= np.isfinite(confidence) & (confidence >= confidence_threshold)
        return valid

    def _make_preview(
        self,
        depth: np.ndarray,
        confidence: Optional[np.ndarray],
        cfg: TrackerConfig,
        result: TrackingResult,
        camera_range_mm: float,
    ) -> np.ndarray:
        clean_depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        normalized = np.clip(clean_depth * (255.0 / max(camera_range_mm, 1.0)), 0, 255).astype(np.uint8)
        image = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)

        quality = self._quality_mask(depth, confidence, cfg.confidence_threshold, camera_range_mm)
        image[~quality] = 0

        # Keep context visible but dim everything outside the user's distance window.
        in_window = quality & (depth >= cfg.min_distance_mm) & (depth <= cfg.max_distance_mm)
        dimmed = (image.astype(np.float32) * 0.25).astype(np.uint8)
        image[~in_window & quality] = dimmed[~in_window & quality]

        h, w = depth.shape[:2]

        if result.detection is not None:
            x, y, bw, bh = result.detection.bbox
            cv2.rectangle(image, (x, y), (x + bw - 1, y + bh - 1), (255, 255, 255), 1)

        if result.active and result.x is not None and result.y is not None:
            cx = int(round(result.x))
            cy = int(round(result.y))
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(image, (cx, cy), 4, (255, 255, 255), 1)
                cv2.drawMarker(image, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 10, 1)

            if result.future_x is not None and result.future_y is not None:
                fx_raw = int(round(result.future_x))
                fy_raw = int(round(result.future_y))
                fx = int(np.clip(fx_raw, 0, w - 1))
                fy = int(np.clip(fy_raw, 0, h - 1))
                start = (int(np.clip(cx, 0, w - 1)), int(np.clip(cy, 0, h - 1)))
                cv2.arrowedLine(image, start, (fx, fy), (255, 255, 255), 1, tipLength=0.18)
                cv2.circle(image, (fx, fy), 5, (0, 0, 0), 1)

        # Short on-frame telemetry; detailed telemetry also appears in the GUI.
        cv2.putText(
            image,
            f"FPS {self.fps_ema:4.1f}",
            (5, 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            f"Range {cfg.min_distance_mm:.0f}-{cfg.max_distance_mm:.0f} mm",
            (5, 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return image

    def run(self) -> None:
        cam = None
        started = False
        try:
            cam = ac.ArducamCamera()
            ret = cam.open(ac.Connection.CSI, 0)
            if ret != 0:
                raise RuntimeError(f"Не удалось открыть ToF-камеру, код ошибки: {ret}")

            ret = cam.start(ac.FrameType.DEPTH)
            if ret != 0:
                raise RuntimeError(f"Не удалось запустить DEPTH-поток, код ошибки: {ret}")
            started = True

            # HQVGA models accept RANGE=2000/4000. Some VGA modes manage range
            # automatically, so an unsupported RANGE write must not kill the app.
            try:
                cam.setControl(ac.Control.RANGE, CAMERA_RANGE_MM)
            except Exception:
                pass

            try:
                camera_range = float(cam.getControl(ac.Control.RANGE))
            except Exception:
                camera_range = float(CAMERA_RANGE_MM)
            if not np.isfinite(camera_range) or camera_range <= 0:
                camera_range = float(CAMERA_RANGE_MM)

            info = cam.getCameraInfo()
            self._put_latest(
                self.error_queue,
                ("info", f"Камера запущена: {info.width}x{info.height}, RANGE={camera_range:.0f} mm"),
            )

            last_t = time.monotonic()

            while not self.stop_event.is_set():
                if self.reset_tracker_event.is_set():
                    self.tracker.reset()
                    self.reset_tracker_event.clear()

                frame = cam.requestFrame(FRAME_TIMEOUT_MS)
                if frame is None:
                    continue

                try:
                    if not isinstance(frame, ac.DepthData):
                        continue

                    # Copy before releaseFrame(): SDK memory may become invalid after release.
                    depth = np.array(frame.depth_data, dtype=np.float32, copy=True)
                    raw_conf = getattr(frame, "confidence_data", None)
                    confidence = (
                        np.array(raw_conf, dtype=np.float32, copy=True)
                        if raw_conf is not None
                        else None
                    )
                finally:
                    cam.releaseFrame(frame)

                if depth.ndim != 2:
                    raise RuntimeError(f"Ожидалась 2D depth-карта, получена форма {depth.shape}")

                now = time.monotonic()
                dt = max(1e-4, now - last_t)
                last_t = now

                instant_fps = 1.0 / dt
                if self.fps_ema <= 0.0:
                    self.fps_ema = instant_fps
                    self.period_ema = dt
                else:
                    self.fps_ema = 0.90 * self.fps_ema + 0.10 * instant_fps
                    self.period_ema = 0.90 * self.period_ema + 0.10 * dt

                cfg = self.settings.get()
                result = self.tracker.step(depth, confidence, cfg, dt, self.period_ema)
                preview = self._make_preview(depth, confidence, cfg, result, camera_range)

                payload = {
                    "image_bgr": preview,
                    "result": result,
                    "fps": self.fps_ema,
                    "period": self.period_ema,
                    "shape": depth.shape,
                    "camera_range": camera_range,
                }
                self._put_latest(self.output_queue, payload)

        except Exception as exc:
            details = traceback.format_exc()
            self._put_latest(self.error_queue, ("error", f"{exc}\n\n{details}"))
        finally:
            if cam is not None:
                if started:
                    try:
                        cam.stop()
                    except Exception:
                        pass
                try:
                    cam.close()
                except Exception:
                    pass


class ToFTrackerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Arducam ToF — ближайший устойчивый объект")
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.settings = SharedSettings(TrackerConfig())
        self.frame_queue: queue.Queue = queue.Queue(maxsize=1)
        self.error_queue: queue.Queue = queue.Queue(maxsize=4)
        self.worker = CameraWorker(self.settings, self.frame_queue, self.error_queue)

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._closing = False

        self._build_ui()
        self.worker.start()
        self.after(15, self._poll)

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=10)
        root.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        root.columnconfigure(0, weight=1)

        preview_frame = ttk.LabelFrame(root, text="Depth / tracking")
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.preview_label = ttk.Label(preview_frame, anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)

        controls = ttk.LabelFrame(root, text="Параметры")
        controls.grid(row=0, column=1, sticky="ns")

        cfg = self.settings.get()
        self.min_distance = tk.DoubleVar(value=cfg.min_distance_mm)
        self.max_distance = tk.DoubleVar(value=cfg.max_distance_mm)
        self.confidence = tk.DoubleVar(value=cfg.confidence_threshold)
        self.min_area = tk.IntVar(value=cfg.min_area_px)
        self.depth_band = tk.DoubleVar(value=cfg.depth_band_mm)
        self.confirm_frames = tk.IntVar(value=cfg.confirm_frames)
        self.max_missed = tk.IntVar(value=cfg.max_missed_frames)
        self.assoc_radius = tk.DoubleVar(value=cfg.association_radius_px)
        self.assoc_depth = tk.DoubleVar(value=cfg.association_depth_mm)
        self.switch_margin = tk.DoubleVar(value=cfg.switch_margin_mm)
        self.prediction_frames = tk.IntVar(value=cfg.prediction_frames)

        row = 0
        ttk.Label(controls, text="Нижняя граница, мм").grid(row=row, column=0, sticky="w", padx=8, pady=(8, 0))
        row += 1
        tk.Scale(
            controls,
            from_=50,
            to=CAMERA_RANGE_MM - 50,
            resolution=10,
            orient=tk.HORIZONTAL,
            length=270,
            variable=self.min_distance,
        ).grid(row=row, column=0, sticky="ew", padx=8)

        row += 1
        ttk.Label(controls, text="Верхняя граница, мм").grid(row=row, column=0, sticky="w", padx=8, pady=(6, 0))
        row += 1
        tk.Scale(
            controls,
            from_=100,
            to=CAMERA_RANGE_MM,
            resolution=10,
            orient=tk.HORIZONTAL,
            length=270,
            variable=self.max_distance,
        ).grid(row=row, column=0, sticky="ew", padx=8)

        row += 1
        sep = ttk.Separator(controls)
        sep.grid(row=row, column=0, sticky="ew", padx=8, pady=8)

        row += 1
        row = self._spin_row(controls, row, "Прогноз через, кадров", self.prediction_frames, 1, 120, 1)
        row = self._spin_row(controls, row, "Confidence threshold", self.confidence, 0, 255, 1)
        row = self._spin_row(controls, row, "Мин. площадь объекта, px", self.min_area, 10, 5000, 10)
        row = self._spin_row(controls, row, "Глубинная полоса, мм", self.depth_band, 30, 800, 10)
        row = self._spin_row(controls, row, "Кадров подтверждения", self.confirm_frames, 1, 20, 1)
        row = self._spin_row(controls, row, "Допустимый пропуск, кадров", self.max_missed, 0, 60, 1)
        row = self._spin_row(controls, row, "Радиус ассоциации, px", self.assoc_radius, 8, 160, 1)
        row = self._spin_row(controls, row, "Допуск по глубине, мм", self.assoc_depth, 30, 1200, 10)
        row = self._spin_row(controls, row, "Порог переключения, мм", self.switch_margin, 0, 1000, 10)

        ttk.Button(controls, text="Применить", command=self.apply_settings).grid(
            row=row, column=0, sticky="ew", padx=8, pady=(10, 4)
        )
        row += 1
        ttk.Button(controls, text="Сбросить трек", command=self.worker.reset_tracker).grid(
            row=row, column=0, sticky="ew", padx=8, pady=4
        )
        row += 1
        ttk.Button(controls, text="Выход", command=self.on_close).grid(
            row=row, column=0, sticky="ew", padx=8, pady=(4, 10)
        )

        telemetry = ttk.LabelFrame(root, text="Телеметрия")
        telemetry.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        self.telemetry_var = tk.StringVar(value="Запуск камеры...")
        ttk.Label(
            telemetry,
            textvariable=self.telemetry_var,
            justify=tk.LEFT,
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        telemetry.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Инициализация...")
        ttk.Label(root, textvariable=self.status_var, anchor="w").grid(
            row=2, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )

    @staticmethod
    def _spin_row(parent, row, label, variable, from_, to, increment) -> int:
        line = ttk.Frame(parent)
        line.grid(row=row, column=0, sticky="ew", padx=8, pady=2)
        line.columnconfigure(0, weight=1)
        ttk.Label(line, text=label).grid(row=0, column=0, sticky="w")
        ttk.Spinbox(
            line,
            textvariable=variable,
            from_=from_,
            to=to,
            increment=increment,
            width=8,
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))
        return row + 1

    def apply_settings(self) -> None:
        try:
            min_d = float(self.min_distance.get())
            max_d = float(self.max_distance.get())
            if not (0 < min_d < max_d <= CAMERA_RANGE_MM):
                raise ValueError(f"Нужно: 0 < нижняя < верхняя <= {CAMERA_RANGE_MM} мм")

            config = TrackerConfig(
                min_distance_mm=min_d,
                max_distance_mm=max_d,
                confidence_threshold=float(self.confidence.get()),
                min_area_px=int(self.min_area.get()),
                depth_band_mm=float(self.depth_band.get()),
                confirm_frames=int(self.confirm_frames.get()),
                max_missed_frames=int(self.max_missed.get()),
                association_radius_px=float(self.assoc_radius.get()),
                association_depth_mm=float(self.assoc_depth.get()),
                switch_margin_mm=float(self.switch_margin.get()),
                prediction_frames=int(self.prediction_frames.get()),
            )

            if config.min_area_px < 1 or config.confirm_frames < 1 or config.prediction_frames < 1:
                raise ValueError("Площадь, подтверждение и горизонт прогноза должны быть >= 1")
            if config.depth_band_mm <= 0 or config.association_radius_px <= 0 or config.association_depth_mm <= 0:
                raise ValueError("Полоса глубины и допуски должны быть > 0")
            if not (0 <= config.confidence_threshold <= 255):
                raise ValueError("Confidence threshold должен быть в диапазоне 0..255")

            self.settings.set(config)
            self.worker.reset_tracker()
            self.status_var.set("Параметры применены; трек переинициализирован.")
        except Exception as exc:
            messagebox.showerror("Некорректные параметры", str(exc))

    @staticmethod
    def _motion_text(vz_mm_s: Optional[float]) -> str:
        if vz_mm_s is None:
            return "—"
        deadband = 35.0
        if vz_mm_s < -deadband:
            return f"ПРИБЛИЖАЕТСЯ ({vz_mm_s:.0f} мм/с)"
        if vz_mm_s > deadband:
            return f"УДАЛЯЕТСЯ (+{vz_mm_s:.0f} мм/с)"
        return f"ПО ГЛУБИНЕ ПОЧТИ НЕПОДВИЖЕН ({vz_mm_s:+.0f} мм/с)"

    def _update_telemetry(self, payload: dict) -> None:
        result: TrackingResult = payload["result"]
        fps = float(payload["fps"])
        h, w = payload["shape"]
        cfg = self.settings.get()

        if not result.active:
            self.telemetry_var.set(
                f"Статус: {result.status}\n"
                f"FPS: {fps:.1f}    Depth: {w}x{h}\n"
                f"Рабочая дальность: {cfg.min_distance_mm:.0f}..{cfg.max_distance_mm:.0f} мм"
            )
            return

        speed_px_s = math.hypot(result.vx_px_s or 0.0, result.vy_px_s or 0.0)
        future = "—"
        if result.future_x is not None and result.future_y is not None and result.future_z_mm is not None:
            future = (
                f"x={result.future_x:.1f}px, y={result.future_y:.1f}px, "
                f"z={result.future_z_mm:.0f}мм"
            )

        self.telemetry_var.set(
            f"Статус: {result.status}\n"
            f"FPS: {fps:.1f}    Depth: {w}x{h}\n"
            f"Текущая точка: x={result.x:.1f}px, y={result.y:.1f}px, z={result.z_mm:.0f}мм\n"
            f"Скорость в кадре: {speed_px_s:.1f}px/с    "
            f"Vx={result.vx_px_s:+.1f}, Vy={result.vy_px_s:+.1f}\n"
            f"По глубине: {self._motion_text(result.vz_mm_s)}\n"
            f"Прогноз через {cfg.prediction_frames} кадров: {future}"
        )

    def _poll(self) -> None:
        if self._closing:
            return

        # Errors/info from the worker.
        while True:
            try:
                kind, text = self.error_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "error":
                self.status_var.set("Ошибка камеры/обработки. Подробности выведены в диалоге.")
                messagebox.showerror("Arducam ToF", text)
            else:
                self.status_var.set(text)

        payload = None
        while True:
            try:
                payload = self.frame_queue.get_nowait()
            except queue.Empty:
                break

        if payload is not None:
            bgr = payload["image_bgr"]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            if DISPLAY_SCALE != 1:
                pil = pil.resize(
                    (pil.width * DISPLAY_SCALE, pil.height * DISPLAY_SCALE),
                    Image.Resampling.NEAREST,
                )
            self._photo = ImageTk.PhotoImage(pil)
            self.preview_label.configure(image=self._photo)
            self._update_telemetry(payload)

        self.after(15, self._poll)

    def on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.status_var.set("Остановка камеры...")
        self.worker.stop()
        self.worker.join(timeout=1.5)
        self.destroy()


def main() -> None:
    app = ToFTrackerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
