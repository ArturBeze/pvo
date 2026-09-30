#!/usr/bin/env python3
"""
Arducam ToF nearest-object tracker with Tkinter GUI.

Features:
- User-configurable minimum and maximum tracking distance (mm)
- Confidence threshold and minimum object area
- Tracks the nearest robust connected object in the selected depth range
- Smooths measurements
- Estimates image-plane velocity (px/s) and depth velocity (mm/s)
- Predicts object centroid/depth N frames into the future
- Draws current object, trail and future trajectory
- Tkinter GUI; camera capture/processing runs in a worker thread

Designed around the ArducamDepthCamera API used by Arducam's Raspberry Pi ToF
Python examples:
    cam.open(ac.Connection.CSI, 0)
    cam.start(ac.FrameType.DEPTH)
    cam.setControl(ac.Control.RANGE, 2000 or 4000)
    frame = cam.requestFrame(...)
    frame.depth_data / frame.confidence_data
    cam.releaseFrame(frame)
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Deque, Optional, Tuple

import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise SystemExit(
        "Pillow is required for the Tkinter preview.\n"
        "Install it with: python3 -m pip install Pillow"
    ) from exc

try:
    import ArducamDepthCamera as ac
except ImportError as exc:
    raise SystemExit(
        "ArducamDepthCamera is not installed.\n"
        "Install the official Arducam ToF SDK/dependencies first."
    ) from exc


# Arducam example notes that RANGE is switchable between 2000 and 4000 mm.
HW_RANGE_NEAR_MM = 2000
HW_RANGE_FAR_MM = 4000


@dataclass(frozen=True)
class TrackerConfig:
    min_distance_mm: int = 200
    max_distance_mm: int = 2000
    confidence_threshold: int = 30
    min_area_px: int = 80
    prediction_frames: int = 10
    history_frames: int = 10
    depth_band_mm: int = 250
    max_jump_px: int = 90
    smoothing_alpha: float = 0.45
    lost_frames_before_reacquire: int = 5
    z_motion_deadband_mm_s: float = 35.0

    def validated(self) -> "TrackerConfig":
        min_d = int(np.clip(self.min_distance_mm, 0, HW_RANGE_FAR_MM - 1))
        max_d = int(np.clip(self.max_distance_mm, 1, HW_RANGE_FAR_MM))
        if min_d >= max_d:
            raise ValueError("Нижняя граница должна быть меньше верхней.")

        return replace(
            self,
            min_distance_mm=min_d,
            max_distance_mm=max_d,
            confidence_threshold=int(np.clip(self.confidence_threshold, 0, 255)),
            min_area_px=max(1, int(self.min_area_px)),
            prediction_frames=max(0, int(self.prediction_frames)),
            history_frames=int(np.clip(self.history_frames, 3, 60)),
            depth_band_mm=int(np.clip(self.depth_band_mm, 20, 1500)),
            max_jump_px=int(np.clip(self.max_jump_px, 5, 500)),
            smoothing_alpha=float(np.clip(self.smoothing_alpha, 0.05, 1.0)),
            lost_frames_before_reacquire=int(
                np.clip(self.lost_frames_before_reacquire, 1, 60)
            ),
            z_motion_deadband_mm_s=max(0.0, float(self.z_motion_deadband_mm_s)),
        )


@dataclass
class TrackMeasurement:
    timestamp: float
    x_px: float
    y_px: float
    z_mm: float


@dataclass
class TrackingResult:
    bbox: Optional[Tuple[int, int, int, int]]
    contour: Optional[np.ndarray]
    current: Optional[TrackMeasurement]
    predicted: Optional[TrackMeasurement]
    vx_px_s: float = 0.0
    vy_px_s: float = 0.0
    vz_mm_s: float = 0.0
    motion_state: str = "нет цели"
    mask: Optional[np.ndarray] = None


@dataclass
class FramePacket:
    image_bgr: np.ndarray
    fps: float
    result: TrackingResult
    camera_range_mm: int
    message: str = ""


class NearestObjectTracker:
    """
    Tracks the nearest robust object inside the configured depth interval.

    x/y are image coordinates. z is measured depth in millimetres.
    Velocity is estimated by linear regression over recent smoothed samples.
    """

    def __init__(self) -> None:
        self.history: Deque[TrackMeasurement] = deque(maxlen=120)
        self.last_bbox: Optional[Tuple[int, int, int, int]] = None
        self.lost_frames = 0

    def reset(self) -> None:
        self.history.clear()
        self.last_bbox = None
        self.lost_frames = 0

    @staticmethod
    def _make_valid_mask(
        depth: np.ndarray,
        confidence: Optional[np.ndarray],
        cfg: TrackerConfig,
    ) -> np.ndarray:
        finite = np.isfinite(depth)
        mask = (
            finite
            & (depth >= float(cfg.min_distance_mm))
            & (depth <= float(cfg.max_distance_mm))
        )

        if confidence is not None and confidence.shape == depth.shape:
            conf = np.nan_to_num(confidence, nan=0.0, posinf=0.0, neginf=0.0)
            mask &= conf >= float(cfg.confidence_threshold)

        return mask

    @staticmethod
    def _nearest_depth_mask(
        depth: np.ndarray,
        valid_mask: np.ndarray,
        cfg: TrackerConfig,
    ) -> np.ndarray:
        """
        Robustly isolate the front-most depth layer.

        Instead of using the absolute minimum (which is sensitive to single-pixel
        ToF noise), choose a low percentile based on the requested minimum object
        area and then keep a configurable depth band behind it.
        """
        values = depth[valid_mask]
        if values.size < cfg.min_area_px:
            return np.zeros(depth.shape, dtype=np.uint8)

        # Keep the seed percentile low, but large enough that a tiny group of
        # outlier pixels cannot become the tracked "object".
        q = 100.0 * cfg.min_area_px / max(1, values.size)
        q = float(np.clip(q, 0.5, 8.0))
        seed_depth = float(np.percentile(values, q))

        upper = min(float(cfg.max_distance_mm), seed_depth + cfg.depth_band_mm)
        target = valid_mask & (depth <= upper)

        target_u8 = (target.astype(np.uint8) * 255)

        kernel = np.ones((3, 3), np.uint8)
        target_u8 = cv2.morphologyEx(
            target_u8, cv2.MORPH_OPEN, kernel, iterations=1
        )
        target_u8 = cv2.morphologyEx(
            target_u8, cv2.MORPH_CLOSE, kernel, iterations=2
        )
        return target_u8

    @staticmethod
    def _components(
        mask_u8: np.ndarray,
        depth: np.ndarray,
        cfg: TrackerConfig,
    ):
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_u8, connectivity=8
        )
        candidates = []

        for label in range(1, n):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < cfg.min_area_px:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[label]

            component_pixels = labels == label
            z_values = depth[component_pixels]
            z_values = z_values[np.isfinite(z_values)]
            if z_values.size == 0:
                continue

            median_z = float(np.median(z_values))

            component_u8 = (component_pixels.astype(np.uint8) * 255)
            contours, _ = cv2.findContours(
                component_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            contour = max(contours, key=cv2.contourArea) if contours else None

            candidates.append(
                {
                    "label": label,
                    "area": area,
                    "bbox": (x, y, w, h),
                    "centroid": (float(cx), float(cy)),
                    "z": median_z,
                    "contour": contour,
                }
            )

        return candidates

    def _select_candidate(
        self,
        candidates,
        cfg: TrackerConfig,
        frame_shape: Tuple[int, int],
    ):
        if not candidates:
            return None

        # No active track: nearest robust component wins.
        if not self.history:
            return min(candidates, key=lambda c: c["z"])

        last = self.history[-1]
        h, w = frame_shape
        diag = max(1.0, math.hypot(w, h))

        gated = []
        for c in candidates:
            cx, cy = c["centroid"]
            spatial = math.hypot(cx - last.x_px, cy - last.y_px)
            if spatial <= cfg.max_jump_px:
                depth_delta = abs(c["z"] - last.z_mm)
                # Dimensionless association score:
                # identity continuity dominates, with a mild preference for
                # similar depth and the nearest object.
                score = (
                    0.60 * (spatial / diag)
                    + 0.25 * (depth_delta / HW_RANGE_FAR_MM)
                    + 0.15 * (c["z"] / HW_RANGE_FAR_MM)
                )
                gated.append((score, c))

        if gated:
            return min(gated, key=lambda item: item[0])[1]

        # Preserve identity for a few missing frames before switching to a new
        # object elsewhere in the image.
        if self.lost_frames < cfg.lost_frames_before_reacquire:
            return None

        self.reset()
        return min(candidates, key=lambda c: c["z"])

    @staticmethod
    def _fit_velocity(samples):
        if len(samples) < 3:
            return 0.0, 0.0, 0.0

        t0 = samples[0].timestamp
        t = np.array([s.timestamp - t0 for s in samples], dtype=np.float64)
        x = np.array([s.x_px for s in samples], dtype=np.float64)
        y = np.array([s.y_px for s in samples], dtype=np.float64)
        z = np.array([s.z_mm for s in samples], dtype=np.float64)

        if float(t[-1] - t[0]) < 1e-3:
            return 0.0, 0.0, 0.0

        # Linear least-squares slope. More stable than a single-frame delta.
        vx = float(np.polyfit(t, x, 1)[0])
        vy = float(np.polyfit(t, y, 1)[0])
        vz = float(np.polyfit(t, z, 1)[0])
        return vx, vy, vz

    @staticmethod
    def _estimate_frame_period(samples) -> float:
        if len(samples) < 2:
            return 1.0 / 30.0
        t = np.array([s.timestamp for s in samples], dtype=np.float64)
        dt = np.diff(t)
        dt = dt[(dt > 1e-4) & (dt < 1.0)]
        if dt.size == 0:
            return 1.0 / 30.0
        return float(np.median(dt))

    def update(
        self,
        depth: np.ndarray,
        confidence: Optional[np.ndarray],
        timestamp: float,
        cfg: TrackerConfig,
    ) -> TrackingResult:
        valid = self._make_valid_mask(depth, confidence, cfg)
        mask_u8 = self._nearest_depth_mask(depth, valid, cfg)
        candidates = self._components(mask_u8, depth, cfg)
        selected = self._select_candidate(candidates, cfg, depth.shape)

        if selected is None:
            self.lost_frames += 1
            if self.lost_frames > cfg.lost_frames_before_reacquire:
                self.reset()
            return TrackingResult(
                bbox=None,
                contour=None,
                current=None,
                predicted=None,
                motion_state="цель потеряна",
                mask=mask_u8,
            )

        self.lost_frames = 0

        cx, cy = selected["centroid"]
        z = selected["z"]

        # Exponential smoothing reduces ToF/centroid jitter before velocity fit.
        if self.history:
            prev = self.history[-1]
            a = cfg.smoothing_alpha
            cx = a * cx + (1.0 - a) * prev.x_px
            cy = a * cy + (1.0 - a) * prev.y_px
            z = a * z + (1.0 - a) * prev.z_mm

        current = TrackMeasurement(timestamp, cx, cy, z)
        self.history.append(current)
        self.last_bbox = selected["bbox"]

        samples = list(self.history)[-cfg.history_frames :]
        vx, vy, vz = self._fit_velocity(samples)

        frame_period = self._estimate_frame_period(samples)
        horizon_s = cfg.prediction_frames * frame_period

        pred_x = current.x_px + vx * horizon_s
        pred_y = current.y_px + vy * horizon_s
        pred_z = current.z_mm + vz * horizon_s

        h, w = depth.shape
        pred_x = float(np.clip(pred_x, 0, w - 1))
        pred_y = float(np.clip(pred_y, 0, h - 1))
        pred_z = float(np.clip(pred_z, 0, HW_RANGE_FAR_MM))

        predicted = TrackMeasurement(
            timestamp + horizon_s, pred_x, pred_y, pred_z
        )

        if vz < -cfg.z_motion_deadband_mm_s:
            motion = "приближается"
        elif vz > cfg.z_motion_deadband_mm_s:
            motion = "удаляется"
        else:
            motion = "дистанция стабильна"

        return TrackingResult(
            bbox=selected["bbox"],
            contour=selected["contour"],
            current=current,
            predicted=predicted,
            vx_px_s=vx,
            vy_px_s=vy,
            vz_mm_s=vz,
            motion_state=motion,
            mask=mask_u8,
        )


class CameraWorker(threading.Thread):
    def __init__(
        self,
        config_getter,
        output_queue: "queue.Queue[FramePacket]",
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="ArducamToFWorker", daemon=True)
        self.config_getter = config_getter
        self.output_queue = output_queue
        self.stop_event = stop_event
        self.tracker = NearestObjectTracker()
        self._reset_requested = threading.Event()

    def request_tracker_reset(self) -> None:
        self._reset_requested.set()

    def _publish(self, packet: FramePacket) -> None:
        # Keep only the newest frame; GUI latency is more important than
        # displaying every captured frame.
        try:
            while True:
                self.output_queue.get_nowait()
        except queue.Empty:
            pass

        try:
            self.output_queue.put_nowait(packet)
        except queue.Full:
            pass

    @staticmethod
    def _desired_hardware_range(max_distance_mm: int) -> int:
        return (
            HW_RANGE_NEAR_MM
            if max_distance_mm <= HW_RANGE_NEAR_MM
            else HW_RANGE_FAR_MM
        )

    @staticmethod
    def _make_preview(
        depth: np.ndarray,
        confidence: Optional[np.ndarray],
        result: TrackingResult,
        cfg: TrackerConfig,
        camera_range_mm: int,
        fps: float,
    ) -> np.ndarray:
        clean = np.nan_to_num(
            depth, nan=0.0, posinf=float(camera_range_mm), neginf=0.0
        )
        depth_8u = np.clip(
            clean * (255.0 / max(1, camera_range_mm)), 0, 255
        ).astype(np.uint8)
        preview = cv2.applyColorMap(depth_8u, cv2.COLORMAP_RAINBOW)

        valid_range = (
            (clean >= cfg.min_distance_mm) & (clean <= cfg.max_distance_mm)
        )

        if confidence is not None and confidence.shape == depth.shape:
            conf = np.nan_to_num(confidence, nan=0.0)
            valid_range &= conf >= cfg.confidence_threshold

        preview[~valid_range] = (20, 20, 20)

        if result.contour is not None:
            cv2.drawContours(preview, [result.contour], -1, (255, 255, 255), 1)

        if result.bbox is not None:
            x, y, w, h = result.bbox
            cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 255, 255), 1)

        if result.current is not None:
            cur = (
                int(round(result.current.x_px)),
                int(round(result.current.y_px)),
            )
            cv2.circle(preview, cur, 4, (255, 255, 255), -1)

            if result.predicted is not None:
                pred = (
                    int(round(result.predicted.x_px)),
                    int(round(result.predicted.y_px)),
                )
                cv2.arrowedLine(
                    preview, cur, pred, (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.2
                )
                cv2.circle(preview, pred, 6, (0, 0, 0), 2)

        # A short text overlay remains readable regardless of GUI panel state.
        cv2.putText(
            preview,
            f"FPS {fps:4.1f}",
            (6, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            preview,
            f"range {cfg.min_distance_mm}-{cfg.max_distance_mm} mm",
            (6, 33),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return preview

    def run(self) -> None:
        cam = None
        camera_started = False
        camera_range_mm = HW_RANGE_FAR_MM

        fps_t0 = time.monotonic()
        fps_count = 0
        fps = 0.0

        try:
            cam = ac.ArducamCamera()

            ret = cam.open(ac.Connection.CSI, 0)
            if ret != 0:
                raise RuntimeError(f"Не удалось открыть камеру, код ошибки: {ret}")

            ret = cam.start(ac.FrameType.DEPTH)
            if ret != 0:
                raise RuntimeError(f"Не удалось запустить камеру, код ошибки: {ret}")
            camera_started = True

            try:
                info = cam.getCameraInfo()
                resolution_msg = f"{info.width}x{info.height}"
            except Exception:
                resolution_msg = "неизвестно"

            cfg = self.config_getter().validated()
            camera_range_mm = self._desired_hardware_range(cfg.max_distance_mm)
            cam.setControl(ac.Control.RANGE, camera_range_mm)

            while not self.stop_event.is_set():
                if self._reset_requested.is_set():
                    self.tracker.reset()
                    self._reset_requested.clear()

                try:
                    cfg = self.config_getter().validated()
                except ValueError:
                    time.sleep(0.03)
                    continue

                desired_range = self._desired_hardware_range(
                    cfg.max_distance_mm
                )
                if desired_range != camera_range_mm:
                    cam.setControl(ac.Control.RANGE, desired_range)
                    camera_range_mm = desired_range
                    self.tracker.reset()

                frame = cam.requestFrame(200)
                if frame is None:
                    continue

                try:
                    if not isinstance(frame, ac.DepthData):
                        continue

                    depth = np.asarray(frame.depth_data, dtype=np.float32).copy()

                    confidence_raw = getattr(frame, "confidence_data", None)
                    confidence = None
                    if confidence_raw is not None:
                        confidence = np.asarray(
                            confidence_raw, dtype=np.float32
                        ).copy()

                    now = time.monotonic()
                    result = self.tracker.update(depth, confidence, now, cfg)

                    fps_count += 1
                    elapsed = now - fps_t0
                    if elapsed >= 0.5:
                        fps = fps_count / elapsed
                        fps_t0 = now
                        fps_count = 0

                    preview = self._make_preview(
                        depth,
                        confidence,
                        result,
                        cfg,
                        camera_range_mm,
                        fps,
                    )

                    self._publish(
                        FramePacket(
                            image_bgr=preview,
                            fps=fps,
                            result=result,
                            camera_range_mm=camera_range_mm,
                            message=f"Камера {resolution_msg}",
                        )
                    )
                finally:
                    cam.releaseFrame(frame)

        except Exception as exc:
            error_image = np.zeros((180, 320, 3), dtype=np.uint8)
            cv2.putText(
                error_image,
                "CAMERA ERROR",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            self._publish(
                FramePacket(
                    image_bgr=error_image,
                    fps=0.0,
                    result=TrackingResult(None, None, None, None),
                    camera_range_mm=camera_range_mm,
                    message=str(exc),
                )
            )
        finally:
            if cam is not None:
                if camera_started:
                    try:
                        cam.stop()
                    except Exception:
                        pass
                try:
                    cam.close()
                except Exception:
                    pass


class ToFTrackerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Arducam ToF — отслеживание ближайшего объекта")
        self.root.minsize(980, 620)

        self._config_lock = threading.Lock()
        self._config = TrackerConfig()

        self.output_queue: "queue.Queue[FramePacket]" = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.worker = CameraWorker(
            self.get_config, self.output_queue, self.stop_event
        )

        self._photo = None

        self._build_gui()
        self._load_config_to_widgets(self._config)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.worker.start()
        self.root.after(30, self._poll_frames)

    def get_config(self) -> TrackerConfig:
        with self._config_lock:
            return self._config

    def set_config(self, cfg: TrackerConfig) -> None:
        with self._config_lock:
            self._config = cfg

    def _build_gui(self) -> None:
        outer = ttk.Frame(self.root, padding=8)
        outer.pack(fill=tk.BOTH, expand=True)

        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.rowconfigure(0, weight=1)

        # Preview area
        preview_frame = ttk.LabelFrame(outer, text="Depth preview")
        preview_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        preview_frame.rowconfigure(0, weight=1)
        preview_frame.columnconfigure(0, weight=1)

        self.preview_label = ttk.Label(
            preview_frame,
            text="Запуск камеры...",
            anchor=tk.CENTER,
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        # Control panel
        controls = ttk.LabelFrame(outer, text="Параметры", padding=10)
        controls.grid(row=0, column=1, sticky="ns")

        self.var_min_distance = tk.IntVar()
        self.var_max_distance = tk.IntVar()
        self.var_confidence = tk.IntVar()
        self.var_min_area = tk.IntVar()
        self.var_prediction = tk.IntVar()
        self.var_history = tk.IntVar()
        self.var_depth_band = tk.IntVar()
        self.var_max_jump = tk.IntVar()
        self.var_smoothing = tk.DoubleVar()

        row = 0
        row = self._add_spin(
            controls,
            row,
            "Мин. дистанция, мм",
            self.var_min_distance,
            0,
            3999,
            10,
        )
        row = self._add_spin(
            controls,
            row,
            "Макс. дистанция, мм",
            self.var_max_distance,
            1,
            4000,
            10,
        )
        row = self._add_spin(
            controls,
            row,
            "Confidence",
            self.var_confidence,
            0,
            255,
            1,
        )
        row = self._add_spin(
            controls,
            row,
            "Мин. площадь, px",
            self.var_min_area,
            1,
            5000,
            10,
        )
        row = self._add_spin(
            controls,
            row,
            "Прогноз, кадров",
            self.var_prediction,
            0,
            120,
            1,
        )
        row = self._add_spin(
            controls,
            row,
            "История, кадров",
            self.var_history,
            3,
            60,
            1,
        )
        row = self._add_spin(
            controls,
            row,
            "Полоса глубины, мм",
            self.var_depth_band,
            20,
            1500,
            10,
        )
        row = self._add_spin(
            controls,
            row,
            "Макс. скачок, px",
            self.var_max_jump,
            5,
            500,
            5,
        )

        ttk.Label(controls, text="Сглаживание").grid(
            row=row, column=0, sticky="w", pady=(6, 0)
        )
        self.smoothing_scale = ttk.Scale(
            controls,
            from_=0.05,
            to=1.0,
            variable=self.var_smoothing,
            orient=tk.HORIZONTAL,
            length=190,
        )
        self.smoothing_scale.grid(
            row=row + 1, column=0, sticky="ew", pady=(0, 8)
        )
        row += 2

        ttk.Button(
            controls, text="Применить параметры", command=self.apply_config
        ).grid(row=row, column=0, sticky="ew", pady=(4, 4))
        row += 1

        ttk.Button(
            controls, text="Сбросить трек", command=self.worker.request_tracker_reset
        ).grid(row=row, column=0, sticky="ew", pady=(0, 10))
        row += 1

        ttk.Separator(controls, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky="ew", pady=6
        )
        row += 1

        self.status_camera = ttk.Label(controls, text="Камера: запуск...")
        self.status_camera.grid(row=row, column=0, sticky="w")
        row += 1

        self.status_fps = ttk.Label(controls, text="FPS: 0.0")
        self.status_fps.grid(row=row, column=0, sticky="w")
        row += 1

        self.status_target = ttk.Label(controls, text="Цель: нет")
        self.status_target.grid(row=row, column=0, sticky="w")
        row += 1

        self.status_depth = ttk.Label(controls, text="Z: —")
        self.status_depth.grid(row=row, column=0, sticky="w")
        row += 1

        self.status_velocity = ttk.Label(controls, text="Vz: —")
        self.status_velocity.grid(row=row, column=0, sticky="w")
        row += 1

        self.status_xy_velocity = ttk.Label(controls, text="Vxy: —")
        self.status_xy_velocity.grid(row=row, column=0, sticky="w")
        row += 1

        self.status_prediction = ttk.Label(
            controls, text="Прогноз: —", wraplength=240
        )
        self.status_prediction.grid(row=row, column=0, sticky="w", pady=(0, 8))
        row += 1

        ttk.Separator(controls, orient=tk.HORIZONTAL).grid(
            row=row, column=0, sticky="ew", pady=6
        )
        row += 1

        help_text = (
            "Белый контур — текущий объект.\n"
            "Стрелка — прогноз центра объекта.\n"
            "Z < 0 по скорости = приближение.\n\n"
            "Важно: X/Y здесь измеряются в пикселях,\n"
            "Z — в миллиметрах."
        )
        ttk.Label(
            controls, text=help_text, justify=tk.LEFT, wraplength=240
        ).grid(row=row, column=0, sticky="w")

    @staticmethod
    def _add_spin(
        parent,
        row: int,
        label: str,
        variable,
        from_: float,
        to: float,
        increment: float,
    ) -> int:
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", pady=(4, 0)
        )
        spin = ttk.Spinbox(
            parent,
            textvariable=variable,
            from_=from_,
            to=to,
            increment=increment,
            width=18,
        )
        spin.grid(row=row + 1, column=0, sticky="ew", pady=(0, 4))
        return row + 2

    def _load_config_to_widgets(self, cfg: TrackerConfig) -> None:
        self.var_min_distance.set(cfg.min_distance_mm)
        self.var_max_distance.set(cfg.max_distance_mm)
        self.var_confidence.set(cfg.confidence_threshold)
        self.var_min_area.set(cfg.min_area_px)
        self.var_prediction.set(cfg.prediction_frames)
        self.var_history.set(cfg.history_frames)
        self.var_depth_band.set(cfg.depth_band_mm)
        self.var_max_jump.set(cfg.max_jump_px)
        self.var_smoothing.set(cfg.smoothing_alpha)

    def apply_config(self) -> None:
        try:
            cfg = TrackerConfig(
                min_distance_mm=int(self.var_min_distance.get()),
                max_distance_mm=int(self.var_max_distance.get()),
                confidence_threshold=int(self.var_confidence.get()),
                min_area_px=int(self.var_min_area.get()),
                prediction_frames=int(self.var_prediction.get()),
                history_frames=int(self.var_history.get()),
                depth_band_mm=int(self.var_depth_band.get()),
                max_jump_px=int(self.var_max_jump.get()),
                smoothing_alpha=float(self.var_smoothing.get()),
                lost_frames_before_reacquire=self.get_config().lost_frames_before_reacquire,
                z_motion_deadband_mm_s=self.get_config().z_motion_deadband_mm_s,
            ).validated()
        except (ValueError, tk.TclError) as exc:
            messagebox.showerror("Ошибка параметров", str(exc))
            return

        self.set_config(cfg)
        self._load_config_to_widgets(cfg)
        self.worker.request_tracker_reset()

    def _poll_frames(self) -> None:
        if self.stop_event.is_set():
            return

        newest = None
        try:
            while True:
                newest = self.output_queue.get_nowait()
        except queue.Empty:
            pass

        if newest is not None:
            self._show_packet(newest)

        self.root.after(30, self._poll_frames)

    def _show_packet(self, packet: FramePacket) -> None:
        image = cv2.cvtColor(packet.image_bgr, cv2.COLOR_BGR2RGB)

        # Enlarge the small ToF image for comfortable viewing while keeping
        # nearest-neighbour-like sharpness.
        h, w = image.shape[:2]
        scale = min(3.0, max(1.0, 720.0 / max(w, h)))
        out_w = max(1, int(w * scale))
        out_h = max(1, int(h * scale))
        image = cv2.resize(
            image, (out_w, out_h), interpolation=cv2.INTER_NEAREST
        )

        pil = Image.fromarray(image)
        self._photo = ImageTk.PhotoImage(pil)
        self.preview_label.configure(image=self._photo, text="")

        self.status_camera.configure(
            text=f"{packet.message} | HW RANGE: {packet.camera_range_mm} мм"
        )
        self.status_fps.configure(text=f"FPS: {packet.fps:.1f}")

        r = packet.result
        if r.current is None:
            self.status_target.configure(text=f"Цель: {r.motion_state}")
            self.status_depth.configure(text="Z: —")
            self.status_velocity.configure(text="Vz: —")
            self.status_xy_velocity.configure(text="Vxy: —")
            self.status_prediction.configure(text="Прогноз: —")
            return

        self.status_target.configure(text=f"Цель: {r.motion_state}")
        self.status_depth.configure(text=f"Z: {r.current.z_mm:.0f} мм")
        self.status_velocity.configure(
            text=f"Vz: {r.vz_mm_s:+.0f} мм/с ({r.vz_mm_s / 1000.0:+.3f} м/с)"
        )
        self.status_xy_velocity.configure(
            text=f"Vx/Vy: {r.vx_px_s:+.1f} / {r.vy_px_s:+.1f} px/s"
        )

        if r.predicted is not None:
            n = self.get_config().prediction_frames
            self.status_prediction.configure(
                text=(
                    f"Через {n} кадр.: "
                    f"x={r.predicted.x_px:.1f}, "
                    f"y={r.predicted.y_px:.1f}, "
                    f"z≈{r.predicted.z_mm:.0f} мм"
                )
            )

    def on_close(self) -> None:
        self.stop_event.set()
        self.root.after(50, self._finish_close)

    def _finish_close(self) -> None:
        if self.worker.is_alive():
            self.worker.join(timeout=0.05)
            if self.worker.is_alive():
                self.root.after(50, self._finish_close)
                return
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = ToFTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
