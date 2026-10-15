"""Pure processing functions; raw camera arrays are never modified."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class FilterSettings:
    confidence: float = 30
    min_mm: float = 100
    max_mm: float = 4000
    smoothing: str = "Нет"
    kernel: int = 3
    sigma_mm: float = 75
    confidence_scale: float = 255


def filter_depth(frame, settings):
    depth = frame.depth
    valid = (np.isfinite(depth) & (depth > 0) &
             (depth >= settings.min_mm) & (depth <= settings.max_mm))
    if frame.confidence is not None:
        valid &= np.isfinite(frame.confidence) & (frame.confidence >= settings.confidence)
    clean = np.where(valid, depth, 0).astype(np.float32)
    k = settings.kernel
    if k not in (3, 5):
        raise ValueError("Kernel must be 3 or 5")
    if settings.smoothing == "Гаусс":
        weights = cv2.GaussianBlur(valid.astype(np.float32), (k, k), 0)
        values = cv2.GaussianBlur(clean, (k, k), 0)
        clean = np.divide(values, weights, out=np.zeros_like(values), where=weights > 1e-6)
    elif settings.smoothing in ("Медианный", "Билатеральный"):
        if settings.smoothing == "Медианный":
            smooth = cv2.medianBlur(clean, k)
        else:
            smooth = cv2.bilateralFilter(clean, k, settings.sigma_mm, k)
        # Do not let invalid zero samples pull neighbours toward the camera.
        interior = cv2.erode(valid.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        clean[interior] = smooth[interior]
    clean[~valid] = np.nan
    return clean


def render(frame, settings, mode, range_mm):
    if mode == "Confidence":
        if frame.confidence is None:
            return np.zeros((*frame.depth.shape, 3), dtype=np.uint8)
        values = np.nan_to_num(frame.confidence, nan=0, posinf=0, neginf=0)
        gray = np.clip(values * 255 / settings.confidence_scale, 0, 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    depth = filter_depth(frame, settings) if mode == "Глубина: фильтр" else frame.depth
    valid = np.isfinite(depth) & (depth > 0)
    values = np.where(valid, depth, 0)
    gray = np.clip(values * 255 / range_mm, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb
