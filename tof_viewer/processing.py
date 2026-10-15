"""Обработка глубины в миллиметрах; NaN означает отсутствие оценки."""
from dataclasses import dataclass
import cv2
import numpy as np


@dataclass(frozen=True)
class Settings:
    confidence: float = 30
    fill_radius: int = 2
    edge_limit: float = 150
    smooth_radius: int = 2
    sigma_depth: float = 60
    alpha: float = 0.65


class Processor:
    def __init__(self):
        self.previous = None
        self.settings = None

    def process(self, depth, confidence, settings, max_depth):
        d = np.asarray(depth, dtype=np.float32)
        c = np.asarray(confidence, dtype=np.float32)
        if d.ndim != 2 or d.shape != c.shape:
            raise ValueError('Depth и confidence должны быть двумерными и одинакового размера')
        valid = np.isfinite(d) & (d > 0) & (d <= max_depth)
        valid &= np.isfinite(c) & (c >= settings.confidence)
        result = np.where(valid, d, 0).astype(np.float32)
        filled = np.zeros(d.shape, bool)
        r = settings.fill_radius
        if r and valid.any():
            kernel = np.ones((2*r+1, 2*r+1), np.uint8)
            count = cv2.boxFilter(valid.astype(np.float32), -1, kernel.shape,
                                  normalize=False, borderType=cv2.BORDER_CONSTANT)
            total = cv2.boxFilter(result, -1, kernel.shape,
                                  normalize=False, borderType=cv2.BORDER_CONSTANT)
            low = cv2.erode(np.where(valid, d, np.inf), kernel,
                            borderType=cv2.BORDER_CONSTANT, borderValue=float('inf'))
            high = cv2.dilate(np.where(valid, d, -np.inf), kernel,
                              borderType=cv2.BORDER_CONSTANT, borderValue=float('-inf'))
            # Не заполняем границы кадра, большие пробелы и переходы глубины.
            filled = ~valid & (count >= max(3, kernel.size * 0.4))
            filled &= (high - low <= settings.edge_limit)
            filled[:r] = False
            filled[-r:] = False
            filled[:, :r] = False
            filled[:, -r:] = False
            result[filled] = (total / np.maximum(count, 1))[filled]
        known = valid | filled
        r = settings.smooth_radius
        if r:
            # Маскированный bilateral: неизвестные пиксели не вносят нулей.
            padded = np.pad(result, r, mode='constant')
            mask = np.pad(known, r, mode='constant')
            numerator = np.zeros_like(result)
            denominator = np.zeros_like(result)
            h, w = d.shape
            for dy in range(-r, r+1):
                for dx in range(-r, r+1):
                    neighbour = padded[r+dy:r+dy+h, r+dx:r+dx+w]
                    present = mask[r+dy:r+dy+h, r+dx:r+dx+w]
                    weight = np.exp(-0.5*((neighbour-result)/settings.sigma_depth)**2
                                    -0.5*(dx*dx+dy*dy)/max(r*r, 1)) * present
                    numerator += weight * neighbour
                    denominator += weight
            result = numerator / np.maximum(denominator, 1e-12)
        result[~known] = np.nan
        if self.settings == settings and self.previous is not None and self.previous.shape == d.shape:
            stable = known & np.isfinite(self.previous)
            stable &= np.abs(result-self.previous) <= settings.edge_limit
            result[stable] = (settings.alpha*result[stable]
                              + (1-settings.alpha)*self.previous[stable])
        self.previous = result.copy()
        self.settings = settings
        return result, valid, filled
