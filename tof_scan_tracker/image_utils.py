"""Фильтрация и отрисовка глубины для сканирования и слежения."""
import cv2
import numpy as np


def valid_mask(depth, confidence, threshold, minimum, maximum):
    return (np.isfinite(depth) & np.isfinite(confidence) & (depth > 0)
             & (confidence >= threshold) & (depth >= minimum) & (depth <= maximum))


def annotate_fragment(image, fragment):
    """Контур на исходной картинке, подпись на отдельной полосе снизу."""
    if fragment is None:
        lines = ("No fragment", "Check thresholds / min-area")
    else:
        contours, _ = cv2.findContours(fragment["mask"].astype(np.uint8),
                                      cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (255, 255, 255), 1)
        lines = (f"Area: {fragment['area']} px | D: {fragment['distance']:.1f} mm",
                 f"Confidence: {fragment['confidence']:.1f} (median)")
    # Стандартные шрифты OpenCV не поддерживают кириллицу.
    output = cv2.copyMakeBorder(image, 0, 44, 0, max(0, 320 - image.shape[1]),
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    for index, line in enumerate(lines):
        cv2.putText(output, line, (5, image.shape[0] + 16 + 19 * index),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
    return output


def render_depth(depth, confidence, threshold, minimum, maximum):
    """Отсечение и цветное BGR-изображение; исходные массивы не меняются."""
    valid = valid_mask(depth, confidence, threshold, minimum, maximum)
    # Цветовая шкала фиксирована по аргументам, не меняется между кадрами.
    scaled = (np.where(valid, depth, minimum) - minimum) * (255.0 / (maximum - minimum))
    gray = np.clip(scaled, 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image
