"""Живая фильтрация Arducam ToF через OpenCV, без tkinter.

python tof_opencv/view_camera.py --confidence 200 --min 300 --max 1200
Глубина задаётся в мм, confidence — в исходных единицах SDK.
Выход: Q (латинская раскладка), Esc или закрытие окна.
"""
import argparse
import math
import sys
from time import monotonic

import cv2
import numpy as np


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confidence", type=float, default=200,
                        help="Минимальный confidence, по умолчанию 200")
    parser.add_argument("--min", type=float, default=300, dest="min_depth",
                        help="Минимальная глубина в мм, по умолчанию 300")
    parser.add_argument("--max", type=float, default=1200, dest="max_depth",
                        help="Максимальная глубина в мм, по умолчанию 1200")
    parser.add_argument("--range", type=int, choices=(2000, 4000), default=4000,
                        dest="range_mm", help="Аппаратный диапазон камеры, по умолчанию 4000 мм")
    parser.add_argument("--index", type=int, default=0, help="Индекс камеры в SDK")
    args = parser.parse_args(argv)
    if not all(math.isfinite(v) for v in (args.confidence, args.min_depth, args.max_depth)):
        parser.error("Пороги должны быть конечными числами")
    if args.confidence < 0:
        parser.error("confidence должен быть неотрицательным")
    if not 0 <= args.min_depth < args.max_depth <= args.range_mm:
        parser.error("Требуется 0 <= min < max <= range (в миллиметрах)")
    if args.index < 0:
        parser.error("Индекс камеры должен быть неотрицательным")
    return args


def valid_mask(depth, confidence, threshold, minimum, maximum):
    return (np.isfinite(depth) & np.isfinite(confidence) & (depth > 0)
             & (confidence >= threshold) & (depth >= minimum) & (depth <= maximum))


def nearest_point(depth, confidence, threshold, minimum, maximum):
    """Расстояние и confidence одной ближайшей валидной точки или None."""
    valid = valid_mask(depth, confidence, threshold, minimum, maximum)
    if not valid.any():
        return None
    index = np.argmin(np.where(valid, depth, np.inf))
    # Python-числа не зависят от памяти SDK после releaseFrame.
    return float(depth.flat[index]), float(confidence.flat[index])


def render_depth(depth, confidence, threshold, minimum, maximum):
    """Отсечение и цветное BGR-изображение; исходные массивы не меняются."""
    valid = valid_mask(depth, confidence, threshold, minimum, maximum)
    # Цветовая шкала фиксирована по аргументам, не меняется между кадрами.
    scaled = (np.where(valid, depth, minimum) - minimum) * (255.0 / (maximum - minimum))
    gray = np.clip(scaled, 0, 255).astype(np.uint8)
    image = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    image[~valid] = 0
    return image


def run(args, sdk=None):
    if sdk is None:
        import ArducamDepthCamera as sdk

    camera = sdk.ArducamCamera()
    opened = started = window_created = False
    window = "Arducam ToF | Q / Esc: exit"
    try:
        print("Открытие камеры…", flush=True)
        result = camera.open(sdk.Connection.CSI, args.index)
        if result != 0:
            raise RuntimeError(f"Не удалось открыть камеру: {result}")
        opened = True
        result = camera.start(sdk.FrameType.DEPTH)
        if result != 0:
            raise RuntimeError(f"Не удалось запустить камеру: {result}")
        started = True
        result = camera.setControl(sdk.Control.RANGE, args.range_mm)
        if result not in (None, 0):
            raise RuntimeError(f"Не удалось установить диапазон: {result}")
        actual_range = camera.getControl(sdk.Control.RANGE)
        if actual_range != args.range_mm:
            raise RuntimeError(f"Камера вернула диапазон {actual_range} вместо {args.range_mm}")

        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        window_created = True
        cv2.resizeWindow(window, 960, 720)
        cv2.imshow(window, np.zeros((180, 240, 3), np.uint8))
        cv2.waitKey(1)
        print(f"confidence >= {args.confidence:g}; глубина {args.min_depth:g}–{args.max_depth:g} мм. "
              "Выход: Q или Esc.", flush=True)
        last_received = monotonic()
        fps_start = last_received
        last_report = None
        count = 0
        while True:
            # Один последовательный цикл без очереди кадров, процессов и
            # передачи массивов. Обработка выполняется векторно в NumPy/OpenCV.
            frame = camera.requestFrame(100)
            if frame is not None:
                try:
                    if not isinstance(frame, sdk.DepthData):
                        raise RuntimeError("SDK вернул кадр без глубины")
                    depth = frame.depth_data
                    confidence = getattr(frame, "confidence_data", None)
                    if depth is None or confidence is None:
                        raise RuntimeError("Камера не предоставила depth и confidence")
                    if (depth.ndim != 2 or not depth.size or confidence.shape != depth.shape
                            or depth.dtype.kind not in "uif" or confidence.dtype.kind not in "uif"):
                        raise ValueError("Некорректные массивы depth/confidence")
                    # Все обращения к буферам SDK — до releaseFrame.
                    # Итоговое цветное изображение имеет собственную память.
                    image = render_depth(depth, confidence, args.confidence,
                                         args.min_depth, args.max_depth)
                    report_due = last_report is None or monotonic() - last_report >= 1
                    if report_due:
                        nearest = nearest_point(depth, confidence, args.confidence,
                                                args.min_depth, args.max_depth)
                finally:
                    camera.releaseFrame(frame)
                last_received = monotonic()
                if report_due:
                    if nearest is None:
                        print("Ближайшая точка: нет точек, прошедших фильтры", flush=True)
                    else:
                        distance, point_confidence = nearest
                        print(f"Ближайшая точка: {distance:.1f} мм | confidence: {point_confidence:.2f}",
                              flush=True)
                    last_report = last_received
                cv2.imshow(window, image)
                count += 1
                if last_received - fps_start >= 2:
                    print(f"Отображение: {count / (last_received - fps_start):.1f} FPS", flush=True)
                    fps_start, count = last_received, 0
            elif monotonic() - last_received > 5:
                raise TimeoutError("Нет кадров камеры более 5 секунд")
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Независимые попытки: ошибка stop не должна помешать close.
        for method, needed in (("stop", started), ("close", opened)):
            if needed:
                try:
                    getattr(camera, method)()
                except Exception as exc:
                    print(f"Ошибка {method}: {exc}", file=sys.stderr)
        if window_created:
            cv2.destroyAllWindows()


def main():
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
