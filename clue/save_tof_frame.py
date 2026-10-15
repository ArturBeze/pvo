"""Сохранить один кадр Arducam ToF без изменения исходных значений.

Запуск: python clue/save_tof_frame.py
Результат: clue/data/depth.npy и clue/data/confidence.npy.
Повторный запуск заменяет предыдущую запись.

Чтение без камеры:
    depth = np.load("clue/data/depth.npy", allow_pickle=False)
    confidence = np.load("clue/data/confidence.npy", allow_pickle=False)

Глубина — в мм, confidence — в исходных единицах SDK, не в процентах.
NPY сохраняет форму, тип и значения массива, включая NaN/Inf, без
нормализации, фильтрации и преобразования в 8-битную картинку.
"""
from pathlib import Path
import sys
from time import monotonic

import numpy as np


CAMERA_INDEX = 0
MAX_DISTANCE = 4000  # 2000 или 4000, как в time-of-flight.py
DATA_DIR = Path(__file__).resolve().parent / "data"


def capture_frame():
    # Отложенный импорт: читать сохранённые файлы можно без SDK камеры.
    import ArducamDepthCamera as ac

    camera = ac.ArducamCamera()
    opened = started = False
    try:
        result = camera.open(ac.Connection.CSI, CAMERA_INDEX)
        if result != 0:
            raise RuntimeError(f"Не удалось открыть камеру: {result}")
        opened = True
        result = camera.start(ac.FrameType.DEPTH)
        if result != 0:
            raise RuntimeError(f"Не удалось запустить камеру: {result}")
        started = True
        result = camera.setControl(ac.Control.RANGE, MAX_DISTANCE)
        if result not in (None, 0):
            raise RuntimeError(f"Не удалось установить диапазон: {result}")

        deadline = monotonic() + 10
        while monotonic() < deadline:
            frame = camera.requestFrame(2000)
            if frame is None:
                continue
            try:
                if not isinstance(frame, ac.DepthData):
                    continue
                depth_data = frame.depth_data
                confidence_data = getattr(frame, "confidence_data", None)
                if depth_data is None or confidence_data is None:
                    raise RuntimeError("Камера не предоставила depth и confidence")
                # Буферы принадлежат SDK: копируем до releaseFrame.
                # Не задаём dtype, чтобы сохранить исходную точность.
                depth = np.array(depth_data, copy=True)
                confidence = np.array(confidence_data, copy=True)
                if (depth.ndim != 2 or not depth.size
                        or confidence.shape != depth.shape
                        or not np.issubdtype(depth.dtype, np.number)
                        or not np.issubdtype(confidence.dtype, np.number)):
                    raise RuntimeError("Некорректные массивы depth/confidence")
                return depth, confidence
            finally:
                camera.releaseFrame(frame)
        raise TimeoutError("Камера не вернула кадр глубины за 10 секунд")
    finally:
        try:
            if started:
                camera.stop()
        finally:
            if opened:
                camera.close()


def main():
    depth, confidence = capture_frame()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Оба массива получены из одного кадра. Никаких цветовых карт и фильтров.
    for name, array in (("depth", depth), ("confidence", confidence)):
        path = DATA_DIR / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        print(f"Сохранено: {path} — {array.shape}, {array.dtype}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Ошибка записи кадра: {exc}", file=sys.stderr)
        sys.exit(1)
