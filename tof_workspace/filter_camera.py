"""Живая глубина Arducam ToF с отсечением по confidence и расстоянию.

Запуск на Raspberry Pi: python tof_workspace/filter_camera.py
Параметры: --index 0 --range 4000 (или 2000).
Нужны ArducamDepthCamera, numpy, opencv-python, Pillow и tkinter.
Остановите другие программы, использующие эту камеру, перед запуском.
Файлы не записываются. Для повторного подключения перезапустите скрипт.
"""
import argparse
import math
from threading import Event, Lock, Thread
from time import monotonic
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from filter_images import depth_mask


class CameraStream:
    """SDK используется только в рабочем потоке; хранится один последний кадр."""
    def __init__(self, index=0, range_mm=4000, sdk=None):
        self.index, self.range_mm, self.sdk = index, range_mm, sdk
        self.stop_event = Event()
        self.lock = Lock()
        self.frame = None
        self.error = None
        self.thread = Thread(target=self.run, name="tof-capture", daemon=True)

    def snapshot(self):
        with self.lock:
            return self.frame, self.error

    def report_error(self, message):
        with self.lock:
            self.error = f"{self.error}; {message}" if self.error else message

    def run(self):
        camera = None
        opened = started = False
        try:
            ac = self.sdk
            if ac is None:
                import ArducamDepthCamera as ac
            camera = ac.ArducamCamera()
            result = camera.open(ac.Connection.CSI, self.index)
            if result != 0:
                raise RuntimeError(f"Не удалось открыть камеру: {result}")
            opened = True
            result = camera.start(ac.FrameType.DEPTH)
            if result != 0:
                raise RuntimeError(f"Не удалось запустить камеру: {result}")
            started = True
            result = camera.setControl(ac.Control.RANGE, self.range_mm)
            if result not in (None, 0):
                raise RuntimeError(f"Не удалось установить диапазон: {result}")
            actual = camera.getControl(ac.Control.RANGE)
            if actual != self.range_mm:
                raise RuntimeError(f"Камера вернула диапазон {actual} вместо {self.range_mm} мм")
            sequence = 0
            last_frame = monotonic()
            while not self.stop_event.is_set():
                frame = camera.requestFrame(200)
                if frame is None:
                    if monotonic() - last_frame > 5:
                        raise TimeoutError("Нет кадров камеры более 5 секунд")
                    self.stop_event.wait(.005)
                    continue
                try:
                    if not isinstance(frame, ac.DepthData):
                        raise RuntimeError("SDK вернул кадр без глубины")
                    depth_data = frame.depth_data
                    confidence_data = getattr(frame, "confidence_data", None)
                    if depth_data is None or confidence_data is None:
                        raise RuntimeError("Камера не предоставила depth и confidence")
                    # Обязательно копируем данные ДО освобождения буфера SDK.
                    depth = np.array(depth_data, copy=True)
                    confidence = np.array(confidence_data, copy=True)
                    if (depth.ndim != 2 or not depth.size or confidence.shape != depth.shape
                            or depth.dtype.kind not in "uif" or confidence.dtype.kind not in "uif"):
                        raise ValueError("Некорректные массивы depth/confidence")
                    depth.setflags(write=False)
                    confidence.setflags(write=False)
                    sequence += 1
                    last_frame = monotonic()
                    with self.lock:
                        self.frame = (sequence, last_frame, depth, confidence)
                finally:
                    camera.releaseFrame(frame)
        except Exception as exc:
            self.report_error(f"{type(exc).__name__}: {exc}")
        finally:
            for method, needed in (("stop", started), ("close", opened)):
                if needed:
                    try:
                        getattr(camera, method)()
                    except Exception as exc:
                        self.report_error(f"{method}: {exc}")


def make_images(depth, confidence, threshold, minimum, maximum, range_mm):
    valid = np.isfinite(depth) & (depth > 0)
    values = np.where(valid, depth, 0).astype(np.float64)
    gray = (np.clip(values / range_mm, 0, 1) * 255).astype(np.uint8)
    original = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    original[~valid] = 0
    mask = depth_mask(depth, confidence, threshold, minimum, maximum)
    filtered = original.copy()
    filtered[~mask] = 0
    return original, filtered, int(mask.sum())


class LiveViewer:
    def __init__(self, root, stream):
        self.root, self.stream = root, stream
        self.closing = False
        self.last_key = None
        self.sources = [None, None]
        self.photos = [None, None]
        self.canvases, self.items = [], []
        self.kept = 0
        self.conf_high = 6000  # Только начальный предел слайдера, не предел SDK.
        root.title("Arducam ToF — фильтрация в реальном времени")
        root.geometry("1100x720")
        root.minsize(700, 520)
        root.rowconfigure(0, weight=1)
        for column, title in enumerate(("Исходная глубина", "После отсечения")):
            root.columnconfigure(column, weight=1, uniform="images")
            panel = ttk.Frame(root, padding=10)
            panel.grid(row=0, column=column, sticky="nsew")
            panel.columnconfigure(0, weight=1)
            panel.rowconfigure(1, weight=1)
            ttk.Label(panel, text=title, font=("TkDefaultFont", 15, "bold")).grid(row=0, column=0)
            canvas = tk.Canvas(panel, background="#181818", highlightthickness=0, width=240, height=180)
            canvas.grid(row=1, column=0, sticky="nsew", pady=8)
            self.canvases.append(canvas)
            self.items.append(canvas.create_image(0, 0, anchor="center"))
            canvas.bind("<Configure>", lambda _event, index=column: self.draw(index))
        controls = ttk.Frame(root, padding=12)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure(1, weight=1)
        self.variables, self.sliders = {}, {}
        for row, (name, title, limit, initial) in enumerate((
                ("confidence", "Минимум confidence (единицы SDK)", self.conf_high, 0),
                ("minimum", "Минимум глубины, мм", stream.range_mm, 0),
                ("maximum", "Максимум глубины, мм", stream.range_mm, stream.range_mm))):
            ttk.Label(controls, text=title).grid(row=row, column=0, sticky="w", padx=(0, 15))
            self.variables[name] = tk.IntVar(value=initial)
            slider = tk.Scale(controls, from_=0, to=limit, resolution=1, orient="horizontal",
                              variable=self.variables[name], highlightthickness=0,
                              command=lambda _value, key=name: self.changed(key))
            slider.grid(row=row, column=1, sticky="ew")
            self.sliders[name] = slider
        self.status = tk.StringVar(value="Подключение камеры…")
        ttk.Label(controls, textvariable=self.status, wraplength=1000).grid(
            row=3, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(controls, text=f"Фиксированная цветовая шкала: 0–{stream.range_mm} мм, синий → красный.\n"
                  "Чёрный — отсечённые или недостоверные точки. Границы включены.\n"
                  "Предел слайдера confidence автоматически растёт по данным камеры.").grid(
                      row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(controls, text="Сбросить пороги", command=self.reset).grid(row=5, column=0, sticky="w", pady=10)
        ttk.Button(controls, text="Закрыть", command=self.close).grid(row=5, column=1, sticky="e")
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Escape>", lambda _event: self.close())
        stream.thread.start()
        self.after_id = root.after(15, self.tick)

    def changed(self, key):
        if "maximum" not in self.variables:
            return
        minimum, maximum = self.variables["minimum"], self.variables["maximum"]
        if minimum.get() > maximum.get():
            if key == "minimum":
                maximum.set(minimum.get())
            else:
                minimum.set(maximum.get())

    def reset(self):
        for name, value in (("confidence", 0), ("minimum", 0), ("maximum", self.stream.range_mm)):
            self.variables[name].set(value)

    def tick(self):
        if self.closing:
            return
        try:
            frame, error = self.stream.snapshot()
            if frame is not None:
                sequence, timestamp, depth, confidence = frame
                settings = tuple(self.variables[name].get() for name in ("confidence", "minimum", "maximum"))
                key = (sequence, settings)
                if key != self.last_key:
                    finite = confidence[np.isfinite(confidence)]
                    peak = float(finite.max()) if finite.size else 0
                    if peak > self.conf_high:
                        self.conf_high = math.ceil(peak / 1000) * 1000
                        self.sliders["confidence"].configure(to=self.conf_high)
                    original, filtered, self.kept = make_images(depth, confidence, *settings, self.stream.range_mm)
                    self.sources = [Image.fromarray(original), Image.fromarray(filtered)]
                    self.draw(0)
                    self.draw(1)
                    self.last_key = key
                age = monotonic() - timestamp
                self.status.set(f"{depth.shape[1]}×{depth.shape[0]} • кадр {sequence} • "
                                f"осталось {self.kept:,} из {depth.size:,} ({self.kept / depth.size:.1%}) • "
                                f"возраст кадра {age * 1000:.0f} мс"
                                + (" — ожидание новых кадров…" if age > 1 else ""))
            if error:
                self.status.set(f"Ошибка камеры: {error}. Для подключения перезапустите скрипт.")
        except Exception as exc:
            self.stream.stop_event.set()
            self.status.set(f"Ошибка отображения: {exc}. Закройте окно и перезапустите скрипт.")
            return
        self.after_id = self.root.after(15, self.tick)

    def draw(self, index):
        source = self.sources[index]
        if source is None:
            return
        canvas = self.canvases[index]
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        scale = min(width / source.width, height / source.height)
        size = max(1, int(source.width * scale)), max(1, int(source.height * scale))
        self.photos[index] = ImageTk.PhotoImage(source.resize(size, Image.Resampling.NEAREST))
        canvas.itemconfigure(self.items[index], image=self.photos[index])
        canvas.coords(self.items[index], width / 2, height / 2)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.root.after_cancel(self.after_id)
        self.stream.stop_event.set()
        self.status.set("Остановка и освобождение камеры…")
        self.wait_for_camera()

    def wait_for_camera(self):
        # Не блокируем Tkinter через join: SDK сам завершает stop/close.
        # Если нативный SDK зависнет, принудительно прервать его поток нельзя.
        if self.stream.thread.is_alive():
            self.root.after(50, self.wait_for_camera)
        else:
            self.stream.thread.join()
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Живая фильтрация Arducam ToF")
    parser.add_argument("--index", type=int, default=0, help="Индекс камеры в SDK")
    parser.add_argument("--range", type=int, choices=(2000, 4000), default=4000, dest="range_mm")
    args = parser.parse_args()
    root = tk.Tk()
    stream = CameraStream(args.index, args.range_mm)
    try:
        LiveViewer(root, stream)
        root.mainloop()
    finally:
        stream.stop_event.set()
        if stream.thread.ident is not None:
            stream.thread.join(timeout=3)


if __name__ == "__main__":
    main()
