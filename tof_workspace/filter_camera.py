"""Живая глубина Arducam ToF с отсечением по confidence и расстоянию.

Запуск на Raspberry Pi: python tof_workspace/filter_camera.py
Параметры: --index 0 --range 4000 (или 2000).
Нужны ArducamDepthCamera, numpy, opencv-python, Pillow и tkinter.
Остановите другие программы, использующие эту камеру, перед запуском.
Камера работает в отдельном процессе; --demo проверяет окно без камеры.
Файлы не записываются. Для повторного подключения перезапустите скрипт.
"""
import argparse
import math
import multiprocessing as mp
from queue import Empty, Full
import traceback
from time import monotonic
import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from filter_images import depth_mask


class CaptureWorker:
    """Все обращения к SDK выполняются внутри дочернего процесса."""
    def __init__(self, index, range_mm, stop_event, frames, messages, sdk=None):
        self.index, self.range_mm, self.sdk = index, range_mm, sdk
        self.stop_event, self.frames, self.messages = stop_event, frames, messages

    def report(self, text, kind="status"):
        print(f"[ToF] {text}", flush=True)
        self.messages.put((kind, text))

    def report_error(self, message):
        self.report(message, "error")

    def publish(self, frame):
        # Очередь ограничена двумя кадрами. При перегрузке удаляем старый.
        try:
            self.frames.put_nowait(frame)
        except Full:
            try:
                self.frames.get_nowait()
            except Empty:
                pass
            try:
                self.frames.put_nowait(frame)
            except Full:
                pass

    def run(self):
        camera = None
        opened = started = False
        try:
            self.report("Загрузка ArducamDepthCamera…")
            ac = self.sdk
            if ac is None:
                import ArducamDepthCamera as ac
            self.report("Создание объекта камеры…")
            camera = ac.ArducamCamera()
            self.report("Открытие камеры…")
            result = camera.open(ac.Connection.CSI, self.index)
            if result != 0:
                raise RuntimeError(f"Не удалось открыть камеру: {result}")
            opened = True
            self.report("Запуск потока глубины…")
            result = camera.start(ac.FrameType.DEPTH)
            if result != 0:
                raise RuntimeError(f"Не удалось запустить камеру: {result}")
            started = True
            self.report("Настройка диапазона…")
            result = camera.setControl(ac.Control.RANGE, self.range_mm)
            if result not in (None, 0):
                raise RuntimeError(f"Не удалось установить диапазон: {result}")
            actual = camera.getControl(ac.Control.RANGE)
            if actual != self.range_mm:
                raise RuntimeError(f"Камера вернула диапазон {actual} вместо {self.range_mm} мм")
            self.report("Ожидание первого кадра…")
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
                    self.publish((sequence, last_frame, depth, confidence))
                    if sequence == 1:
                        self.report("Получен первый кадр")
                finally:
                    camera.releaseFrame(frame)
        except Exception as exc:
            self.report_error(f"{type(exc).__name__}: {exc}")
        finally:
            for method, needed in (("stop", started), ("close", opened)):
                if needed:
                    try:
                        self.report(f"Освобождение камеры: {method}…")
                        getattr(camera, method)()
                    except Exception as exc:
                        self.report_error(f"{method}: {exc}")


def capture_process(index, range_mm, stop_event, frames, messages, demo):
    # Не ждём отправки кадров в уже закрытое окно при завершении процесса.
    frames.cancel_join_thread()
    worker = CaptureWorker(index, range_mm, stop_event, frames, messages)
    if not demo:
        worker.run()
        return
    worker.report("Демонстрация без камеры")
    y, x = np.mgrid[:180, :240]
    sequence = 0
    while not stop_event.is_set():
        sequence += 1
        depth = np.full(x.shape, range_mm * .7, dtype=np.float32)
        center = 120 + 60 * np.sin(sequence / 30)
        depth[(x - center)**2 + (y - 90)**2 < 35**2] = range_mm * .25
        confidence = np.full(x.shape, 500, dtype=np.float32)
        confidence[:, :60] = 20
        worker.publish((sequence, monotonic(), depth, confidence))
        stop_event.wait(1 / 30)


class CameraStream:
    """Интерфейс окна к камере; spawn не наследует состояние Tcl/Tk."""
    def __init__(self, index=0, range_mm=4000, demo=False):
        self.range_mm = range_mm
        context = mp.get_context("spawn")
        self.stop_event = context.Event()
        self.frames = context.Queue(maxsize=2)
        self.messages = context.Queue()
        self.process = context.Process(target=capture_process,
            args=(index, range_mm, self.stop_event, self.frames, self.messages, demo),
            name="tof-camera", daemon=True)
        self.frame = self.error = None
        self.status = "Окно готово. Запуск камеры…"
        self.started = False

    def start(self):
        self.process.start()
        self.started = True

    def snapshot(self):
        for _ in range(32):
            try:
                kind, message = self.messages.get_nowait()
            except Empty:
                break
            if kind == "error":
                self.error = message
            else:
                self.status = message
        for _ in range(2):
            try:
                self.frame = self.frames.get_nowait()
            except Empty:
                break
        if self.started and not self.process.is_alive() and not self.stop_event.is_set():
            self.error = self.error or f"Процесс камеры завершился (код {self.process.exitcode})"
        return self.frame, self.error

    def dispose(self):
        self.frames.close()
        self.messages.close()


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
        self.start_id = root.after(200, self.start_camera)
        self.after_id = root.after(15, self.tick)
        root.report_callback_exception = self.callback_error

    def start_camera(self):
        if self.closing:
            return
        try:
            self.stream.start()
        except Exception as exc:
            self.stream.error = f"Не удалось запустить процесс камеры: {exc}"
            print(self.stream.error, flush=True)

    def callback_error(self, exc_type, exc, tb):
        traceback.print_exception(exc_type, exc, tb)
        self.status.set(f"Ошибка интерфейса: {exc}. Подробности в консоли.")

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
            if frame is None:
                self.status.set(self.stream.status)
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
        self.root.after_cancel(self.start_id)
        self.close_started = monotonic()
        self.stream.stop_event.set()
        self.status.set("Остановка и освобождение камеры…")
        self.wait_for_camera()

    def wait_for_camera(self):
        process = self.stream.process
        if self.stream.started and process.is_alive():
            if monotonic() - self.close_started > 3:
                self.status.set("SDK не отвечает. Завершение процесса камеры…")
                if monotonic() - self.close_started > 5:
                    process.kill()
                else:
                    process.terminate()
            self.root.after(50, self.wait_for_camera)
        else:
            if self.stream.started:
                process.join()
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Живая фильтрация Arducam ToF")
    parser.add_argument("--index", type=int, default=0, help="Индекс камеры в SDK")
    parser.add_argument("--range", type=int, choices=(2000, 4000), default=4000, dest="range_mm")
    parser.add_argument("--demo", action="store_true", help="Проверить окно и обмен кадрами без SDK камеры")
    args = parser.parse_args()
    root = tk.Tk()
    stream = CameraStream(args.index, args.range_mm, args.demo)
    try:
        LiveViewer(root, stream)
        root.mainloop()
    finally:
        stream.stop_event.set()
        if stream.started:
            stream.process.join(timeout=3)
            if stream.process.is_alive():
                stream.process.terminate()
                stream.process.join(timeout=2)
            if stream.process.is_alive():
                stream.process.kill()
                stream.process.join(timeout=1)
        stream.dispose()


if __name__ == "__main__":
    mp.freeze_support()
    main()
