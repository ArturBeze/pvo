"""Run: python tof_workspace/app.py [--demo]."""
import argparse
from dataclasses import replace
from threading import Event, Lock, Thread
from time import monotonic
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from camera import ToFCamera
from processing import FilterSettings, render


class Viewer:
    def __init__(self, root, camera):
        self.root, self.camera = root, camera
        self.stop_event = Event()
        self.lock = Lock()
        self.settings = FilterSettings(max_mm=camera.range_mm)
        self.mode = "Глубина: фильтр"
        self.result = None
        self.error = None
        self.last_key = None
        self.last_display = monotonic()
        self.closing = False
        root.title("Arducam ToF" + (" — ДЕМОНСТРАЦИЯ" if camera.demo else ""))
        root.geometry("1050x700")
        root.minsize(800, 540)
        controls = ttk.Frame(root, padding=12)
        controls.pack(side="right", fill="y")
        self.image_label = ttk.Label(root, anchor="center")
        self.image_label.pack(fill="both", expand=True)
        self.status = tk.StringVar(value="Запуск камеры…")
        ttk.Label(controls, text="Изображение").pack(anchor="w")
        self.mode_var = tk.StringVar(value=self.mode)
        combo = ttk.Combobox(controls, textvariable=self.mode_var, state="readonly",
                             values=("Глубина: фильтр", "Глубина: исходная", "Confidence"))
        combo.pack(fill="x", pady=(0, 10))
        combo.bind("<<ComboboxSelected>>", self.update_settings)
        self.variables = {}
        self.slider(controls, "Порог confidence (единицы SDK)", "confidence", 0, 255, 30)
        self.slider(controls, "Минимум глубины, мм", "min_mm", 0, camera.range_mm, 100)
        self.slider(controls, "Максимум глубины, мм", "max_mm", 0, camera.range_mm, camera.range_mm)
        ttk.Label(controls, text="Сглаживание глубины").pack(anchor="w")
        self.smoothing = tk.StringVar(value="Нет")
        smooth = ttk.Combobox(controls, textvariable=self.smoothing, state="readonly",
                              values=("Нет", "Медианный", "Гаусс", "Билатеральный"))
        smooth.pack(fill="x")
        smooth.bind("<<ComboboxSelected>>", self.update_settings)
        ttk.Label(controls, text="Размер окна").pack(anchor="w", pady=(8, 0))
        self.kernel = tk.StringVar(value="3")
        kernel = ttk.Combobox(controls, textvariable=self.kernel, state="readonly", values=("3", "5"))
        kernel.pack(fill="x")
        kernel.bind("<<ComboboxSelected>>", self.update_settings)
        self.slider(controls, "Билатеральный: sigma, мм", "sigma_mm", 1, 300, 75)
        self.slider(controls, "Белый уровень confidence", "confidence_scale", 1, 1024, 255)
        ttk.Label(controls, text="Фильтры применяются только к\nрежиму «Глубина: фильтр».\nЧёрный — нет достоверной глубины.\nГлубина: синий → красный = ближе → дальше.",
                  wraplength=270).pack(anchor="w", pady=12)
        ttk.Label(controls, textvariable=self.status, wraplength=270).pack(anchor="w", pady=10)
        ttk.Button(controls, text="Закрыть", command=self.close).pack(fill="x")
        root.protocol("WM_DELETE_WINDOW", self.close)
        camera.start()
        self.worker = Thread(target=self.process, name="tof-processing", daemon=True)
        self.worker.start()
        root.after(15, self.tick)

    def slider(self, parent, title, name, low, high, initial):
        value = tk.DoubleVar(value=initial)
        self.variables[name] = value
        label = ttk.Label(parent, text=f"{title}: {initial}")
        label.pack(anchor="w", pady=(8, 0))

        def changed(_):
            label.configure(text=f"{title}: {value.get():.0f}")
            self.update_settings()

        ttk.Scale(parent, from_=low, to=high, variable=value, command=changed).pack(fill="x")

    def update_settings(self, _event=None):
        if not hasattr(self, "kernel"):
            return
        values = {name: value.get() for name, value in self.variables.items()}
        if values["min_mm"] >= values["max_mm"]:
            self.status.set("Минимум должен быть меньше максимума. Сохранены предыдущие настройки.")
            return
        with self.lock:
            self.settings = replace(self.settings, **values, smoothing=self.smoothing.get(),
                                    kernel=int(self.kernel.get()))
            self.mode = self.mode_var.get()

    def process(self):
        previous = None
        try:
            while not self.stop_event.is_set():
                frame = self.camera.latest()
                with self.lock:
                    settings, mode = self.settings, self.mode
                if frame is None:
                    self.stop_event.wait(0.01)
                    continue
                key = (frame.sequence, settings, mode)
                if key == previous:
                    self.stop_event.wait(0.005)
                    continue
                rgb = render(frame, settings, mode, self.camera.range_mm)
                with self.lock:
                    self.result = (key, frame, rgb)
                previous = key
        except Exception as exc:
            with self.lock:
                self.error = f"Ошибка обработки: {exc}"

    def tick(self):
        if self.closing:
            return
        with self.lock:
            result, error = self.result, self.error
        if result is not None:
            key, frame, rgb = result
            size = (max(1, self.image_label.winfo_width()), max(1, self.image_label.winfo_height()))
            display_key = (key, size)
            if display_key != self.last_key:
                image = Image.fromarray(rgb)
                image.thumbnail(size, Image.Resampling.NEAREST)
                self.photo = ImageTk.PhotoImage(image)
                self.image_label.configure(image=self.photo)
                self.last_key = display_key
            age = monotonic() - frame.timestamp
            confidence = "" if frame.confidence is not None else "\nConfidence недоступен: порог не применяется."
            self.status.set(f"Кадр {frame.sequence} • {rgb.shape[1]}×{rgb.shape[0]}\n"
                            f"Возраст кадра: {age * 1000:.0f} мс\nДиапазон: 0–{self.camera.range_mm} мм"
                            + confidence + ("\nОжидание новых кадров…" if age > 1 else ""))
        if self.camera.error or error:
            self.status.set(self.camera.error or error)
        self.root.after(15, self.tick)

    def close(self):
        if self.closing:
            return
        self.closing = True
        self.stop_event.set()
        self.camera.stop()
        self.close_started = monotonic()
        self.status.set("Остановка камеры…")
        self.wait_for_close()

    def wait_for_close(self):
        if self.camera.alive or self.worker.is_alive():
            if monotonic() - self.close_started > 3:
                self.status.set("SDK ещё завершает работу. Ожидание освобождения камеры…")
            self.root.after(50, self.wait_for_close)
            return
        self.camera.join()
        self.worker.join()
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Arducam ToF viewer")
    parser.add_argument("--demo", action="store_true", help="Synthetic frames without hardware")
    parser.add_argument("--index", type=int, default=0, help="SDK camera index")
    parser.add_argument("--range", type=int, choices=(2000, 4000), default=4000, dest="range_mm")
    args = parser.parse_args()
    root = tk.Tk()
    Viewer(root, ToFCamera(index=args.index, range_mm=args.range_mm, demo=args.demo))
    root.mainloop()


if __name__ == "__main__":
    main()
