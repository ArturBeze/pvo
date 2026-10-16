"""Интерактивное отсечение глубины из clue/data без камеры.

Запуск: python tof_workspace/filter_images.py
Зависимости: numpy, opencv-python, Pillow, tkinter.
Файлы не изменяются. Пороги включительные; confidence — единицы SDK.
"""
from pathlib import Path
import math
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


DATA_DIR = Path(__file__).resolve().parent.parent / "clue" / "data"


def load_arrays():
    arrays = []
    for name in ("depth", "confidence"):
        data = np.load(DATA_DIR / f"{name}.npy", allow_pickle=False)
        if data.ndim != 2 or not data.size or data.dtype.kind not in "uif":
            raise ValueError(f"{name}: ожидается непустой двумерный числовой массив")
        arrays.append(data)
    if arrays[0].shape != arrays[1].shape:
        raise ValueError("Размеры depth и confidence не совпадают")
    return arrays


def depth_mask(depth, confidence, threshold, minimum, maximum):
    """True — оставить точку. Значения на границах сохраняются."""
    return (np.isfinite(depth) & (depth > 0) & np.isfinite(confidence)
            & (confidence >= threshold) & (depth >= minimum) & (depth <= maximum))


def colorize(depth):
    """Одна цветовая шкала для исходного и отфильтрованного изображений."""
    valid = np.isfinite(depth) & (depth > 0)
    gray = np.zeros(depth.shape, np.uint8)
    low = high = 0.0
    if valid.any():
        values = depth[valid].astype(np.float64)
        low, high = float(values.min()), float(values.max())
        if high > low:
            gray[valid] = np.clip((values - low) / (high - low) * 255, 0, 255).astype(np.uint8)
        else:
            gray[valid] = 128
    rgb = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb, low, high


class FilterViewer:
    def __init__(self, root, depth, confidence):
        self.root, self.depth, self.confidence = root, depth, confidence
        self.rgb, low, high = colorize(depth)
        self.depth_limit = max(1, math.ceil(high))
        finite = confidence[np.isfinite(confidence)]
        self.conf_low = min(0, math.floor(float(finite.min()))) if finite.size else 0
        self.conf_high = max(self.conf_low + 1, math.ceil(float(finite.max()))) if finite.size else 1
        self.pending = None
        self.photos = [None, None]
        self.sources = [Image.fromarray(self.rgb), Image.fromarray(self.rgb)]
        self.canvases = []
        self.items = []
        root.title("Фильтрация сохранённой карты глубины")
        root.geometry("1100x720")
        root.minsize(700, 520)
        root.rowconfigure(0, weight=1)
        for col, title in enumerate(("Исходная глубина", "После отсечения")):
            root.columnconfigure(col, weight=1, uniform="images")
            panel = ttk.Frame(root, padding=10)
            panel.grid(row=0, column=col, sticky="nsew")
            panel.rowconfigure(1, weight=1)
            panel.columnconfigure(0, weight=1)
            ttk.Label(panel, text=title, font=("TkDefaultFont", 15, "bold")).grid(row=0, column=0)
            canvas = tk.Canvas(panel, background="#181818", highlightthickness=0, width=240, height=180)
            canvas.grid(row=1, column=0, sticky="nsew", pady=8)
            self.canvases.append(canvas)
            self.items.append(canvas.create_image(0, 0, anchor="center"))
            canvas.bind("<Configure>", lambda _event, index=col: self.draw(index))

        controls = ttk.Frame(root, padding=12)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew")
        controls.columnconfigure(1, weight=1)
        self.variables = {}
        settings = (("confidence", "Минимум confidence (единицы SDK)", self.conf_low, self.conf_high, self.conf_low),
                    ("minimum", "Минимум глубины, мм", 0, self.depth_limit, 0),
                    ("maximum", "Максимум глубины, мм", 0, self.depth_limit, self.depth_limit))
        for row, (name, title, start, end, initial) in enumerate(settings):
            ttk.Label(controls, text=title).grid(row=row, column=0, sticky="w", padx=(0, 15))
            var = tk.IntVar(value=initial)
            self.variables[name] = var
            tk.Scale(controls, from_=start, to=end, resolution=1, orient="horizontal",
                     variable=var, command=lambda _value, key=name: self.changed(key),
                     highlightthickness=0).grid(row=row, column=1, sticky="ew")
        self.status = tk.StringVar()
        ttk.Label(controls, textvariable=self.status).grid(row=3, column=0, columnspan=2, sticky="w", pady=8)
        ttk.Label(controls, text=f"Цветовая шкала фиксирована: {low:.1f}–{high:.1f} мм, синий → красный.\n"
                  "Чёрный — отсечённые или недостоверные точки. Значения на границах сохраняются.\n"
                  "Исходные файлы не изменяются.").grid(row=4, column=0, columnspan=2, sticky="w")
        ttk.Button(controls, text="Сбросить пороги", command=self.reset).grid(row=5, column=0, sticky="w", pady=(10, 0))
        root.bind("<Escape>", lambda _event: root.destroy())
        self.refresh()

    def changed(self, key):
        if not hasattr(self, "status"):
            return
        minimum, maximum = self.variables["minimum"], self.variables["maximum"]
        if minimum.get() > maximum.get():
            if key == "minimum":
                maximum.set(minimum.get())
            else:
                minimum.set(maximum.get())
        # Не пересчитываем каждый промежуточный шаг быстрого движения мыши.
        if self.pending is None:
            self.pending = self.root.after(16, self.refresh)

    def reset(self):
        self.variables["confidence"].set(self.conf_low)
        self.variables["minimum"].set(0)
        self.variables["maximum"].set(self.depth_limit)
        self.changed("minimum")

    def refresh(self):
        self.pending = None
        mask = depth_mask(self.depth, self.confidence, self.variables["confidence"].get(),
                          self.variables["minimum"].get(), self.variables["maximum"].get())
        filtered = self.rgb.copy()
        filtered[~mask] = 0
        self.sources[1] = Image.fromarray(filtered)
        count = int(mask.sum())
        self.status.set(f"Осталось точек: {count:,} из {mask.size:,} ({count / mask.size:.1%})")
        self.draw(1)

    def draw(self, index):
        canvas, source = self.canvases[index], self.sources[index]
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        scale = min(width / source.width, height / source.height)
        size = max(1, int(source.width * scale)), max(1, int(source.height * scale))
        self.photos[index] = ImageTk.PhotoImage(source.resize(size, Image.Resampling.NEAREST))
        canvas.itemconfigure(self.items[index], image=self.photos[index])
        canvas.coords(self.items[index], width / 2, height / 2)


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        depth, confidence = load_arrays()
    except Exception as exc:
        messagebox.showerror("Не удалось загрузить данные", f"{DATA_DIR}\n\n{exc}", parent=root)
        root.destroy()
        return
    FilterViewer(root, depth, confidence)
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
