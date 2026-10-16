"""Показать clue/data/depth.npy и confidence.npy без подключения камеры.

Запуск из любой директории: python /путь/к/tof_workspace/view_images.py
Зависимости: numpy, opencv-python, Pillow и tkinter.
Исходные файлы не изменяются; нормализация нужна только для отображения.
"""
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk


DATA_DIR = Path(__file__).resolve().parent.parent / "clue" / "data"


def load_image(path, *, depth=False):
    """Вернуть RGB-картинку для просмотра и описание исходных данных."""
    data = np.load(path, allow_pickle=False)
    # print(data.shape)
    if data.ndim != 2 or not data.size or data.dtype.kind not in "uif":
        raise ValueError(f"{path.name}: ожидается непустой двумерный числовой массив")

    valid = np.isfinite(data)
    if depth:
        valid &= data > 0
    gray = np.zeros(data.shape, dtype=np.uint8)
    if valid.any():
        values = data[valid].astype(np.float64)
        low, high = values.min(), values.max()
        if high > low:
            gray[valid] = np.clip((values - low) / (high - low) * 255, 0, 255).astype(np.uint8)
        else:
            gray[valid] = 128
        units = " мм" if depth else " (единицы SDK)"
        description = f"Диапазон: {low:.2f}–{high:.2f}{units}"
    else:
        description = "Нет корректных значений"

    if depth:
        rgb = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    rgb[~valid] = 0
    description += f"\n{data.shape[1]} × {data.shape[0]} • {data.dtype}"
    return Image.fromarray(rgb), description


def main():
    root = tk.Tk()
    root.withdraw()
    try:
        images = [load_image(DATA_DIR / "depth.npy", depth=True),
                  load_image(DATA_DIR / "confidence.npy")]
        print(images[0][0].size)
        print(images[1][0].size)
    except Exception as exc:
        messagebox.showerror("Не удалось открыть изображения", f"{DATA_DIR}\n\n{exc}", parent=root)
        root.destroy()
        return

    root.title("Сохранённые изображения Arducam ToF")
    root.geometry("1100x550")
    root.minsize(640, 360)
    root.rowconfigure(0, weight=1)
    for column, (title, (source, description)) in enumerate(zip(
            ("Глубина", "Confidence"), images)):
        root.columnconfigure(column, weight=1, uniform="images")
        panel = ttk.Frame(root, padding=12)
        panel.grid(row=0, column=column, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(1, weight=1)
        ttk.Label(panel, text=title, font=("TkDefaultFont", 16, "bold")).grid(row=0, column=0)
        preview = ttk.Label(panel, anchor="center")
        preview.grid(row=1, column=0, sticky="nsew", pady=8)

        def resize(event, label=preview, original=source):
            scale = min(max(1, event.width) / original.width,
                        max(1, event.height) / original.height)
            size = (max(1, int(original.width * scale)), max(1, int(original.height * scale)))
            # Ближайший сосед сохраняет видимую структуру исходных пикселей.
            photo = ImageTk.PhotoImage(original.resize(size, Image.Resampling.NEAREST))
            label.configure(image=photo)
            label.image = photo  # Tkinter требует сохранять ссылку.

        preview.bind("<Configure>", resize)
        legend = ("Синий → красный: ближе → дальше. Чёрный: нет глубины."
                  if column == 0 else "Темнее → светлее: меньше → больше confidence.")
        ttk.Label(panel, text=f"{description}\n{legend}", anchor="center",
                  justify="center", wraplength=450).grid(row=2, column=0, sticky="ew")
    ttk.Label(root, text="Масштаб цвета рассчитан отдельно по диапазону каждого массива. "
              "Исходные данные не изменены.", anchor="center").grid(
                  row=1, column=0, columnspan=2, sticky="ew", pady=10)
    root.bind("<Escape>", lambda _event: root.destroy())
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
