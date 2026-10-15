"""Запуск: python tof_viewer/app.py [--demo]."""
import argparse
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from camera import Camera, DemoCamera
from processing import Processor, Settings


class App:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.settings = Settings()
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.latest = None
        self.closing = False
        root.title('Arducam ToF — обработка глубины')
        root.geometry('1150x760')
        root.minsize(900, 650)
        panel = ttk.Frame(root, padding=12)
        panel.pack(side='left', fill='y')
        self.canvas = tk.Canvas(root, bg='#161b22', highlightthickness=0)
        self.canvas.pack(side='right', fill='both', expand=True)
        ttk.Label(panel, text='Демонстрация' if args.demo else 'Arducam CSI',
                  font=('', 16, 'bold')).pack(anchor='w', pady=(0, 12))
        ttk.Label(panel, text='Показывать').pack(anchor='w')
        self.view = tk.StringVar(value='Обработанная глубина')
        views = ['Обработанная глубина', 'Исходная глубина', 'Confidence',
                 'Надёжные измерения', 'Восстановленные пиксели']
        ttk.Combobox(panel, textvariable=self.view, values=views, state='readonly',
                     width=29).pack(fill='x', pady=(0, 12))
        self.variables = {}
        controls = [
            ('confidence', 'Порог confidence (единицы SDK)', 0, 255, 30),
            ('fill_radius', 'Радиус восстановления (px; 0 — выкл.)', 0, 5, 2),
            ('edge_limit', 'Допустимый перепад глубины (мм)', 1, 500, 150),
            ('smooth_radius', 'Радиус сглаживания (px; 0 — выкл.)', 0, 4, 2),
            ('sigma_depth', 'Сила сглаживания по глубине (мм)', 1, 250, 60),
            ('alpha', 'Доля нового кадра (1 — без накопления)', 0.05, 1, 0.65),
        ]
        for name, label, low, high, default in controls:
            ttk.Label(panel, text=label).pack(anchor='w')
            var = tk.DoubleVar(value=default)
            self.variables[name] = var
            tk.Scale(panel, variable=var, from_=low, to=high, orient='horizontal',
                     resolution=0.05 if name == 'alpha' else 1,
                     command=self.update_settings, length=290).pack(fill='x')
        ttk.Button(panel, text='Сбросить фильтры', command=self.reset).pack(fill='x', pady=8)
        ttk.Label(panel, text='Чёрный: нет достоверной глубины.\n'
                  'Глубина: синий → красный, 0 → максимум.\n'
                  'Confidence: чёрный → белый, 0 → 255.\n'
                  'Восстановление: белый = оценка,\n'
                  'а не измерение камеры.', wraplength=290).pack(anchor='w', pady=8)
        self.status = tk.StringVar(value='Подключение…')
        ttk.Label(panel, textvariable=self.status, wraplength=290).pack(anchor='w', pady=8)
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(30, self.tick)

    def update_settings(self, _=None):
        if len(self.variables) != 6:
            return
        values = {k: v.get() for k, v in self.variables.items()}
        for name in ('fill_radius', 'smooth_radius'):
            values[name] = int(values[name])
        with self.lock:
            self.settings = Settings(**values)

    def reset(self):
        defaults = Settings()
        for name, var in self.variables.items():
            var.set(getattr(defaults, name))
        self.update_settings()

    def worker(self):
        source = None
        try:
            source = (DemoCamera if self.args.demo else Camera)(self.args.range, self.args.index)
            processor = Processor()
            last_frame = time.monotonic()
            while not self.stop.is_set():
                start = time.monotonic()
                frame = source.read()
                if frame is None:
                    if time.monotonic() - last_frame > 5:
                        raise RuntimeError('Нет кадров более 5 секунд. Проверьте подключение камеры.')
                    continue
                last_frame = time.monotonic()
                depth, confidence = frame
                with self.lock:
                    settings = self.settings
                processed, valid, filled = processor.process(depth, confidence, settings, source.max_depth)
                item = (depth, confidence, processed, valid, filled, source.max_depth,
                        time.monotonic()-start)
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
                self.frames.put_nowait(item)
        except Exception as exc:
            self.events.put(f'Ошибка: {exc}\nЗакройте окно и повторите запуск.\n'
                            'Без оборудования используйте --demo.')
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception as exc:
                    self.events.put(f'Ошибка закрытия камеры: {exc}')

    def tick(self):
        if self.closing:
            if self.thread.is_alive():
                self.root.after(50, self.tick)
            else:
                self.root.destroy()
            return
        try:
            self.latest = self.frames.get_nowait()
            d, c, p, valid, filled, maximum, elapsed = self.latest
            finite = c[np.isfinite(c)]
            conf_range = f'{finite.min():.1f}…{finite.max():.1f}' if finite.size else 'нет'
            self.status.set(f'{d.shape[1]} × {d.shape[0]} | {1/max(elapsed, 1e-6):.1f} FPS\n'
                            f'Диапазон: 0…{maximum} мм\n'
                            f'Надёжные: {valid.mean():.1%}\nВосстановлены: {filled.mean():.1%}\n'
                            f'Confidence в кадре: {conf_range}')
        except queue.Empty:
            pass
        try:
            self.status.set(self.events.get_nowait())
        except queue.Empty:
            pass
        if self.latest is not None:
            self.draw()
        self.root.after(40, self.tick)

    def draw(self):
        d, c, p, valid, filled, maximum, elapsed = self.latest
        mode = self.view.get()
        if mode == 'Confidence':
            gray = np.clip(np.nan_to_num(c, nan=0, posinf=0, neginf=0), 0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        elif mode in ('Надёжные измерения', 'Восстановленные пиксели'):
            mask = valid if mode == 'Надёжные измерения' else filled
            rgb = cv2.cvtColor(mask.astype(np.uint8)*255, cv2.COLOR_GRAY2RGB)
        else:
            values = d if mode == 'Исходная глубина' else p
            known = np.isfinite(values) & (values > 0) & (values <= maximum)
            gray = (np.clip(np.where(known, values, 0)/maximum, 0, 1)*255).astype(np.uint8)
            rgb = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
            rgb[~known] = 0
        width, height = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        scale = min(width/rgb.shape[1], height/rgb.shape[0])
        size = (max(1, int(rgb.shape[1]*scale)), max(1, int(rgb.shape[0]*scale)))
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize(size, Image.Resampling.NEAREST))
        self.canvas.delete('all')
        self.canvas.create_image(width/2, height/2, image=self.photo)

    def close(self):
        self.closing = True
        self.stop.set()
        self.status.set('Остановка камеры…')


def main():
    parser = argparse.ArgumentParser(description='Просмотр и фильтрация Arducam ToF')
    parser.add_argument('--demo', action='store_true', help='Синтетическая камера')
    parser.add_argument('--range', type=int, choices=(2000, 4000), default=4000, help='Диапазон в мм')
    parser.add_argument('--index', type=int, default=0, help='Индекс CSI камеры в SDK')
    args = parser.parse_args()
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
