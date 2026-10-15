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


class App:
    def __init__(self, root, args):
        self.root, self.args = root, args
        self.stop = threading.Event()
        self.frames = queue.Queue(maxsize=1)
        self.events = queue.Queue()
        self.latest = None
        self.closing = False
        root.title('Arducam ToF — просмотр')
        root.geometry('900x650')
        root.minsize(640, 400)
        panel = ttk.Frame(root, padding=12)
        panel.pack(side='left', fill='y')
        self.canvas = tk.Canvas(root, bg='#161b22', highlightthickness=0)
        self.canvas.pack(side='right', fill='both', expand=True)
        ttk.Label(panel, text='Демонстрация' if args.demo else 'Arducam CSI',
                  font=('', 16, 'bold')).pack(anchor='w', pady=(0, 12))
        ttk.Label(panel, text='Показывать').pack(anchor='w')
        self.view = tk.StringVar(value='Глубина')
        ttk.Combobox(panel, textvariable=self.view,
                     values=['Глубина', 'Confidence'], state='readonly',
                     width=20).pack(fill='x', pady=(0, 12))
        ttk.Label(panel, text='Глубина: цветная карта.\n'
                  'Confidence: оттенки серого,\n'
                  'шкала 0…255.', wraplength=200).pack(anchor='w')
        self.status = tk.StringVar(value='Подключение…')
        ttk.Label(panel, textvariable=self.status, wraplength=290).pack(anchor='w', pady=8)
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()
        root.protocol('WM_DELETE_WINDOW', self.close)
        root.after(30, self.tick)

    def worker(self):
        source = None
        try:
            source = (DemoCamera if self.args.demo else Camera)(self.args.range, self.args.index)
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
                item = (depth, confidence, source.max_depth, time.monotonic()-start)
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
            d, c, maximum, elapsed = self.latest
            finite = c[np.isfinite(c)]
            conf_range = f'{finite.min():.1f}…{finite.max():.1f}' if finite.size else 'нет'
            self.status.set(f'{d.shape[1]} × {d.shape[0]} | {1/max(elapsed, 1e-6):.1f} FPS\n'
                            f'Диапазон: 0…{maximum} мм\n'
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
        d, c, maximum, elapsed = self.latest
        mode = self.view.get()
        if mode == 'Confidence':
            gray = np.clip(np.nan_to_num(c, nan=0, posinf=0, neginf=0), 0, 255).astype(np.uint8)
            rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            values = d
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
    parser = argparse.ArgumentParser(description='Просмотр Arducam ToF')
    parser.add_argument('--demo', action='store_true', help='Синтетическая камера')
    parser.add_argument('--range', type=int, choices=(2000, 4000), default=4000, help='Диапазон в мм')
    parser.add_argument('--index', type=int, default=0, help='Индекс CSI камеры в SDK')
    args = parser.parse_args()
    root = tk.Tk()
    App(root, args)
    root.mainloop()


if __name__ == '__main__':
    main()
