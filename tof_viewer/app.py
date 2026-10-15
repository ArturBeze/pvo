"""Прямой просмотр OpenCV: python tof_viewer/app.py [--demo]."""
import argparse
import sys
import threading
import time

import cv2
import numpy as np
from camera import Camera, DemoCamera

WINDOW = 'Arducam ToF'
SELECTOR = 'View: 0 Depth / 1 Confidence'


class Capture:
    """Публикует только последний кадр, не накапливая задержку отображения."""
    def __init__(self, factory, maximum, index):
        self.factory, self.maximum, self.index = factory, maximum, index
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.count = 0
        self.error = None
        self.thread = threading.Thread(target=self.run, name='tof-capture', daemon=True)

    def snapshot(self):
        with self.lock:
            return self.latest, self.count, self.error

    def run(self):
        camera = None
        try:
            camera = self.factory(self.maximum, self.index)
            last = time.monotonic()
            while not self.stop.is_set():
                frame = camera.read()
                now = time.monotonic()
                if frame is None:
                    if now-last > 5:
                        raise RuntimeError('Нет кадров более 5 секунд. Проверьте камеру.')
                    continue
                last = now
                # Массивы уже скопированы из SDK до releaseFrame, их не меняем.
                with self.lock:
                    self.count += 1
                    self.latest = (*frame, camera.max_depth, now)
        except Exception as exc:
            with self.lock:
                self.error = str(exc)
        finally:
            if camera is not None:
                try:
                    camera.close()
                except Exception as exc:
                    with self.lock:
                        self.error = f'Ошибка закрытия камеры: {exc}'


def render(depth, confidence, maximum, mode):
    """Возвращает BGR/серый кадр исходного размера, не меняя данные SDK."""
    if mode == 1:
        return np.clip(np.nan_to_num(confidence, nan=0, posinf=0, neginf=0),
                       0, 255).astype(np.uint8)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= maximum)
    gray = (np.where(valid, depth, 0) * (255.0/maximum)).astype(np.uint8)
    image = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    image[~valid] = 0
    return image


def run(args):
    capture = Capture(DemoCamera if args.demo else Camera, args.range, args.index)
    mode = 0

    def select(value):
        nonlocal mode
        mode = value

    opened = False
    started = False
    try:
        # Все GUI-вызовы выполняются в главном потоке.
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        opened = True
        cv2.resizeWindow(WINDOW, 720, 540)
        cv2.createTrackbar(SELECTOR, WINDOW, 0, 1, select)
        cv2.imshow(WINDOW, np.zeros((180, 240), np.uint8))
        capture.thread.start()
        started = True
        print('0 / D: глубина; 1 / C: confidence; пробел: переключить; Q / Esc: выход.')
        previous_count, previous_mode = -1, -1
        stats_time = time.monotonic()
        stats_count = displayed = 0
        prepare_seconds = age_seconds = 0.0
        while True:
            frame, count, error = capture.snapshot()
            if error is not None:
                raise RuntimeError(error)
            if frame is not None and (count != previous_count or mode != previous_mode):
                depth, confidence, maximum, received = frame
                start = time.monotonic()
                image = render(depth, confidence, maximum, mode)
                prepare_seconds += time.monotonic()-start
                age_seconds += time.monotonic()-received
                cv2.imshow(WINDOW, image)
                displayed += 1
                previous_count, previous_mode = count, mode
            now = time.monotonic()
            if now-stats_time >= 1:
                interval = now-stats_time
                print(f'Захват: {(count-stats_count)/interval:.1f} FPS | '
                      f'Вывод: {displayed/interval:.1f} FPS | '
                      f'Подготовка: {prepare_seconds/max(displayed, 1)*1000:.2f} мс | '
                      f'Возраст кадра до вывода: {age_seconds/max(displayed, 1)*1000:.1f} мс',
                      flush=True)
                stats_time, stats_count = now, count
                displayed = 0
                prepare_seconds = age_seconds = 0.0
            # Обработка событий без прежней фиксированной паузы 40 мс.
            # waitKey(1) также не даёт пустому циклу загрузить ядро целиком.
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('0'), ord('d'), ord('D')):
                mode = 0
                cv2.setTrackbarPos(SELECTOR, WINDOW, mode)
            elif key in (ord('1'), ord('c'), ord('C'), ord(' ')):
                mode = 1-mode if key == ord(' ') else 1
                cv2.setTrackbarPos(SELECTOR, WINDOW, mode)
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break  # Окно уже уничтожено пользователем.
    finally:
        capture.stop.set()
        if opened:
            cv2.destroyAllWindows()
        if started:
            capture.thread.join(timeout=3)
            if capture.thread.is_alive():
                print('SDK не завершил захват за 3 секунды.', file=sys.stderr)
            elif capture.error:
                print(f'Камера: {capture.error}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Быстрый просмотр Arducam ToF')
    parser.add_argument('--demo', action='store_true', help='Синтетическая камера')
    parser.add_argument('--range', type=int, choices=(2000, 4000), default=4000, help='Диапазон в мм')
    parser.add_argument('--index', type=int, default=0, help='Индекс CSI камеры в SDK')
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
