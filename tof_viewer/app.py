"""Минимальный просмотр Arducam ToF на основе clue/time-of-flight.py."""
import argparse
import sys
import time

import cv2
import numpy as np

WINDOW = 'Arducam ToF'
VIEW = 'View: 0 Depth / 1 Confidence'


def make_preview(frame, maximum, view, threshold):
    """Результат не ссылается на память кадра SDK."""
    depth = np.asarray(frame.depth_data)
    confidence = np.asarray(frame.confidence_data)
    if depth.ndim != 2 or depth.size == 0 or confidence.shape != depth.shape:
        raise RuntimeError('Камера не предоставила совместимые depth и confidence.')
    if view == 1:
        # Как в примере, нормализуем confidence по кадру, но не меняем буфер SDK.
        clean = np.nan_to_num(confidence, nan=0, posinf=0, neginf=0).astype(np.float32, copy=False)
        return cv2.normalize(clean, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    valid = np.isfinite(depth) & (depth > 0) & (depth <= maximum)
    valid &= np.isfinite(confidence) & (confidence >= threshold)
    gray = (np.where(valid, depth, 0) * (255.0 / maximum)).astype(np.uint8)
    image = cv2.applyColorMap(gray, cv2.COLORMAP_RAINBOW)
    image[~valid] = 0
    return image


def check(code, operation):
    if code != 0:
        raise RuntimeError(f'Arducam {operation}: код ошибки {code}')


def run(args):
    import ArducamDepthCamera as ac

    cam = ac.ArducamCamera()
    opened = started = window_open = False
    view, threshold = 0, args.confidence

    def select(value):
        nonlocal view
        view = value

    def set_threshold(value):
        nonlocal threshold
        threshold = value

    try:
        check(cam.open(ac.Connection.CSI, args.index), 'open')
        opened = True
        check(cam.start(ac.FrameType.DEPTH), 'start')
        started = True
        check(cam.setControl(ac.Control.RANGE, args.range), 'RANGE')
        maximum = float(cam.getControl(ac.Control.RANGE))
        if not np.isfinite(maximum) or maximum <= 0:
            raise RuntimeError('SDK вернул некорректный диапазон глубины.')
        info = cam.getCameraInfo()
        print(f'Arducam: {info.width}x{info.height}, диапазон {maximum:g} мм')
        print('0 / D: глубина; 1 / C: confidence; пробел: переключение; Q / Esc: выход.')
        # Исходное разрешение: без дополнительных копий для увеличения картинки.
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
        window_open = True
        cv2.createTrackbar(VIEW, WINDOW, view, 1, select)
        cv2.createTrackbar('Confidence threshold', WINDOW, threshold, 255, set_threshold)
        cv2.imshow(WINDOW, np.zeros((info.height, info.width), np.uint8))
        last_frame = stats_start = time.monotonic()
        frames = 0
        while True:
            # Ограниченный таймаут: при пропаже камеры события окна не ждут 2 секунды.
            frame = cam.requestFrame(100)
            if frame is not None:
                try:
                    if not isinstance(frame, ac.DepthData):
                        raise RuntimeError('SDK вернул кадр, отличный от DepthData.')
                    image = make_preview(frame, maximum, view, threshold)
                finally:
                    cam.releaseFrame(frame)
                # Здесь image уже независим от памяти SDK.
                cv2.imshow(WINDOW, image)
                last_frame = time.monotonic()
                frames += 1
            now = time.monotonic()
            if now-last_frame > 5:
                raise RuntimeError('Нет кадров более 5 секунд. Проверьте подключение камеры.')
            if args.stats and now-stats_start >= 1:
                print(f'Вывод: {frames/(now-stats_start):.1f} FPS', flush=True)
                frames = 0
                stats_start = now
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('0'), ord('d'), ord('D')):
                cv2.setTrackbarPos(VIEW, WINDOW, 0)
            elif key in (ord('1'), ord('c'), ord('C')):
                cv2.setTrackbarPos(VIEW, WINDOW, 1)
            elif key == ord(' '):
                cv2.setTrackbarPos(VIEW, WINDOW, 1-view)
            try:
                if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) == 0:
                    break
            except cv2.error:
                break
    finally:
        # Каждую операцию выполняем независимо, сохраняя первоначальную ошибку.
        cleanup = []
        if started:
            cleanup.append(('stop', cam.stop))
        if opened:
            cleanup.append(('close', cam.close))
        if window_open:
            cleanup.append(('окно', lambda: cv2.destroyWindow(WINDOW)))
        for name, operation in cleanup:
            try:
                result = operation()
                if result is not None and result != 0:
                    print(f'Ошибка завершения {name}: {result}', file=sys.stderr)
            except Exception as exc:
                print(f'Ошибка завершения {name}: {exc}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description='Просмотр Arducam ToF')
    parser.add_argument('--range', type=int, choices=(2000, 4000), default=4000)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--confidence', type=int, choices=range(256), default=30, metavar='0..255')
    parser.add_argument('--stats', action='store_true', help='Печатать FPS раз в секунду')
    args = parser.parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
