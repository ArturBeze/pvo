"""ToF + Arduino Uno: сканирование змейкой и слежение за фрагментом.

python tof_scan_tracker/scan_track.py --port /dev/ttyACM0 --min 400 --max 500
Выход: Q / Esc. Требуется StandardFirmata и pyFirmata2.
"""
import argparse
import math
import sys
from time import monotonic

import cv2
import numpy as np
from image_utils import valid_mask, render_depth, annotate_fragment


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for flag, default, help_text in (
        ('confidence', 200, 'Минимальный confidence'),
        ('min', 400, 'Минимальная глубина, мм'),
        ('max', 500, 'Максимальная глубина, мм'),
        ('tolerance', 50, 'Допуск глубины соседей, мм'),
        ('lost-timeout', 3, 'Ожидание после потери цели, секунды'),
        ('x-min', 20, 'Минимальный угол X'), ('x-max', 160, 'Максимальный угол X'),
        ('y-min', 30, 'Минимальный угол Y'), ('y-max', 150, 'Максимальный угол Y'),
        ('scan-speed', 20, 'Скорость сканирования, градусов/сек'),
        ('scan-step', 10, 'Шаг между строками по Y, градусов'),
        ('track-speed', 40, 'Максимальная скорость слежения, градусов/сек'),
        ('gain', 30, 'Скорость слежения при смещении на половину кадра'),
        ('dead-zone', .10, 'Мёртвая зона, доля половины ширины/высоты кадра'),
        ('servo-hz', 20, 'Максимальная частота команд Arduino'),
        ('match-radius', .30, 'Допустимое смещение центра цели, доля размера кадра'),
        ('match-depth', 100, 'Допустимое изменение глубины цели, мм'),
    ):
        p.add_argument('--' + flag, type=float, default=default, help=help_text)
    p.add_argument('--range', dest='range_mm', type=int, choices=(2000, 4000), default=4000)
    p.add_argument('--index', type=int, default=0, help='Индекс камеры')
    p.add_argument('--min-area', type=int, default=80, help='Минимальная площадь, пиксели')
    p.add_argument('--port', default='auto', help='Порт Arduino или auto для автоматического поиска')
    p.add_argument('--scan-only', action='store_true', help='Только сканирование змейкой, без слежения')
    p.add_argument('--invert-x', action='store_true', help='Обратить направление слежения по X')
    p.add_argument('--invert-y', action='store_true', help='Обратить направление слежения по Y')
    p.add_argument('--no-servos', action='store_true', help='Камера и расчёт углов без подключения Arduino')
    a = p.parse_args(argv)
    if any(isinstance(v, float) and not math.isfinite(v) for v in vars(a).values()):
        p.error('Все числовые значения должны быть конечными')
    if not 0 <= a.min < a.max <= a.range_mm:
        p.error('Требуется 0 <= min < max <= range')
    if a.confidence < 0 or a.tolerance < 0 or a.min_area < 1 or a.index < 0 or a.lost_timeout < 0:
        p.error('Проверьте confidence, tolerance, min-area, index и lost-timeout')
    if not (20 <= a.x_min < a.x_max <= 160 and 30 <= a.y_min < a.y_max <= 150):
        p.error('Углы должны оставаться в пределах примера: X 20–160, Y 30–150')
    if min(a.scan_speed, a.scan_step, a.track_speed, a.gain, a.servo_hz, a.match_radius, a.match_depth) <= 0:
        p.error('Скорости, шаг и параметры сопоставления должны быть положительными')
    if a.servo_hz > 50 or not 0 <= a.dead_zone < 1 or a.match_radius > 1:
        p.error('servo-hz <= 50, 0 <= dead-zone < 1, match-radius <= 1')
    return a


def resolve_port(port='auto', ports=None):
    """Выбор по USB-описаниям без открытия посторонних последовательных портов."""
    if port != 'auto':
        return port
    if ports is None:
        from serial.tools import list_ports
        ports = list_ports.comports()
    arduino, bridges = set(), set()
    for item in ports:
        description = ' '.join(str(getattr(item, key, '') or '')
                               for key in ('description', 'manufacturer', 'product')).lower()
        vid = getattr(item, 'vid', None)
        pid = getattr(item, 'pid', None)
        if 'arduino' in description or vid in (0x2341, 0x2A03):
            arduino.add(item.device)
        elif ((vid, pid) in ((0x1A86, 0x7523), (0x1A86, 0x5523),
                             (0x10C4, 0xEA60), (0x0403, 0x6001))
              or any(name in description for name in ('ch340', 'ch341', 'cp210', 'ft232'))):
            bridges.add(item.device)
    candidates = sorted(arduino or bridges)
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise RuntimeError('Arduino не найден автоматически. Подключите плату или задайте --port /dev/ttyACM0')
    raise RuntimeError('Найдено несколько возможных Arduino: ' + ', '.join(candidates)
                       + '. Выберите нужный через --port')


def find_fragments(depth, confidence, args):
    """Связные компоненты по глубине и их центры для слежения."""
    valid = valid_mask(depth, confidence, args.confidence, args.min, args.max)
    if not valid.any():
        return []
    h, w = depth.shape
    graph = np.zeros((2*h-1, 2*w-1), np.uint8)
    graph[::2, ::2] = valid
    safe = np.where(valid, depth, 0).astype(np.float32)
    graph[::2, 1::2] = (valid[:, :-1] & valid[:, 1:] &
                        (np.abs(safe[:, 1:] - safe[:, :-1]) <= args.tolerance))
    graph[1::2, ::2] = (valid[:-1] & valid[1:] &
                        (np.abs(safe[1:] - safe[:-1]) <= args.tolerance))
    _, labels = cv2.connectedComponents(graph, connectivity=4)
    labels = labels[::2, ::2]
    areas = np.bincount(labels.ravel())
    fragments = []
    for label in np.flatnonzero(areas >= args.min_area):
        if label == 0:
            continue
        mask = labels == label
        y, x = np.nonzero(mask)
        fragments.append(dict(mask=mask, area=int(areas[label]),
            distance=float(np.median(depth[mask])), confidence=float(np.median(confidence[mask])),
            center=(float(x.mean()) / max(1, w-1), float(y.mean()) / max(1, h-1))))
    return fragments


def approach(current, target, step):
    return current + max(-step, min(step, target-current))


class ScanTracker:
    """Контроллер без аппаратных зависимостей; время передаётся явно для тестов."""
    def __init__(self, args):
        self.a = args
        self.x = min(args.x_max, max(args.x_min, 90.))
        self.y = min(args.y_max, max(args.y_min, 90.))
        self.state = 'SCAN'
        self.phase = 'HOME'
        self.x_direction = self.y_direction = 1
        self.row_target = args.y_min
        self.target = None
        self.last_seen = self.last_update = None

    def update(self, fragments, now):
        dt = 0 if self.last_update is None else min(.1, max(0, now-self.last_update))
        self.last_update = now
        if self.a.scan_only:
            self.state, self.target = 'SCAN', None
            self.scan(dt)
            return None
        if self.state == 'SCAN':
            target = min(fragments, key=lambda f: (f['distance'], -f['area']), default=None)
        else:
            # Не перескакиваем на любой новый ближайший предмет: ищем рядом
            # с предыдущим центром и глубиной. Это геометрическое сопоставление.
            candidates = []
            for f in fragments:
                dx = f['center'][0] - self.target['center'][0]
                dy = f['center'][1] - self.target['center'][1]
                offset = math.hypot(dx, dy)
                delta = abs(f['distance'] - self.target['distance'])
                if offset <= self.a.match_radius and delta <= self.a.match_depth:
                    candidates.append((offset / self.a.match_radius + delta / self.a.match_depth, f))
            target = min(candidates, key=lambda item: item[0])[1] if candidates else None
        if target is not None:
            self.target = {k: target[k] for k in ('center', 'distance')}
            self.last_seen = now
            self.state = 'TRACK'
            for axis, center, invert, low, high in (
                ('x', target['center'][0], self.a.invert_x, self.a.x_min, self.a.x_max),
                ('y', target['center'][1], self.a.invert_y, self.a.y_min, self.a.y_max)):
                error = 2 * (center - .5)
                if abs(error) <= self.a.dead_zone:
                    continue
                speed = min(self.a.track_speed, self.a.gain * abs(error))
                delta = math.copysign(speed * dt, error) * (-1 if invert else 1)
                setattr(self, axis, min(high, max(low, getattr(self, axis)+delta)))
            return target
        if self.state != 'SCAN':
            self.state = 'WAIT'
            if now - self.last_seen < self.a.lost_timeout:
                return None  # Держим последние углы, не продолжаем старую коррекцию.
            self.state, self.target = 'SCAN', None
            # Возобновляем горизонтальный проход с текущего положения.
            self.phase = 'SWEEP'
            self.x_direction = 1 if self.x < (self.a.x_min+self.a.x_max)/2 else -1
            self.y_direction = 1 if self.y < (self.a.y_min+self.a.y_max)/2 else -1
        self.scan(dt)
        return None

    def scan(self, dt):
        step = self.a.scan_speed * dt
        if self.phase == 'HOME':
            self.x = approach(self.x, self.a.x_min, step)
            self.y = approach(self.y, self.a.y_min, step)
            if self.x == self.a.x_min and self.y == self.a.y_min:
                self.phase = 'SWEEP'
        elif self.phase == 'SWEEP':
            edge = self.a.x_max if self.x_direction > 0 else self.a.x_min
            self.x = approach(self.x, edge, step)
            if self.x == edge:
                if self.y >= self.a.y_max:
                    self.y_direction = -1
                elif self.y <= self.a.y_min:
                    self.y_direction = 1
                self.row_target = min(self.a.y_max, max(self.a.y_min,
                    self.y + self.y_direction * self.a.scan_step))
                self.phase = 'ROW'
        else:
            self.y = approach(self.y, self.row_target, step)
            if self.y == self.row_target:
                self.x_direction *= -1
                self.phase = 'SWEEP'


class ServoRig:
    def __init__(self, args):
        self.args = args
        self.board = None
        self.pins = None
        self.last_command = None
        self.last_time = -math.inf
        if args.no_servos:
            return
        import pyfirmata2
        port = resolve_port(args.port)
        print(f'Порт Arduino: {port}', flush=True)
        self.board = pyfirmata2.Arduino(port)
        try:
            # servo_config задаёт начальные углы сразу: get_pin(':s') в
            # некоторых версиях библиотеки сначала устанавливает угол 0.
            x = min(args.x_max, max(args.x_min, 90))
            y = min(args.y_max, max(args.y_min, 90))
            self.board.servo_config(9, angle=int(round(x)))
            self.board.servo_config(10, angle=int(round(y)))
            self.pins = self.board.digital[9], self.board.digital[10]
        except Exception:
            self.board.exit()
            raise

    def write(self, x, y, now):
        if now - self.last_time < 1 / self.args.servo_hz:
            return
        angles = (int(round(min(self.args.x_max, max(self.args.x_min, x)))),
                  int(round(min(self.args.y_max, max(self.args.y_min, y)))))
        if self.pins:
            for index, pin in enumerate(self.pins):
                if self.last_command is None or angles[index] != self.last_command[index]:
                    pin.write(angles[index])
        self.last_command, self.last_time = angles, now

    def close(self):
        if self.board:
            self.board.exit()


def run(args):
    import ArducamDepthCamera as ac
    camera = ac.ArducamCamera()
    opened = started = False
    rig = None
    window = 'ToF scan / track | Q / Esc: exit'
    tracker = ScanTracker(args)
    try:
        result = camera.open(ac.Connection.CSI, args.index)
        if result != 0:
            raise RuntimeError(f'Открытие камеры: {result}')
        opened = True
        result = camera.start(ac.FrameType.DEPTH)
        if result != 0:
            raise RuntimeError(f'Запуск камеры: {result}')
        started = True
        result = camera.setControl(ac.Control.RANGE, args.range_mm)
        if result not in (None, 0) or camera.getControl(ac.Control.RANGE) != args.range_mm:
            raise RuntimeError('Не удалось установить диапазон камеры')
        print('Подключение Arduino…' if not args.no_servos else 'Режим без сервоприводов', flush=True)
        rig = ServoRig(args)
        cv2.namedWindow(window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(window, 960, 720)
        last_received = last_report = monotonic()
        # Начальная команда позиционирования не имеет обратной связи.
        ready_at = last_received + 1
        print(f'Поиск {args.min:g}–{args.max:g} мм; X=9, Y=10; ожидание {args.lost_timeout:g} с', flush=True)
        while True:
            frame = camera.requestFrame(100)
            if frame is not None:
                try:
                    if not isinstance(frame, ac.DepthData):
                        raise RuntimeError('Камера вернула кадр без глубины')
                    depth, confidence = frame.depth_data, getattr(frame, 'confidence_data', None)
                    if (depth is None or confidence is None or depth.ndim != 2 or not depth.size
                            or confidence.shape != depth.shape or depth.dtype.kind not in 'uif'
                            or confidence.dtype.kind not in 'uif'):
                        raise RuntimeError('Камера не предоставила корректные depth/confidence')
                    now = monotonic()
                    fragments = [] if args.scan_only else find_fragments(depth, confidence, args)
                    target = tracker.update(fragments, now) if now >= ready_at else None
                    image = render_depth(depth, confidence, args.confidence, args.min, args.max)
                    h, w = depth.shape
                    cv2.drawMarker(image, (w//2, h//2), (255,255,255), cv2.MARKER_CROSS, 9, 1)
                    if target is not None:
                        cx, cy = target['center']
                        cv2.circle(image, (round(cx*(w-1)), round(cy*(h-1))), 3, (255,255,255), 1)
                    if not args.scan_only:
                        image = annotate_fragment(image, target)
                    state = tracker.state if now >= ready_at else 'START'
                    remaining = max(0, args.lost_timeout-(now-tracker.last_seen)) if state == 'WAIT' else 0
                    text = f'{state} X:{tracker.x:.0f} Y:{tracker.y:.0f}'
                    if args.scan_only: text += ' | scan only'
                    if state == 'WAIT': text += f' wait:{remaining:.1f}s'
                    image = cv2.copyMakeBorder(image, 22, 0, 0, 0, cv2.BORDER_CONSTANT)
                    cv2.putText(image, text, (4,15), cv2.FONT_HERSHEY_SIMPLEX, .4, (255,255,255), 1)
                finally:
                    camera.releaseFrame(frame)
                last_received = monotonic()
                if now >= ready_at:
                    rig.write(tracker.x, tracker.y, last_received)
                cv2.imshow(window, image)
                if last_received-last_report >= 1:
                    details = (f" | {target['distance']:.1f} мм | confidence {target['confidence']:.1f}"
                               f" | площадь {target['area']} px") if target else ''
                    print(text + details, flush=True)
                    last_report = last_received
            else:
                # Без нового кадра не двигаем серво по устаревшей цели.
                if monotonic()-last_received > 5:
                    raise TimeoutError('Нет кадров более 5 секунд; сканирование остановлено')
            if cv2.waitKey(1) & 0xFF in (27, ord('q'), ord('Q')):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        # Сначала прекращаем управление приводами, затем освобождаем камеру.
        actions = []
        if rig is not None: actions.append(rig.close)
        if started: actions.append(camera.stop)
        if opened: actions.append(camera.close)
        for action in actions:
            try:
                action()
            except Exception as exc:
                print(f'Ошибка завершения: {exc}', file=sys.stderr)
        cv2.destroyAllWindows()


def main():
    args = parse_args()
    try:
        run(args)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f'Ошибка: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
