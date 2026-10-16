"""Проверка двух сервоприводов без камеры, OpenCV и tkinter.

python tof_scan_tracker/servo_sweep.py
Остановка: Ctrl+C. Arduino Uno со StandardFirmata, X=9, Y=10.
"""
import argparse
import math
import sys
from time import monotonic, sleep


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--port', default='auto', help='Порт Arduino или auto')
    for name, default, text in (
        ('x-min',20,'Минимальный угол X'), ('x-max',160,'Максимальный угол X'),
        ('y-min',30,'Минимальный угол Y'), ('y-max',150,'Максимальный угол Y'),
        ('scan-speed',20,'Скорость движения, градусов/сек'),
        ('scan-step',10,'Расстояние между строками по Y, градусов'),
        ('servo-hz',20,'Максимальная частота команд')):
        p.add_argument('--'+name, type=float, default=default, help=text)
    p.add_argument('--dry-run', dest='no_servos', action='store_true',
                   help='Расчёт траектории без подключения Arduino')
    a = p.parse_args(argv)
    if any(isinstance(v,float) and not math.isfinite(v) for v in vars(a).values()):
        p.error('Значения должны быть конечными')
    if not (20 <= a.x_min < a.x_max <= 160 and 30 <= a.y_min < a.y_max <= 150):
        p.error('Допустимые границы: X 20–160°, Y 30–150°')
    if math.ceil(a.x_min) > math.floor(a.x_max) or math.ceil(a.y_min) > math.floor(a.y_max):
        p.error('Диапазон каждой оси должен содержать хотя бы один целый угол')
    if min(a.scan_speed,a.scan_step,a.servo_hz) <= 0 or a.servo_hz > 50:
        p.error('Скорость и шаг должны быть положительными, частота команд — больше 0 и не больше 50 Гц')
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


def approach(current, target, step):
    return current + max(-step, min(step, target-current))


def servo_angle(value, minimum, maximum):
    """Firmata принимает целые градусы: округление не должно выйти за границы."""
    return max(math.ceil(minimum), min(math.floor(maximum), int(round(value))))


class SnakePath:
    def __init__(self, args):
        self.a = args
        self.x = min(args.x_max, max(args.x_min, 90.))
        self.y = min(args.y_max, max(args.y_min, 90.))
        self.phase = 'HOME'
        self.x_direction = self.y_direction = 1
        self.row_target = args.y_min

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


def open_board(port, arduino_class=None):
    """Закрыть USB даже при ошибке Firmata и не повторять exit из __del__."""
    if arduino_class is None:
        from pyfirmata2 import Arduino
        arduino_class = Arduino

    class ManagedArduino(arduino_class):
        def __init__(self, device):
            self._connection_closed = False
            self._connection_ready = False
            try:
                super().__init__(device)
                self._connection_ready = True
            except BaseException:
                self.exit()
                raise

        def exit(self):
            if self._connection_closed:
                return
            self._connection_closed = True
            serial_port = getattr(self, 'sp', None)
            try:
                if self._connection_ready:
                    super().exit()
            except Exception as exc:
                # Вторичная ошибка не должна скрывать исходную ошибку записи.
                print(f'Не удалось отключить серво через Firmata: {exc}', file=sys.stderr)
            finally:
                if serial_port is not None:
                    try:
                        serial_port.close()
                    except Exception as exc:
                        print(f'Не удалось закрыть порт {port}: {exc}', file=sys.stderr)

        def __del__(self):
            # Финализатор может вызываться для частично созданного объекта
            # или при завершении Python, когда модули уже выгружены.
            try:
                self.exit()
            except Exception:
                pass

    try:
        return ManagedArduino(port)
    except Exception as exc:
        raise RuntimeError(f'Arduino {port}: ошибка подключения/инициализации: {exc}') from exc


class ServoRig:
    def __init__(self, args):
        self.args = args
        self.board = None
        self.pins = None
        self.last_command = None
        self.last_time = -math.inf
        if args.no_servos:
            return
        port = resolve_port(args.port)
        self.port = port
        print(f'Порт Arduino: {port}', flush=True)
        self.board = open_board(port)
        try:
            # servo_config задаёт начальные углы сразу: get_pin(':s') в
            # некоторых версиях библиотеки сначала устанавливает угол 0.
            x = min(args.x_max, max(args.x_min, 90))
            y = min(args.y_max, max(args.y_min, 90))
            print('Настройка сервопривода X (pin 9)…', flush=True)
            self.board.servo_config(9, angle=servo_angle(x, args.x_min, args.x_max))
            print('Настройка сервопривода Y (pin 10)…', flush=True)
            self.board.servo_config(10, angle=servo_angle(y, args.y_min, args.y_max))
            self.pins = self.board.digital[9], self.board.digital[10]
        except BaseException:
            self.board.exit()
            raise

    def write(self, x, y, now):
        if now - self.last_time < 1 / self.args.servo_hz:
            return
        angles = (servo_angle(x, self.args.x_min, self.args.x_max),
                  servo_angle(y, self.args.y_min, self.args.y_max))
        if self.pins:
            for index, pin in enumerate(self.pins):
                if self.last_command is None or angles[index] != self.last_command[index]:
                    try:
                        pin.write(angles[index])
                    except Exception as exc:
                        raise RuntimeError(f'Arduino {self.port}: запись угла {angles[index]}° '
                                           f'на pin {9 + index} не выполнена: {exc}') from exc
        self.last_command, self.last_time = angles, now

    def close(self):
        if self.board:
            self.board.exit()

def run(args):
    path = SnakePath(args)
    rig = ServoRig(args)
    try:
        print('Только расчёт, Arduino не подключается.' if args.no_servos else
              'Сервоприводы подключены: X=9, Y=10.', flush=True)
        print('Начальное положение: X=%.0f°, Y=%.0f°. Остановка: Ctrl+C.' % (path.x,path.y), flush=True)
        # Даём приводам время на начальную команду. Нет обратной связи по углу.
        sleep(1)
        previous = monotonic()
        last_report = -math.inf
        while True:
            now = monotonic()
            # После задержки не компенсируем всё пропущенное время скачком угла.
            dt = min(.1, max(0, now-previous))
            previous = now
            path.scan(dt)
            rig.write(path.x,path.y,now)
            if now-last_report >= 1:
                print(f'{path.phase}: X={path.x:.1f}° | Y={path.y:.1f}°', flush=True)
                last_report = now
            sleep(1 / (2*args.servo_hz))
    finally:
        rig.close()
        print('Проверка завершена.', flush=True)


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
