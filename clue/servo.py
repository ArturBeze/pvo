import pyfirmata2
import time

# Порт Arduino
PORT = "/dev/ttyACM0"

# Пины сервоприводов
SERVO1_PIN = 9
SERVO2_PIN = 10

# Ограничения углов
SERVO1_MIN = 20
SERVO1_MAX = 160

SERVO2_MIN = 30
SERVO2_MAX = 150


def clamp(angle, min_angle, max_angle):
    return max(min_angle, min(angle, max_angle))


# Подключаемся к Arduino
board = pyfirmata2.Arduino(PORT)

print("Arduino connected")

# Настраиваем сервоприводы
servo1 = board.get_pin(f"d:{SERVO1_PIN}:s")
servo2 = board.get_pin(f"d:{SERVO2_PIN}:s")


def set_servo1(angle):
    angle = clamp(angle, SERVO1_MIN, SERVO1_MAX)
    servo1.write(angle)
    print(f"Servo 1: {angle}°")


def set_servo2(angle):
    angle = clamp(angle, SERVO2_MIN, SERVO2_MAX)
    servo2.write(angle)
    print(f"Servo 2: {angle}°")


try:
    # Начальное положение
    set_servo1(90)
    set_servo2(90)

    time.sleep(1)

    while True:
        print()
        print("1 <angle> - управление Servo 1")
        print("2 <angle> - управление Servo 2")
        print("q         - выход")

        command = input("> ")

        if command.lower() == "q":
            break

        parts = command.split()

        if len(parts) != 2:
            print("Пример: 1 120")
            continue

        servo_number = parts[0]

        try:
            angle = float(parts[1])
        except ValueError:
            print("Угол должен быть числом")
            continue

        if servo_number == "1":
            set_servo1(angle)

        elif servo_number == "2":
            set_servo2(angle)

        else:
            print("Неизвестный сервопривод")


finally:
    board.exit()
    print("Arduino disconnected")