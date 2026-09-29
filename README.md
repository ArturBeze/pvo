
## Посмотри версию Raspberry Pi OS

```
cat /etc/os-release
```
и
```
uname -m
```

## Установи зависимости для сборки Python

```
sudo apt update
```
```
sudo apt upgrade
```

```
sudo apt install -y \
    libssl-dev \
    libbz2-dev \
    libffi-dev \
    libreadline-dev \
    libncurses-dev \
    zlib1g-dev \
    libsqlite3-dev \
    liblzma-dev \
    tk-dev \
    uuid-dev \
    libgdbm-dev \
    libnss3-dev
```

```
sudo apt install -y build-essential
```

## Установи pyenv

```
curl https://pyenv.run | bash
```

```
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

```
source ~/.bashrc
```

```
pyenv --version
```

## Установи Python 3.11

```
pyenv install 3.11.11
```

```
pyenv versions
```

## Создаём environment

```
pyenv local 3.11.11
```

```
python --version
```

```
python -m venv .venv
```

```
source .venv/bin/activate
```

```
python --version
```

```
which python
```

## Для обновления программного обеспечения ArduCam ToF в /boot/firmware/config.txt:
```
[all]
# Disable automatic camera detection
camera_auto_detect=0
# Enable the IMX219 camera on cam0
dtoverlay=imx219,cam0
# Enable the Arducam Pivariety camera on cam1
dtoverlay=arducam-pivariety,cam1
```

In the example Python code, the only change I made was in this function:
```
	ac.TOFConnect.CSI, ac.TOFOutput.DEPTH, 0)
	ret = cam.open(ac.Connection.CSI, 0)
```

Исправление ошибок при старте камеры (errors and warnings)
```
wget -O install_pivariety_pkgs.sh https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
chmod +x install_pivariety_pkgs.sh
./install_pivariety_pkgs.sh -p libcamera_dev
./install_pivariety_pkgs.sh -p libcamera_apps
```

```
sudo apt-mark manual qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
sudo apt install qtbase5-dev qtchooser qt5-qmake qtbase5-dev-tools
pip3 install --verbose pyqt6==6.3.0
pip3 install pyqt5 --config-settings --confirm-license= --verbose
pip install PyOpenGL
```

Для установки PiCamera2 в виртуальном пространстве:
Running on Bookworm, use the following commands to install the prequisites
```
sudo apt update && sudo apt upgrade
```

все:
```
sudo apt install libcap-dev libatlas-base-dev ffmpeg libopenjp2-7 libcamera-dev libkms++-dev libfmt-dev libdrm-dev
```

Then activate your virtual environment and run the following commands
```
pip install --upgrade pip
pip install wheel
pip install rpi-libcamera rpi-kms picamera2
```

# You can check that libcamera is working by opening a command window and typing:
```
rpicam-hello
```

```
rpicam-hello --list-cameras
```

