
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