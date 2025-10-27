# Используем официальный образ Python 3.12 как базовый
FROM python:3.12

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Копируем requirements.txt (на случай, если потребуется доустановка)
COPY requirements.txt .

# Устанавливаем зависимости из requirements.txt (если что-то отсутствует в .venv)
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код приложения
COPY . .

# Указываем команду по умолчанию (замените на ваш скрипт)
CMD ["python", "main.py"]