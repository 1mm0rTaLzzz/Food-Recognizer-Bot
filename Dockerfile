# Используем официальный образ Python с поддержкой CUDA
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Создаем скрипт для запуска
RUN echo '#!/bin/bash\n\
if [ ! -f .env ]; then\n\
    if [ -z "$BOT_TOKEN" ]; then\n\
        echo "Ошибка: BOT_TOKEN не установлен. Пожалуйста, установите переменную окружения BOT_TOKEN или создайте файл .env"\n\
        exit 1\n\
    fi\n\
    echo "BOT_TOKEN=$BOT_TOKEN" > .env\n\
fi\n\
python3 main.py' > /app/start.sh && chmod +x /app/start.sh

CMD ["/app/start.sh"] 