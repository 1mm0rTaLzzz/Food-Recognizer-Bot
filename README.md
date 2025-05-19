# Food Recognizer Bot 🤖

Telegram-бот для распознавания блюд на фотографиях с помощью искусственного интеллекта. Бот использует модель ViT (Vision Transformer), обученную на датасете Food101, для определения типа блюда и оценки его примерной калорийности.

## 🚀 Возможности

- 📸 Распознавание блюд на фотографиях
- 🍽 Определение типа блюда с вероятностью
- 🔥 Оценка примерной калорийности
- 💬 Удобный интерфейс в Telegram

## 🛠 Технологии

- Python 3.x
- PyTorch
- Transformers (Hugging Face)
- Aiogram 3.x
- Pillow
- CUDA (опционально, для ускорения на GPU)

## 📋 Требования

- Python 3.8 или выше
- Telegram Bot Token
- Доступ к интернету

## 🔧 Установка

### Вариант 1: Локальная установка

1. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/Food-Recognizer-Bot.git
cd Food-Recognizer-Bot
```

2. Создайте виртуальное окружение и активируйте его:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/Mac
# или
venv\Scripts\activate  # для Windows
```

3. Установите зависимости:
```bash
pip install -r requirements.txt
```

4. Создайте файл `.env` в корневой директории и добавьте в него токен вашего бота:
```
BOT_TOKEN=your_telegram_bot_token_here
```

### Вариант 2: Установка с помощью Docker

1. Установите Docker:
   - Для Windows: [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Для Linux: [Docker Engine](https://docs.docker.com/engine/install/)

2. Установите NVIDIA Container Toolkit (для поддержки GPU):
   ```bash
   # Для Ubuntu/Debian:
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker

   # Для Windows:
   # Убедитесь, что у вас установлены:
   # 1. NVIDIA GPU с поддержкой CUDA
   # 2. Последние драйверы NVIDIA
   # 3. Docker Desktop с включенной поддержкой WSL 2
   ```

3. Проверьте установку NVIDIA Container Toolkit:
   ```bash
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

4. Клонируйте репозиторий:
```bash
git clone https://github.com/yourusername/Food-Recognizer-Bot.git
cd Food-Recognizer-Bot
```

5. Создайте файл `.env` в корневой директории и добавьте в него токен вашего бота:
```
BOT_TOKEN=your_telegram_bot_token_here
```

6. Соберите Docker образ:
```bash
docker build -t food-recognizer-bot .
```

## 🚀 Запуск

### Вариант 1: Локальный запуск

1. Активируйте виртуальное окружение (если еще не активировано)
2. Запустите бота:
```bash
python main.py
```

### Вариант 2: Запуск в Docker

1. Запустите контейнер одним из способов:

   a. Используя файл `.env`:
   ```bash
   # Для запуска с поддержкой GPU:
   docker run --gpus all --env-file .env food-recognizer-bot

   # Для запуска без GPU:
   docker run --env-file .env food-recognizer-bot
   ```

   b. Передавая токен напрямую:
   ```bash
   # Для запуска с поддержкой GPU:
   docker run --gpus all -e BOT_TOKEN=your_telegram_bot_token_here food-recognizer-bot

   # Для запуска без GPU:
   docker run -e BOT_TOKEN=your_telegram_bot_token_here food-recognizer-bot
   ```

2. Проверьте логи контейнера:
   ```bash
   docker logs -f $(docker ps -q --filter ancestor=food-recognizer-bot)
   ```

## 💡 Использование

1. Найдите бота в Telegram по его username
2. Отправьте команду `/start` для начала работы
3. Отправьте фотографию блюда
4. Получите результат распознавания и оценку калорийности

## 📝 Примечания

- Бот использует предобученную модель ViT, обученную на датасете Food101
- Оценка калорийности является приблизительной и основана на базовых предположениях
- Для лучшей производительности рекомендуется использовать GPU

## 🤝 Вклад в проект

Если вы хотите внести свой вклад в проект:
1. Форкните репозиторий
2. Создайте ветку для ваших изменений
3. Внесите изменения и создайте pull request

## 📄 Лицензия

MIT License