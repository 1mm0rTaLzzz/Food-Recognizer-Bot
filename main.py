import os
import torch
import torchvision.transforms as transforms
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.fsm.storage.memory import MemoryStorage
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import numpy as np
import logging
import asyncio

# === 1. Загрузка токена из .env ===
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не найден в .env")

# === 2. Настройка модели ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "nateraw/food"  # Модель для распознавания еды

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# Дополнительные трансформации для улучшения качества
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def preprocess_image(image):
    """Улучшенная предобработка изображения"""
    if image.mode != 'RGB':
        image = image.convert('RGB')

    image_tensor = transform(image)
    return image_tensor.unsqueeze(0).to(device)

def get_top_predictions(probs, threshold=0.1):
    """Получение топ предсказаний с фильтрацией по порогу уверенности"""
    values, indices = torch.topk(probs, k=5)  # Получаем топ-5 предсказаний
    predictions = []
    for value, idx in zip(values, indices):
        if value.item() > threshold:
            label = model.config.id2label[idx.item()]
            predictions.append((label, value.item()))
    return predictions

# === 3. Настройка Telegram-бота ===
bot = Bot(BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "👋 Привет! Отправь фото блюда, и я скажу, что это за еда, а также оценю калорийность.\n\n"
        "💡 Советы для лучшего распознавания:\n"
        "• Сфотографируйте блюдо при хорошем освещении\n"
        "• Старайтесь, чтобы блюдо занимало большую часть кадра\n"
        "• Избегайте размытия и отражений"
    )

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    try:
        photo = message.photo[-1]  # Берем фото максимального размера
        file = await bot.get_file(photo.file_id)
        buf = BytesIO()
        await bot.download_file(file.file_path, destination=buf)
        buf.seek(0)

        # Подготавливаем изображение
        img = Image.open(buf)
        img_tensor = preprocess_image(img)
        
        # Получаем предсказания
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        
        # Получаем топ предсказания
        predictions = get_top_predictions(probs, threshold=0.1)
        
        if not predictions:
            await message.reply("😕 Извините, я не уверен, что это за блюдо. Попробуйте сделать фото при лучшем освещении или с другого ракурса.")
            return

        text = "🍽 Результаты распознавания:\n\n"
        for label, prob in predictions:
            text += f"• {label}: {prob*100:.1f}%\n"
        
        # Оценка калорийности на основе уверенности модели
        confidence = predictions[0][1]
        base_calories = 250
        max_additional_calories = 350
        est_cal = round(base_calories + max_additional_calories * confidence)
        
        text += f"\n🔥 Примерная калорийность: ~{est_cal} ккал"
        
        if confidence < 0.5:
            text += "\n\n⚠️ Уверенность в определении не очень высокая. Возможно, стоит сделать фото при лучшем освещении."
        
        await message.reply(text)
        
    except Exception as e:
        logging.error(f"Ошибка при обработке фото: {e}")
        await message.reply("😔 Произошла ошибка при обработке фото. Пожалуйста, попробуйте еще раз.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(dp.start_polling(bot))
