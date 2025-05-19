import os
import torch
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.fsm.storage.memory import MemoryStorage
from transformers import AutoFeatureExtractor, AutoModelForImageClassification

import logging
import asyncio

# === 1. Загрузка токена из .env ===
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не найден в .env")

# === 2. Настройка ViT на Food101 ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "ashaduzzaman/vit-finetuned-food101"


feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_NAME)
model = AutoModelForImageClassification.from_pretrained(MODEL_NAME).to(device)
model.eval()

# === 3. Настройка Telegram-бота ===
bot = Bot(BOT_TOKEN)
dp = Dispatcher(storage=MemoryStorage())

@dp.message(CommandStart())
async def cmd_start(message: types.Message):
    await message.answer(
        "👋 Привет! Отправь фото блюда, и я скажу, что это за еда, а также оценю калорийность."
    )

@dp.message(F.photo)
async def handle_photo(message: types.Message):
    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    buf = BytesIO()
    await bot.download_file(file.file_path, destination=buf)
    buf.seek(0)

    # Подготавливаем изображение
    img = Image.open(buf).convert("RGB")
    inputs = feature_extractor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
        top1 = torch.topk(probs, 1)

    text = "🍽 Я думаю, это:\n"
    for idx, p in zip(top1.indices, top1.values):
        label = model.config.id2label[idx.item()]
        text += f"• {label}: {p.item()*100:.2f}%\n"

    # Примитивная оценка калорийности
    est_cal = round(250 + 350 * top1.values[0].item())
    text += f"\n🔥 Примерная калорийность: ~{est_cal} ккал"

    await message.reply(text)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(dp.start_polling(bot))
