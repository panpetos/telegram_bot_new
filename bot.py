
import os
import io
import json
import time
import base64
import logging
import re
from uuid import uuid4
from typing import Tuple, List, Dict, Any
import asyncio

import aiohttp
from dotenv import load_dotenv
from PIL import Image, ImageDraw
import numpy as np

from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup, Message
from telegram.ext import Application, MessageHandler, CommandHandler, ContextTypes, filters, CallbackQueryHandler

# ───────── конфиг ─────────
load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY", "")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "https://funfishinggame.store").rstrip("/")
UPLOAD_DIR = "/var/www/funfishinggame.store/html/uploads"

# переводчики (любой из них опционален)
DEEPL_API_KEY = os.getenv("DEEPL_API_KEY", "").strip()
DEEPL_API_URL = os.getenv("DEEPL_API_URL", "https://api-free.deepl.com/v2/translate").strip()
LIBRETRANSLATE_URL = os.getenv("LIBRETRANSLATE_URL", "https://libretranslate.com").rstrip("/")
LIBRETRANSLATE_API_KEY = os.getenv("LIBRETRANSLATE_API_KEY", "").strip()

if not BOT_TOKEN or not STABILITY_API_KEY:
    raise SystemExit("BOT_TOKEN и/или STABILITY_API_KEY не заданы в .env")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs("references", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("window_replacement_bot")

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=180, connect=30)
MAX_IMAGE_DIMENSION = 2048  # пиксели
MAX_IMAGE_PAYLOAD = 7 * 1024 * 1024  # ~7 МБ
LAST_IMAGE_URL: dict[int, str] = {}
USER_SESSIONS: dict[int, dict] = {}
CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

# Категории окон
WINDOW_CATEGORIES = {
    "standard": {
        "name": "🏠 Стандартные окна",
        "description": "Классические прямоугольные окна",
        "prompt": "standard rectangular window with white frame"
    },
    "guillotine": {
        "name": "🪟 Гильотинные окна", 
        "description": "Окна с вертикальным подъемом",
        "prompt": "guillotine sash window with vertical sliding mechanism"
    },
    "frameless": {
        "name": "🔳 Безрамные окна",
        "description": "Современные панорамные окна без видимых рам",
        "prompt": "frameless panoramic window with minimal frame"
    },
    "arched": {
        "name": "🏛️ Арочные окна",
        "description": "Окна с арочным верхом",
        "prompt": "arched window with curved top frame"
    },
    "bay": {
        "name": "🏘️ Эркерные окна",
        "description": "Выступающие многосекционные окна",
        "prompt": "bay window with multiple sections protruding from wall"
    },
    "french": {
        "name": "🇫🇷 Французские окна",
        "description": "Высокие окна-двери до пола",
        "prompt": "french window door extending to floor with glass panels"
    }
}

# ───────── утилиты ─────────
async def download_bytes(url: str) -> bytes:
    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as sess:
        async with sess.get(url) as r:
            r.raise_for_status()
            return await r.read()

async def save_tg_file_to_uploads(file_id: str, context: ContextTypes.DEFAULT_TYPE) -> tuple[str, str]:
    f = await context.bot.get_file(file_id)
    tg_url = f.file_path if f.file_path.startswith("http") else f"https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEg6u4lEdcUdsrmBZnmKYfynYMcaXnnvgxY_Xm578Xy6JybJvdHLKdF0468L1sKMLinW2ssAxKCH-QKhtFS36N3_38zSYFvexVhLLaK9sr1Z1D0Tis48IhphDVmbjgTqVlVH0o4E3HiVivHU/d/hqdefault.jpg"

    ext = os.path.splitext(f.file_path)[1].lower() or ".jpg"
    name = f"{int(time.time())}_{uuid4().hex}{ext}"
    local_path = os.path.join(UPLOAD_DIR, name)

    data = await download_bytes(tg_url)

    # webp -> png
    if ext == ".webp":
        try:
            im = Image.open(io.BytesIO(data)).convert("RGB")
            buf = io.BytesIO()
            im.save(buf, format="PNG")
            data = buf.getvalue()
            name = name.rsplit(".", 1)[0] + ".png"
            local_path = os.path.join(UPLOAD_DIR, name)
        except Exception:
            pass

    with open(local_path, "wb") as out:
        out.write(data)

    return local_path, f"{PUBLIC_BASE_URL}/uploads/{name}"

# ───────── детекция окон ─────────
class WindowDetector:
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Загружаем модель YOLO для детекции окон"""
        try:
            from ultralytics import YOLO
            # Используем предобученную модель YOLOv8
            self.model = YOLO('yolov8n.pt')
            log.info("Модель YOLO загружена успешно")
        except Exception as e:
            log.error(f"Ошибка загрузки модели YOLO: {e}")
            self.model = None
    
    def detect_windows(self, image_path: str) -> List[Dict[str, Any]]:
        """Детектирует окна на изображении"""
        if not self.model:
            return self._fallback_detection(image_path)
        
        try:
            # Запускаем детекцию
            results = self.model(image_path)
            windows = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Фильтруем только окна (класс может варьироваться)
                        # В стандартной COCO модели нет класса "window", поэтому используем fallback
                        pass
            
            # Если YOLO не нашла окна, используем fallback
            if not windows:
                return self._fallback_detection(image_path)
                
            return windows
            
        except Exception as e:
            log.error(f"Ошибка детекции YOLO: {e}")
            return self._fallback_detection(image_path)
    
    def _fallback_detection(self, image_path: str) -> List[Dict[str, Any]]:
        """Упрощенная детекция окон с помощью OpenCV"""
        try:
            import cv2
            
            # Загружаем изображение
            img = cv2.imread(image_path)
            if img is None:
                return []
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Простая эвристика для поиска прямоугольных областей (окон)
            # Применяем размытие и поиск контуров
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 50, 150)
            
            # Находим контуры
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            windows = []
            for contour in contours:
                # Аппроксимируем контур
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Ищем прямоугольные формы
                if len(approx) >= 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Фильтруем по размеру (окна должны быть достаточно большими)
                    if w > width * 0.05 and h > height * 0.05 and w < width * 0.8 and h < height * 0.8:
                        # Проверяем соотношение сторон (окна обычно вытянуты по вертикали или горизонтали)
                        aspect_ratio = w / h
                        if 0.3 < aspect_ratio < 3.0:
                            windows.append({
                                'bbox': [x, y, w, h],
                                'confidence': 0.7,  # Фиксированная уверенность для fallback
                                'area': w * h
                            })
            
            # Сортируем по площади (большие окна сначала)
            windows.sort(key=lambda x: x['area'], reverse=True)
            
            # Ограничиваем количество найденных окон
            return windows[:10]
            
        except Exception as e:
            log.error(f"Ошибка fallback детекции: {e}")
            return []

# Глобальный детектор
window_detector = WindowDetector()

def create_window_mask(image_path: str, windows: List[Dict[str, Any]]) -> Image.Image:
    """Создает маску для найденных окон"""
    try:
        # Загружаем исходное изображение для получения размеров
        with Image.open(image_path) as img:
            width, height = img.size
        
        # Создаем маску
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Рисуем прямоугольники для каждого окна
        for window in windows:
            bbox = window['bbox']
            x, y, w, h = bbox
            # Рисуем белый прямоугольник (255) на черном фоне (0)
            draw.rectangle([x, y, x + w, y + h], fill=255)
        
        return mask
        
    except Exception as e:
        log.error(f"Ошибка создания маски: {e}")
        # Возвращаем пустую маску
        return Image.new('L', (512, 512), 0)

async def call_stability_inpaint(image_bytes: bytes, mask_png: bytes, prompt: str) -> bytes:
    """Вызов Stability API для замены окон"""
    url = "https://api.stability.ai/v2beta/stable-image/edit/inpaint"
    headers = {"Authorization": f"Bearer {STABILITY_API_KEY}", "Accept": "image/*"}

    form = aiohttp.FormData()
    form.add_field("image", image_bytes, filename="image.png", content_type="image/png")
    form.add_field("mask",  mask_png,   filename="mask.png",  content_type="image/png")
    if prompt.strip():
        form.add_field("prompt", prompt.strip())
    form.add_field("output_format", "png")

    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as sess:
        async with sess.post(url, headers=headers, data=form) as resp:
            if resp.status == 413:
                raise RuntimeError(
                    "Stability API вернул 413 (слишком большой запрос). Попробуйте отправить изображение меньшего размера."
                )
            if resp.status != 200:
                txt = await resp.text()
                raise RuntimeError(f"Ошибка Stability API: {resp.status} {txt}")
            ctype = resp.headers.get("Content-Type", "")
            if "image/" not in ctype:
                txt = await resp.text()
                raise RuntimeError(f"Неожиданный content-type: {ctype} | тело: {txt[:400]}")
        return await resp.read()


def _resize_image_and_mask(
    image: Image.Image,
    mask: Image.Image,
    scale: float,
) -> tuple[Image.Image, Image.Image]:
    """Вспомогательная функция для масштабирования изображения и маски."""

    if scale >= 0.999:
        return image, mask

    new_size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    resized_image = image.resize(new_size, Image.LANCZOS)
    resized_mask = mask.resize(new_size, Image.NEAREST)
    return resized_image, resized_mask


def prepare_inpaint_payload(
    image_path: str,
    mask: Image.Image,
    max_dimension: int = MAX_IMAGE_DIMENSION,
    max_bytes: int = MAX_IMAGE_PAYLOAD,
) -> tuple[bytes, bytes]:
    """
    Приводит изображение и маску к приемлемому размеру для Stability API.

    API возвращает 413 при слишком больших payload'ах. Мы уменьшаем изображение и маску,
    сохраняя пропорции, пока размер PNG не станет приемлемым.
    """

    with Image.open(image_path) as src:
        image = src.convert("RGB")

    mask = mask.convert("L")

    # Ограничиваем максимальную сторону
    longest_side = max(image.width, image.height)
    if longest_side > max_dimension:
        scale = max_dimension / float(longest_side)
        image, mask = _resize_image_and_mask(image, mask, scale)

    def encode(img: Image.Image, mk: Image.Image) -> tuple[bytes, bytes]:
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG", optimize=True)
        mask_buf = io.BytesIO()
        mk.save(mask_buf, format="PNG")
        return img_buf.getvalue(), mask_buf.getvalue()

    image_bytes, mask_bytes = encode(image, mask)

    # Если все еще слишком большой payload, уменьшаем постепенно
    while len(image_bytes) + len(mask_bytes) > max_bytes and min(image.width, image.height) > 512:
        image, mask = _resize_image_and_mask(image, mask, 0.85)
        image_bytes, mask_bytes = encode(image, mask)

    return image_bytes, mask_bytes

# ───────── перевод RU→EN ─────────
async def translate_ru_to_en(text: str) -> tuple[str, str | None]:
    """
    Возвращает (перевод, провайдер) либо (исходный, None) если перевода не требуется/не вышло.
    """
    if not text or not CYRILLIC_RE.search(text):
        return text, None

    # 1) DeepL (если есть ключ)
    if DEEPL_API_KEY:
        try:
            headers = {"Authorization": f"DeepL-Auth-Key {DEEPL_API_KEY}"}
            data = aiohttp.FormData()
            data.add_field("text", text)
            data.add_field("target_lang", "EN")
            data.add_field("source_lang", "RU")
            async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as sess:
                async with sess.post(DEEPL_API_URL, headers=headers, data=data) as r:
                    if r.status == 200:
                        j = await r.json()
                        tr = j.get("translations", [{}])[0].get("text")
                        if tr:
                            return tr, "deepl"
                    else:
                        log.warning("Ошибка DeepL %s: %s", r.status, await r.text())
        except Exception as e:
            log.warning("Исключение DeepL: %s", e)

    # 2) LibreTranslate (по умолчанию публичный инстанс)
    try:
        url = f"{LIBRETRANSLATE_URL}/translate"
        payload = {"q": text, "source": "ru", "target": "en", "format": "text"}
        if LIBRETRANSLATE_API_KEY:
            payload["api_key"] = LIBRETRANSLATE_API_KEY
        async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as sess:
            async with sess.post(url, json=payload) as r:
                if r.status == 200:
                    j = await r.json()
                    tr = j.get("translatedText")
                    if tr:
                        return tr, "libre"
                else:
                    log.warning("Ошибка LibreTranslate %s: %s", r.status, await r.text())
    except Exception as e:
        log.warning("Исключение LibreTranslate: %s", e)

    # если всё мимо — возвращаем исходник
    return text, None

# ───────── хендлеры ─────────
async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    welcome_text = """🏠 Добро пожаловать в бот для замены окон на фотографиях домов!

🤖 Что я умею:
• Автоматически находить окна на фото дома
• Заменять их на окна выбранной категории
• Работать с различными архитектурными стилями

📋 Как пользоваться:
1. Отправьте мне фото дома
2. Выберите категорию окон из предложенных
3. Получите результат с замененными окнами

Начните с отправки фотографии дома! 📸"""
    
    await update.message.reply_text(welcome_text)

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = """📖 Справка по использованию бота:

🔹 /start - Начать работу с ботом
🔹 /help - Показать эту справку
🔹 /ping - Проверить работу бота

🏠 Как заменить окна:
1️⃣ Отправьте фото дома боту
2️⃣ Выберите категорию окон из списка
3️⃣ Дождитесь автоматической обработки
4️⃣ Получите результат

🪟 Доступные категории окон:
• Стандартные окна
• Гильотинные окна  
• Безрамные окна
• Арочные окна
• Эркерные окна
• Французские окна

💡 Советы:
• Используйте четкие фото фасадов зданий
• Лучше всего работает с фронтальными видами
• Поддерживаются форматы: JPG, PNG, WebP
• Максимальный размер файла: 20 МБ"""
    
    await update.message.reply_text(help_text)

async def ping_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🟢 Бот работает нормально!")

def create_window_categories_keyboard():
    """Создает клавиатуру с категориями окон"""
    keyboard = []
    for key, category in WINDOW_CATEGORIES.items():
        keyboard.append([InlineKeyboardButton(
            category["name"], 
            callback_data=f"window_category:{key}"
        )])
    
    keyboard.append([InlineKeyboardButton("❌ Отмена", callback_data="cancel")])
    return InlineKeyboardMarkup(keyboard)

async def got_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = update.message
    user_id = update.effective_user.id
    
    if msg.photo:
        file_id = msg.photo[-1].file_id
    else:
        if not (msg.document and msg.document.mime_type and msg.document.mime_type.startswith("image/")):
            await msg.reply_text("❌ Пожалуйста, отправьте изображение дома в виде фото или документа.")
            return
        file_id = msg.document.file_id

    # Показываем индикатор загрузки
    loading_msg = await msg.reply_text("⏳ Загружаю и анализирую изображение...")

    try:
        local_path, public_url = await save_tg_file_to_uploads(file_id, context)
        LAST_IMAGE_URL[user_id] = public_url
        
        # Детектируем окна
        windows = window_detector.detect_windows(local_path)
        
        if not windows:
            await loading_msg.edit_text("❌ На изображении не найдено окон. Попробуйте загрузить фото фасада здания с четко видимыми окнами.")
            return
        
        # Сохраняем информацию о сессии пользователя
        USER_SESSIONS[user_id] = {
            'image_path': local_path,
            'image_url': public_url,
            'windows': windows,
            'timestamp': time.time()
        }
        
        log.info(f"Найдено {len(windows)} окон для пользователя {user_id}")
        
        await loading_msg.edit_text(
            f"✅ Изображение обработано!\n\n"
            f"🔍 Найдено окон: {len(windows)}\n\n"
            f"Выберите категорию окон для замены:",
            reply_markup=create_window_categories_keyboard()
        )
        
    except Exception as e:
        log.error("Ошибка при обработке изображения: %s", e)
        await loading_msg.edit_text(f"❌ Произошла ошибка при обработке изображения: {str(e)}")

async def handle_window_category(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка выбора категории окон"""
    query = update.callback_query
    await query.answer()
    
    user_id = update.effective_user.id
    
    if query.data == "cancel":
        await query.edit_message_text("❌ Операция отменена.")
        return
    
    if not query.data.startswith("window_category:"):
        return
    
    category_key = query.data.split(":", 1)[1]
    
    if category_key not in WINDOW_CATEGORIES:
        await query.edit_message_text("❌ Неизвестная категория окон.")
        return
    
    if user_id not in USER_SESSIONS:
        await query.edit_message_text("❌ Сессия истекла. Пожалуйста, загрузите изображение заново.")
        return
    
    session = USER_SESSIONS[user_id]
    category = WINDOW_CATEGORIES[category_key]
    
    # Показываем индикатор обработки
    await query.edit_message_text(
        f"🎨 Заменяю окна на {category['name'].lower()}...\n\n"
        f"⏳ Это может занять до 2 минут, пожалуйста подождите."
    )
    
    try:
        # Создаем маску для найденных окон
        mask = create_window_mask(session['image_path'], session['windows'])

        # Готовим payload для Stability API (учитывая ограничения размера)
        image_bytes, mask_bytes = prepare_inpaint_payload(session['image_path'], mask)
        
        # Переводим промпт
        prompt = category['prompt']
        translated_prompt, provider = await translate_ru_to_en(prompt)
        if provider and translated_prompt:
            prompt = translated_prompt
        
        # Вызываем Stability API
        result_png = await call_stability_inpaint(image_bytes, mask_bytes, prompt)
        
        # Отправляем результат
        caption = f"✨ Готово! Окна заменены на {category['name'].lower()}\n\n"
        caption += f"📝 Категория: {category['description']}\n"
        caption += f"🔍 Обработано окон: {len(session['windows'])}"
        
        await context.bot.send_photo(
            chat_id=query.message.chat_id,
            photo=InputFile(io.BytesIO(result_png), filename="result.png"),
            caption=caption
        )
        
        # Предлагаем попробовать другую категорию
        await context.bot.send_message(
            chat_id=query.message.chat_id,
            text="🔄 Хотите попробовать другую категорию окон для этого же изображения?",
            reply_markup=create_window_categories_keyboard()
        )
        
        # Удаляем сообщение с индикатором
        try:
            await query.message.delete()
        except:
            pass
            
    except Exception as e:
        log.error(f"Ошибка при замене окон: {e}")
        await query.edit_message_text(f"❌ Ошибка при обработке: {str(e)}")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка текстовых сообщений"""
    text = update.message.text.lower()
    
    if any(word in text for word in ["привет", "hello", "hi", "старт"]):
        await start_cmd(update, context)
    elif any(word in text for word in ["помощь", "справка", "help"]):
        await help_cmd(update, context)
    elif any(word in text for word in ["пинг", "ping", "работает"]):
        await ping_cmd(update, context)
    else:
        help_text = """❓ Я не понимаю эту команду.

🏠 Отправьте мне фото дома для замены окон или используйте команды:
• /start - Начать работу
• /help - Справка  
• /ping - Проверить работу бота"""
        await update.message.reply_text(help_text)

async def log_every_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = getattr(update, "message", None)
    if not msg:
        return
    t = (msg.text or "")[:80]
    log.info("[получено] чат=%s пользователь=%s текст=%s", msg.chat_id, update.effective_user.id, t)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    log.exception("Необработанная ошибка", exc_info=context.error)

# ───────── запуск ─────────
def main():
    app = Application.builder().token(BOT_TOKEN).build()

    # Команды
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("ping", ping_cmd))

    # Обработка callback запросов (кнопки)
    app.add_handler(CallbackQueryHandler(handle_window_category))
    
    # Обработка изображений
    app.add_handler(MessageHandler(filters.PHOTO | filters.Document.IMAGE, got_image))
    
    # Обработка текстовых сообщений
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))
    
    # Логирование всех сообщений
    app.add_handler(MessageHandler(filters.ALL, log_every_message))

    app.add_error_handler(error_handler)

    log.info("🚀 Бот для замены окон запущен")
    app.run_polling(allowed_updates=None, drop_pending_updates=False)

    print("✅ Bot updated via GitHub Actions")


if __name__ == "__main__":
    main()
