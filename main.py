import logging
import os
from pathlib import Path
import time

from threading import Thread, Lock
from queue import Queue
from typing import Tuple

import telebot
from dotenv import load_dotenv

from speech_recognizer_fast import SpeechRecognizerFast
from telebot.apihelper import ApiTelegramException
from queue import Queue
import gc
import torch
from user_manager import UserManager

# ────────────────────────────────
# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("bot")

# ────────────────────────────────
# Настройка окружения
load_dotenv()
API_KEY = os.getenv("API_KEY")

if not API_KEY:
    raise ValueError("Переменная API_KEY не найдена в .env")

bot = telebot.TeleBot(API_KEY)
AUDIO_SAVE_PATH = Path("audio_files/input")
AUDIO_SAVE_PATH.mkdir(parents=True, exist_ok=True)

ADMIN_ID = int(os.getenv("ADMIN_ID"))  # замени на свой Telegram ID
user_manager = UserManager(admin_id=ADMIN_ID)

recognizer = SpeechRecognizerFast()


# Очередь задач: (message, file_path)
task_queue = Queue()
queue_lock = Lock()
is_processing = False

# ────────────────────────────────
# Обработчики


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    logger.info(f"Команда {message.text} от {message.chat.id}")
    bot.reply_to(message, "🎙 Отправь голосовое сообщение, и я его расшифрую!")


@bot.message_handler(commands=["queue"])
def show_queue(message):
    if message.chat.id != ADMIN_ID:
        return
    size = task_queue.qsize()
    status = "обрабатывается" if is_processing else "свободен"
    bot.reply_to(message, f"Очередь: {size} задач | Статус: {status}")


@bot.message_handler(commands=["adduser"])
def add_user_command(message):
    if message.chat.id != user_manager.admin_id:
        bot.reply_to(
            message, "🚫 Только администратор может добавлять пользователей.")
        return

    try:
        _, new_user_id = message.text.split(maxsplit=1)
        new_user_id = int(new_user_id)
        user_manager.add_user(new_user_id)
        bot.reply_to(message, f"✅ Пользователь {new_user_id} добавлен.")
    except Exception:
        bot.reply_to(message, "Использование: /adduser <user_id>")


@bot.message_handler(commands=["listusers"])
def list_users_command(message):
    if message.chat.id != user_manager.admin_id:
        bot.reply_to(
            message, "🚫 Только администратор может смотреть список пользователей.")
        return

    users = user_manager.list_users()
    if not users:
        bot.reply_to(message, "📭 Список пуст.")
    else:
        bot.reply_to(message, "📜 Разрешённые пользователи:\n" +
                     "\n".join(map(str, users)))


def audio_worker(audio_path: str, result_queue: Queue):
    """
    Должна быть определена на верхнем уровне модуля.
    Ничего не принимает из главного процесса, кроме примитивов.
    """
    try:
        # Создаём recognizer ЗДЕСЬ, в дочернем процессе
        # recognizer = SpeechRecognizerFast()
        text = recognizer.transcribe_audio(audio_path)
        result_queue.put(text)
    except Exception as e:
        result_queue.put(f"[ОШИБКА] {e}")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, message.text)


def split_text_by_chars(text: str, max_len: int):
    """Разбивает текст на куски до max_len символов, не разрывая слова."""
    chunks = []
    start = 0
    while start < len(text):
        if len(text) - start <= max_len:
            # Остаток текста меньше лимита — добавляем всё
            chunks.append(text[start:].strip())
            break

        # Ищем ближайший пробел перед границей max_len
        end = text.rfind(" ", start, start + max_len)
        if end == -1:
            # Если пробела нет, просто режем по лимиту
            end = start + max_len
        chunks.append(text[start:end].strip())
        start = end + 1  # начинаем после пробела
    return chunks


def start_bot():

    while True:
        try:
            logger.info("Бот запущен, ожидание сообщений...")
            # bot.polling(none_stop=True)
            bot.polling(none_stop=True, interval=3, timeout=20)
        except ApiTelegramException as e:
            logger.error(f"Ошибка Telegram API: {str(e)}")
            time.sleep(15)  # Ждём перед перезапуском
        except Exception as e:
            logger.error(f"Общая ошибка: {str(e)}")
            time.sleep(15)  # Ждём перед перезапуском


@bot.message_handler(content_types=["audio", "voice", "video"])
def handle_audio(message):
    try:
        user_id = message.chat.id

        # Проверяем разрешён ли пользователь
        if not user_manager.is_allowed(user_id):
            bot.reply_to(
                message, "⛔ У вас нет доступа к этому боту. Запрос отправлен администратору.")

            # Уведомляем администратора
            try:
                bot.send_message(
                    user_manager.admin_id,
                    f"🚫 Неизвестный пользователь пытается использовать бота:\n"
                    f"👤 Имя: {message.from_user.full_name}\n"
                    f"💬 Username: @{message.from_user.username or '—'}\n"
                    f"🆔 ID: {user_id}\n\n"
                    f"Добавить его можно командой:\n/adduser {user_id}"
                )
            except Exception as e:
                logger.error(f"Ошибка при уведомлении администратора: {e}")
                return

            # Определяем тип файла и параметры
        if message.audio:
            file_id = message.audio.file_id
            file_name = message.audio.file_name or f"audio_{message.message_id}"
            file_size = message.audio.file_size
            file_type = "audio"
        elif message.voice:
            file_id = message.voice.file_id
            file_name = f"voice_{message.message_id}"
            file_size = message.voice.file_size
            file_type = "voice"
        elif message.video:
            file_id = message.video.file_id
            file_name = message.video.file_name or f"video_{message.message_id}"
            file_size = message.video.file_size
            file_type = "video"
        else:
            bot.reply_to(message, "Неизвестный тип файла.")
            return

        logger.info(
            f"Получен файл: {file_name}, file_id: {file_id}, размер: {file_size} байт")

        logger.info(f"Сообщение от {message.chat.id}")

        file_info = bot.get_file(
            message.audio.file_id if message.audio else message.voice.file_id
        )

        original_extension = os.path.splitext(
            file_info.file_path)[1].lower()

        # Проверка формата
        supported_formats = ['.ogg', '.oga',
                             '.mp3', '.wav', '.m4a', '.flac']

        if original_extension not in supported_formats:
            bot.reply_to(
                message,
                f"Формат файла {original_extension} не поддерживается.\n"
                f"Поддерживаемые форматы: {', '.join(f.upper() for f in supported_formats)}."
            )
            return

            # ext = ".ogg" if message.voice else ".mp3"
            # file_name = f"{message.chat.id}_{message.message_id}{ext}"

            # Получаем file_path и скачиваем файл
        file_info = bot.get_file(file_id)
        file_path = AUDIO_SAVE_PATH / file_name

        print(file_path)
        print(file_name)
        # ... (весь код до скачивания файла) ...

        downloaded_file = bot.download_file(file_info.file_path)
        file_path = AUDIO_SAVE_PATH / file_name
        with open(file_path, "wb") as f:
            f.write(downloaded_file)

        # Проверяем, есть ли уже кто-то в очереди
        queue_size = task_queue.qsize()
        if queue_size == 0:
            bot.reply_to(message, "Принял голосовое. Начинаю распознавание...")
        else:
            bot.reply_to(
                message, f"В очереди {queue_size} запрос(ов). Ожидайте...")

        # Добавляем в очередь
        task_queue.put((message, file_path))

    except Exception as e:
        logger.exception("Ошибка при приёме файла")
        bot.reply_to(message, f"Ошибка: {e}")


def transcription_worker():
    global is_processing
    while True:
        message, file_path = task_queue.get()
        if message is None:  # сигнал остановки
            break

        try:
            with queue_lock:
                is_processing = True

            bot.send_message(message.chat.id, "Обрабатываю ваш запрос...")

            start_time = time.time()
            text = recognizer.transcribe_audio(str(file_path))

            duration = time.time() - start_time
            duration_text = f"Время распознавания: {duration:.2f} сек."

            bot.send_message(message.chat.id, duration_text)

            MAX_LEN = 4000
            chunks = split_text_by_chars(text, MAX_LEN)
            for chunk in chunks:
                bot.send_message(
                    message.chat.id, f"Распознанный текст:\n{chunk}")

        except Exception as e:
            bot.send_message(message.chat.id, f"Ошибка: {e}")
            logger.exception("Ошибка в воркере")
        finally:
            try:
                file_path.unlink()
            except:
                pass
            with queue_lock:
                is_processing = False
            task_queue.task_done()


# Запускаем при старте
worker_thread = Thread(target=transcription_worker, daemon=True)
worker_thread.start()


if __name__ == "__main__":
    start_bot()
