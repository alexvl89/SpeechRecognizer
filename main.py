import logging
import os
from pathlib import Path
import time

import telebot
from dotenv import load_dotenv

from speech_recognizer_fast import SpeechRecognizerFast
from telebot.apihelper import ApiTelegramException


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

recognizer = SpeechRecognizerFast()

# ────────────────────────────────
# Обработчики


@bot.message_handler(commands=["start", "help"])
def send_welcome(message):
    logger.info(f"Команда {message.text} от {message.chat.id}")
    bot.reply_to(message, "🎙 Отправь голосовое сообщение, и я его расшифрую!")


@bot.message_handler(content_types=["audio", "voice", "video"])
def handle_audio(message):
    try:
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

        original_extension = os.path.splitext(file_info.file_path)[1].lower()

        # Проверка формата
        supported_formats = ['.ogg', '.oga', '.mp3', '.wav', '.m4a', '.flac']

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

        downloaded_file = bot.download_file(file_info.file_path)

        with open(file_path, "wb") as f:
            f.write(downloaded_file)

        bot.reply_to(message, "🎧 Распознаю аудио, подожди немного...")
        text = recognizer.transcribe_audio(str(file_path))

        print(text)

        # максимальная длина текста
        MAX_LEN = 4000
        for chunk in [text[i:i+MAX_LEN] for i in range(0, len(text), MAX_LEN)]:
            # bot.send_message(chat_id, chunk)
            bot.reply_to(message, f"🗣 Распознанный текст:\n{chunk}")

        # summary = recognizer.summarize_text(text)
        # if summary:
        #     bot.reply_to(message, f"📝 Краткий пересказ:\n{summary}")

    except Exception as e:
        logger.exception("Ошибка при обработке аудио")
        bot.reply_to(message, f"⚠️ Ошибка: {e}")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, message.text)


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


if __name__ == "__main__":
    start_bot()
