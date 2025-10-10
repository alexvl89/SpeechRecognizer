import telebot
from dotenv import load_dotenv
import os
import logging  # Импортируем модуль logging

from speech_recognizer import SpeechRecognizer


# Настройка логгера
logging.basicConfig(
    # Уровень логирования (можно изменить на DEBUG для более подробного вывода)
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",  # Формат сообщений
    handlers=[
        logging.StreamHandler()  # Вывод логов в консоль
    ]
)
logger = logging.getLogger(__name__)  # Создаем экземпляр логгера


# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение токена из переменной окружения
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise ValueError(
        "API_KEY не найден. Убедитесь, что он указан в .env файле.")

# Создание экземпляра бота с полученным токеном
bot = telebot.TeleBot(API_KEY)

# Каталог для сохранения аудиофайлов
AUDIO_SAVE_PATH = "audio_files\\input"

# Создание каталога, если он не существует
os.makedirs(AUDIO_SAVE_PATH, exist_ok=True)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    logger.info(
        f"Получена команда: {message.text} от пользователя {message.chat.id}")
    bot.reply_to(message, "Привет! Я ваш бот.")


@bot.message_handler(content_types=['audio', 'voice'])
def handle_audio(message):
    try:
        logger.info(
            f"Получено аудиосообщение от пользователя {message.chat.id}")
        # Проверка, является ли сообщение пересланным
        if message.forward_from or message.forward_from_chat:
            bot.reply_to(
                message, "Вы отправили пересланное аудио. Обрабатываю...")

        # Получение файла
        file_info = bot.get_file(
            message.audio.file_id if message.audio else message.voice.file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        # Определение имени файла
        file_extension = ".ogg" if message.voice else ".mp3"
        file_name = f"{message.chat.id}_{message.message_id}{file_extension}"
        file_path = os.path.join(AUDIO_SAVE_PATH, file_name)

        # Сохранение файла
        with open(file_path, "wb") as audio_file:
            audio_file.write(downloaded_file)

        logger.info(f"Аудиофайл сохранен: {file_path}")
        bot.reply_to(
            message, f"Аудиофайл сохранен: {file_name}. Начинаем распознавание")

        text = SpeechRecognizer.transcribe_audio(
            file_path)

        logger.info(f"Распознанный текст: {text}")
        bot.reply_to(message, f"распознанные слова: {text}")

        logger.info(f"Распознавание завершено. ожидание")


    except Exception as e:
        bot.reply_to(
            message, f"Произошла ошибка при обработке аудиофайла: {str(e)}")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, message.text)


def start_bot():
    print("Бот запущен...")
    bot.polling()


if __name__ == "__main__":
    start_bot()
