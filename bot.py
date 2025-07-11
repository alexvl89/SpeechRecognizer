import telebot
from dotenv import load_dotenv
import os

# Загрузка переменных окружения из .env файла
load_dotenv()

# Получение токена из переменной окружения
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    raise ValueError(
        "API_KEY не найден. Убедитесь, что он указан в .env файле.")

# Создание экземпляра бота с полученным токеном
bot = telebot.TeleBot(API_KEY)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Привет! Я ваш бот.")


@bot.message_handler(func=lambda message: True)
def echo_all(message):
    bot.reply_to(message, message.text)


def start_bot():
    print("Бот запущен...")
    bot.polling()


if __name__ == "__main__":
    start_bot()
