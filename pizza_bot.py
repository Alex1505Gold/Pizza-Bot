from dotenv import load_dotenv
from pathlib import Path
import os
import logging
import re
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, Filters, MessageHandler, ConversationHandler

import numpy as np
import tensorflow as tf
import keras
from keras.preprocessing import image

dotenv_path = Path('.env')
load_dotenv(dotenv_path=dotenv_path)

TOKEN = os.getenv('BOT_TOKEN')

model = keras.models.load_model('pizza.keras')

good_probability = 0.865

# Подключаем логирование
logging.basicConfig(
    filename='logfile.txt', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)

def start(update: Update, context):
    user = update.effective_user
    update.message.reply_text(f'Привет {user.full_name}!')

def helpCommand(update: Update, context):
    update.message.reply_text('Доступны следующие команды:\n\
/classify_image\n\
')

def receiveImg(update: Update, context):
    update.message.reply_text('Отправьте картинку, чтобы узнать пицца это или нет ')
    return 'classify_image_state'

def classifyImg(update: Update, context):
    file_img = update.message.photo[-1].file_id
    obj = context.bot.get_file(file_img)
    obj.download("input_img.jpg")
    img_width, img_height = 224, 224
    img = image.load_img('input_img.jpg', target_size = (img_width, img_height))
    logging.info(type(img))
    img = image.img_to_array(img)
    
    img = np.expand_dims(img, axis = 0)
    logging.info(type(img))
    img = img.reshape(1, 224, 224, 3)
    img /= 255
    prediction = model.predict(img, batch_size=1)
    probability = prediction[0][0]
    logging.info(probability)
    if probability < good_probability:
        update.message.reply_text('Не пицца')
    else:
        update.message.reply_text('Пицца')
    return ConversationHandler.END

def documentReply(update: Update, context):
    update.message.reply_text('Бот не поддерживает работу с документами ')
    return 'classify_image_state'

def defaultReply(update: Update, context):
    update.message.reply_text('Чтобы увидеть список доступных команд введите /help')

def main():
    updater = Updater(TOKEN, use_context=True)

    # Получаем диспетчер для регистрации обработчиков
    dp = updater.dispatcher

    # Обработчик диалога
    convHandlers = ConversationHandler(
        entry_points=[CommandHandler('classify_image', receiveImg),
                      ],
        states={
            'classify_image_state': [MessageHandler(Filters.photo, classifyImg),
                             MessageHandler(Filters.document, documentReply),
                             MessageHandler(Filters.text & ~Filters.command, receiveImg),
                             ],
        },
        fallbacks=[]
    )

	# Регистрируем обработчики команд
    dp.add_handler(CommandHandler("start", start))
    dp.add_handler(CommandHandler("help", helpCommand))
    dp.add_handler(convHandlers)
		
	# Регистрируем обработчик текстовых сообщений
    dp.add_handler(MessageHandler(Filters.text & ~Filters.command, defaultReply))
	# Бездиалоговая обработка фотографий
    dp.add_handler(MessageHandler(Filters.photo, classifyImg))
	# Запускаем бота
    updater.start_polling()

	# Останавливаем бота при нажатии Ctrl+C
    updater.idle()


if __name__ == '__main__':
    main()
