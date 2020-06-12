import telepot


def message_telegram(text):
	"""
	Function to write me a message in Telegram thought a bot.
	:param text: (str) Text to write me
	"""
	bot = telepot.Bot('990722479:AAFes17zw8t4S9oSH8-2B_W4StoODQBxnlU')  # Load my bot API
	bot.sendMessage(909417112, text)  # Write the message


def image_telegram(file_path):
	"""
	Function to send me a image in Telegram thought a bot.
	:param file_path: (str) Path to the image that the used wants to send
	"""
	bot = telepot.Bot('990722479:AAFes17zw8t4S9oSH8-2B_W4StoODQBxnlU')  # Load my bot API
	image = open(file_path, 'rb')
	bot.sendPhoto(909417112, image)  # Write the message
