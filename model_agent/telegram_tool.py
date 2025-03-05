# %%
from telegram import Update
from telegram.ext import (
    MessageHandler,
    filters,
    Application,
    CommandHandler,
    ContextTypes,
)

import os
from pathlib import Path
import asyncio
import tempfile
import ffmpeg
import pytesseract
from PIL import Image
from googletrans import Translator
import speech_recognition as sr
from openai import OpenAI


translator = Translator()
BOT_USERNAME = "@ArbitrageurBot"
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORGANIZATION_ID"),
    project=os.getenv("OPENAI_PROJECT_ID"),
)


# %%


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Sends a welcome message when the bot starts."""
    await update.message.reply_text(
        "Send me text, voice, photo, video, or a document for analysis."
    )


async def download_file(file, file_name: str) -> str:
    """Downloads a file from Telegram and returns the local file path."""
    file_path = Path(tempfile.gettempdir()) / file_name
    await file.download_to_drive(file_path)
    return str(file_path)


def extract_text_from_image(image_path: str) -> str:
    """Extracts text from an image using Tesseract OCR."""
    img = Image.open(image_path)
    return pytesseract.image_to_string(img)


async def transcribe_audio(audio_path):
    """Transcribe audio file using OpenAI Whisper"""
    with open(audio_path, "rb") as f:
        result = client.audio.transcriptions.create(
            model="whisper-1", file=f, language="en"
        )

    return result.text


async def extract_text_from_video(file_path: str) -> str:
    """Extracts audio from a video and transcribes it to text."""
    audio_path = file_path.replace(".mp4", ".wav")
    os.system(f"ffmpeg -i {file_path} {audio_path}")
    return transcribe_audio(audio_path)


async def process_messages(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Processes all incoming messages and sends them to OpenAI LLM for analysis."""
    extracted_text = ""
    chat_id = update.message.chat_id
    message = update.message

    if message.text:
        extracted_text = message.text

    elif message.voice:
        file = await message.voice.get_file()
        file_path = await download_file(file, "voice.ogg")
        extracted_text = await transcribe_audio(file_path)

    elif message.photo:
        file = await message.photo[-1].get_file()
        file_path = await download_file(file, "photo.jpg")
        extracted_text = await extract_text_from_image(file_path)

    elif message.video:
        file = await message.video.get_file()
        file_path = await download_file(file, "video.mp4")
        extracted_text = extract_text_from_video(file_path)

    # elif message.document:
    #     file_extension = message.document.file_name.split(".")[-1].lower()
    #     file_path = await download_file(
    #         update, message.document.file_id, f"document.{file_extension}"
    #     )

    #     if file_extension in ["pdf"]:
    #         extracted_text = extract_text_from_pdf(file_path)
    #     else:
    #         extracted_text = (
    #             f"Received a {file_extension} file. Processing not implemented."
    #         )

    else:
        extracted_text = "Unknown message type received."

    if extracted_text:
        detector = await translator.detect(extracted_text)
        lang = detector.lang
        if lang != "en":
            extracted_text = translator.translate(
                extracted_text, src=lang, dest="en"
            ).text

        response = extracted_text

    else:
        response = "Could not extract useful text from the message."

    message.reply_text(response)
    # context.bot.send_message(chat_id, response)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle incoming voice messages"""
    try:
        # Get voice message
        voice_file = await context.bot.get_file(update.message.voice.file_id)

        # Download the voice file
        file_path = f"voice_{update.message.message_id}.ogg"
        await voice_file.download_to_drive(file_path)

        # Convert to wav format (Whisper requires wav/mp3)
        wav_path = f"voice_{update.message.message_id}.wav"
        os.system(f"ffmpeg -i {file_path} {wav_path}")

        # Transcribe the audio
        transcription = await transcribe_audio(wav_path)

        # Send transcription back to user
        await update.message.reply_text(f"Transcription: {transcription}")

        # Clean up temporary files
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)

    except Exception as e:
        await update.message.reply_text(f"Error processing voice message: {str(e)}")


def main():
    """Main function to start the bot"""

    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Set up message handler
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.ALL, process_messages))

    app.run_polling(poll_interval=5)
    print("Bot is polling...")


if __name__ == "__main__":
    # Check for ffmpeg installation
    # if os.system("ffmpeg -version") != 0:
    #     print("Error: FFmpeg is required for audio conversion. Please install FFmpeg.")
    #     exit(1)

    main()
