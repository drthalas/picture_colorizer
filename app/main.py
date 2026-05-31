import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message

from app.config import load_settings
from app.image_pipeline import DEFAULT_MODE, colorize_document


dp = Dispatcher()


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Send a black-and-white paper document, note, stamp, letter, or archival fragment. "
        "I will return a softly colorized vintage version while keeping text readable."
    )


@dp.message(F.photo | F.document)
async def handle_image(message: Message, bot: Bot) -> None:
    settings = load_settings()
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_path = temp_path / "input_image"
        output_path = output_dir / f"colorized-{uuid4().hex}.jpg"

        if message.photo:
            file_id = message.photo[-1].file_id
        elif message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
            file_id = message.document.file_id
        else:
            await message.answer("Please send an image file or photo.")
            return

        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, destination=input_path)

        try:
            colorize_document(input_path, output_path, mode=DEFAULT_MODE)
        except Exception as exc:
            await message.answer(f"Could not process this image: {exc}")
            return

        data = output_path.read_bytes()
        await message.answer_photo(
            BufferedInputFile(data, filename=output_path.name),
            caption="Soft vintage colorization complete.",
        )


@dp.message()
async def fallback(message: Message) -> None:
    await message.answer("Send a black-and-white image of a paper document, note, stamp, or archival fragment.")


async def main() -> None:
    settings = load_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Copy .env.example to .env and fill it in.")

    bot = Bot(token=settings.telegram_bot_token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
