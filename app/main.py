import asyncio
import logging
from pathlib import Path
from uuid import uuid4

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.config import load_settings
from app.image_pipeline import colorize_document


dp = Dispatcher()
logger = logging.getLogger(__name__)

MODE_BUTTONS: tuple[tuple[str, str], ...] = (
    ("archive_document_4050", "📜 Архивный документ 40–50-х"),
    ("vintage", "🖼 Фото/портрет"),
    ("clean", "⚙️ Стандартная обработка"),
)
MODE_DESCRIPTIONS: dict[str, str] = {
    "archive_document_4050": "Архивный документ 40–50-х",
    "vintage": "Фото/портрет",
    "clean": "Стандартная обработка",
}
PENDING_IMAGES: dict[str, Path] = {}


def build_mode_keyboard(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=label, callback_data=f"process:{token}:{mode}")]
            for mode, label in MODE_BUTTONS
        ]
    )


@dp.message(CommandStart())
async def start(message: Message) -> None:
    await message.answer(
        "Send a black-and-white paper document, note, stamp, letter, or archival fragment. "
        "I will ask you to choose a processing mode before returning the result."
    )


@dp.message(F.photo | F.document)
async def handle_image(message: Message, bot: Bot) -> None:
    settings = load_settings()
    output_dir = Path(settings.output_dir)
    upload_dir = output_dir / "uploads"
    output_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    if message.photo:
        file_id = message.photo[-1].file_id
    elif message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
        file_id = message.document.file_id
    else:
        await message.answer("Please send an image file or photo.")
        return

    token = uuid4().hex
    input_path = upload_dir / f"input-{token}"
    file = await bot.get_file(file_id)
    await bot.download_file(file.file_path, destination=input_path)
    PENDING_IMAGES[token] = input_path

    await message.answer(
        "Выберите режим обработки:",
        reply_markup=build_mode_keyboard(token),
    )


@dp.callback_query(F.data.startswith("process:"))
async def handle_mode_choice(callback: CallbackQuery) -> None:
    if not callback.data:
        return

    try:
        _, token, mode = callback.data.split(":", 2)
    except ValueError:
        await callback.answer("Не удалось прочитать выбранный режим.", show_alert=True)
        return

    input_path = PENDING_IMAGES.get(token)
    if input_path is None or not input_path.exists():
        await callback.message.answer("Исходное изображение не найдено. Отправьте фото ещё раз.")
        await callback.answer()
        return

    settings = load_settings()
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"processed-{mode}-{uuid4().hex}.jpg"
    mode_description = MODE_DESCRIPTIONS.get(mode, mode)

    await callback.answer()
    await callback.message.answer(f"Обрабатываю в режиме: {mode_description}…")

    try:
        await asyncio.to_thread(colorize_document, input_path, output_path, mode)
    except Exception:
        logger.exception("Image processing failed for mode %s", mode)
        await callback.message.answer(
            "Не получилось обработать изображение. Попробуйте другое фото или режим стандартной обработки."
        )
        return

    data = output_path.read_bytes()
    await callback.message.answer_photo(
        BufferedInputFile(data, filename=output_path.name),
        caption=f"Готово: {mode_description}",
    )


@dp.message()
async def fallback(message: Message) -> None:
    await message.answer("Send a black-and-white image of a paper document, note, stamp, or archival fragment.")


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Copy .env.example to .env and fill it in.")

    bot = Bot(token=settings.telegram_bot_token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
