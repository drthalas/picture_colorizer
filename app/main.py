import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
from uuid import uuid4

from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.access_control import AccessStore, TelegramUserInfo
from app.config import load_settings
from app.image_pipeline import colorize_document, get_auto_vlm_warning


dp = Dispatcher()
logger = logging.getLogger(__name__)

MODE_BUTTONS: tuple[tuple[str, str], ...] = (
    ("auto_vlm", "🤖 Авто-анализ локальной моделью"),
    ("vintage", "🖼 Фото/портрет"),
    ("clean", "⚙️ Стандартная обработка"),
)
MODE_DESCRIPTIONS: dict[str, str] = {
    "auto_vlm": "Авто-анализ локальной моделью",
    "archive_document_4050": "Архивный документ 40–50-х",
    "archive_document": "Архивный документ",
    "document_readability": "Читаемость документа",
    "standard": "Стандартная обработка",
    "vintage": "Фото/портрет",
    "clean": "Стандартная обработка",
}


@dataclass(frozen=True)
class PendingImage:
    path: Path
    user_id: int


PENDING_IMAGES: dict[str, PendingImage] = {}


def get_access_store() -> AccessStore:
    settings = load_settings()
    return AccessStore(Path(settings.authorized_users_path), settings.admin_user_id)


def build_mode_keyboard(token: str) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=label, callback_data=f"process:{token}:{mode}")]
            for mode, label in MODE_BUTTONS
        ]
    )


def build_connect_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Подключить", callback_data="access:request")],
        ]
    )


def build_access_review_keyboard(user_id: int) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Подтвердить", callback_data=f"access:approve:{user_id}"),
                InlineKeyboardButton(text="Отклонить", callback_data=f"access:reject:{user_id}"),
            ],
        ]
    )


async def answer_connect_prompt(message: Message) -> None:
    await message.answer(
        "Бот работает по подтверждению доступа. Нажмите «Подключить», и я отправлю запрос администратору.",
        reply_markup=build_connect_keyboard(),
    )


@dp.message(CommandStart())
async def start(message: Message) -> None:
    if not get_access_store().is_allowed(message.from_user.id if message.from_user else None):
        await answer_connect_prompt(message)
        return

    await message.answer(
        "Send a black-and-white paper document, note, stamp, letter, or archival fragment. "
        "I will ask you to choose a processing mode before returning the result."
    )


@dp.message(F.photo | F.document)
async def handle_image(message: Message, bot: Bot) -> None:
    user_id = message.from_user.id if message.from_user else None
    if not get_access_store().is_allowed(user_id):
        await answer_connect_prompt(message)
        return

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
    PENDING_IMAGES[token] = PendingImage(path=input_path, user_id=user_id)

    await message.answer(
        "Выберите режим обработки:",
        reply_markup=build_mode_keyboard(token),
    )


@dp.callback_query(F.data == "access:request")
async def handle_access_request(callback: CallbackQuery, bot: Bot) -> None:
    settings = load_settings()
    access_store = AccessStore(Path(settings.authorized_users_path), settings.admin_user_id)
    user = callback.from_user

    if access_store.is_allowed(user.id):
        await callback.answer("Доступ уже подключён.", show_alert=True)
        return

    if settings.admin_user_id is None:
        await callback.answer("Администратор бота не настроен.", show_alert=True)
        return

    user_info = TelegramUserInfo(
        user_id=user.id,
        full_name=user.full_name,
        username=user.username,
    )
    await bot.send_message(
        settings.admin_user_id,
        f"Запрос доступа к боту:\n{user_info.display_name()}",
        reply_markup=build_access_review_keyboard(user.id),
    )
    await callback.answer("Запрос отправлен администратору.", show_alert=True)


@dp.callback_query(F.data.startswith("access:approve:"))
@dp.callback_query(F.data.startswith("access:reject:"))
async def handle_access_review(callback: CallbackQuery, bot: Bot) -> None:
    if not callback.data:
        return

    settings = load_settings()
    access_store = AccessStore(Path(settings.authorized_users_path), settings.admin_user_id)

    if not access_store.is_admin(callback.from_user.id):
        await callback.answer("Это действие доступно только администратору.", show_alert=True)
        return

    try:
        _, action, user_id_raw = callback.data.split(":", 2)
        user_id = int(user_id_raw)
    except ValueError:
        await callback.answer("Не удалось прочитать запрос.", show_alert=True)
        return

    if action == "approve":
        access_store.approve(user_id)
        await callback.answer("Доступ подтверждён.")
        if callback.message:
            await callback.message.edit_text(f"Доступ подтверждён для пользователя id: {user_id}")
        try:
            await bot.send_message(user_id, "Доступ подтверждён. Теперь можно отправлять фото для обработки.")
        except Exception:
            logger.exception("Failed to notify approved user %s", user_id)
        return

    if action == "reject":
        await callback.answer("Запрос отклонён.")
        if callback.message:
            await callback.message.edit_text(f"Запрос отклонён для пользователя id: {user_id}")
        try:
            await bot.send_message(user_id, "Запрос доступа отклонён.")
        except Exception:
            logger.exception("Failed to notify rejected user %s", user_id)
        return

    await callback.answer("Неизвестное действие.", show_alert=True)


@dp.callback_query(F.data.startswith("process:"))
async def handle_mode_choice(callback: CallbackQuery) -> None:
    if not callback.data:
        return

    user_id = callback.from_user.id
    if not get_access_store().is_allowed(user_id):
        await callback.answer("Сначала нужно подключить доступ.", show_alert=True)
        return

    try:
        _, token, mode = callback.data.split(":", 2)
    except ValueError:
        await callback.answer("Не удалось прочитать выбранный режим.", show_alert=True)
        return

    pending_image = PENDING_IMAGES.get(token)
    if pending_image is None or not pending_image.path.exists():
        await callback.message.answer("Исходное изображение не найдено. Отправьте фото ещё раз.")
        await callback.answer()
        return
    if pending_image.user_id != user_id:
        await callback.answer("Это изображение привязано к другому пользователю.", show_alert=True)
        return

    settings = load_settings()
    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"processed-{mode}-{uuid4().hex}.jpg"
    mode_description = MODE_DESCRIPTIONS.get(mode, mode)

    await callback.answer()
    await callback.message.answer(f"Обрабатываю в режиме: {mode_description}…")

    try:
        await asyncio.to_thread(colorize_document, pending_image.path, output_path, mode)
    except Exception as exc:
        logger.exception("Image processing failed for mode %s", mode)
        if mode == "auto_vlm":
            await callback.message.answer(
                f"Локальный авто-анализ сейчас недоступен: {exc}. Попробуйте режим стандартной обработки."
            )
            return
        await callback.message.answer(
            "Не получилось обработать изображение. Попробуйте другое фото или режим стандартной обработки."
        )
        return

    data = output_path.read_bytes()
    warning = get_auto_vlm_warning(output_path) if mode == "auto_vlm" else None
    if warning:
        await callback.message.answer(warning)

    await callback.message.answer_photo(
        BufferedInputFile(data, filename=output_path.name),
        caption=f"Готово: {mode_description}" if not warning else f"Готово с предупреждением: {mode_description}",
    )


@dp.message()
async def fallback(message: Message) -> None:
    if not get_access_store().is_allowed(message.from_user.id if message.from_user else None):
        await answer_connect_prompt(message)
        return

    await message.answer("Send a black-and-white image of a paper document, note, stamp, or archival fragment.")


async def main() -> None:
    logging.basicConfig(level=logging.INFO)
    settings = load_settings()
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is missing. Copy .env.example to .env and fill it in.")
    if settings.admin_user_id is None:
        raise RuntimeError("ADMIN_USER_ID is missing. Add your Telegram user id to .env.")

    bot = Bot(token=settings.telegram_bot_token)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
