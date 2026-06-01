from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    admin_user_id: int | None = None
    output_dir: str = "data/output"
    authorized_users_path: str = "data/authorized_users.json"
    debug_image_pipeline: bool = False
    image_max_long_side: int = 2800


def _parse_optional_int(value: str | None) -> int | None:
    if value is None or not value.strip():
        return None
    return int(value)


def load_settings() -> Settings:
    load_dotenv()
    admin_user_id = os.getenv("ADMIN_USER_ID") or os.getenv("TELEGRAM_ADMIN_USER_ID")

    return Settings(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        admin_user_id=_parse_optional_int(admin_user_id),
        output_dir=os.getenv("OUTPUT_DIR", "data/output"),
        authorized_users_path=os.getenv("AUTHORIZED_USERS_PATH", "data/authorized_users.json"),
        debug_image_pipeline=os.getenv("DEBUG_IMAGE_PIPELINE", "").lower() in {"1", "true", "yes", "on"},
        image_max_long_side=int(os.getenv("IMAGE_MAX_LONG_SIDE", "2800")),
    )
