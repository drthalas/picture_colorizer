from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    output_dir: str = "data/output"
    debug_image_pipeline: bool = False
    image_max_long_side: int = 2800


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        output_dir=os.getenv("OUTPUT_DIR", "data/output"),
        debug_image_pipeline=os.getenv("DEBUG_IMAGE_PIPELINE", "").lower() in {"1", "true", "yes", "on"},
        image_max_long_side=int(os.getenv("IMAGE_MAX_LONG_SIDE", "2800")),
    )
