from dataclasses import dataclass
import os

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    telegram_bot_token: str
    output_dir: str = "data/output"


def load_settings() -> Settings:
    load_dotenv()
    return Settings(
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        output_dir=os.getenv("OUTPUT_DIR", "data/output"),
    )
