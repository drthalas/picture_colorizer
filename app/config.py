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
    local_vlm_enabled: bool = False
    local_vlm_provider: str = "ollama"
    local_vlm_base_url: str = "http://127.0.0.1:11434"
    local_vlm_model: str = "qwen2.5vl:7b"
    local_vlm_fallback_model: str = "qwen2.5vl:3b"
    local_vlm_timeout_seconds: int = 180
    local_ocr_enabled: bool = False
    local_ocr_provider: str = "paddleocr"
    local_ocr_lang: str = "ru"
    local_ocr_min_confidence: float = 0.35
    local_ocr_min_similarity: float = 0.58
    local_ocr_max_text_drop_ratio: float = 0.35
    document_restorer_provider: str = "opencv"


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
        local_vlm_enabled=os.getenv("LOCAL_VLM_ENABLED", "").lower() in {"1", "true", "yes", "on"},
        local_vlm_provider=os.getenv("LOCAL_VLM_PROVIDER", "ollama"),
        local_vlm_base_url=os.getenv("LOCAL_VLM_BASE_URL", "http://127.0.0.1:11434"),
        local_vlm_model=os.getenv("LOCAL_VLM_MODEL", "qwen2.5vl:7b"),
        local_vlm_fallback_model=os.getenv("LOCAL_VLM_FALLBACK_MODEL", "qwen2.5vl:3b"),
        local_vlm_timeout_seconds=int(os.getenv("LOCAL_VLM_TIMEOUT_SECONDS", "180")),
        local_ocr_enabled=os.getenv("LOCAL_OCR_ENABLED", "").lower() in {"1", "true", "yes", "on"},
        local_ocr_provider=os.getenv("LOCAL_OCR_PROVIDER", "paddleocr"),
        local_ocr_lang=os.getenv("LOCAL_OCR_LANG", "ru"),
        local_ocr_min_confidence=float(os.getenv("LOCAL_OCR_MIN_CONFIDENCE", "0.35")),
        local_ocr_min_similarity=float(os.getenv("LOCAL_OCR_MIN_SIMILARITY", "0.58")),
        local_ocr_max_text_drop_ratio=float(os.getenv("LOCAL_OCR_MAX_TEXT_DROP_RATIO", "0.35")),
        document_restorer_provider=os.getenv("DOCUMENT_RESTORER_PROVIDER", "opencv"),
    )
