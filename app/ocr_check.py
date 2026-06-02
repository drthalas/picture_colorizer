from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from app.config import load_settings
from app.services.ocr import PaddleOCRTextReader, unavailable_ocr_result


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Check optional local PaddleOCR text recognition on an image.")
    parser.add_argument("image", type=Path, help="Path to the image to analyze.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    reader = PaddleOCRTextReader(
        lang=settings.local_ocr_lang,
        min_confidence=settings.local_ocr_min_confidence,
    )

    try:
        result = reader.read_text(args.image)
    except Exception as exc:
        result = unavailable_ocr_result(reader.provider, args.image, exc)

    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
