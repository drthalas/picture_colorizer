from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from app.config import load_settings
from app.services.local_vlm import LocalVLMClient


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Check local Ollama Qwen2.5-VL image analysis.")
    parser.add_argument("input", type=Path, help="Path to an image for VLM analysis.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    client = LocalVLMClient(
        base_url=settings.local_vlm_base_url,
        model=settings.local_vlm_model,
        timeout_seconds=settings.local_vlm_timeout_seconds,
        fallback_model=settings.local_vlm_fallback_model,
    )

    if not client.is_available():
        raise SystemExit(
            f"Local VLM is not available at {settings.local_vlm_base_url}. "
            f"Check Ollama and pull {settings.local_vlm_model}."
        )

    analysis = client.analyze_image(args.input)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
