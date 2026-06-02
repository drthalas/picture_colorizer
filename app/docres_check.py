from __future__ import annotations

from argparse import ArgumentParser
import json
from pathlib import Path

from app.config import load_settings
from app.services.document_restorer import DocumentRestorerError, create_document_restorer


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Check optional external DocRes document restoration backend.")
    parser.add_argument("input", type=Path, help="Path to the source image.")
    parser.add_argument("output", type=Path, help="Path for the restored image.")
    parser.add_argument("--task", default=None, help="DocRes task: appearance, dewarping, deshadowing, etc.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = load_settings()
    task = args.task or settings.docres_task
    restorer = create_document_restorer(
        "docres",
        repo_dir=settings.docres_repo_dir,
        python_executable=settings.docres_python,
        task=task,
        timeout_seconds=settings.docres_timeout_seconds,
        save_dtsprompt=settings.docres_save_dtsprompt,
    )

    try:
        restored_path = restorer.restore(args.input, args.output)
        result = {
            "available": True,
            "provider": restorer.provider,
            "task": task,
            "output_path": str(restored_path),
        }
    except DocumentRestorerError as exc:
        result = {
            "available": False,
            "provider": "docres",
            "task": task,
            "error": str(exc),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
