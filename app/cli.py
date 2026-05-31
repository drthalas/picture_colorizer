from argparse import ArgumentParser
from pathlib import Path

from app.image_pipeline import AVAILABLE_MODES, DEFAULT_MODE, colorize_document


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Enhance a paper document image while preserving readable text.")
    parser.add_argument("input", type=Path, help="Path to the source image.")
    parser.add_argument("output", type=Path, help="Path for the processed image.")
    parser.add_argument(
        "--mode",
        choices=AVAILABLE_MODES,
        default=DEFAULT_MODE,
        help=f"Processing mode. Defaults to {DEFAULT_MODE}.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = colorize_document(args.input, args.output, mode=args.mode)
    print(f"Saved {args.mode} image to {output}")


if __name__ == "__main__":
    main()
