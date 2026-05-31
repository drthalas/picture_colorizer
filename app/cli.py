from argparse import ArgumentParser
from pathlib import Path

from app.image_pipeline import colorize_document


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Softly colorize a black-and-white archival paper image.")
    parser.add_argument("input", type=Path, help="Path to the source image.")
    parser.add_argument("output", type=Path, help="Path for the colorized image.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = colorize_document(args.input, args.output)
    print(f"Saved colorized image to {output}")


if __name__ == "__main__":
    main()
