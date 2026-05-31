from pathlib import Path
from contextlib import redirect_stderr
from io import StringIO
from unittest import TestCase

from PIL import Image, ImageDraw

from app.cli import build_parser
from app.image_pipeline import AVAILABLE_MODES, DEFAULT_MODE, colorize_document, validate_mode


class ModeTests(TestCase):
    def test_default_mode_is_vintage(self) -> None:
        self.assertEqual(DEFAULT_MODE, "vintage")

    def test_validate_mode_accepts_all_supported_modes(self) -> None:
        for mode in AVAILABLE_MODES:
            self.assertEqual(validate_mode(mode), mode)

    def test_validate_mode_rejects_unknown_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported mode"):
            validate_mode("portrait")

    def test_cli_accepts_supported_modes(self) -> None:
        parser = build_parser()

        for mode in AVAILABLE_MODES:
            args = parser.parse_args(["input.jpg", "output.jpg", "--mode", mode])
            self.assertEqual(args.mode, mode)

    def test_cli_defaults_to_vintage(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["input.jpg", "output.jpg"])

        self.assertEqual(args.mode, DEFAULT_MODE)

    def test_cli_rejects_unknown_mode(self) -> None:
        parser = build_parser()

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["input.jpg", "output.jpg", "--mode", "portrait"])

    def test_colorize_document_writes_output_for_each_mode(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.png"
            image = Image.new("RGB", (220, 140), "white")
            draw = ImageDraw.Draw(image)
            draw.text((20, 35), "NOTE 1943", fill="black")
            draw.ellipse((150, 30, 200, 80), outline=(120, 30, 25), width=3)
            image.save(input_path)

            for mode in AVAILABLE_MODES:
                output_path = temp_path / f"{mode}.jpg"
                colorize_document(input_path, output_path, mode=mode)

                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)
