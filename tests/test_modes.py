from pathlib import Path
from contextlib import redirect_stderr
from io import StringIO
from unittest import TestCase

import numpy as np
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

    def test_vintage_keeps_noisy_document_readable(self) -> None:
        from tempfile import TemporaryDirectory

        rng = np.random.default_rng(1943)
        paper = np.full((220, 320), 178, dtype=np.int16)
        noisy_paper = np.clip(paper + rng.normal(0, 28, paper.shape), 0, 255).astype(np.uint8)
        image = Image.fromarray(noisy_paper).convert("RGB")
        draw = ImageDraw.Draw(image)
        draw.text((24, 24), "ARCHIVE NOTE", fill="black")
        draw.line((24, 78, 296, 78), fill="black", width=2)
        draw.text((24, 100), "Text must stay readable", fill="black")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "noisy.png"
            output_path = temp_path / "vintage.jpg"
            image.save(input_path)

            colorize_document(input_path, output_path, mode="vintage")

            result = np.asarray(Image.open(output_path).convert("L"))
            blank_region = result[130:190, 35:285]
            text_region = result[18:115, 18:302]

            self.assertGreater(float(blank_region.mean()), 155.0)
            self.assertLess(float(blank_region.std()), 28.0)
            self.assertLess(float(text_region.min()), 60.0)

    def test_vintage_preserves_faint_document_marks(self) -> None:
        from tempfile import TemporaryDirectory

        image = Image.new("L", (320, 220), 205)
        draw = ImageDraw.Draw(image)
        draw.text((28, 28), "DARK HEADER", fill=25)
        draw.text((28, 82), "faint printed text", fill=105)
        draw.ellipse((34, 132, 120, 206), outline=112, width=4)
        draw.text((52, 164), "STAMP", fill=112)

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "faint.png"
            output_path = temp_path / "vintage.jpg"
            image.convert("RGB").save(input_path)

            colorize_document(input_path, output_path, mode="vintage")

            result = np.asarray(Image.open(output_path).convert("L"))
            paper_region = result[140:200, 190:290]
            faint_text_region = result[76:108, 24:180]
            stamp_region = result[128:210, 24:130]

            self.assertLess(float(np.percentile(faint_text_region, 5)), float(paper_region.mean()) - 20.0)
            self.assertLess(float(np.percentile(stamp_region, 20)), float(paper_region.mean()) - 12.0)
