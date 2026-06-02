from pathlib import Path
from contextlib import redirect_stderr
from io import StringIO
from unittest import TestCase
from unittest.mock import patch
import json

import numpy as np
from PIL import Image, ImageDraw

from app.access_control import AccessStore
from app.cli import build_parser
from app.image_pipeline import (
    AVAILABLE_MODES,
    DEFAULT_MODE,
    _select_vlm_pipeline_mode,
    colorize_document,
    get_auto_vlm_warning,
    validate_mode,
)
from app.main import build_access_review_keyboard, build_connect_keyboard, build_mode_keyboard
from app.pipeline.archive_document_4050 import build_text_mask, preprocess_document, settings_from_env
from app.services.local_vlm import LocalVLMClient, parse_json_response


class ModeTests(TestCase):
    def test_default_mode_is_vintage(self) -> None:
        self.assertEqual(DEFAULT_MODE, "vintage")

    def test_archive_document_4050_mode_is_registered(self) -> None:
        self.assertIn("archive_document_4050", AVAILABLE_MODES)

    def test_auto_vlm_mode_is_registered(self) -> None:
        self.assertIn("auto_vlm", AVAILABLE_MODES)

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
                if mode == "auto_vlm":
                    continue
                output_path = temp_path / f"{mode}.jpg"
                colorize_document(input_path, output_path, mode=mode)

                self.assertTrue(output_path.exists())
                self.assertGreater(output_path.stat().st_size, 0)

    def test_archive_document_pipeline_preserves_size(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "document.png"
            output_path = temp_path / "archive.jpg"
            image = Image.new("RGB", (260, 180), (220, 220, 215))
            draw = ImageDraw.Draw(image)
            draw.text((24, 24), "DOCUMENT 1945", fill="black")
            draw.line((24, 80, 236, 80), fill="black", width=2)
            draw.text((24, 108), "handwritten note", fill=(35, 35, 35))
            image.save(input_path)

            colorize_document(input_path, output_path, mode="archive_document_4050")

            with Image.open(output_path) as result:
                self.assertEqual(result.size, image.size)
            self.assertGreater(output_path.stat().st_size, 0)

    def test_archive_document_text_mask_shape(self) -> None:
        image = Image.new("RGB", (240, 160), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 28, 92, 52), fill="black")
        draw.line((20, 84, 220, 84), fill="black", width=4)

        rgb = np.asarray(image)
        preprocessed = preprocess_document(rgb, settings_from_env())
        mask = build_text_mask(preprocessed)

        self.assertEqual(mask.shape, preprocessed.shape)
        self.assertGreater(int(mask.max()), 0)

    def test_telegram_mode_keyboard_contains_only_public_callbacks(self) -> None:
        keyboard = build_mode_keyboard("abc123")
        callback_data = [button.callback_data for row in keyboard.inline_keyboard for button in row]

        self.assertEqual(
            callback_data,
            [
                "process:abc123:auto_vlm",
                "process:abc123:vintage",
                "process:abc123:clean",
            ],
        )

    def test_connect_keyboard_requests_access(self) -> None:
        keyboard = build_connect_keyboard()
        button = keyboard.inline_keyboard[0][0]

        self.assertEqual(button.text, "Подключить")
        self.assertEqual(button.callback_data, "access:request")

    def test_access_review_keyboard_contains_admin_actions(self) -> None:
        keyboard = build_access_review_keyboard(12345)
        callback_data = [button.callback_data for row in keyboard.inline_keyboard for button in row]

        self.assertEqual(callback_data, ["access:approve:12345", "access:reject:12345"])

    def test_access_store_approves_users_and_always_allows_admin(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            store = AccessStore(Path(temp_dir) / "users.json", admin_user_id=7)

            self.assertTrue(store.is_allowed(7))
            self.assertFalse(store.is_allowed(42))

            store.approve(42)

            self.assertTrue(store.is_allowed(42))

    def test_parse_json_response_accepts_plain_json(self) -> None:
        self.assertEqual(parse_json_response('{"ok": true}'), {"ok": True})

    def test_parse_json_response_accepts_markdown_json_block(self) -> None:
        response = '```json\n{"readability_changed": "better"}\n```'

        self.assertEqual(parse_json_response(response), {"readability_changed": "better"})

    def test_parse_json_response_rejects_invalid_json(self) -> None:
        with self.assertRaises(json.JSONDecodeError):
            parse_json_response("not json")

    def test_local_vlm_is_available_with_mocked_http(self) -> None:
        class FakeClient(LocalVLMClient):
            def _request_json(self, path: str, payload: dict | None, timeout_seconds: int) -> dict:
                return {"models": [{"name": "qwen2.5vl:7b"}]}

        self.assertTrue(FakeClient().is_available())

    def test_non_vlm_pipeline_does_not_call_model_when_disabled(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "input.png"
            output_path = temp_path / "output.jpg"
            Image.new("RGB", (80, 60), "white").save(input_path)

            with patch("app.image_pipeline.LocalVLMClient", side_effect=AssertionError("VLM should not be called")):
                colorize_document(input_path, output_path, mode="standard")

            self.assertTrue(output_path.exists())

    def test_auto_vlm_uses_readability_risk_to_choose_conservative_mode(self) -> None:
        selected = _select_vlm_pipeline_mode(
            {
                "recommended_mode": "archive_document",
                "readability_risk": "high",
                "recommended_settings": {"colorization_strength": 0.12},
            }
        )

        self.assertEqual(selected, "document_readability")

    def test_auto_vlm_routes_text_documents_to_readability_mode(self) -> None:
        selected = _select_vlm_pipeline_mode(
            {
                "image_type": "document",
                "has_text": True,
                "has_stamp": True,
                "recommended_mode": "archive_document",
                "readability_risk": "low",
                "recommended_settings": {"colorization_strength": 0.35},
            }
        )

        self.assertEqual(selected, "document_readability")

    def test_auto_vlm_rejection_warning_is_exposed(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "result.jpg"
            compare_path = Path(temp_dir) / "result.vlm_compare.json"
            compare_path.write_text(
                json.dumps(
                    {
                        "should_accept_result": False,
                        "reason": "text became less readable",
                    }
                ),
                encoding="utf-8",
            )

            warning = get_auto_vlm_warning(output_path)

            self.assertIsNotNone(warning)
            self.assertIn("text became less readable", warning or "")

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

    def test_vintage_tints_handwritten_strokes_blue(self) -> None:
        from tempfile import TemporaryDirectory

        image = Image.new("L", (360, 220), 210)
        draw = ImageDraw.Draw(image)
        draw.text((28, 24), "PRINTED HEADER", fill=20)
        draw.line((28, 62, 330, 62), fill=20, width=2)
        draw.line((28, 120, 330, 120), fill=20, width=2)
        draw.line((28, 62, 28, 178), fill=20, width=2)
        draw.line((330, 62, 330, 178), fill=20, width=2)
        draw.line((56, 92, 116, 82, 176, 96, 236, 84, 300, 98), fill=35, width=4, joint="curve")

        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_path = temp_path / "handwriting.png"
            output_path = temp_path / "vintage.jpg"
            image.convert("RGB").save(input_path)

            colorize_document(input_path, output_path, mode="vintage")

            result = np.asarray(Image.open(output_path).convert("RGB"))
            handwriting_region = result[76:108, 48:310]
            blue_pixels = handwriting_region[:, :, 2] > handwriting_region[:, :, 0] + 16

            self.assertGreater(float(blue_pixels.mean()), 0.015)
