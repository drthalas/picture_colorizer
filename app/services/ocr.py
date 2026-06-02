from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from itertools import zip_longest
import importlib
import importlib.util
from pathlib import Path
import re
from typing import Any

import numpy as np


class OCRUnavailableError(RuntimeError):
    """Raised when the configured OCR backend cannot run locally."""


@dataclass(frozen=True)
class OCRItem:
    text: str
    confidence: float
    box: list[list[float]] | None = None


@dataclass(frozen=True)
class OCRResult:
    provider: str
    image_path: str
    available: bool
    items: list[OCRItem]
    error: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(item.text for item in self.items if item.text.strip())

    @property
    def mean_confidence(self) -> float:
        if not self.items:
            return 0.0
        return sum(item.confidence for item in self.items) / len(self.items)

    def to_dict(self) -> dict:
        return {
            "provider": self.provider,
            "image_path": self.image_path,
            "available": self.available,
            "text": self.text,
            "text_count": len(self.items),
            "mean_confidence": self.mean_confidence,
            "items": [asdict(item) for item in self.items],
            "error": self.error,
        }


@dataclass
class PaddleOCRTextReader:
    lang: str = "ru"
    min_confidence: float = 0.35
    provider: str = "paddleocr"

    def __post_init__(self) -> None:
        self._ocr: Any | None = None

    def is_available(self) -> bool:
        return importlib.util.find_spec("paddleocr") is not None

    def read_text(self, image_path: str | Path) -> OCRResult:
        image_path = Path(image_path)
        if not self.is_available():
            raise OCRUnavailableError("paddleocr is not installed")

        raw_result = self._run_ocr(image_path)
        items = [
            item
            for item in parse_paddleocr_result(raw_result)
            if item.text.strip() and item.confidence >= self.min_confidence
        ]
        return OCRResult(
            provider=self.provider,
            image_path=str(image_path),
            available=True,
            items=items,
        )

    def _run_ocr(self, image_path: Path) -> Any:
        ocr = self._get_ocr()
        if hasattr(ocr, "predict"):
            return ocr.predict(input=str(image_path))
        if hasattr(ocr, "ocr"):
            try:
                return ocr.ocr(str(image_path), cls=True)
            except TypeError:
                return ocr.ocr(str(image_path))
        raise OCRUnavailableError("PaddleOCR object does not expose predict() or ocr()")

    def _get_ocr(self) -> Any:
        if self._ocr is not None:
            return self._ocr

        module = importlib.import_module("paddleocr")
        paddle_ocr = getattr(module, "PaddleOCR")
        init_attempts = (
            {"lang": self.lang, "use_textline_orientation": True},
            {"lang": self.lang, "use_angle_cls": True, "show_log": False},
            {"lang": self.lang},
        )
        last_error: Exception | None = None
        for kwargs in init_attempts:
            try:
                self._ocr = paddle_ocr(**kwargs)
                return self._ocr
            except TypeError as exc:
                last_error = exc

        raise OCRUnavailableError(f"Could not initialize PaddleOCR: {last_error}")


def parse_paddleocr_result(raw_result: Any) -> list[OCRItem]:
    items = _parse_v3_result(raw_result)
    if items:
        return items
    return _parse_v2_result(raw_result)


def compare_ocr_results(
    before: OCRResult,
    after: OCRResult,
    min_similarity: float = 0.58,
    max_text_drop_ratio: float = 0.35,
) -> dict:
    if not before.available or not after.available:
        return {
            "available": False,
            "should_accept_result": True,
            "reason": "OCR unavailable; skipped text preservation check.",
        }

    before_text = normalize_ocr_text(before.text)
    after_text = normalize_ocr_text(after.text)
    before_chars = len(before_text)
    after_chars = len(after_text)

    if before_chars == 0:
        return {
            "available": True,
            "should_accept_result": True,
            "similarity": 1.0,
            "text_drop_ratio": 0.0,
            "before_chars": before_chars,
            "after_chars": after_chars,
            "reason": "OCR found no source text to protect.",
        }

    similarity = SequenceMatcher(None, before_text, after_text).ratio()
    text_drop_ratio = max(0.0, (before_chars - after_chars) / before_chars)
    should_accept = similarity >= min_similarity and text_drop_ratio <= max_text_drop_ratio
    reason = "OCR text preservation check passed."
    if not should_accept:
        reason = (
            "OCR detected possible text loss: "
            f"similarity={similarity:.3f}, text_drop_ratio={text_drop_ratio:.3f}."
        )

    return {
        "available": True,
        "should_accept_result": should_accept,
        "similarity": similarity,
        "text_drop_ratio": text_drop_ratio,
        "before_chars": before_chars,
        "after_chars": after_chars,
        "before_text_count": len(before.items),
        "after_text_count": len(after.items),
        "before_mean_confidence": before.mean_confidence,
        "after_mean_confidence": after.mean_confidence,
        "reason": reason,
    }


def normalize_ocr_text(text: str) -> str:
    normalized = text.lower().replace("ё", "е")
    return re.sub(r"\s+", " ", normalized).strip()


def unavailable_ocr_result(provider: str, image_path: str | Path, error: Exception) -> OCRResult:
    return OCRResult(
        provider=provider,
        image_path=str(image_path),
        available=False,
        items=[],
        error=str(error),
    )


def _parse_v3_result(raw_result: Any) -> list[OCRItem]:
    items: list[OCRItem] = []
    pages = raw_result if isinstance(raw_result, list) else [raw_result]
    for page in pages:
        data = _result_mapping(page)
        texts = _get_result_value(data, "rec_texts") or _get_result_value(data, "texts")
        scores = _get_result_value(data, "rec_scores") or _get_result_value(data, "scores")
        boxes = _get_result_value(data, "rec_polys") or _get_result_value(data, "rec_boxes") or []
        if not isinstance(texts, list):
            continue
        if not isinstance(scores, list):
            scores = [1.0] * len(texts)
        for text, score, box in zip_longest(texts, scores, boxes, fillvalue=None):
            if text is None:
                continue
            items.append(OCRItem(text=str(text), confidence=_safe_float(score, 1.0), box=_normalize_box(box)))
    return items


def _parse_v2_result(raw_result: Any) -> list[OCRItem]:
    items: list[OCRItem] = []
    for entry in _flatten_ocr_entries(raw_result):
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        box = _normalize_box(entry[0])
        text_score = entry[1]
        if isinstance(text_score, (list, tuple)) and text_score:
            text = str(text_score[0])
            score = _safe_float(text_score[1] if len(text_score) > 1 else 1.0, 1.0)
            items.append(OCRItem(text=text, confidence=score, box=box))
    return items


def _flatten_ocr_entries(value: Any) -> list:
    if not isinstance(value, list):
        return []
    if _looks_like_v2_entry(value):
        return [value]
    entries: list = []
    for item in value:
        entries.extend(_flatten_ocr_entries(item))
    return entries


def _looks_like_v2_entry(value: list) -> bool:
    return (
        len(value) >= 2
        and isinstance(value[0], (list, tuple))
        and isinstance(value[1], (list, tuple))
        and value[1]
        and isinstance(value[1][0], str)
    )


def _result_mapping(value: Any) -> Any:
    if isinstance(value, dict):
        return value
    if hasattr(value, "json"):
        try:
            json_value = value.json
            if callable(json_value):
                json_value = json_value()
            if isinstance(json_value, dict):
                if isinstance(json_value.get("res"), dict):
                    return json_value["res"]
                return json_value
        except Exception:
            return value
    return value


def _get_result_value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_box(box: Any) -> list[list[float]] | None:
    if box is None:
        return None
    if isinstance(box, np.ndarray):
        box = box.tolist()
    if not isinstance(box, (list, tuple)):
        return None
    if box and all(isinstance(value, (int, float)) for value in box):
        if len(box) == 4:
            x1, y1, x2, y2 = [float(value) for value in box]
            return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        return None
    normalized: list[list[float]] = []
    for point in box:
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            normalized.append([_safe_float(point[0], 0.0), _safe_float(point[1], 0.0)])
    return normalized or None
