from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
import urllib.error
import urllib.request


logger = logging.getLogger(__name__)

ANALYZE_SYSTEM_PROMPT = (
    "Ты — модуль анализа старых фотографий и архивных документов. "
    "Твоя задача — не фантазировать, а оценивать изображение технически. "
    "Если на изображении есть текст, номера, даты, подписи, печати или таблицы, "
    "главный приоритет — сохранить читаемость и не менять содержание. "
    "Отвечай только валидным JSON без markdown."
)

ANALYZE_USER_PROMPT = """
Проанализируй изображение для обработки в Telegram-боте реставрации.
Верни JSON строго такого вида:

{
  "image_type": "document | photo | portrait | mixed | unknown",
  "has_text": true,
  "has_handwriting": true,
  "has_table": true,
  "has_stamp": false,
  "document_quality": "good | medium | bad",
  "readability_risk": "low | medium | high",
  "recommended_mode": "document_readability | archive_document | photo_color | standard",
  "recommended_settings": {
    "preserve_text_strength": 0.9,
    "contrast_strength": 0.7,
    "denoise_strength": 0.3,
    "colorization_strength": 0.25,
    "vintage_strength": 0.35,
    "sharpen_strength": 0.4
  },
  "warnings": [
    "string"
  ],
  "short_explanation": "string"
}

Критерии:
- Если это документ с текстом, recommended_mode должен быть document_readability или archive_document.
- Если есть риск ухудшить читаемость, colorization_strength должен быть низким.
- Не предлагай агрессивную цветизацию для документов.
- Не предлагай перерисовывать буквы, цифры, подписи и даты.
- Для архивного стиля допускается тёплая бумага, мягкий сине-серый оттенок, слабая винтажность.
- Главный приоритет — читаемость.
""".strip()

COMPARE_USER_PROMPT = """
Сравни исходное изображение и результат обработки.
Оцени, стало ли лучше или хуже с точки зрения читаемости.
Верни JSON строго такого вида:

{
  "readability_changed": "better | same | worse",
  "text_preserved": true,
  "contrast_changed": "better | same | worse",
  "noise_changed": "better | same | worse",
  "color_quality": "good | acceptable | bad | not_applicable",
  "should_accept_result": true,
  "reason": "string"
}

Правило:
Если текст, цифры, даты, подписи или таблицы стали хуже читаемыми, should_accept_result=false,
даже если картинка выглядит красивее.
""".strip()


class LocalVLMError(RuntimeError):
    """Raised when the local VLM cannot complete the requested analysis."""


@dataclass(frozen=True)
class LocalVLMClient:
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen2.5vl:7b"
    timeout_seconds: int = 180
    fallback_model: str = "qwen2.5vl:3b"

    def is_available(self) -> bool:
        try:
            return any(self._model_available(model) for model in self._candidate_models())
        except LocalVLMError:
            return False

    def analyze_image(self, image_path: str | Path) -> dict:
        return self._generate_json([Path(image_path)], ANALYZE_USER_PROMPT)

    def compare_before_after(self, before_path: str | Path, after_path: str | Path) -> dict:
        return self._generate_json([Path(before_path), Path(after_path)], COMPARE_USER_PROMPT)

    def suggest_processing_settings(self, image_path: str | Path) -> dict:
        analysis = self.analyze_image(image_path)
        return {
            "mode": analysis.get("recommended_mode", "standard"),
            "preserve_text_strength": analysis.get("recommended_settings", {}).get("preserve_text_strength", 0.8),
            "contrast_strength": analysis.get("recommended_settings", {}).get("contrast_strength", 0.6),
            "denoise_strength": analysis.get("recommended_settings", {}).get("denoise_strength", 0.3),
            "colorization_strength": analysis.get("recommended_settings", {}).get("colorization_strength", 0.25),
            "vintage_strength": analysis.get("recommended_settings", {}).get("vintage_strength", 0.35),
            "risk_level": analysis.get("readability_risk", "medium"),
            "explanation": analysis.get("short_explanation", ""),
        }

    def _generate_json(self, image_paths: list[Path], prompt: str) -> dict:
        last_error: Exception | None = None
        for model in self._candidate_models():
            if not self._model_available(model):
                continue
            try:
                raw = self._generate(model, image_paths, prompt)
                try:
                    parsed = parse_json_response(raw)
                except ValueError:
                    logger.debug("Raw local VLM response was not valid JSON: %s", raw)
                    raise
                parsed.setdefault("_model", model)
                return parsed
            except (LocalVLMError, ValueError) as exc:
                logger.warning("Local VLM request failed with model %s: %s", model, exc)
                last_error = exc

        raise LocalVLMError(f"Local VLM request failed: {last_error}")

    def _candidate_models(self) -> tuple[str, ...]:
        if self.fallback_model and self.fallback_model != self.model:
            return (self.model, self.fallback_model)
        return (self.model,)

    def _model_available(self, model: str) -> bool:
        data = self._request_json("/api/tags", None, timeout_seconds=10)
        models = data.get("models", [])
        model_names = {item.get("name") for item in models if isinstance(item, dict)}
        return model in model_names

    def _generate(self, model: str, image_paths: list[Path], prompt: str) -> str:
        payload = {
            "model": model,
            "stream": False,
            "system": ANALYZE_SYSTEM_PROMPT,
            "prompt": prompt,
            "images": [_encode_image(path) for path in image_paths],
            "format": "json",
        }
        data = self._request_json("/api/generate", payload, timeout_seconds=self.timeout_seconds)
        response = data.get("response")
        if not isinstance(response, str) or not response.strip():
            raise LocalVLMError("Ollama returned an empty response")
        return response

    def _request_json(self, path: str, payload: dict | None, timeout_seconds: int) -> dict:
        url = self.base_url.rstrip("/") + path
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LocalVLMError(str(exc)) from exc


def parse_json_response(raw_response: str) -> dict:
    cleaned = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    if not cleaned.startswith("{"):
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("VLM response JSON must be an object")
    return data


def _encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")
