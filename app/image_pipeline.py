from pathlib import Path
import json
from typing import Literal, cast

import cv2
import numpy as np
from PIL import Image

from app.config import load_settings
from app.pipeline.archive_document_4050 import process_archive_document_4050
from app.services.local_vlm import LocalVLMClient

ProcessingMode = Literal[
    "clean",
    "vintage",
    "strong_1940s",
    "stamp_focus",
    "archive_document_4050",
    "archive_document",
    "document_readability",
    "standard",
    "auto_vlm",
]
AVAILABLE_MODES: tuple[ProcessingMode, ...] = (
    "clean",
    "vintage",
    "strong_1940s",
    "stamp_focus",
    "archive_document_4050",
    "archive_document",
    "document_readability",
    "standard",
    "auto_vlm",
)
DEFAULT_MODE: ProcessingMode = "vintage"

VLM_RECOMMENDED_MODE_MAP: dict[str, ProcessingMode] = {
    "document_readability": "document_readability",
    "archive_document": "archive_document",
    "photo_color": "vintage",
    "standard": "standard",
}


_MODE_SETTINGS: dict[ProcessingMode, dict[str, object]] = {
    "clean": {
        "clahe": 1.7,
        "denoise": 12,
        "paper": ((82, 79, 72), (198, 190, 172), (246, 241, 228)),
        "ink": (24, 24, 23),
        "soft_ink": (80, 76, 70),
        "handwriting_ink": (42, 36, 80),
        "stamp_ink": (72, 76, 92),
        "ink_start": 0.50,
        "ink_width": 0.28,
        "ink_threshold": 64,
        "ink_area": 16,
        "soft_start": 0.18,
        "soft_width": 0.52,
        "soft_strength": 0.36,
        "paper_smooth": 4.0,
        "texture": 0.35,
        "grain": 0.08,
        "vignette": 0.0,
    },
    "vintage": {
        "clahe": 1.45,
        "denoise": 8,
        "paper": ((69, 56, 36), (190, 166, 105), (242, 229, 180)),
        "ink": (34, 25, 38),
        "soft_ink": (65, 54, 78),
        "handwriting_ink": (38, 31, 96),
        "stamp_ink": (62, 66, 92),
        "ink_start": 0.48,
        "ink_width": 0.30,
        "ink_threshold": 56,
        "ink_area": 10,
        "soft_start": 0.12,
        "soft_width": 0.62,
        "soft_strength": 0.62,
        "paper_smooth": 8.0,
        "texture": 0.06,
        "grain": 0.01,
        "background_smooth": 0.9,
        "vignette": 0.08,
    },
    "strong_1940s": {
        "clahe": 1.55,
        "denoise": 8,
        "paper": ((76, 51, 31), (187, 143, 75), (240, 207, 141)),
        "ink": (28, 20, 18),
        "soft_ink": (76, 54, 58),
        "handwriting_ink": (38, 30, 88),
        "stamp_ink": (66, 60, 84),
        "ink_start": 0.46,
        "ink_width": 0.30,
        "ink_threshold": 54,
        "ink_area": 14,
        "soft_start": 0.15,
        "soft_width": 0.58,
        "soft_strength": 0.48,
        "paper_smooth": 3.2,
        "texture": 1.05,
        "grain": 0.2,
        "vignette": 0.14,
    },
    "stamp_focus": {
        "clahe": 1.55,
        "denoise": 8,
        "paper": ((78, 61, 42), (196, 170, 112), (244, 229, 181)),
        "ink": (29, 23, 24),
        "soft_ink": (76, 62, 80),
        "handwriting_ink": (38, 31, 98),
        "stamp_ink": (54, 60, 102),
        "ink_start": 0.46,
        "ink_width": 0.30,
        "ink_threshold": 54,
        "ink_area": 16,
        "soft_start": 0.15,
        "soft_width": 0.58,
        "soft_strength": 0.5,
        "paper_smooth": 3.4,
        "texture": 0.85,
        "grain": 0.16,
        "vignette": 0.08,
    },
}


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def validate_mode(mode: str) -> ProcessingMode:
    if mode not in AVAILABLE_MODES:
        choices = ", ".join(AVAILABLE_MODES)
        raise ValueError(f"Unsupported mode: {mode}. Expected one of: {choices}")
    return cast(ProcessingMode, mode)


def _apply_vignette(result_rgb: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0.0:
        return result_rgb

    height, width = result_rgb.shape[:2]
    x_kernel = cv2.getGaussianKernel(width, width / 1.6)
    y_kernel = cv2.getGaussianKernel(height, height / 1.6)
    mask = y_kernel @ x_kernel.T
    mask = mask / mask.max()
    vignette = 1.0 - strength * (1.0 - mask)
    return result_rgb * vignette[..., None]


def _percentile_normalize(image: np.ndarray, low: float = 1.0, high: float = 99.4) -> np.ndarray:
    low_value, high_value = np.percentile(image, (low, high))
    if high_value <= low_value:
        return image.astype(np.uint8)
    normalized = (image.astype(np.float32) - low_value) * (255.0 / (high_value - low_value))
    return np.clip(normalized, 0, 255).astype(np.uint8)


def _enhance_luminance(gray: np.ndarray, denoise_strength: float, clahe_limit: float) -> np.ndarray:
    denoised = cv2.fastNlMeansDenoising(
        gray,
        None,
        h=denoise_strength,
        templateWindowSize=7,
        searchWindowSize=25,
    )

    height, width = gray.shape[:2]
    background_sigma = max(24.0, min(height, width) / 18.0)
    background = cv2.GaussianBlur(denoised, (0, 0), sigmaX=background_sigma, sigmaY=background_sigma)
    background_target = float(np.percentile(background, 82))
    flat = denoised.astype(np.float32) + (background_target - background.astype(np.float32)) * 0.85
    flat = _percentile_normalize(flat, low=1.0, high=98.8)

    clahe = cv2.createCLAHE(clipLimit=clahe_limit, tileGridSize=(8, 8))
    contrast = clahe.apply(flat)
    return cv2.addWeighted(flat, 0.72, contrast, 0.28, 0)


def _paper_gradient(normalized: np.ndarray, settings: dict[str, object]) -> np.ndarray:
    shadow_rgb, mid_rgb, highlight_rgb = settings["paper"]
    shadow = np.array(shadow_rgb, dtype=np.float32)
    mid = np.array(mid_rgb, dtype=np.float32)
    highlight = np.array(highlight_rgb, dtype=np.float32)

    lower = normalized[..., None] * 2.0
    upper = (normalized[..., None] - 0.5) * 2.0
    return np.where(
        normalized[..., None] < 0.5,
        shadow * (1.0 - lower) + mid * lower,
        mid * (1.0 - upper) + highlight * upper,
    )


def _ink_masks(normalized: np.ndarray, settings: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dark = 1.0 - normalized
    strong = np.clip((dark - float(settings["ink_start"])) / float(settings["ink_width"]), 0.0, 1.0)
    strong_u8 = np.clip(strong * 255.0, 0, 255).astype(np.uint8)
    strong_u8 = cv2.medianBlur(strong_u8, 3)

    binary = strong_u8 > int(settings["ink_threshold"])
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    keep = np.zeros(binary.shape, dtype=np.uint8)
    handwriting = np.zeros(binary.shape, dtype=np.uint8)
    stamp = np.zeros(binary.shape, dtype=np.uint8)
    min_area = int(settings["ink_area"])
    image_height, image_width = binary.shape
    for label in range(1, labels_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        width = int(stats[label, cv2.CC_STAT_WIDTH])
        height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        keep[labels == label] = 1

        aspect = width / max(height, 1)
        fill = area / max(width * height, 1)
        touches_edge = x <= 2 or y <= 2 or x + width >= image_width - 2 or y + height >= image_height - 2
        is_long_rule = aspect > 4.2 and height <= 13 and fill < 0.22
        is_table_line = is_long_rule or (aspect > 8.0 and height <= 16) or (aspect > 5.0 and height <= 12) or (
            aspect < 0.24 and width <= 22
        )
        is_header_print = y < image_height * 0.18 and area < 420 and height <= 42
        is_small_print = area < 180 and width <= 42 and height <= 34 and fill > 0.12
        is_signature_zone = y > image_height * 0.68 and width >= 22 and height >= 6 and area >= 42
        is_cursive = (
            not is_table_line
            and not is_header_print
            and not is_small_print
            and not touches_edge
            and width >= 18
            and height >= 6
            and area >= 34
            and (width >= 30 or is_signature_zone or fill < 0.28)
        )
        is_stamp_like = (
            not is_table_line
            and not touches_edge
            and y > image_height * 0.55
            and width >= 18
            and height >= 10
            and fill < 0.42
        )

        if is_cursive:
            handwriting[labels == label] = 1
        if is_stamp_like:
            stamp[labels == label] = 1

    strong_filtered = (strong_u8.astype(np.float32) / 255.0) * keep.astype(np.float32)
    strong_filtered = cv2.GaussianBlur(strong_filtered, (0, 0), sigmaX=0.35, sigmaY=0.35)
    strong_filtered = np.clip(strong_filtered, 0.0, 1.0)

    soft = np.clip((dark - float(settings["soft_start"])) / float(settings["soft_width"]), 0.0, 1.0)
    soft = cv2.GaussianBlur(soft, (0, 0), sigmaX=0.45, sigmaY=0.45)
    soft *= float(settings["soft_strength"]) * (1.0 - strong_filtered)
    handwriting = cv2.GaussianBlur(handwriting.astype(np.float32), (0, 0), sigmaX=0.45, sigmaY=0.45)
    stamp = cv2.GaussianBlur(stamp.astype(np.float32), (0, 0), sigmaX=0.65, sigmaY=0.65)
    return (
        strong_filtered[..., None],
        np.clip(soft, 0.0, 1.0)[..., None],
        np.clip(handwriting, 0.0, 1.0)[..., None],
        np.clip(stamp, 0.0, 1.0)[..., None],
    )


def _stamp_tint(
    source_bgr: np.ndarray,
    normalized: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    red_like = ((hue <= 12) | (hue >= 165)) & (saturation > 35) & (value < 230)
    blue_like = (hue >= 85) & (hue <= 135) & (saturation > 28) & (value < 235)
    line_like = normalized < 0.78

    red_mask = cv2.GaussianBlur((red_like & line_like).astype(np.float32), (0, 0), sigmaX=1.0)
    blue_mask = cv2.GaussianBlur((blue_like & line_like).astype(np.float32), (0, 0), sigmaX=1.0)

    broad_dark = cv2.GaussianBlur((normalized < 0.42).astype(np.float32), (0, 0), sigmaX=4.0, sigmaY=4.0)
    grayscale_stamp = np.clip((broad_dark - 0.34) / 0.66, 0.0, 0.08)

    red_mask = np.maximum(np.clip(red_mask, 0.0, 0.45), grayscale_stamp * 0.25)[..., None]
    blue_mask = np.maximum(np.clip(blue_mask, 0.0, 0.40), grayscale_stamp)[..., None]
    red_tint = np.array([122, 53, 43], dtype=np.float32)
    blue_tint = np.array([54, 57, 96], dtype=np.float32)
    return red_mask, blue_mask, red_tint, blue_tint


def colorize_document(
    input_path: str | Path,
    output_path: str | Path,
    mode: ProcessingMode | str = DEFAULT_MODE,
) -> Path:
    """Enhance a paper document image while preserving existing text and marks."""
    mode = validate_mode(mode)
    if mode == "auto_vlm":
        return process_auto_vlm(input_path, output_path)
    if mode in {"archive_document_4050", "archive_document"}:
        return process_archive_document_4050(input_path, output_path)
    if mode == "document_readability":
        mode = "clean"
    elif mode == "standard":
        mode = "clean"

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    settings = _MODE_SETTINGS[mode]

    source_bgr = _read_image(input_path)
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)
    enhanced = _enhance_luminance(gray, float(settings["denoise"]), float(settings["clahe"]))

    ink_normalized = enhanced.astype(np.float32) / 255.0
    paper_luminance = cv2.GaussianBlur(
        enhanced,
        (0, 0),
        sigmaX=float(settings["paper_smooth"]),
        sigmaY=float(settings["paper_smooth"]),
    )
    paper_luminance = cv2.bilateralFilter(paper_luminance, d=0, sigmaColor=18, sigmaSpace=18)
    paper_normalized = paper_luminance.astype(np.float32) / 255.0
    warm = _paper_gradient(paper_normalized, settings)

    ink_mask, soft_ink_mask, handwriting_mask, stamp_mask = _ink_masks(ink_normalized, settings)
    ink = np.array(settings["ink"], dtype=np.float32)
    soft_ink = np.array(settings["soft_ink"], dtype=np.float32)
    handwriting_ink = np.array(settings["handwriting_ink"], dtype=np.float32)
    stamp_ink = np.array(settings["stamp_ink"], dtype=np.float32)
    colored_ink = ink * (1.0 - handwriting_mask) + handwriting_ink * handwriting_mask
    colored_ink = colored_ink * (1.0 - stamp_mask * 0.45) + stamp_ink * (stamp_mask * 0.45)

    result_rgb = warm * (1.0 - soft_ink_mask) + soft_ink * soft_ink_mask
    result_rgb = result_rgb * (1.0 - ink_mask) + colored_ink * ink_mask

    if mode == "stamp_focus":
        red_mask, blue_mask, red_tint, blue_tint = _stamp_tint(source_bgr, ink_normalized)
        result_rgb = result_rgb * (1.0 - red_mask) + red_tint * red_mask
        result_rgb = result_rgb * (1.0 - blue_mask) + blue_tint * blue_mask

    paper_texture = cv2.GaussianBlur(paper_luminance, (0, 0), sigmaX=10).astype(np.float32)
    paper_texture = (paper_texture - paper_texture.mean()) / (paper_texture.std() + 1e-6)
    result_rgb += paper_texture[..., None] * float(settings["texture"])
    fine_texture = enhanced.astype(np.float32) - cv2.GaussianBlur(enhanced, (0, 0), sigmaX=2.2).astype(np.float32)
    result_rgb += fine_texture[..., None] * float(settings["grain"])
    background_mask = 1.0 - np.clip(ink_mask * 1.45 + soft_ink_mask * 0.8, 0.0, 1.0)
    smoothed_rgb = cv2.GaussianBlur(result_rgb, (0, 0), sigmaX=1.9, sigmaY=1.9)
    smooth_weight = background_mask * float(settings.get("background_smooth", 0.0))
    result_rgb = result_rgb * (1.0 - smooth_weight) + smoothed_rgb * smooth_weight
    result_rgb = _apply_vignette(result_rgb, float(settings["vignette"]))

    result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
    Image.fromarray(result_rgb).save(output_path, quality=95)
    return output_path


def process_auto_vlm(input_path: str | Path, output_path: str | Path) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    settings = load_settings()
    if not settings.local_vlm_enabled:
        raise RuntimeError("LOCAL_VLM_ENABLED=false. Enable it in .env to use auto_vlm.")
    if settings.local_vlm_provider != "ollama":
        raise RuntimeError(f"Unsupported LOCAL_VLM_PROVIDER={settings.local_vlm_provider!r}. Expected 'ollama'.")

    client = LocalVLMClient(
        base_url=settings.local_vlm_base_url,
        model=settings.local_vlm_model,
        timeout_seconds=settings.local_vlm_timeout_seconds,
        fallback_model=settings.local_vlm_fallback_model,
    )
    if not client.is_available():
        raise RuntimeError(
            f"Local VLM is not available. Start Ollama and pull {settings.local_vlm_model} "
            f"or {settings.local_vlm_fallback_model}."
        )

    analysis = client.analyze_image(input_path)
    _write_json(_sidecar_path(output_path, "vlm_analysis"), analysis)

    selected_mode = _select_vlm_pipeline_mode(analysis)
    colorize_document(input_path, output_path, mode=selected_mode)

    comparison = client.compare_before_after(input_path, output_path)
    comparison["selected_pipeline_mode"] = selected_mode

    if comparison.get("should_accept_result") is False and selected_mode != "document_readability":
        initial_comparison = comparison
        selected_mode = "document_readability"
        colorize_document(input_path, output_path, mode=selected_mode)
        comparison = client.compare_before_after(input_path, output_path)
        comparison["selected_pipeline_mode"] = selected_mode
        comparison["fallback_from_pipeline_mode"] = initial_comparison.get("selected_pipeline_mode")
        comparison["initial_comparison"] = initial_comparison

    if comparison.get("should_accept_result") is False:
        comparison["warning"] = "VLM marked the processed result as risky for readability."

    _write_json(_sidecar_path(output_path, "vlm_compare"), comparison)

    return output_path


def _select_vlm_pipeline_mode(analysis: dict) -> ProcessingMode:
    recommended = str(analysis.get("recommended_mode", "standard"))
    selected_mode = VLM_RECOMMENDED_MODE_MAP.get(recommended, "standard")
    recommended_settings = analysis.get("recommended_settings", {})
    if not isinstance(recommended_settings, dict):
        recommended_settings = {}

    try:
        colorization_strength = float(recommended_settings.get("colorization_strength", 0.25))
    except (TypeError, ValueError):
        colorization_strength = 0.25

    readability_risk = str(analysis.get("readability_risk", "medium")).lower()
    if selected_mode in {"archive_document", "vintage"} and (
        readability_risk == "high" or colorization_strength < 0.18
    ):
        return "document_readability"
    return selected_mode


def get_auto_vlm_warning(output_path: str | Path) -> str | None:
    compare_path = _sidecar_path(Path(output_path), "vlm_compare")
    if not compare_path.exists():
        return None

    try:
        data = json.loads(compare_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if data.get("should_accept_result") is False:
        reason = data.get("reason") or data.get("warning") or "модель отметила риск ухудшения читаемости"
        return f"Внимание: локальная модель считает результат рискованным для читаемости. Причина: {reason}"
    return None


def _sidecar_path(output_path: Path, suffix: str) -> Path:
    return output_path.with_name(f"{output_path.stem}.{suffix}.json")


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
