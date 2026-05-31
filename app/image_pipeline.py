from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
from PIL import Image

ProcessingMode = Literal["clean", "vintage", "strong_1940s", "stamp_focus"]
AVAILABLE_MODES: tuple[ProcessingMode, ...] = ("clean", "vintage", "strong_1940s", "stamp_focus")
DEFAULT_MODE: ProcessingMode = "vintage"


_MODE_SETTINGS: dict[ProcessingMode, dict[str, object]] = {
    "clean": {
        "clahe": 1.7,
        "denoise": 12,
        "paper": ((82, 79, 72), (198, 190, 172), (246, 241, 228)),
        "ink": (24, 24, 23),
        "ink_start": 0.42,
        "ink_width": 0.30,
        "ink_threshold": 48,
        "ink_area": 14,
        "paper_smooth": 6.0,
        "texture": 0.25,
        "vignette": 0.0,
    },
    "vintage": {
        "clahe": 1.45,
        "denoise": 13,
        "paper": ((75, 62, 42), (195, 172, 112), (244, 231, 182)),
        "ink": (31, 25, 28),
        "ink_start": 0.44,
        "ink_width": 0.30,
        "ink_threshold": 52,
        "ink_area": 20,
        "paper_smooth": 6.5,
        "texture": 0.35,
        "vignette": 0.04,
    },
    "strong_1940s": {
        "clahe": 1.55,
        "denoise": 12,
        "paper": ((76, 51, 31), (187, 143, 75), (240, 207, 141)),
        "ink": (28, 20, 18),
        "ink_start": 0.40,
        "ink_width": 0.32,
        "ink_threshold": 44,
        "ink_area": 14,
        "paper_smooth": 5.8,
        "texture": 0.55,
        "vignette": 0.12,
    },
    "stamp_focus": {
        "clahe": 1.55,
        "denoise": 12,
        "paper": ((78, 61, 42), (196, 170, 112), (244, 229, 181)),
        "ink": (29, 23, 24),
        "ink_start": 0.39,
        "ink_width": 0.32,
        "ink_threshold": 44,
        "ink_area": 16,
        "paper_smooth": 5.8,
        "texture": 0.3,
        "vignette": 0.04,
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


def _ink_mask(normalized: np.ndarray, settings: dict[str, object]) -> np.ndarray:
    dark = 1.0 - normalized
    mask = np.clip((dark - float(settings["ink_start"])) / float(settings["ink_width"]), 0.0, 1.0)
    mask_u8 = np.clip(mask * 255.0, 0, 255).astype(np.uint8)
    mask_u8 = cv2.medianBlur(mask_u8, 3)

    binary = mask_u8 > int(settings["ink_threshold"])
    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    keep = np.zeros(binary.shape, dtype=np.uint8)
    min_area = int(settings["ink_area"])
    for label in range(1, labels_count):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            keep[labels == label] = 1

    filtered = (mask_u8.astype(np.float32) / 255.0) * keep.astype(np.float32)
    filtered = cv2.GaussianBlur(filtered, (0, 0), sigmaX=0.35, sigmaY=0.35)
    return np.clip(filtered, 0.0, 1.0)[..., None]


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
    paper_normalized = paper_luminance.astype(np.float32) / 255.0
    warm = _paper_gradient(paper_normalized, settings)

    ink_mask = _ink_mask(ink_normalized, settings)
    ink = np.array(settings["ink"], dtype=np.float32)

    result_rgb = warm * (1.0 - ink_mask) + ink * ink_mask

    if mode == "stamp_focus":
        red_mask, blue_mask, red_tint, blue_tint = _stamp_tint(source_bgr, ink_normalized)
        result_rgb = result_rgb * (1.0 - red_mask) + red_tint * red_mask
        result_rgb = result_rgb * (1.0 - blue_mask) + blue_tint * blue_mask

    paper_texture = cv2.GaussianBlur(paper_luminance, (0, 0), sigmaX=10).astype(np.float32)
    paper_texture = (paper_texture - paper_texture.mean()) / (paper_texture.std() + 1e-6)
    result_rgb += paper_texture[..., None] * float(settings["texture"])
    result_rgb = _apply_vignette(result_rgb, float(settings["vignette"]))

    result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
    Image.fromarray(result_rgb).save(output_path, quality=95)
    return output_path
