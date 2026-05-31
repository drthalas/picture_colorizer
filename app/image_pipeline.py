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
        "clahe": 2.2,
        "paper": ((58, 55, 51), (182, 174, 158), (244, 239, 225)),
        "ink": (24, 24, 23),
        "ink_start": 0.34,
        "ink_width": 0.38,
        "texture": 1.2,
        "vignette": 0.0,
    },
    "vintage": {
        "clahe": 1.8,
        "paper": ((92, 66, 43), (194, 160, 105), (239, 222, 177)),
        "ink": (30, 25, 21),
        "ink_start": 0.38,
        "ink_width": 0.42,
        "texture": 2.5,
        "vignette": 0.0,
    },
    "strong_1940s": {
        "clahe": 1.9,
        "paper": ((82, 52, 31), (181, 132, 73), (236, 203, 139)),
        "ink": (27, 20, 16),
        "ink_start": 0.35,
        "ink_width": 0.40,
        "texture": 3.2,
        "vignette": 0.13,
    },
    "stamp_focus": {
        "clahe": 2.0,
        "paper": ((88, 65, 45), (196, 164, 112), (240, 224, 184)),
        "ink": (29, 24, 20),
        "ink_start": 0.36,
        "ink_width": 0.40,
        "texture": 2.0,
        "vignette": 0.02,
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

    red_mask = np.clip(red_mask, 0.0, 0.45)[..., None]
    blue_mask = np.clip(blue_mask, 0.0, 0.40)[..., None]
    red_tint = np.array([126, 55, 42], dtype=np.float32)
    blue_tint = np.array([58, 76, 112], dtype=np.float32)
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

    clahe = cv2.createCLAHE(clipLimit=float(settings["clahe"]), tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(contrast, None, h=5, templateWindowSize=7, searchWindowSize=21)

    normalized = denoised.astype(np.float32) / 255.0

    shadow_rgb, mid_rgb, highlight_rgb = settings["paper"]
    shadow = np.array(shadow_rgb, dtype=np.float32)
    mid = np.array(mid_rgb, dtype=np.float32)
    highlight = np.array(highlight_rgb, dtype=np.float32)

    lower = normalized[..., None] * 2.0
    upper = (normalized[..., None] - 0.5) * 2.0
    warm = np.where(
        normalized[..., None] < 0.5,
        shadow * (1.0 - lower) + mid * lower,
        mid * (1.0 - upper) + highlight * upper,
    )

    ink_mask = 1.0 - normalized
    ink_mask = np.clip((ink_mask - float(settings["ink_start"])) / float(settings["ink_width"]), 0.0, 1.0)[
        ..., None
    ]
    ink = np.array(settings["ink"], dtype=np.float32)

    result_rgb = warm * (1.0 - ink_mask) + ink * ink_mask

    if mode == "stamp_focus":
        red_mask, blue_mask, red_tint, blue_tint = _stamp_tint(source_bgr, normalized)
        result_rgb = result_rgb * (1.0 - red_mask) + red_tint * red_mask
        result_rgb = result_rgb * (1.0 - blue_mask) + blue_tint * blue_mask

    paper_texture = cv2.GaussianBlur(gray, (0, 0), sigmaX=13).astype(np.float32)
    paper_texture = (paper_texture - paper_texture.mean()) / (paper_texture.std() + 1e-6)
    result_rgb += paper_texture[..., None] * float(settings["texture"])
    result_rgb = _apply_vignette(result_rgb, float(settings["vignette"]))

    result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
    Image.fromarray(result_rgb).save(output_path, quality=95)
    return output_path
