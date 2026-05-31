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
        "ink_start": 0.50,
        "ink_width": 0.28,
        "ink_threshold": 62,
        "ink_area": 14,
        "soft_start": 0.14,
        "soft_width": 0.60,
        "soft_strength": 0.52,
        "paper_smooth": 3.6,
        "texture": 0.9,
        "grain": 0.16,
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
        is_table_line = (aspect > 8.0 and height <= 24) or (aspect > 5.0 and height <= 16) or (
            aspect < 0.24 and width <= 22
        )
        is_header_print = y < image_height * 0.18 and area < 420 and height <= 42
        is_small_print = area < 180 and width <= 42 and height <= 34 and fill > 0.12
        is_signature_zone = y > image_height * 0.68 and width >= 18 and height >= 5
        is_cursive = (
            not is_table_line
            and not is_header_print
            and not is_small_print
            and not touches_edge
            and width >= 12
            and height >= 5
            and (area >= 24 or width >= 26 or is_signature_zone)
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
    result_rgb = _apply_vignette(result_rgb, float(settings["vignette"]))

    result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
    Image.fromarray(result_rgb).save(output_path, quality=95)
    return output_path
