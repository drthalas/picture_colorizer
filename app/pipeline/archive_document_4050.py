from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image, ImageOps
from dotenv import load_dotenv


@dataclass(frozen=True)
class ArchiveDocument4050Settings:
    warmth: float = 0.35
    blue_folder_tint: float = 0.25
    contrast_strength: float = 0.8
    grain_strength: float = 0.08
    sharpness_strength: float = 0.5
    preserve_text_strength: float = 0.9
    debug: bool = False
    max_long_side: int = 2800


class DocumentRestorer(Protocol):
    """Extension point for future document restoration models such as DocRes."""

    def restore(self, image_rgb: np.ndarray) -> np.ndarray:
        ...


class Colorizer(Protocol):
    """Extension point for future neural colorizers such as DDColor."""

    def colorize(self, image_rgb: np.ndarray, readability_layer: np.ndarray) -> np.ndarray:
        ...


class LocalDocumentRestorer:
    """Current lightweight OpenCV restoration backend."""

    def restore(self, image_rgb: np.ndarray) -> np.ndarray:
        return image_rgb


class LocalVintageColorizer:
    """Current deterministic color backend for archival documents."""

    def __init__(self, settings: ArchiveDocument4050Settings) -> None:
        self.settings = settings

    def colorize(self, image_rgb: np.ndarray, readability_layer: np.ndarray) -> np.ndarray:
        return create_controlled_vintage_color_layer(image_rgb, readability_layer, self.settings)


def settings_from_env() -> ArchiveDocument4050Settings:
    load_dotenv()
    return ArchiveDocument4050Settings(
        debug=os.getenv("DEBUG_IMAGE_PIPELINE", "").lower() in {"1", "true", "yes", "on"},
        max_long_side=int(os.getenv("IMAGE_MAX_LONG_SIDE", "2800")),
    )


def load_image(input_path: str | Path, max_long_side: int = 2800) -> np.ndarray:
    image = Image.open(input_path)
    image = ImageOps.exif_transpose(image).convert("RGB")

    width, height = image.size
    long_side = max(width, height)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        image = image.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)

    return np.asarray(image)


def preprocess_document(image_rgb: np.ndarray, settings: ArchiveDocument4050Settings) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)

    height, width = gray.shape[:2]
    sigma = max(24.0, min(height, width) / 18.0)
    background = cv2.GaussianBlur(denoised, (0, 0), sigmaX=sigma, sigmaY=sigma)
    target = float(np.percentile(background, 82))
    flattened = denoised.astype(np.float32) + (target - background.astype(np.float32)) * 0.72

    low, high = np.percentile(flattened, (1.0, 99.0))
    if high > low:
        flattened = (flattened - low) * (255.0 / (high - low))

    flattened_u8 = np.clip(flattened, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.2 + settings.contrast_strength * 0.8, tileGridSize=(8, 8))
    contrast = clahe.apply(flattened_u8)
    return cv2.addWeighted(flattened_u8, 0.55, contrast, 0.45, 0)


def build_text_mask(preprocessed_gray: np.ndarray) -> np.ndarray:
    blurred = cv2.medianBlur(preprocessed_gray, 3)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        45,
        18,
    )
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    dark_cutoff = max(65, min(122, int(np.percentile(blurred, 34))))
    dark = (blurred < dark_cutoff).astype(np.uint8) * 255
    mask = cv2.bitwise_and(cv2.bitwise_or(adaptive, otsu), dark)

    small_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, small_kernel, iterations=1)

    join_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, join_kernel, iterations=1)

    labels_count, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    filtered = np.zeros_like(mask)
    height, width = mask.shape[:2]
    image_area = height * width
    for label in range(1, labels_count):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        component_width = int(stats[label, cv2.CC_STAT_WIDTH])
        component_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 14 or area > image_area * 0.18:
            continue

        aspect = component_width / max(component_height, 1)
        fill = area / max(component_width * component_height, 1)
        touches_most_edges = x <= 1 and y <= 1 and x + component_width >= width - 1 and y + component_height >= height - 1
        line_like = (aspect > 3.5 and component_height <= 14) or (aspect < 0.35 and component_width <= 14)
        text_like = area >= 42 and component_width >= 3 and component_height >= 3 and fill > 0.14
        plausible_text_or_line = (
            component_width >= 3
            and component_height >= 3
            and not touches_most_edges
            and (text_like or line_like)
        )
        if plausible_text_or_line:
            filtered[labels == label] = 255

    return cv2.GaussianBlur(filtered, (0, 0), sigmaX=0.45, sigmaY=0.45)


def enhance_readability_layer(preprocessed_gray: np.ndarray, settings: ArchiveDocument4050Settings) -> np.ndarray:
    blur = cv2.GaussianBlur(preprocessed_gray, (0, 0), sigmaX=1.0)
    sharpened = cv2.addWeighted(
        preprocessed_gray,
        1.0 + settings.sharpness_strength,
        blur,
        -settings.sharpness_strength,
        0,
    )
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def create_controlled_vintage_color_layer(
    image_rgb: np.ndarray,
    readability_layer: np.ndarray,
    settings: ArchiveDocument4050Settings,
) -> np.ndarray:
    luminance = cv2.GaussianBlur(readability_layer, (0, 0), sigmaX=8.0, sigmaY=8.0).astype(np.float32) / 255.0

    shadow = np.array([94, 83, 66], dtype=np.float32)
    mid = np.array([190, 174, 132], dtype=np.float32)
    highlight = np.array([239, 229, 193], dtype=np.float32)
    blue_folder = np.array([156, 166, 174], dtype=np.float32)

    lower = luminance[..., None] * 2.0
    upper = (luminance[..., None] - 0.5) * 2.0
    paper = np.where(
        luminance[..., None] < 0.5,
        shadow * (1.0 - lower) + mid * lower,
        mid * (1.0 - upper) + highlight * upper,
    )

    warmth = np.clip(settings.warmth, 0.0, 1.0)
    paper = paper * (0.85 + warmth * 0.15) + np.array([245, 225, 180], dtype=np.float32) * (warmth * 0.08)

    low_detail = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=16.0, sigmaY=16.0).astype(np.float32)
    saturation_hint = np.std(low_detail, axis=2, keepdims=True)
    folder_mask = np.clip((saturation_hint - 4.0) / 28.0, 0.0, 1.0)
    paper = paper * (1.0 - folder_mask * settings.blue_folder_tint) + blue_folder * (
        folder_mask * settings.blue_folder_tint
    )

    return np.clip(paper, 0, 255).astype(np.uint8)


def _stamp_tint_mask(image_rgb: np.ndarray, text_mask: np.ndarray, readability_layer: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    blue_or_red = (
        (((hue >= 85) & (hue <= 135)) | (hue <= 12) | (hue >= 165))
        & (saturation > 24)
        & (value < 238)
    )
    broad_dark = cv2.GaussianBlur((readability_layer < 118).astype(np.float32), (0, 0), sigmaX=4.0)
    stamp_like = np.maximum(blue_or_red.astype(np.float32), np.clip((broad_dark - 0.35) / 0.6, 0.0, 0.35))
    return np.clip(stamp_like * (text_mask.astype(np.float32) / 255.0), 0.0, 1.0)


def blend_text_back(
    color_layer_rgb: np.ndarray,
    readability_layer: np.ndarray,
    text_mask: np.ndarray,
    image_rgb: np.ndarray,
    settings: ArchiveDocument4050Settings,
) -> np.ndarray:
    mask = np.clip(text_mask.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
    darkness = (255.0 - readability_layer.astype(np.float32)) / 255.0
    darkness = np.clip(darkness * (0.88 + settings.preserve_text_strength * 0.35), 0.0, 1.0)[..., None]

    dark_ink = np.array([32, 29, 27], dtype=np.float32)
    stamp_ink = np.array([55, 58, 96], dtype=np.float32)
    stamp_mask = _stamp_tint_mask(image_rgb, text_mask, readability_layer)[..., None]
    ink_color = dark_ink * (1.0 - stamp_mask * 0.55) + stamp_ink * (stamp_mask * 0.55)

    ink_layer = color_layer_rgb.astype(np.float32) * (1.0 - darkness) + ink_color * darkness
    result = color_layer_rgb.astype(np.float32) * (1.0 - mask) + ink_layer * mask
    return np.clip(result, 0, 255).astype(np.uint8)


def add_vintage_finish(
    image_rgb: np.ndarray,
    text_mask: np.ndarray,
    settings: ArchiveDocument4050Settings,
) -> np.ndarray:
    result = image_rgb.astype(np.float32)
    text = np.clip(text_mask.astype(np.float32) / 255.0, 0.0, 1.0)[..., None]
    background = 1.0 - np.clip(text * 1.25, 0.0, 1.0)

    if settings.grain_strength > 0:
        rng = np.random.default_rng(4050)
        grain = rng.normal(0.0, 5.0 * settings.grain_strength, result.shape[:2]).astype(np.float32)
        result += grain[..., None] * background

    height, width = result.shape[:2]
    x_kernel = cv2.getGaussianKernel(width, width / 1.5)
    y_kernel = cv2.getGaussianKernel(height, height / 1.5)
    vignette = y_kernel @ x_kernel.T
    vignette = vignette / vignette.max()
    vignette_strength = 0.06
    result *= (1.0 - vignette_strength * (1.0 - vignette))[..., None]

    blurred = cv2.GaussianBlur(result, (0, 0), sigmaX=0.75)
    result = cv2.addWeighted(result, 1.08, blurred, -0.08, 0)
    return np.clip(result, 0, 255).astype(np.uint8)


def save_result(image_rgb: np.ndarray, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_rgb).save(output_path, quality=95)
    return output_path


def _save_debug(debug_dir: Path, name: str, image: np.ndarray) -> None:
    debug_dir.mkdir(parents=True, exist_ok=True)
    if image.ndim == 2:
        Image.fromarray(image).save(debug_dir / f"{name}.png")
    else:
        Image.fromarray(image).save(debug_dir / f"{name}.png")


def process_archive_document_4050(
    input_path: str | Path,
    output_path: str | Path,
    settings: ArchiveDocument4050Settings | None = None,
    restorer: DocumentRestorer | None = None,
    colorizer: Colorizer | None = None,
) -> Path:
    settings = settings or settings_from_env()
    restorer = restorer or LocalDocumentRestorer()
    colorizer = colorizer or LocalVintageColorizer(settings)

    image_rgb = load_image(input_path, max_long_side=settings.max_long_side)
    restored_rgb = restorer.restore(image_rgb)
    preprocessed = preprocess_document(restored_rgb, settings)
    text_mask = build_text_mask(preprocessed)
    readability = enhance_readability_layer(preprocessed, settings)
    color_layer = colorizer.colorize(restored_rgb, readability)
    blended = blend_text_back(color_layer, readability, text_mask, restored_rgb, settings)
    finished = add_vintage_finish(blended, text_mask, settings)

    output_path = save_result(finished, output_path)

    if settings.debug:
        debug_dir = output_path.parent / f"{output_path.stem}-debug"
        _save_debug(debug_dir, "01_preprocessed", preprocessed)
        _save_debug(debug_dir, "02_text_mask", text_mask)
        _save_debug(debug_dir, "03_readability", readability)
        _save_debug(debug_dir, "04_color_layer", color_layer)
        _save_debug(debug_dir, "05_blended", blended)

    return output_path
