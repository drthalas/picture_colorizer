from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not load image: {path}")
    return image


def colorize_document(input_path: str | Path, output_path: str | Path) -> Path:
    """Create a warm archival-paper colorization while preserving dark ink."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_bgr = _read_image(input_path)
    gray = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(contrast, None, h=5, templateWindowSize=7, searchWindowSize=21)

    normalized = denoised.astype(np.float32) / 255.0

    # Warm paper base, with subtle tonal variation from the original luminance.
    shadow = np.array([92, 66, 43], dtype=np.float32)
    mid = np.array([194, 160, 105], dtype=np.float32)
    highlight = np.array([239, 222, 177], dtype=np.float32)

    lower = normalized[..., None] * 2.0
    upper = (normalized[..., None] - 0.5) * 2.0
    warm = np.where(
        normalized[..., None] < 0.5,
        shadow * (1.0 - lower) + mid * lower,
        mid * (1.0 - upper) + highlight * upper,
    )

    ink_mask = 1.0 - normalized
    ink_mask = np.clip((ink_mask - 0.38) / 0.42, 0.0, 1.0)[..., None]
    ink = np.array([30, 25, 21], dtype=np.float32)

    result_rgb = warm * (1.0 - ink_mask) + ink * ink_mask

    paper_texture = cv2.GaussianBlur(gray, (0, 0), sigmaX=13).astype(np.float32)
    paper_texture = (paper_texture - paper_texture.mean()) / (paper_texture.std() + 1e-6)
    result_rgb += paper_texture[..., None] * 2.5

    result_rgb = np.clip(result_rgb, 0, 255).astype(np.uint8)
    Image.fromarray(result_rgb, mode="RGB").save(output_path, quality=95)
    return output_path
