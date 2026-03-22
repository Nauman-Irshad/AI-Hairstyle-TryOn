"""Image preprocessing helpers for the try-on pipeline."""

from typing import Tuple

import cv2
import numpy as np


def ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
    """Ensure HxWx3 uint8 BGR."""
    if image is None or image.size == 0:
        raise ValueError("Empty image")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def resize_long_edge(image: np.ndarray, long_edge: int) -> Tuple[np.ndarray, float]:
    """Resize so longest side == long_edge; return scale factor applied to original dimensions."""
    h, w = image.shape[:2]
    m = max(h, w)
    if m <= long_edge:
        return image.copy(), 1.0
    scale = long_edge / float(m)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    out = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
    return out, scale


def mask_to_float01(mask_u8: np.ndarray) -> np.ndarray:
    """HxW or HxWx1 uint8 -> HxW float32 [0,1]."""
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[:, :, 0]
    return (mask_u8.astype(np.float32) / 255.0).clip(0, 1)
