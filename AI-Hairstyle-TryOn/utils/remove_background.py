"""Remove portrait background; composite subject on white for downstream face/hair pipeline."""

from __future__ import annotations

import logging
import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def remove_background_bgr(bgr: np.ndarray) -> np.ndarray:
    """
    Run rembg (U²-Net) on the image and composite the subject on white.

    Falls back to the input if rembg is missing or fails.
    """
    if bgr is None or bgr.size == 0:
        return bgr
    try:
        from rembg import remove
    except ImportError:
        LOGGER.warning("rembg not installed — skip background removal (pip install rembg).")
        return bgr

    try:
        ok, buf = cv2.imencode(".png", bgr)
        if not ok:
            return bgr
        out = remove(buf.tobytes())
        arr = np.frombuffer(out, dtype=np.uint8)
        bgra = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if bgra is None:
            return bgr
        if bgra.shape[2] == 3:
            return bgra
        alpha = bgra[:, :, 3:4].astype(np.float32) / 255.0
        fg = bgra[:, :, :3].astype(np.float32)
        white = np.full_like(fg, 255.0)
        comp = (fg * alpha + white * (1.0 - alpha)).astype(np.uint8)
        return comp
    except Exception as e:
        LOGGER.warning("Background removal failed: %s — using original image.", e)
        return bgr


def remove_background_png_bytes(bgr: np.ndarray) -> bytes:
    """
    Run rembg and return PNG bytes (RGBA, transparent background).
    Faster path for /remove-background — no compositing to white.
    """
    if bgr is None or bgr.size == 0:
        raise ValueError("empty image")
    try:
        from rembg import remove
    except ImportError as e:
        raise RuntimeError("Install rembg: pip install rembg") from e
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("could not encode input PNG")
    return remove(buf.tobytes())


def downscale_max_side(bgr: np.ndarray, max_side: int = 1280) -> np.ndarray:
    """Resize so longest edge is at most max_side (faster rembg on huge photos)."""
    h, w = bgr.shape[:2]
    m = max(h, w)
    if m <= max_side:
        return bgr
    scale = max_side / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
