"""
Predict where the user's scalp + natural hair live so we can inpaint that region before
placing the try-on hairstyle. Combines BiSeNet hair segmentation with a landmark-based
hairline arc (where forehead skin meets hair).

Uses MediaPipe Face Landmarker indices aligned with utils.alignment.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from utils.alignment import IDX_CHIN, IDX_FOREHEAD, IDX_LEFT_TEMPLE, IDX_RIGHT_TEMPLE

# Inner eyebrow / upper face — hairline sits above these (approximate).
_IDX_BROW_INNER_LEFT = 107
_IDX_BROW_INNER_RIGHT = 336


def _lm_xy(lm, idx: int, w: int, h: int) -> np.ndarray:
    p = lm[idx]
    return np.array([p.x * w, p.y * h], dtype=np.float64)


def _brow_band_lower_y(lm, w: int, h: int) -> Optional[float]:
    """Largest y (lowest on image) among eyebrow-related points — hairline must stay above this band."""
    ys: List[float] = []
    for idx in (_IDX_BROW_INNER_LEFT, _IDX_BROW_INNER_RIGHT, 66, 296):
        if idx < len(lm):
            ys.append(float(lm[idx].y * h))
    return max(ys) if ys else None


def scalp_hairline_arc_mask_u8(height: int, width: int, landmarker_result: Any) -> np.ndarray:
    """
    Binary mask for the upper head region bounded by a quadratic Bezier hairline from
    temple to temple (bulging toward the top of the head). This approximates scalp +
    forehead strip where natural hair grows, above the eyebrows.
    """
    out = np.zeros((height, width), dtype=np.uint8)
    if not landmarker_result or not getattr(landmarker_result, "face_landmarks", None):
        return out
    lm = landmarker_result.face_landmarks[0]
    w, h = width, height

    p234 = _lm_xy(lm, IDX_LEFT_TEMPLE, w, h)
    p454 = _lm_xy(lm, IDX_RIGHT_TEMPLE, w, h)
    p10 = _lm_xy(lm, IDX_FOREHEAD, w, h)
    p152 = _lm_xy(lm, IDX_CHIN, w, h)

    chin_y = float(p152[1])
    y10 = float(p10[1])
    face_h = max(8.0, chin_y - y10)

    # Control point: bulge upward between temples (smaller y = higher on face).
    mid = (p234 + p454) * 0.5
    C = np.array([mid[0], mid[1] - 0.17 * face_h], dtype=np.float64)

    # Quadratic Bezier from p454 (right temple) to p234 (left temple), bulging toward the crown.
    n = 40
    t = np.linspace(0.0, 1.0, n)
    arc: List[Tuple[float, float]] = []
    for ti in t:
        q = (1.0 - ti) ** 2 * p454 + 2.0 * (1.0 - ti) * ti * C + ti**2 * p234
        arc.append((float(q[0]), float(q[1])))

    arc = [(float(np.clip(x, 0, w - 1)), float(np.clip(y, 0, h - 1))) for x, y in arc]
    y_brow = _brow_band_lower_y(lm, w, h)
    if y_brow is not None:
        margin = 0.025 * face_h
        cap_y = y_brow - margin
        arc = [(x, float(min(y, cap_y))) for x, y in arc]

    # Closed polygon: top bar → down right image edge → along hairline arc (right→left) → up left edge.
    y_r = float(np.clip(p454[1], 0, h - 1))
    y_l = float(np.clip(p234[1], 0, h - 1))
    poly: List[Tuple[float, float]] = [
        (0.0, 0.0),
        (float(w - 1), 0.0),
        (float(w - 1), y_r),
    ]
    for pt in arc:
        poly.append(pt)
    poly.extend([(0.0, y_l), (0.0, 0.0)])
    pts = np.array(poly, dtype=np.float32).reshape(-1, 1, 2)
    cv2.fillPoly(out, [pts.astype(np.int32)], 255)
    return out


def predict_user_hair_and_scalp_mask_u8(
    bisenet_mask_u8: np.ndarray,
    landmarker_result: Any,
) -> np.ndarray:
    """
    Full mask to inpaint: BiSeNet hair + landmark scalp/hairline band, then light dilation.
    This removes the user's natural hair and fills the visible scalp so the try-on hair sits on skin.
    """
    h, w = bisenet_mask_u8.shape[:2]
    m = bisenet_mask_u8.copy()
    if landmarker_result is not None:
        arc = scalp_hairline_arc_mask_u8(h, w, landmarker_result)
        m = cv2.bitwise_or(m, arc)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    m = cv2.dilate(m, k, iterations=2)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
    return m


def combine_bisenet_and_hairline_cap(
    bisenet_mask_u8: np.ndarray,
    landmarker_result: Any,
) -> np.ndarray:
    """OR the segmentation mask with the landmark upper-head cap (same resolution)."""
    return predict_user_hair_and_scalp_mask_u8(bisenet_mask_u8, landmarker_result)


def augment_dark_hair_mask_u8(
    image_bgr: np.ndarray,
    mask_u8: np.ndarray,
    landmarker_result: Any,
) -> np.ndarray:
    """
    BiSeNet often under-segments jet-black / near-black hair (low contrast vs background).
    Union very dark pixels that lie in the upper-head arc (above the hairline), so LaMa
    can remove them. Tune with env HAIR_DARK_GRAY_MAX (default 52, lower = stricter “pure black”).
    """
    if landmarker_result is None:
        return mask_u8
    try:
        max_gray = int(os.environ.get("HAIR_DARK_GRAY_MAX", "52").strip())
    except ValueError:
        max_gray = 52
    max_gray = max(20, min(90, max_gray))

    h, w = image_bgr.shape[:2]
    arc = scalp_hairline_arc_mask_u8(h, w, landmarker_result)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    dark = (gray < max_gray).astype(np.uint8) * 255
    candidate = cv2.bitwise_and(dark, arc)
    # Drop 1–2 px salt noise; keep thin strands
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, k3, iterations=1)
    # Strands just outside BiSeNet: dark near existing mask, still inside arc
    band = cv2.dilate(mask_u8, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 19)), iterations=1)
    near_strand = cv2.bitwise_and(dark, cv2.bitwise_and(arc, band))
    out = cv2.bitwise_or(mask_u8, candidate)
    out = cv2.bitwise_or(out, near_strand)
    return out


# Back-compat alias
upper_head_hairline_mask_u8 = scalp_hairline_arc_mask_u8
