"""
Smooth blending of warped hairstyle onto inpainted head (feathered alpha + optional edge smoothing).

Premultiplied RGBA warping avoids white/gray halos: straight-alpha warp interpolates opaque hair
colors with transparent (black) neighbors, which produces a milky fringe on the scalp.
"""

from typing import Tuple

import cv2
import numpy as np


def feather_alpha(alpha: np.ndarray, radius: int = 5) -> np.ndarray:
    """Gaussian blur on alpha to soften edges (reduces visible mask boundaries)."""
    if radius <= 0:
        return alpha
    a = np.clip(alpha, 0, 1).astype(np.float32)
    k = radius * 2 + 1
    blurred = cv2.GaussianBlur(a, (k, k), 0)
    return np.clip(blurred, 0, 1)


def warp_rgba_affine_premultiplied(
    hair_bgra: np.ndarray,
    affine_2x3: np.ndarray,
    out_wh: Tuple[int, int],
    interp: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Affine-warp RGBA using premultiplied colors so edges don't pick up white/gray from
    interpolating RGB against transparent black (common 'halo' on hair overlays).

    Returns (bgr uint8 HxWx3, alpha float32 HxW in [0,1]).
    """
    iw, ih = out_wh
    bgr = hair_bgra[:, :, :3].astype(np.float32)
    a = hair_bgra[:, :, 3].astype(np.float32) / 255.0
    a = np.clip(a, 0.0, 1.0)
    premul = bgr * a[..., None]
    warped_pm = cv2.warpAffine(
        premul,
        affine_2x3,
        (iw, ih),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )
    warped_a = cv2.warpAffine(
        a,
        affine_2x3,
        (iw, ih),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_a = np.clip(warped_a, 0.0, 1.0)
    eps = 1e-3
    denom = np.maximum(warped_a[..., None], eps)
    straight = np.zeros_like(warped_pm)
    np.divide(warped_pm, denom, out=straight, where=warped_a[..., None] > eps)
    straight = np.clip(straight, 0, 255).astype(np.uint8)
    return straight, warped_a.astype(np.float32)


def warp_rgba_perspective_premultiplied(
    hair_bgra: np.ndarray,
    perspective_3x3: np.ndarray,
    out_wh: Tuple[int, int],
    interp: int = cv2.INTER_LINEAR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perspective-warp RGBA with premultiplied colors (same halo fix as affine)."""
    iw, ih = out_wh
    bgr = hair_bgra[:, :, :3].astype(np.float32)
    a = hair_bgra[:, :, 3].astype(np.float32) / 255.0
    a = np.clip(a, 0.0, 1.0)
    premul = bgr * a[..., None]
    warped_pm = cv2.warpPerspective(
        premul,
        perspective_3x3,
        (iw, ih),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0.0, 0.0, 0.0),
    )
    warped_a = cv2.warpPerspective(
        a,
        perspective_3x3,
        (iw, ih),
        flags=interp,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_a = np.clip(warped_a, 0.0, 1.0)
    eps = 1e-3
    denom = np.maximum(warped_a[..., None], eps)
    straight = np.zeros_like(warped_pm)
    np.divide(warped_pm, denom, out=straight, where=warped_a[..., None] > eps)
    straight = np.clip(straight, 0, 255).astype(np.uint8)
    return straight, warped_a.astype(np.float32)


def blend_bgr_over(
    background_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """
    alpha: HxW float32 [0,1]
    """
    a = alpha[..., None] if alpha.ndim == 2 else alpha
    bg = background_bgr.astype(np.float32)
    fg = overlay_bgr.astype(np.float32)
    out = fg * a + bg * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def seamless_blend_optional(
    background_bgr: np.ndarray,
    overlay_bgr: np.ndarray,
    mask_u8: np.ndarray,
    center: Tuple[int, int],
) -> np.ndarray:
    """
    OpenCV seamlessClone when mask has sufficient support; fallback to alpha blend.
    """
    h, w = background_bgr.shape[:2]
    if mask_u8.ndim == 3:
        mask_u8 = mask_u8[:, :, 0]
    if mask_u8.max() < 10:
        return background_bgr
    try:
        cx = int(np.clip(center[0], 0, w - 1))
        cy = int(np.clip(center[1], 0, h - 1))
        return cv2.seamlessClone(overlay_bgr, background_bgr, mask_u8, (cx, cy), cv2.NORMAL_CLONE)
    except cv2.error:
        a = (mask_u8.astype(np.float32) / 255.0)
        return blend_bgr_over(background_bgr, overlay_bgr, feather_alpha(a, 3))
