"""
Apply a hairstyle PNG overlay aligned to MediaPipe landmarks, with GPU refinement pass.
"""

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from utils.alignment import (
    _triangle_signed_area,
    default_hair_template_quad,
    default_hair_template_triangle,
    hair_dst_quad_scalp_fit,
    hair_dst_triangle_scalp_fit,
)
from utils.blending import blend_bgr_over, warp_rgba_affine_premultiplied, warp_rgba_perspective_premultiplied

from .gan_model import HairRefinementNet

LOGGER = logging.getLogger(__name__)


def _scale_dst_points(dst: np.ndarray, scale: float, iw: int, ih: int) -> np.ndarray:
    """Uniform scale of landmark destination points about their centroid (user fine-tune)."""
    if scale <= 0 or abs(scale - 1.0) < 1e-9:
        return dst.astype(np.float32)
    c = dst.mean(axis=0)
    out = c + (dst - c) * scale
    out[:, 0] = np.clip(out[:, 0], 0, iw - 1)
    out[:, 1] = np.clip(out[:, 1], 0, ih - 1)
    return out.astype(np.float32)


def load_hairstyle_rgba(path: Path) -> np.ndarray:
    """Load PNG with alpha; BGR + A."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Hairstyle asset not found: {path}")
    if img.shape[2] == 3:
        bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255
        return bgra
    return img


@torch.inference_mode()
def refine_composite_gpu(
    composite_bgr: np.ndarray,
    net: Optional[HairRefinementNet],
    device: torch.device,
) -> np.ndarray:
    """Optional light CNN pass on GPU (near-identity when untrained)."""
    if net is None:
        return composite_bgr
    h, w = composite_bgr.shape[:2]
    t = torch.from_numpy(composite_bgr[:, :, ::-1].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    t = t.to(device)
    t = F.interpolate(t, size=(min(512, h), min(512, w)), mode="bilinear", align_corners=False)
    out = net(t)
    out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
    rgb = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    bgr = (rgb[:, :, ::-1] * 255.0).astype(np.uint8)
    return bgr


def apply_hairstyle_overlay(
    base_bgr: np.ndarray,
    hair_rgba: np.ndarray,
    landmarker_result,
    device: torch.device,
    refinement: Optional[HairRefinementNet] = None,
    *,
    hair_scale: float = 1.0,
) -> np.ndarray:
    """
    Warp hair RGBA onto base_bgr using Face Landmarker; blend with feathered alpha; optional refinement.
    hair_scale scales the destination shape about its centroid (1.0 = face fit from landmarks).
    """
    ih, iw = base_bgr.shape[:2]
    dst_quad = hair_dst_quad_scalp_fit(landmarker_result, iw, ih)
    if dst_quad is None:
        LOGGER.warning("No face landmarks — skipping hairstyle overlay")
        return base_bgr
    hh, ww = hair_rgba.shape[:2]
    src_quad = default_hair_template_quad(hh, ww).copy()
    dst = _scale_dst_points(dst_quad.copy(), hair_scale, iw, ih)
    # Same winding on src/dst quads (triangle crown–left–bottom).
    if _triangle_signed_area(src_quad[:3]) * _triangle_signed_area(dst[:3]) < 0:
        src_quad = src_quad[[0, 3, 2, 1]]
        dst = dst[[0, 3, 2, 1]]
    bgr = None
    alpha = None
    try:
        H = cv2.getPerspectiveTransform(src_quad.astype(np.float32), dst.astype(np.float32))
        if not np.isfinite(H).all() or abs(float(np.linalg.det(H))) < 1e-12:
            raise ValueError("degenerate homography")
        bgr, alpha = warp_rgba_perspective_premultiplied(hair_rgba, H, (iw, ih), interp=cv2.INTER_LINEAR)
    except (cv2.error, ValueError) as e:
        LOGGER.debug("Perspective hair warp fallback to affine: %s", e)
        dst_tri_raw = hair_dst_triangle_scalp_fit(landmarker_result, iw, ih)
        if dst_tri_raw is None:
            return base_bgr
        dst_tri = _scale_dst_points(dst_tri_raw, hair_scale, iw, ih)
        src_tri = default_hair_template_triangle(hh, ww)
        M = cv2.getAffineTransform(src_tri.astype(np.float32), dst_tri.astype(np.float32))
        bgr, alpha = warp_rgba_affine_premultiplied(hair_rgba, M, (iw, ih), interp=cv2.INTER_LINEAR)
    out = blend_bgr_over(base_bgr, bgr, alpha)
    out = refine_composite_gpu(out, refinement, device)
    return out
