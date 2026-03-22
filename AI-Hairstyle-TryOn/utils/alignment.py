"""
Align hairstyle PNG overlays to the face using MediaPipe face mesh landmarks.
Maps a canonical triangle on the hairstyle asset to stable points on the forehead / sides.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FaceLandmarks2D:
    """Normalized image coordinates (pixel space) for a few stable indices."""

    points: np.ndarray  # Nx2 float32


# MediaPipe FaceMesh landmark indices (stable for frontal faces).
# Forehead center, left temple, right temple (approximate hairline anchors).
IDX_FOREHEAD = 10
IDX_LEFT_TEMPLE = 234
IDX_RIGHT_TEMPLE = 454
IDX_CHIN = 152  # chin tip — for face height to scale hair triangle to scalp


def landmarks_from_face_landmarker(
    landmarker_result, image_width: int, image_height: int
) -> Optional[FaceLandmarks2D]:
    """Convert Face Landmarker Task API result to pixel coordinates (478 landmarks, same indices as FaceMesh)."""
    if not landmarker_result or not landmarker_result.face_landmarks:
        return None
    lm = landmarker_result.face_landmarks[0]
    pts = []
    for idx in (IDX_FOREHEAD, IDX_LEFT_TEMPLE, IDX_RIGHT_TEMPLE):
        p = lm[idx]
        pts.append([p.x * image_width, p.y * image_height])
    return FaceLandmarks2D(points=np.array(pts, dtype=np.float32))


def _triangle_signed_area(pts: np.ndarray) -> float:
    """Signed area * 2 for winding (image coords, y down)."""
    p0, p1, p2 = pts[0], pts[1], pts[2]
    return (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])


def hair_dst_triangle_scalp_fit(
    landmarker_result,
    image_width: int,
    image_height: int,
    temple_scale: float = 1.12,
) -> Optional[np.ndarray]:
    """
    3x2 destination points for affine hair warp onto the scalp.

    Important: MediaPipe index 10 is the glabella (between brows), NOT the hairline. Mapping
    the top of the wig asset to lm[10] places hair on the eyes/nose. We anchor the top to an
    estimated hairline above the glabella and center X between the temples.
    """
    fl = landmarks_from_face_landmarker(landmarker_result, image_width, image_height)
    if fl is None or not landmarker_result.face_landmarks:
        return None
    lm = landmarker_result.face_landmarks[0]
    lh = fl.points[1].copy()
    rh = fl.points[2].copy()
    x10 = float(lm[IDX_FOREHEAD].x * image_width)
    y10 = float(lm[IDX_FOREHEAD].y * image_height)
    chin_y = float(lm[IDX_CHIN].y * image_height)
    face_h = max(8.0, chin_y - y10)
    tw = float(np.linalg.norm(rh - lh))
    lift = 0.145 + min(0.04, max(0.0, (tw / max(image_width, 1.0) - 0.28) * 0.15))
    hairline_y = y10 - lift * face_h
    mid_x = float((lh[0] + rh[0]) * 0.5)
    fh = np.array([mid_x, hairline_y], dtype=np.float32)
    fh[0] = float(np.clip(fh[0], 0, image_width - 1))
    fh[1] = max(0.0, min(fh[1], image_height - 1))
    # Stretch from glabella toward temples so scale matches head width.
    glabella = np.array([x10, y10], dtype=np.float32)
    lh2 = fh + (lh - glabella) * temple_scale
    rh2 = fh + (rh - glabella) * temple_scale
    lh2[0] = float(np.clip(lh2[0], 0, image_width - 1))
    lh2[1] = float(np.clip(lh2[1], 0, image_height - 1))
    rh2[0] = float(np.clip(rh2[0], 0, image_width - 1))
    rh2[1] = float(np.clip(rh2[1], 0, image_height - 1))
    dst = np.stack([fh, lh2, rh2], axis=0).astype(np.float32)
    # Match winding order to default hair template (top, bottom-left, bottom-right).
    hh, ww = 512, 512  # shape only matters for winding sign of src template
    src = default_hair_template_triangle(hh, ww)
    if _triangle_signed_area(src) * _triangle_signed_area(dst) < 0:
        dst = np.stack([fh, rh2, lh2], axis=0).astype(np.float32)
    return dst


def landmarks_from_face_mesh(mesh_result, image_width: int, image_height: int):
    """Deprecated name — use landmarks_from_face_landmarker."""
    return landmarks_from_face_landmarker(mesh_result, image_width, image_height)


def default_hair_template_triangle(h: int, w: int) -> np.ndarray:
    """
    Three points on the hairstyle PNG: top-center, bottom-left, bottom-right of upper region.
    Coordinates in pixel space (x, y).
    """
    return np.array(
        [
            [w * 0.5, h * 0.08],
            [w * 0.12, h * 0.55],
            [w * 0.88, h * 0.55],
        ],
        dtype=np.float32,
    )


def default_hair_template_quad(h: int, w: int) -> np.ndarray:
    """
    Four corners on the hairstyle PNG (clockwise: top → left → bottom → right) so a perspective
    warp can match crown, temples, and the lower hair mass to the face.
    """
    return np.array(
        [
            [w * 0.5, h * 0.035],  # crown
            [w * 0.065, h * 0.48],  # left side
            [w * 0.5, h * 0.90],  # bottom of asset (long styles)
            [w * 0.935, h * 0.48],  # right side
        ],
        dtype=np.float32,
    )


def hair_dst_quad_scalp_fit(
    landmarker_result,
    image_width: int,
    image_height: int,
    temple_scale: float = 1.12,
) -> Optional[np.ndarray]:
    """
    4x2 destination points for a perspective hair warp: crown, left temple, bottom anchor, right temple.
    Bottom anchor sits near the chin so long PNGs follow the face length.
    """
    fl = landmarks_from_face_landmarker(landmarker_result, image_width, image_height)
    if fl is None or not landmarker_result.face_landmarks:
        return None
    lm = landmarker_result.face_landmarks[0]
    lh = fl.points[1].copy()
    rh = fl.points[2].copy()
    x10 = float(lm[IDX_FOREHEAD].x * image_width)
    y10 = float(lm[IDX_FOREHEAD].y * image_height)
    chin_y = float(lm[IDX_CHIN].y * image_height)
    face_h = max(8.0, chin_y - y10)
    tw = float(np.linalg.norm(rh - lh))
    lift = 0.145 + min(0.04, max(0.0, (tw / max(image_width, 1.0) - 0.28) * 0.15))
    hairline_y = y10 - lift * face_h
    mid_x = float((lh[0] + rh[0]) * 0.5)
    fh = np.array([mid_x, hairline_y], dtype=np.float32)
    fh[0] = float(np.clip(fh[0], 0, image_width - 1))
    fh[1] = max(0.0, min(fh[1], image_height - 1))
    glabella = np.array([x10, y10], dtype=np.float32)
    lh2 = fh + (lh - glabella) * temple_scale
    rh2 = fh + (rh - glabella) * temple_scale
    lh2[0] = float(np.clip(lh2[0], 0, image_width - 1))
    lh2[1] = float(np.clip(lh2[1], 0, image_height - 1))
    rh2[0] = float(np.clip(rh2[0], 0, image_width - 1))
    rh2[1] = float(np.clip(rh2[1], 0, image_height - 1))
    # Lower anchor: below chin — pulls long hair assets down the neck/chest naturally.
    y_bottom = chin_y + 0.11 * face_h
    y_bottom = float(np.clip(y_bottom, fh[1] + 0.22 * face_h, image_height - 1))
    q_bottom = np.array([mid_x, y_bottom], dtype=np.float32)
    # Order matches default_hair_template_quad: top, left, bottom, right.
    dst = np.stack([fh, lh2, q_bottom, rh2], axis=0).astype(np.float32)
    return dst


def estimate_affine(
    src_tri: np.ndarray,
    dst_tri: np.ndarray,
) -> np.ndarray:
    """2x3 affine from 3 point correspondences (OpenCV getAffineTransform)."""
    return cv2.getAffineTransform(src_tri.astype(np.float32), dst_tri.astype(np.float32))


def warp_hairstyle_overlay(
    hair_rgba: np.ndarray,
    face_pts_dst: np.ndarray,
    src_tri: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp RGBA hairstyle asset to match face_pts_dst (3x2).
    Returns warped BGRA and alpha channel (H,W) float32.
    """
    hh, ww = hair_rgba.shape[:2]
    src = src_tri if src_tri is not None else default_hair_template_triangle(hh, ww)
    M = estimate_affine(src, face_pts_dst)
    dsize = (hair_rgba.shape[1], hair_rgba.shape[0])
    warped = cv2.warpAffine(hair_rgba, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if warped.shape[2] == 3:
        alpha = np.ones((warped.shape[0], warped.shape[1]), dtype=np.float32)
        bgr = warped
    else:
        bgr = warped[:, :, :3]
        alpha = warped[:, :, 3].astype(np.float32) / 255.0
    return cv2.merge([bgr[:, :, 0], bgr[:, :, 1], bgr[:, :, 2], (alpha * 255).astype(np.uint8)]), alpha


def warp_hairstyle_to_image_size(
    hair_rgba: np.ndarray,
    image_shape: Tuple[int, int],
    face_pts_dst: np.ndarray,
    src_tri: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Warp hair asset into full image dimensions (H,W) from face_pts_dst already in image pixel coords.
    """
    h, w = image_shape[:2]
    hh, ww = hair_rgba.shape[:2]
    src = src_tri if src_tri is not None else default_hair_template_triangle(hh, ww)
    M = estimate_affine(src, face_pts_dst)
    warped = cv2.warpAffine(hair_rgba, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if warped.shape[2] == 3:
        alpha = np.ones((h, w), dtype=np.float32)
        bgr = warped
    else:
        bgr = warped[:, :, :3]
        alpha = warped[:, :, 3].astype(np.float32) / 255.0
    return bgr, alpha
