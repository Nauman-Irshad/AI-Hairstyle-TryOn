"""
End-to-end AI Hairstyle Try-On pipeline (GPU PyTorch + MediaPipe).

Flow:
  0) Optional: rembg — remove portrait background, composite on white
  1) Face landmarks (MediaPipe) — hairline / temples for scalp region
  2) Hair segmentation (BiSeNet) + landmark hairline arc → mask of native hair + scalp
  3) LaMa inpainting — remove user hair and fill a clean scalp/forehead
  4) Hairstyle PNG warped + blended on the inpainted result
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from app.backend.model_loader import ModelBundle
from pipeline.step1_face import detect_face_mesh
from pipeline.step2_segmentation import segment_hair
from pipeline.step3_inpaint import inpaint_hair_region
from pipeline.step4_hairstyle import apply_style
from utils.helpers import next_result_path, ensure_dir
from utils.remove_background import remove_background_bgr

LOGGER = logging.getLogger(__name__)


def run_try_on(
    image_bgr: np.ndarray,
    hairstyle_filename: str,
    bundle: ModelBundle,
    hairstyles_dir: Path,
    save_result: bool = True,
    results_dir: Optional[Path] = None,
    *,
    hair_scale: float = 1.0,
    remove_background: bool = True,
) -> Tuple[np.ndarray, Path]:
    """
    Execute full pipeline. Returns (result_bgr, saved_path or placeholder path).
    """
    if remove_background:
        LOGGER.info("Removing portrait background (rembg → white)…")
        image_bgr = remove_background_bgr(image_bgr)

    landmarker_result = detect_face_mesh(image_bgr, bundle.face_landmarker)
    LOGGER.info("Predicting scalp / native hair mask (BiSeNet + hairline landmarks)…")
    hair_mask = segment_hair(bundle.segmentation_net, image_bgr, bundle.device, landmarker_result)
    LOGGER.info("Inpainting hair region (LaMa)…")
    inpainted = inpaint_hair_region(bundle.inpainter, image_bgr, hair_mask)
    hair_path = hairstyles_dir / hairstyle_filename
    if not hair_path.is_file():
        raise FileNotFoundError(f"Hairstyle not found: {hair_path}")
    LOGGER.info("Applying hairstyle overlay…")
    final_bgr = apply_style(
        inpainted,
        landmarker_result,
        hair_path,
        bundle.device,
        bundle.refinement_net,
        hair_scale=hair_scale,
    )
    saved = Path("")
    if save_result and results_dir is not None:
        ensure_dir(results_dir)
        import cv2

        saved = next_result_path(results_dir)
        cv2.imwrite(str(saved), final_bgr)
        LOGGER.info("Saved result to %s", saved)
    return final_bgr, saved


def run_remove_hair_only(
    image_bgr: np.ndarray,
    bundle: ModelBundle,
    *,
    remove_background: bool = True,
) -> np.ndarray:
    """
    Face landmarks → hair mask (BiSeNet) → LaMa inpaint.
    Same prep as try-on but no hairstyle overlay. Returns BGR uint8.
    """
    if remove_background:
        LOGGER.info("Removing portrait background (rembg → white)…")
        image_bgr = remove_background_bgr(image_bgr)

    landmarker_result = detect_face_mesh(image_bgr, bundle.face_landmarker)
    LOGGER.info("Predicting scalp / native hair mask (BiSeNet + hairline landmarks)…")
    hair_mask = segment_hair(bundle.segmentation_net, image_bgr, bundle.device, landmarker_result)
    LOGGER.info("Inpainting hair region (LaMa)…")
    return inpaint_hair_region(bundle.inpainter, image_bgr, hair_mask)
