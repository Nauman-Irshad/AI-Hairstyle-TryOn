"""Step 4: Apply selected hairstyle overlay aligned to face landmarks."""

import logging
from pathlib import Path

import numpy as np

from models.hairstyle.inference import apply_hairstyle_overlay, load_hairstyle_rgba

LOGGER = logging.getLogger(__name__)


def apply_style(
    inpainted_bgr: np.ndarray,
    landmarker_result,
    hairstyle_path: Path,
    device,
    refinement_net=None,
    *,
    hair_scale: float = 1.0,
) -> np.ndarray:
    """Load PNG hairstyle and composite using step1 Face Landmarker result."""
    hair = load_hairstyle_rgba(hairstyle_path)
    return apply_hairstyle_overlay(
        inpainted_bgr, hair, landmarker_result, device, refinement_net, hair_scale=hair_scale
    )
