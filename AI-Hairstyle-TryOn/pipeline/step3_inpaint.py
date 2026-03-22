"""Step 3: Remove hair region via LaMa inpainting."""

import logging

import numpy as np

from models.inpainting.inference import LaMaInpainter

LOGGER = logging.getLogger(__name__)


def inpaint_hair_region(inpainter: LaMaInpainter, image_bgr: np.ndarray, hair_mask_u8: np.ndarray) -> np.ndarray:
    """Inpaint masked region (hair) to get a clean scalp / forehead area."""
    return inpainter.inpaint(image_bgr, hair_mask_u8)
