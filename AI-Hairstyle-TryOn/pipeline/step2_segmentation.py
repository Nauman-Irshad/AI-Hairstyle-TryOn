"""Step 2: Hair segmentation with BiSeNet."""

import logging
from typing import Any, Optional

import numpy as np

from models.segmentation.inference import predict_hair_mask, refine_hair_mask
from utils.hair_mask import augment_dark_hair_mask_u8, combine_bisenet_and_hairline_cap

LOGGER = logging.getLogger(__name__)


def segment_hair(
    net,
    image_bgr: np.ndarray,
    device,
    landmarker_result: Optional[Any] = None,
) -> np.ndarray:
    """
    Return uint8 hair mask 255=hair.
    When face landmarks are available, union BiSeNet with an upper-head cap so the
    hairline / forehead strip is included for cleaner removal before overlay.
    """
    mask = predict_hair_mask(net, image_bgr, device)
    mask = refine_hair_mask(mask, dilate_iter=2, erode_iter=0)
    if landmarker_result is not None:
        mask = combine_bisenet_and_hairline_cap(mask, landmarker_result)
        mask = augment_dark_hair_mask_u8(image_bgr, mask, landmarker_result)
    return mask
