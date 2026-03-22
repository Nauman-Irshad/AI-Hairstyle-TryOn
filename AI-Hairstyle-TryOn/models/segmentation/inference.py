"""
BiSeNet inference: RGB image -> hair mask (binary float).
"""

import logging
from typing import Tuple

import cv2
import numpy as np
import torch

from .load_model import HAIR_CLASS_INDEX

LOGGER = logging.getLogger(__name__)


def _preprocess_bgr(bgr: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
    """Resize to model input size, ImageNet normalize, NCHW float."""
    h, w = size
    img = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    return torch.from_numpy(img)


@torch.inference_mode()
def predict_hair_mask(
    net: torch.nn.Module,
    image_bgr: np.ndarray,
    device: torch.device,
    input_size: Tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    Returns binary hair mask (H, W) uint8 {0,255} at original image resolution.
    """
    orig_h, orig_w = image_bgr.shape[:2]
    h, w = input_size
    # Step 1: normalize and run BiSeNet
    inp = _preprocess_bgr(image_bgr, (h, w)).to(device)
    out, _, _ = net(inp)
    # Step 2: argmax over classes -> per-pixel label
    pred = out.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
    # Step 3: extract hair class
    hair = (pred == HAIR_CLASS_INDEX).astype(np.uint8) * 255
    # Step 4: resize mask back to original size
    hair = cv2.resize(hair, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return hair


def refine_hair_mask(mask_u8: np.ndarray, dilate_iter: int = 2, erode_iter: int = 0) -> np.ndarray:
    """Slightly dilate hair mask so inpainting covers hair boundary (reduces edge seams)."""
    m = mask_u8.copy()
    if dilate_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.dilate(m, k, iterations=dilate_iter)
    if erode_iter > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        m = cv2.erode(m, k, iterations=erode_iter)
    return m
