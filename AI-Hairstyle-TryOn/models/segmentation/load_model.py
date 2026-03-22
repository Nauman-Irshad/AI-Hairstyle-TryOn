"""
Load BiSeNet face-parsing weights (79999_iter.pth) from disk or Hugging Face.
"""

import logging
from pathlib import Path

import torch

from .bisenet import BiSeNet

LOGGER = logging.getLogger(__name__)

N_CLASSES = 19
# Face-parsing (CelebAMask-style) class index for hair — see face-parsing.PyTorch labels.
HAIR_CLASS_INDEX = 13


def build_bisenet(device: torch.device) -> BiSeNet:
    """Construct BiSeNet with 19 semantic classes (face parsing)."""
    net = BiSeNet(N_CLASSES)
    net = net.to(device)
    net.eval()
    return net


def load_bisenet_weights(net: BiSeNet, weights_path: Path, device: torch.device) -> None:
    """Load pretrained state_dict; handles DataParallel 'module.' prefix."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"BiSeNet weights not found: {weights_path}")
    try:
        state = torch.load(weights_path, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    new_state = {}
    for k, v in state.items():
        nk = k.replace("module.", "") if k.startswith("module.") else k
        new_state[nk] = v
    missing, unexpected = net.load_state_dict(new_state, strict=False)
    LOGGER.info("BiSeNet loaded. missing=%s unexpected=%s", missing, unexpected)
