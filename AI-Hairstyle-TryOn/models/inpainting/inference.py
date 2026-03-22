"""
LaMa inpainting inference on GPU: image + binary mask -> inpainted RGB.

Supports:
- PyTorch Lightning checkpoints (dict with state_dict / generator.* keys)
- TorchScript / JIT .pt (many Hugging Face `big-lama.pt` files are exported this way)
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .model import build_big_lama_generator

LOGGER = logging.getLogger(__name__)

DEFAULT_INPAINT_SIZE = 512


def _torch_load_any(path: Path, device: torch.device):
    """PyTorch 2.6+ safe load for trusted checkpoints."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_generator_state_dict(ckpt_obj: dict) -> dict:
    """Extract generator weights from Lightning-style checkpoint dict."""
    if "state_dict" in ckpt_obj:
        sd = ckpt_obj["state_dict"]
    else:
        sd = ckpt_obj
    out = {}
    for k, v in sd.items():
        if k.startswith("generator."):
            out[k[len("generator.") :]] = v
    if not out:
        for k, v in sd.items():
            if not k.startswith("discriminator"):
                out[k] = v
    return out


def load_checkpoint_into_generator(generator: torch.nn.Module, weights_path: Path, device: torch.device) -> None:
    ckpt = _torch_load_any(weights_path, device)
    if isinstance(ckpt, torch.jit.ScriptModule):
        raise TypeError("JIT checkpoint — use load_lama_checkpoint instead")
    if not isinstance(ckpt, dict):
        raise ValueError(f"Expected dict checkpoint, got {type(ckpt)}")
    gen_sd = extract_generator_state_dict(ckpt)
    missing, unexpected = generator.load_state_dict(gen_sd, strict=False)
    LOGGER.info("LaMa generator (state_dict) loaded. missing=%s unexpected=%s", missing, unexpected)


def load_lama_checkpoint(weights_path: Path, device: torch.device) -> Tuple[str, Union[torch.nn.Module, None], Optional[torch.jit.ScriptModule]]:
    """
    Returns ("eager", generator, None) or ("jit", None, jit_module).
    """
    if not weights_path.is_file():
        raise FileNotFoundError(f"LaMa weights not found: {weights_path}")

    # 1) TorchScript / exported LaMa — use torch.jit.load first (avoids torch.load zip warning).
    try:
        jit = torch.jit.load(str(weights_path), map_location=device)
        jit.eval()
        LOGGER.info("LaMa TorchScript model loaded via torch.jit.load (%s)", weights_path.name)
        return "jit", None, jit
    except Exception as e:
        LOGGER.debug("Not a TorchScript archive (trying pickle checkpoint): %s", e)

    # 2) PyTorch Lightning / pickle checkpoint dict
    ckpt = _torch_load_any(weights_path, device)
    if isinstance(ckpt, dict):
        gen = build_big_lama_generator().to(device)
        gen_sd = extract_generator_state_dict(ckpt)
        missing, unexpected = gen.load_state_dict(gen_sd, strict=False)
        gen.eval()
        LOGGER.info("LaMa eager generator loaded. missing=%s unexpected=%s", missing, unexpected)
        return "eager", gen, None

    raise ValueError(f"Unrecognized LaMa checkpoint type: {type(ckpt)}")


def _pad_to_multiple(img: np.ndarray, mask: np.ndarray, multiple: int = 8) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    h, w = img.shape[:2]
    nh = (h + multiple - 1) // multiple * multiple
    nw = (w + multiple - 1) // multiple * multiple
    if nh == h and nw == w:
        return img, mask, (h, w)
    img_p = cv2.copyMakeBorder(img, 0, nh - h, 0, nw - w, cv2.BORDER_REFLECT_101)
    mask_p = cv2.copyMakeBorder(mask, 0, nh - h, 0, nw - w, cv2.BORDER_CONSTANT, value=0)
    return img_p, mask_p, (h, w)


class LaMaInpainter:
    """Inpainting: masked RGB + mask (4ch) -> prediction; composite with original."""

    def __init__(self, device: torch.device, weights_path: Optional[Path] = None):
        self.device = device
        self.generator: Optional[torch.nn.Module] = None
        self._jit: Optional[torch.jit.ScriptModule] = None

        if weights_path is not None and Path(weights_path).is_file():
            mode, gen, jit_m = load_lama_checkpoint(Path(weights_path), device)
            self.generator = gen
            self._jit = jit_m
            if mode == "eager" and self.generator is None:
                raise RuntimeError("Eager mode but generator is None")
            if mode == "jit" and self._jit is None:
                raise RuntimeError("JIT mode but model is None")
        else:
            self.generator = build_big_lama_generator().to(device)
            self.generator.eval()
            LOGGER.warning("LaMa weights missing — using random-init generator (poor quality)")

    def _forward(
        self,
        masked_rgb: torch.Tensor,
        mask_t: torch.Tensor,
        x_four_ch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Eager LaMa FFC: single 4-channel tensor (masked RGB + mask).
        Exported TorchScript LaMa: forward(image, mask) with 3ch + 1ch tensors.
        """
        if self._jit is not None:
            # JIT: forward(Tensor image, Tensor mask) — image is typically masked RGB [B,3,H,W]
            out = self._jit(masked_rgb, mask_t)
            if isinstance(out, (tuple, list)):
                out = out[0]
            return out
        if self.generator is not None:
            return self.generator(x_four_ch)
        raise RuntimeError("No inpainting model loaded")

    @torch.inference_mode()
    def inpaint(
        self,
        image_bgr: np.ndarray,
        mask_u8: np.ndarray,
        max_side: int = DEFAULT_INPAINT_SIZE,
    ) -> np.ndarray:
        orig_h, orig_w = image_bgr.shape[:2]
        if mask_u8.ndim == 3:
            mask_u8 = mask_u8[:, :, 0]
        scale = min(max_side / max(orig_h, orig_w), 1.0)
        rh, rw = int(orig_h * scale), int(orig_w * scale)
        rh = max(8, (rh // 8) * 8)
        rw = max(8, (rw // 8) * 8)
        img = cv2.resize(image_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
        m = cv2.resize(mask_u8, (rw, rh), interpolation=cv2.INTER_NEAREST)
        img, m, orig_hw = _pad_to_multiple(img, m, 8)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_f = (m.astype(np.float32) / 255.0)[..., None]
        im_t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)
        mask_t = torch.from_numpy(mask_f).permute(2, 0, 1).unsqueeze(0).to(self.device)
        masked_img = im_t * (1.0 - mask_t)
        x_four = torch.cat([masked_img, mask_t], dim=1)
        pred = self._forward(masked_img, mask_t, x_four)
        inp = mask_t * pred + (1.0 - mask_t) * im_t
        out = inp.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out = (out * 255.0).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        out = out[: orig_hw[0], : orig_hw[1]]
        out = cv2.resize(out, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return out
