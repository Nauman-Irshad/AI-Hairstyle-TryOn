"""
Load all deep models once (BiSeNet, LaMa, optional refinement) on CUDA when available.
Downloads pretrained weights into models/weights/ on first run.
"""

import logging
import sys
from pathlib import Path

# Project root on path when this module is imported first (e.g. tests).
_ml = Path(__file__).resolve().parents[2]
if str(_ml) not in sys.path:
    sys.path.insert(0, str(_ml))
from dataclasses import dataclass
from typing import Optional

import torch

from models.inpainting.inference import LaMaInpainter
from models.segmentation.load_model import build_bisenet, load_bisenet_weights
from pipeline.step1_face import FaceLandmarkerRunner

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"

# Hugging Face Hub sources (public mirrors of common checkpoints).
BIENET_URL = "https://huggingface.co/vivym/face-parsing-bisenet/resolve/main/79999_iter.pth"
LAMA_FILENAME = "big-lama.pt"
LAMA_REPO = "fashn-ai/LaMa"
# MediaPipe Tasks — Face Landmarker (replaces legacy solutions.face_mesh)
FACE_LANDMARKER_NAME = "face_landmarker.task"
FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class ModelBundle:
    """Holds all runtime models for the try-on pipeline."""

    device: torch.device
    segmentation_net: torch.nn.Module
    inpainter: LaMaInpainter
    face_landmarker: FaceLandmarkerRunner
    refinement_net: Optional[torch.nn.Module]


def _download(url: str, dest: Path) -> None:
    """Download a file with progress logging."""
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Downloading %s -> %s", url, dest)

    def reporthook(block, block_size, total):
        if total > 0:
            pct = min(100, block * block_size * 100 / total)
            if block % 50 == 0:
                LOGGER.info("Download progress: %.1f%%", pct)

    urllib.request.urlretrieve(url, str(dest), reporthook=reporthook)


def download_face_landmarker_model() -> Path:
    """Download MediaPipe Face Landmarker .task file (Tasks API)."""
    dest = WEIGHTS_DIR / FACE_LANDMARKER_NAME
    if dest.is_file():
        return dest
    _download(FACE_LANDMARKER_URL, dest)
    return dest


def download_bisenet_weights() -> Path:
    """Ensure BiSeNet face-parsing weights exist."""
    dest = WEIGHTS_DIR / "79999_iter.pth"
    if not dest.is_file():
        _download(BIENET_URL, dest)
    return dest


def download_lama_weights() -> Path:
    """Ensure LaMa big-lama checkpoint exists (PyTorch Lightning checkpoint .pt)."""
    dest = WEIGHTS_DIR / LAMA_FILENAME
    if dest.is_file():
        return dest
    # Hugging Face direct resolve URL (follows redirects to LFS).
    url = f"https://huggingface.co/{LAMA_REPO}/resolve/main/{LAMA_FILENAME}"
    _download(url, dest)
    return dest


def load_models(device: Optional[torch.device] = None) -> ModelBundle:
    """
    Build networks, load weights, return bundle.
    Call once at FastAPI startup.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Using device: %s", device)

    # Step 1: MediaPipe Face Landmarker (Tasks API; runs on CPU, small model).
    fl_path = download_face_landmarker_model()
    face_landmarker = FaceLandmarkerRunner(fl_path)

    # Step 2: BiSeNet segmentation
    seg_path = download_bisenet_weights()
    seg_net = build_bisenet(device)
    load_bisenet_weights(seg_net, seg_path, device)

    # Step 3: LaMa inpainting
    lama_path = download_lama_weights()
    inpainter = LaMaInpainter(device=device, weights_path=lama_path)

    # Step 4: optional HairRefinementNet — must stay None unless you load trained weights.
    # An untrained random CNN maps the whole image to ~flat gray (sigmoid ~0.5), which ruins output.
    return ModelBundle(
        device=device,
        segmentation_net=seg_net,
        inpainter=inpainter,
        face_landmarker=face_landmarker,
        refinement_net=None,
    )
