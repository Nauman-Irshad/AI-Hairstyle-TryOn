"""Logging, device selection, filesystem helpers."""

import logging
import sys
from pathlib import Path
from typing import Optional

import torch


def setup_logging(log_dir: Optional[Path] = None, level: int = logging.INFO) -> None:
    """Console + optional file logging under outputs/logs/."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        fh.setLevel(level)
        handlers.append(fh)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )


def get_torch_device() -> torch.device:
    """Prefer CUDA for full GPU pipeline."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def next_result_path(results_dir: Path, prefix: str = "result") -> Path:
    ensure_dir(results_dir)
    i = 0
    while True:
        cand = results_dir / f"{prefix}_{i:05d}.png"
        if not cand.exists():
            return cand
        i += 1
