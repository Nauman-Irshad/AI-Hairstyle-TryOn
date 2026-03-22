"""
Download hair-only (wig / human-hair style) transparent PNGs for try-on overlays.

Sources: pngimg.com wig category — CC BY-NC 4.0 (non-commercial; attribution).
See https://pngimg.com/license — keep this notice if you redistribute assets.

Run from project root:
  python scripts/download_hairstyle_assets.py

Writes resized copies (max long edge 1024px, alpha preserved) under data/hairstyles/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hairstyles"
MAX_EDGE = 1024

# Curated pngimg.com wig PNGs (transparent background, hair/wig only in frame).
ASSETS: list[tuple[str, str]] = [
    ("https://pngimg.com/d/wig_PNG155.png", "hair_online_short_dark.png"),
    ("https://pngimg.com/d/wig_PNG157.png", "hair_online_bob_brown.png"),
    ("https://pngimg.com/d/wig_PNG159.png", "hair_online_long_wavy_dark.png"),
    ("https://pngimg.com/d/wig_PNG160.png", "hair_online_long_straight_brunette.png"),
    ("https://pngimg.com/d/wig_PNG162.png", "hair_online_midlength_copper.png"),
    ("https://pngimg.com/d/wig_PNG164.png", "hair_online_curly_volume.png"),
]


def _fetch(url: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": "AI-Hairstyle-TryOn/1.0 (asset download)"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _resize_keep_alpha(bgra: np.ndarray, max_edge: int) -> np.ndarray:
    h, w = bgra.shape[:2]
    m = max(h, w)
    if m <= max_edge:
        return bgra
    scale = max_edge / float(m)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    return cv2.resize(bgra, (nw, nh), interpolation=cv2.INTER_AREA)


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    OUT.mkdir(parents=True, exist_ok=True)
    for url, fname in ASSETS:
        dest = OUT / fname
        print(f"Fetching {url} -> {dest.name} ...")
        raw = _fetch(url)
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"  SKIP: could not decode image from {url}")
            continue
        if img.ndim != 3:
            print(f"  SKIP: unexpected shape {img.shape}")
            continue
        if img.shape[2] == 3:
            bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
            bgra[:, :, 3] = 255
        else:
            bgra = img
        bgra = _resize_keep_alpha(bgra, MAX_EDGE)
        cv2.imwrite(str(dest), bgra)
        print(f"  OK {bgra.shape[1]}x{bgra.shape[0]} -> {dest}")
    print("Done.")


if __name__ == "__main__":
    main()
