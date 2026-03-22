"""
Create default PNG hairstyle overlays (RGBA) under data/hairstyles/.

Run: python scripts/create_default_hairstyles.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "hairstyles"


def draw_hair(h: int, w: int, color_bgr: tuple, style: str) -> np.ndarray:
    """Draw a simple hair silhouette with alpha falloff (artistic placeholder assets)."""
    bgra = np.zeros((h, w, 4), dtype=np.uint8)
    center = (w // 2, int(h * 0.35))
    axes = (int(w * 0.42), int(h * 0.38))
    cv2.ellipse(bgra, center, axes, 0, 180, 360, (*color_bgr, 220), thickness=-1)
    if style == "wavy":
        for i in range(5):
            off = int((i - 2) * w * 0.08)
            cv2.ellipse(
                bgra,
                (center[0] + off, center[1] + 20),
                (int(w * 0.12), int(h * 0.1)),
                0,
                0,
                180,
                (*color_bgr, 180),
                thickness=6,
            )
    elif style == "short":
        cv2.rectangle(bgra, (int(w * 0.2), int(h * 0.15)), (int(w * 0.8), int(h * 0.45)), (*color_bgr, 200), -1)
    else:  # long
        cv2.ellipse(bgra, (center[0], center[1] + 80), (int(w * 0.35), int(h * 0.45)), 0, 0, 180, (*color_bgr, 200), -1)
    # Feather alpha at bottom
    alpha = bgra[:, :, 3].astype(np.float32)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigmaX=5, sigmaY=5)
    bgra[:, :, 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return bgra


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    h, w = 512, 512
    assets = [
        ("style_wavy_brown.png", (40, 28, 18), "wavy"),
        ("style_short_black.png", (15, 15, 20), "short"),
        ("style_long_blonde.png", (60, 72, 120), "long"),
    ]
    for name, col, st in assets:
        img = draw_hair(h, w, col, st)
        cv2.imwrite(str(OUT / name), img)
        print("Wrote", OUT / name)


if __name__ == "__main__":
    main()
