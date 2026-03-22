"""
Train inpainting (LaMa-style generator) on FFHQ crops with synthetic masks.

Run: python scripts/train_inpainting.py --csv data/processed/ffhq_sample.csv --epochs 1
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.inpainting.model import build_big_lama_generator

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def load_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("image_path"):
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "processed" / "ffhq_sample.csv")
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gen = build_big_lama_generator().to(device)
    opt = optim.AdamW(gen.parameters(), lr=1e-4)
    crit = nn.L1Loss()

    rows = load_rows(args.csv)
    if not rows:
        LOGGER.warning("No rows — run scripts/download_datasets.py first.")
        return

    LOGGER.info("Inpainting training stub: %d rows", len(rows))
    for ep in range(args.epochs):
        LOGGER.info("Epoch %d — plug in Dataset that loads RGB + mask, forward masked input.", ep + 1)
    out = ROOT / "models" / "weights" / "inpainting_finetuned_stub.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(gen.state_dict(), out)
    LOGGER.info("Saved %s", out)


if __name__ == "__main__":
    main()
