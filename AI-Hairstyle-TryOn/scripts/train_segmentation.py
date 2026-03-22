"""
Train / fine-tune BiSeNet on CelebAMask-HQ-style data (CSV: image_path, mask_path, labels).

Run: python scripts/train_segmentation.py --csv data/processed/celebamask_sample.csv --epochs 1
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

from models.segmentation.bisenet import BiSeNet

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def load_rows(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("image_path"):
                rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=ROOT / "data" / "processed" / "celebamask_sample.csv")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = BiSeNet(19).to(device)
    opt = optim.AdamW(net.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss(ignore_index=255)

    rows = load_rows(args.csv)
    if not rows:
        LOGGER.warning("No rows in CSV — nothing to train. Run scripts/download_datasets.py first.")
        return

    LOGGER.info("Training stub: %d rows, device=%s", len(rows), device)
    for ep in range(args.epochs):
        # Real training would load image/mask pairs from disk; this loop demonstrates the wiring only.
        LOGGER.info("Epoch %d — replace this with DataLoader over paired images and masks.", ep + 1)
    torch.save(net.state_dict(), ROOT / "models" / "weights" / "bisenet_finetuned_stub.pth")
    LOGGER.info("Saved stub checkpoint to models/weights/bisenet_finetuned_stub.pth")


if __name__ == "__main__":
    main()
