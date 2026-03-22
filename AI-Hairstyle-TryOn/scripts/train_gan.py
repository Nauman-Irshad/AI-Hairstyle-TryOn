"""
Train / fine-tune the lightweight HairRefinementNet on composited pairs (optional).

Run: python scripts/train_gan.py --epochs 1
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.hairstyle.gan_model import build_refinement_net

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = build_refinement_net(device)
    opt = optim.AdamW(net.parameters(), lr=1e-4)
    crit = nn.MSELoss()

    LOGGER.info("GAN/refinement stub on %s — supply paired (input, target) tensors for real training.", device)
    for ep in range(args.epochs):
        x = torch.rand(1, 3, 128, 128, device=device)
        y = net(x)
        loss = crit(y, x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        LOGGER.info("epoch %d loss=%.6f", ep + 1, loss.item())

    out = ROOT / "models" / "weights" / "refinement_net_stub.pth"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out)
    LOGGER.info("Saved %s", out)


if __name__ == "__main__":
    main()
