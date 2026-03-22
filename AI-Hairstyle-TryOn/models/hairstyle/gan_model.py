"""
Lightweight refinement CNN (optional) to smooth compositing artifacts after overlay.
Runs on GPU; small enough for real-time use alongside the main pipeline.
"""

import torch
import torch.nn as nn


class HairRefinementNet(nn.Module):
    """
    Residual bottleneck that maps BGR 3xHxW -> 3xHxW in [0,1], trained to reduce edge halos.
    When untrained, behaves like a near-identity (still differentiable for future fine-tuning).
    """

    def __init__(self, channels: int = 32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.dec = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x in [0,1]
        e = self.enc(x)
        r = self.res(e) + e
        y = self.dec(r)
        return y


def build_refinement_net(device: torch.device) -> HairRefinementNet:
    net = HairRefinementNet().to(device)
    net.eval()
    return net
