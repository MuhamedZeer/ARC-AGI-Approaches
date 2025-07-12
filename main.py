"""
Minimal, ready‑to‑train implementation of the convolutional baseline
we tried before pivoting away from CNNs.

* Three stride‑2 Conv‑Res blocks compress a 30×30×11 grid to a 512‑d
  latent.
* A key MLP fuses three (input, output) example pairs into a single
  task vector.
* A map MLP converts (test‑latent + key) into an output latent.
* A mirrored deconvolutional decoder reconstructs the grid.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------

def pad_to_30x30(grid: List[List[int]], pad_value: int = 10) -> torch.Tensor:
    h, w = len(grid), len(grid[0])
    canvas = torch.full((30, 30), pad_value, dtype=torch.long)
    canvas[:h, :w] = torch.tensor(grid, dtype=torch.long)
    return canvas


def one_hot(imgs: torch.Tensor, num_classes: int = 11) -> torch.Tensor:
    return F.one_hot(imgs, num_classes).permute(0, 3, 1, 2).float()

class ResBlock(nn.Module):
    def __init__(self, ch: int, p: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1), nn.Dropout(p),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)


class ConvBlock(nn.Module):
    def __init__(self, mode: str, c_in: int, c_out: int, p: float = 0.1):
        super().__init__()
        if mode == "down":
            conv = nn.Conv2d(c_in, c_out, 4, stride=2)
        elif mode == "up":
            conv = nn.ConvTranspose2d(c_in, c_out, 4, stride=2)
        elif mode == "same":
            conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        else:
            raise ValueError("mode must be down/up/same")
        self.seq = nn.Sequential(conv, nn.BatchNorm2d(c_out), nn.ReLU(), nn.Dropout(p))

    def forward(self, x):
        return self.seq(x)


class Encoder(nn.Module):
    def __init__(self, chans=(256, 512, 512), latent_dim=512):
        super().__init__()
        c0, c1, c2 = chans
        self.conv1 = ConvBlock("down", 11, c0)
        self.res1 = ResBlock(c0)
        self.conv2 = ConvBlock("down", c0, c1)
        self.res2 = ResBlock(c1)
        self.conv3 = ConvBlock("down", c1, c2)
        self.fc = nn.Linear(c2 * 2 * 2, latent_dim)

    def forward(self, img):
        x = one_hot(img)
        skips = []
        x = self.conv1(x); x = self.res1(x); skips.append(x)
        x = self.conv2(x); x = self.res2(x); skips.append(x)
        x = self.conv3(x); skips.append(x)
        z = self.fc(x.flatten(1))
        return z, skips


class Decoder(nn.Module):
    def __init__(self, chans=(256, 512, 512), latent_dim=512):
        super().__init__()
        c0, c1, c2 = chans
        self.fc = nn.Linear(latent_dim, c2 * 2 * 2)
        self.up3 = ConvBlock("up", c2 * 2, c1)
        self.res3 = ResBlock(c1)
        self.up2 = ConvBlock("up", c1 * 2, c0)
        self.res2 = ResBlock(c0)
        self.up1 = ConvBlock("up", c0 * 2, c0)
        self.head = nn.Conv2d(c0, 11, 3, padding=1)

    def forward(self, z, skips):
        x = self.fc(z).view(z.size(0), -1, 2, 2)
        x = torch.cat((x, skips[2]), 1)
        x = self.up3(x); x = self.res3(x)
        x = torch.cat((x, skips[1]), 1)
        x = self.up2(x); x = self.res2(x)
        x = torch.cat((x, skips[0]), 1)
        x = self.up1(x)
        return self.head(x)


class MLP(nn.Sequential):
    def __init__(self, d_in, d_out, hidden=512, p=0.1):
        super().__init__(
            nn.Linear(d_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden, d_out),
        )


class ARCAutoEncoder(nn.Module):
    def __init__(self, latent_dim: object = 512) -> object:
        super().__init__()
        self.enc = Encoder(latent_dim=latent_dim)
        self.dec = Decoder(latent_dim=latent_dim)
        self.key_mlp = MLP(latent_dim * 6, latent_dim)
        self.map_mlp = MLP(latent_dim * 2, latent_dim)

    def _encode_pair(self, grids):
        z_in, _ = self.enc(grids[0])
        z_out, _ = self.enc(grids[1])
        return z_in, z_out

    def forward(self, test_in: torch.Tensor, examples):
        pair_latents = [self._encode_pair(p) for p in examples]
        key = self.key_mlp(torch.cat([torch.cat(z) for z in pair_latents], 1))
        z_test, skips = self.enc(test_in)
        z_out = self.map_mlp(torch.cat((z_test, key), 1))
        logits = self.dec(z_out, skips)
        return logits.argmax(1)



# Dummy single‑task run (for smoke test)
if __name__ == "__main__":
    # random toy example
    torch.manual_seed(0)
    ex_in = torch.randint(0, 11, (1, 30, 30))
    ex_out = ex_in.clone()
    ex_pairs = [(ex_in, ex_out)] * 3
    test_in = torch.randint(0, 11, (1, 30, 30))
    model = ARCAutoEncoder()
    pred = model(test_in, ex_pairs)
    print("Output shape:", pred.shape)
