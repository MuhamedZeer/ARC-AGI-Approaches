"""pixel_vit.py
A self‑contained scaffolding for the *pixel‑level* Vision Transformer stack that forms the
front‑end of an ARC‑AGI‑2 solver.  It covers:

1. **PixelEmbedder** – colour + absolute position → fixed‑width token
2. **PixelViT** – ViT encoder with prepended `[CLS]` token (returns its embedding)
3. **GridEncoder** – pads a *batch* of variable‑sized integer grids, builds masks,
   and pushes them through PixelViT to obtain one vector per grid
4. **TaskSynthesiser** – another Transformer that pools the six (input/output)
   grid vectors into a single *task* embedding

The module is deliberately lightweight yet flexible (colour vocab, embedding size,
layer counts, etc.) and ships with a mini smoke‑test plus bare‑bones *pytest* cases.
All tensors use **-1** as the *pad* value.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "PixelEmbedder",
    "PixelViT",
    "GridEncoder",
    "TaskSynthesiser",
    "default_pixel_vit",  # quick helper
]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

_PAD_VAL = -1  # sentinel for padded cells


def _make_pad_mask(batch: torch.Tensor, pad_val: int = _PAD_VAL) -> torch.BoolTensor:
    """Return *True* where `batch` contains the padding value (any shape)."""
    return batch == pad_val


# -----------------------------------------------------------------------------
# 1. Pixel‑wise embedding (colour + X + Y)
# -----------------------------------------------------------------------------


class PixelEmbedder(nn.Module):
    """Embed a H×W integer grid into a (H, W, d) tensor.

    * **Colour embedding** – learned lookup of size ``num_colours``
    * **Positional embedding** – sum of learned *x* and *y* vectors
      (max length ``max_grid_size`` each)

    The final embedding is *(colour + x + y)*.  Positions whose colour index is
    *_PAD_VAL* are set to **0** so they do not leak into attention.
    """

    def __init__(
        self,
        num_colours: int = 10,
        emb_dim: int = 128,
        max_grid_size: int = 30,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.num_colours = num_colours
        self.colour_emb = nn.Embedding(num_colours, emb_dim)
        self.x_emb = nn.Embedding(max_grid_size, emb_dim)
        self.y_emb = nn.Embedding(max_grid_size, emb_dim)

        # Cache coordinate ranges so we never recompute tensors on every call.
        self.register_buffer("_x_idx", torch.arange(max_grid_size), persistent=False)
        self.register_buffer("_y_idx", torch.arange(max_grid_size), persistent=False)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        """Args
        -----
        grid: **LongTensor** of shape *(H, W)* with ints in ``[0, num_colours-1]``
              or *_PAD_VAL* for padded cells.
        Returns
        -------
        emb: *(H, W, emb_dim)* float tensor.
        """
        if grid.dim() != 2:
            raise ValueError("PixelEmbedder expects a 2‑D grid (H×W).")
        h, w = grid.shape
        if h > self.x_emb.num_embeddings or w > self.y_emb.num_embeddings:
            raise ValueError(
                f"Grid {grid.shape} exceeds max_grid_size=({self.x_emb.num_embeddings},"
                f" {self.y_emb.num_embeddings})."
            )

        # Colour embedding (masked)
        colour_idx = grid.clamp(min=0)  # negative (pad) → 0 so Embedding lookup is safe
        colour_vec = self.colour_emb(colour_idx)  # (H,W,d)

        pad_mask = grid == _PAD_VAL
        colour_vec[pad_mask] = 0.0  # zero out padded cells

        # Positional embedding – broadcast & add
        x_vec = self.x_emb(self._x_idx[:w])  # (W, d)
        y_vec = self.y_emb(self._y_idx[:h])  # (H, d)
        pos = x_vec.unsqueeze(0) + y_vec.unsqueeze(1)  # (H, W, d)

        return colour_vec + pos  # (H,W,d)


# -----------------------------------------------------------------------------
# 2. Vision Transformer operating on pixel tokens
# -----------------------------------------------------------------------------


class PixelViT(nn.Module):
    """Transformer encoder over flattened pixel tokens with a prepend `[CLS]`."""

    def __init__(
        self,
        emb_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.cls = nn.Parameter(torch.zeros(1, 1, emb_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(
        self,
        tokens: torch.Tensor,  # (B, L, d)
        pad_mask: torch.BoolTensor | None = None,  # (B, L)
    ) -> torch.Tensor:
        b, _, d = tokens.shape
        assert d == self.emb_dim

        cls_tok = self.cls.expand(b, -1, -1)  # (B,1,d)
        tokens = torch.cat([cls_tok, tokens], dim=1)  # (B, L+1, d)

        if pad_mask is not None:
            pad_mask = torch.cat(
                [torch.zeros(b, 1, dtype=torch.bool, device=tokens.device), pad_mask], dim=1
            )  # (B, L+1)

        out = self.transformer(tokens, src_key_padding_mask=pad_mask)  # (B,L+1,d)
        return out[:, 0]  # cls embedding


# -----------------------------------------------------------------------------
# 3. GridEncoder – variable‑sized batch handling
# -----------------------------------------------------------------------------


@dataclass
class GridEncoderCfg:
    emb_dim: int = 128
    depth: int = 4
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    num_colours: int = 10
    max_grid_size: int = 30


class GridEncoder(nn.Module):
    """Embed a *list* of integer grids into a batch of CLS vectors."""

    def __init__(self, cfg: GridEncoderCfg):
        super().__init__()
        self.cfg = cfg
        self.embedder = PixelEmbedder(cfg.num_colours, cfg.emb_dim, cfg.max_grid_size)
        self.vit = PixelViT(cfg.emb_dim, cfg.depth, cfg.num_heads, cfg.mlp_ratio, cfg.dropout)

    def _pad_and_stack(self, grids: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.BoolTensor]:
        """Pad variable‑sized grids into a single 3‑D batch tensor."""
        hs = [g.shape[0] for g in grids]
        ws = [g.shape[1] for g in grids]
        h_max, w_max = max(hs), max(ws)
        pad_batch = []
        for g in grids:
            pad_h, pad_w = h_max - g.shape[0], w_max - g.shape[1]
            pad_grid = F.pad(g, (0, pad_w, 0, pad_h), value=_PAD_VAL)  # (Hmax,Wmax)
            pad_batch.append(pad_grid)
        batch = torch.stack(pad_batch, dim=0)  # (B,Hmax,Wmax)
        mask = _make_pad_mask(batch)  # (B,Hmax,Wmax)
        return batch, mask

    def forward(self, grids: Sequence[torch.Tensor]) -> torch.Tensor:
        """Returns (B, emb_dim) CLS embeddings for *each* grid."""
        batch, mask = self._pad_and_stack(grids)  # (B,H,W)
        b, h, w = batch.shape
        feat = []
        for i in range(b):
            feat.append(self.embedder(batch[i]))  # list[(H,W,d)]
        feat = torch.stack([f.flatten(0, 1) for f in feat], dim=0)  # (B, L, d)
        mask = mask.flatten(1)  # (B, L)
        return self.vit(feat, mask)  # (B,d)


# -----------------------------------------------------------------------------
# 4. Task synthesiser – condense six grid vectors into a *rule* vector
# -----------------------------------------------------------------------------


class TaskSynthesiser(nn.Module):
    """Attention‑based pooling of example grid embeddings → single task vector."""

    def __init__(
        self,
        emb_dim: int = 128,
        depth: int = 2,
        num_heads: int = 4,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.role_emb = nn.Embedding(2, emb_dim)  # 0=input,1=output
        self.cls_task = nn.Parameter(torch.zeros(1, 1, emb_dim))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=emb_dim * mlp_ratio,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=depth)

    def forward(self, grid_vecs: torch.Tensor, roles: torch.Tensor) -> torch.Tensor:
        """Args
        -----
        grid_vecs: *(N, d)* – one CLS per example grid.
        roles: *(N,)*  – 0 for *input* grids, 1 for *output* grids.
        Returns
        -------
        task_vec: *(d,)* summary vector capturing the transformation rule.
        """
        if grid_vecs.ndim != 2:
            raise ValueError("grid_vecs must be (N,d)")
        n, d = grid_vecs.shape
        role_added = grid_vecs + self.role_emb(roles)  # (N,d)
        seq = torch.cat([self.cls_task, role_added.unsqueeze(0)], dim=1)  # (1,N+1,d)
        out = self.encoder(seq)  # (1,N+1,d)
        return out[:, 0, :]


# -----------------------------------------------------------------------------
# Convenience factory
# -----------------------------------------------------------------------------

def default_pixel_vit(num_colours: int = 10) -> tuple[GridEncoder, TaskSynthesiser]:
    cfg = GridEncoderCfg(num_colours=num_colours)
    return GridEncoder(cfg), TaskSynthesiser(cfg.emb_dim)


# -----------------------------------------------------------------------------
# Pytest unit tests (run `pytest pixel_vit.py -q`)
# -----------------------------------------------------------------------------


def _rand_grid(h: int, w: int, num_colours: int) -> torch.Tensor:
    return torch.randint(0, num_colours, (h, w), dtype=torch.long)


def test_variable_sizes():
    enc, _ = default_pixel_vit()
    grids = [_rand_grid(10, 10, 10), _rand_grid(15, 7, 10), _rand_grid(3, 22, 10)]
    out = enc(grids)
    assert out.shape == (3, enc.cfg.emb_dim)


def test_colour_vocab_expand():
    num_colours = 256
    enc, _ = default_pixel_vit(num_colours)
    g = _rand_grid(5, 5, num_colours)
    out = enc([g])
    assert out.shape == (1, enc.cfg.emb_dim)


def test_masking_logic():
    enc, _ = default_pixel_vit()
    g = torch.full((4, 4), _PAD_VAL, dtype=torch.long)
    out = enc([g])
    # CLS token of an *all‑pad* grid should be near‑zero as only positional terms survive.
    assert torch.allclose(out.abs().mean(), torch.zeros(1), atol=1e-3)


def test_shapes_end_to_end():
    enc, synth = default_pixel_vit()
    grids = [_rand_grid(10, 10, 10) for _ in range(6)]
    roles = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    grid_vecs = enc(grids)
    task_vec = synth(grid_vecs, roles)
    assert task_vec.shape == (enc.cfg.emb_dim,)


# -----------------------------------------------------------------------------
# Smoke‑test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    enc, synth = default_pixel_vit(num_colours=12)

    # Six dummy (input, output) example grids with *different* sizes
    examples = [
        _rand_grid(10, 10, 12),  # input‑1
        _rand_grid(10, 10, 12),  # output‑1
        _rand_grid(5, 7, 12),  # input‑2
        _rand_grid(5, 7, 12),  # output‑2
        _rand_grid(13, 3, 12),  # input‑3
        _rand_grid(13, 3, 12),  # output‑3
    ]
    roles = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)

    grid_vecs = enc(examples)
    task_vec = synth(grid_vecs, roles)

    print("Grid CLS vectors:", grid_vecs.shape)  # (6,128)
    print("Task vector:", task_vec.shape)  # (128,)
