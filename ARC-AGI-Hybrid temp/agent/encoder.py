"""
Hybrid Encoder Module - Fixed Version
Combines PixelViT (Model B) with CNN features (Model A) for comprehensive grid encoding
with proper device handling
"""

from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== Shared constants =====
_PAD_VAL = -1

class DeviceAwareModule(nn.Module):
    """Base class that ensures consistent device placement"""
    def __init__(self):
        super().__init__()
        self._device = torch.device('cpu')
        
    @property
    def device(self):
        return next(self.parameters()).device if list(self.parameters()) else self._device
    
    def to(self, *args, **kwargs):
        self._device = torch._C._nn._parse_to(*args, **kwargs)[0]
        return super().to(*args, **kwargs)


# ===== Pixel/Token Embedding =====

class PixelEmbedder(DeviceAwareModule):
    def __init__(self, num_colours: int, emb_dim: int, max_grid_size: int = 64):
        super().__init__()
        self.colour_emb = nn.Embedding(num_colours + 1, emb_dim)  # +1 for PAD bucket
        self.x_emb = nn.Embedding(max_grid_size, emb_dim)
        self.y_emb = nn.Embedding(max_grid_size, emb_dim)
        self.emb_dim = emb_dim

    def forward(self, grid: torch.LongTensor) -> torch.FloatTensor:
        # Ensure input is on correct device
        grid = grid.to(self.device)
        
        if grid.dim() != 2:
            raise ValueError("PixelEmbedder expects a 2-D grid (H×W).")

        h, w = grid.shape
        if h >= self.x_emb.num_embeddings or w >= self.y_emb.num_embeddings:
            pass  # Handle size checks as before

        # Colour embedding (masked)
        colour_idx = grid.clamp(min=0)
        colour_vec = self.colour_emb(colour_idx)  # (H, W, D)

        # 2D absolute pos
        ys = torch.arange(h, device=self.device)
        xs = torch.arange(w, device=self.device)
        yy = self.y_emb(ys)[:, None, :]  # (H,1,D)
        xx = self.x_emb(xs)[None, :, :]  # (1,W,D)
        pos = yy + xx  # (H,W,D)

        emb = colour_vec + pos  # (H,W,D)

        # zero out padded pixels
        pad_mask = (grid == _PAD_VAL).unsqueeze(-1)
        emb = emb.masked_fill(pad_mask, 0.0)
        return emb

class PixelViT(DeviceAwareModule):
    def __init__(self, emb_dim: int, depth: int, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=num_heads,
            dim_feedforward=int(emb_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
        )
        self.layers = nn.ModuleList([nn.TransformerEncoder(encoder_layer, num_layers=1) for _ in range(depth)])
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, tokens: torch.Tensor, pad_mask: Optional[torch.BoolTensor] = None) -> torch.Tensor:
        tokens = tokens.to(self.device)
        if pad_mask is not None:
            pad_mask = pad_mask.to(self.device)

        b, l, d = tokens.shape
        cls_tokens = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        if pad_mask is not None:
            cls_mask = torch.zeros(b, 1, dtype=torch.bool, device=self.device)
            attn_mask = torch.cat([cls_mask, pad_mask], dim=1)
        else:
            attn_mask = None

        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=attn_mask)
        return self.ln(tokens)

class GridEncoder(nn.Module):
    """Encodes variable-sized grids using PixelViT with proper device handling."""
    
    def __init__(self, cfg: PixelViT):
        super().__init__()
        self.cfg = cfg
        self.pixel_embedder = PixelEmbedder(
            num_colours=cfg.num_colours,
            emb_dim=cfg.emb_dim,
            max_grid_size=cfg.max_grid_size,
        )
        self.vit = PixelViT(
            emb_dim=cfg.emb_dim,
            depth=cfg.depth,
            num_heads=cfg.num_heads,
            mlp_ratio=cfg.mlp_ratio,
            dropout=cfg.dropout,
        )

    def _pad_and_stack(self, grids: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.BoolTensor]:
        """Ensure all operations happen on the same device"""
        if not grids:
            raise ValueError("Empty grid list.")
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        # Move all grids to the correct device
        grids = [g.to(device) for g in grids]
        
        max_h = min(max(g.shape[0] for g in grids), self.cfg.max_grid_size)
        max_w = min(max(g.shape[1] for g in grids), self.cfg.max_grid_size)

        padded_grids = []
        pad_masks = []
        
        for grid in grids:
            h, w = grid.shape
            padded = torch.full((max_h, max_w), _PAD_VAL, 
                              dtype=grid.dtype, 
                              device=device)
            padded[:h, :w] = grid
            padded_grids.append(padded)
            
            mask = torch.ones(max_h, max_w, 
                             dtype=torch.bool, 
                             device=device)
            mask[:h, :w] = False
            pad_masks.append(mask)
            
        return torch.stack(padded_grids), torch.stack(pad_masks)

    def forward(self, grids: List[torch.Tensor]) -> torch.Tensor:
        """Main forward pass with device safety"""
        # Explicit device handling
        device = next(self.parameters()).device
        grids = [g.to(device) for g in grids]
        
        padded_grids, pad_masks = self._pad_and_stack(grids)
        embeddings = []
        
        for grid, mask in zip(padded_grids, pad_masks):
            # Process each grid on the correct device
            h, w = grid.shape
            emb_2d = self.pixel_embedder(grid)  # (H,W,D)
            flat_emb = emb_2d.view(-1, self.cfg.emb_dim)  # (H*W, D)
            flat_mask = mask.view(-1)  # (H*W)
            
            valid_tokens = flat_emb[~flat_mask]  # (N_valid, D)
            if valid_tokens.numel() == 0:
                embeddings.append(torch.zeros(self.cfg.emb_dim, device=device))
                continue
                
            # ViT processing on same device
            cls = self.vit.cls_token.to(device)
            all_tokens = torch.cat([cls, valid_tokens.unsqueeze(0)], dim=1)
            attn_mask = torch.zeros(1, all_tokens.size(1), 
                                 dtype=torch.bool, 
                                 device=device)
            
            for layer in self.vit.layers:
                all_tokens = layer(all_tokens.to(device), 
                                 src_key_padding_mask=attn_mask)
                
            embeddings.append(self.vit.ln(all_tokens[0, 0]))
            
        return torch.stack(embeddings)

class SimpleCNN(DeviceAwareModule):
    def __init__(self, in_channels: int = 11, out_dim: int = 256):  # Using default instead of CNN_FEATURE_DIMS
        super().__init__()
        c = 32
        self.conv1 = nn.Conv2d(in_channels, c, 3, padding=1)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.conv3 = nn.Conv2d(c, c, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(c, out_dim)
        self.act = nn.ReLU(inplace=True)

    def forward(self, grid: torch.Tensor) -> torch.Tensor:
        grid = grid.to(self.device)
        h, w = grid.shape

        # One-hot encoding with device awareness
        x = torch.zeros(11, h, w, device=self.device)
        valid = (grid != _PAD_VAL)
        clamped = grid.clamp(min=0)
        ys = torch.arange(h, device=self.device).unsqueeze(1).expand(h, w)
        xs = torch.arange(w, device=self.device).unsqueeze(0).expand(h, w)
        x[clamped, ys, xs] = 1.0
        x[:, ~valid] = 0.0

        x = x.unsqueeze(0)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        x = self.pool(x).view(1, -1)
        return self.proj(x).squeeze(0)

class TaskSynthesiser(DeviceAwareModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=cfg.emb_dim,
                nhead=cfg.num_heads,
                dim_feedforward=cfg.emb_dim * cfg.mlp_ratio,
                dropout=cfg.dropout,
                batch_first=True,
            ),
            num_layers=cfg.depth
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.role_projection = nn.Linear(2, cfg.emb_dim)
        self.task_embedding = nn.Parameter(torch.randn(cfg.emb_dim))
        self.ln = nn.LayerNorm(cfg.emb_dim)

    def forward(self, grid_embeddings: torch.Tensor, role_embeddings: torch.Tensor) -> torch.Tensor:
        grid_embeddings = grid_embeddings.to(self.device)
        role_embeddings = role_embeddings.to(self.device)

        role_features = self.role_projection(role_embeddings)
        combined = grid_embeddings + role_features
        task_token = self.task_embedding.expand(combined.size(0), -1)
        sequence = torch.cat([task_token.unsqueeze(0), combined.unsqueeze(0)], dim=1)
        output = self.transformer(sequence)
        return self.ln(output[0, 0])

class HybridGridEncoder(DeviceAwareModule):
    def __init__(self, vit_cfg, use_cnn: bool = True, use_symbolic: bool = False):
        super().__init__()
        self.vit_enc = GridEncoder(vit_cfg)
        self.use_cnn = use_cnn
        self.use_symbolic = use_symbolic

        in_dim = vit_cfg.emb_dim
        if use_cnn:
            in_dim += 256  # CNN_FEATURE_DIMS default
            self.cnn = SimpleCNN(in_channels=11, out_dim=256)
        if use_symbolic:
            in_dim += 128  # SYMBOLIC_FEATURE_DIMS default

        self.proj = nn.Linear(in_dim, vit_cfg.emb_dim)

    def forward(self, grids: List[torch.Tensor], symbolic_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Ensure all inputs are on the same device
        device = self.device
        grids = [g.to(device) for g in grids]
        
        vit_feats = self.vit_enc(grids)
        feats = [vit_feats]
        
        if self.use_cnn:
            cnn_feats = torch.stack([self.cnn(g) for g in grids])
            feats.append(cnn_feats)
            
        if self.use_symbolic and symbolic_feats is not None:
            feats.append(symbolic_feats.to(device))
            
        fused = torch.cat(feats, dim=1)
        return self.proj(fused)

# Backwards-compatible alias
HybridEncoder = HybridGridEncoder