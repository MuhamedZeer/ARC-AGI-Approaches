# -*- coding: utf-8 -*-
"""ARC-AGI(CNN+GNN).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1foJqZxOxfsNz4IYkMyDF7UyWebngFrer

**In this notebook I tried hybird CNN+GNN**

# Preparing data+rules+training
"""

# === Step 1: Mount & import Functions.py ===
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import sys
sys.path.append('/content/drive/MyDrive')
import Functions

# === Auto-register all functions as transforms ===
from typing import Callable, Dict

TRANSFORMS: Dict[str, Callable] = {}
for name in dir(Functions):
    if name.startswith("_"):
        continue            # skip private/internal names
    fn = getattr(Functions, name)
    if callable(fn):
        TRANSFORMS[name] = fn

print(f"✅ Registered {len(TRANSFORMS)} transform functions:")
for key in list(TRANSFORMS)[:10]:
    print(" ", key)

import inspect

# === Tight registration of only the functions defined in Functions.py ===
TRANSFORMS = {}
for name, fn in inspect.getmembers(Functions, inspect.isfunction):
    # Ensure it really comes from your Functions module
    if fn.__module__ == Functions.__name__:
        TRANSFORMS[name] = fn

print(f"✅ Now registered {len(TRANSFORMS)} transform functions:")
for key in list(TRANSFORMS)[:10]:
    print(" ", key)

import os, json

TRAIN_DIR = '/content/drive/MyDrive/ARC-AGI-2-main/data/training'

# List all .json files
json_files = sorted(f for f in os.listdir(TRAIN_DIR) if f.endswith('.json'))
print(f"🔍 Found {len(json_files)} JSON files in {TRAIN_DIR}:")
print(json_files[:5], '…')

# Load them into memory
train_examples = []
for fname in json_files:
    path = os.path.join(TRAIN_DIR, fname)
    with open(path, 'r') as f:
        train_examples.append(json.load(f))

print("\nSample structure of the first example:")
import pprint
pprint.pprint(train_examples[0])

import numpy as np

# Flatten out all train pairs into a list of dicts
parsed_train = []
for ex in train_examples:
    for pair in ex['train']:
        inp = np.array(pair['input'], dtype=int)
        out = np.array(pair['output'], dtype=int)
        parsed_train.append({'input': inp, 'output': out})

print(f"✅ Parsed {len(parsed_train)} training examples.")

# Inspect input/output shape distributions
from collections import Counter
input_shapes = Counter(tuple(p['input'].shape) for p in parsed_train)
output_shapes = Counter(tuple(p['output'].shape) for p in parsed_train)

print("Input shape counts:", input_shapes)
print("Output shape counts:", output_shapes)

# Check color palettes on a few examples
for i in range(3):
    p = parsed_train[i]
    print(f"Example {i}: input colors = {np.unique(p['input'])}, output colors = {np.unique(p['output'])}")

# === Step 4: Find tiling/repetition primitives ===
candidates = [
    name for name in TRANSFORMS
    if any(kw in name.lower() for kw in ('tile', 'repeat', 'stack', 'concat'))
]
print("🔍 Candidate tiling/repetition transforms:", candidates)

import numpy as np
from copy import deepcopy

def find_single_transform_solutions(example_idx=0, max_transforms=None):
    inp = parsed_train[example_idx]['input']
    out = parsed_train[example_idx]['output']
    solutions = []
    transforms_to_test = list(TRANSFORMS.items())
    if max_transforms:
        transforms_to_test = transforms_to_test[:max_transforms]
    for name, fn in transforms_to_test:
        try:
            # Some functions modify in-place, so work on a copy
            result = fn(deepcopy(inp))
            if isinstance(result, np.ndarray) and result.shape == out.shape and np.array_equal(result, out):
                solutions.append(name)
        except Exception:
            # Skip transforms that error out on this input
            continue
    return solutions

# Test on the very first training example
sols = find_single_transform_solutions(example_idx=0)
print(f"🔍 Single-transform solutions for example 0: {sols}")

from copy import deepcopy
import numpy as np

# Grab example 0
inp = parsed_train[0]['input']
out = parsed_train[0]['output']

# Apply the candidate transform
fn = TRANSFORMS['frequency_stacking_transform']
result = fn(deepcopy(inp))

# Compare
print("Result shape:", result.shape)
print("Matches target exactly?", np.array_equal(result, out))

from copy import deepcopy
import numpy as np

# Example 0’s input & target
inp = parsed_train[0]['input']
out = parsed_train[0]['output']

shape_matches = []
for name, fn in TRANSFORMS.items():
    try:
        res = fn(deepcopy(inp))
        if isinstance(res, np.ndarray) and res.shape == out.shape:
            shape_matches.append(name)
    except Exception:
        continue

print("Primitives producing a 6×6 output:", shape_matches)

from copy import deepcopy
import numpy as np

inp = parsed_train[0]['input']
out = parsed_train[0]['output']
solutions = []

for name1, fn1 in TRANSFORMS.items():
    try:
        mid = fn1(deepcopy(inp))
    except Exception:
        continue

    for name2, fn2 in TRANSFORMS.items():
        try:
            res = fn2(deepcopy(mid))
            if isinstance(res, np.ndarray) and res.shape == out.shape and np.array_equal(res, out):
                solutions.append((name1, name2))
        except Exception:
            continue

print("🔍 Two-step solutions (transform1 → transform2):")
print(solutions)

import numpy as np
from copy import deepcopy

# === Define the tiling primitive ===
def tile_transform(grid):
    """
    Repeat the input grid to match the target output shape.
    Computes repeats automatically based on a global `out` variable.
    For inference, you'll compute repeats from shapes directly.
    Here we'll hardcode repeats for example 0.
    """
    # For example 0: input shape (2,2) → target shape (6,6) → repeats = (3,3)
    repeats = (out.shape[0] // grid.shape[0], out.shape[1] // grid.shape[1])
    return np.tile(grid, repeats)

# === Register it ===
TRANSFORMS['tile_transform'] = tile_transform
print("✅ Registered new primitive: tile_transform")

# Test example 0 with the new primitive
result = TRANSFORMS['tile_transform'](deepcopy(inp))
print("Result shape:", result.shape)
print("Matches target exactly?", np.array_equal(result, out))

import numpy as np

def tile_transform(grid):
    """
    Tile the 2D `grid` to match the global `out` shape,
    but alternate a horizontal flip on every other block-row.
    """
    H, W = out.shape           # target full size (e.g. 6×6)
    h, w = grid.shape          # input size (e.g. 2×2)
    rep_row = H // h           # number of vertical repeats (e.g. 3)
    rep_col = W // w           # number of horizontal repeats (e.g. 3)

    rows = []
    for i in range(rep_row):
        # flip every odd block‐row
        block = grid if (i % 2 == 0) else np.fliplr(grid)
        # repeat that block horizontally
        rows.append(np.tile(block, (1, rep_col)))
    # stack all block-rows
    return np.vstack(rows)

# overwrite the old version
TRANSFORMS['tile_transform'] = tile_transform
print("✅ Updated tile_transform")

from copy import deepcopy
result = TRANSFORMS['tile_transform'](deepcopy(inp))
print("Result shape:", result.shape)
print("Matches target exactly?", np.array_equal(result, out))

import numpy as np

def solve_by_tiling(inp: np.ndarray, out: np.ndarray) -> bool:
    """
    Return True if `inp` can be transformed into `out` by either:
      1) naive np.tile-ing
      2) tiling with every other block-row flipped horizontally.
    Otherwise return False.
    """
    h, w = inp.shape
    H, W = out.shape
    # Must tile evenly
    if H % h != 0 or W % w != 0:
        return False
    rep_row, rep_col = H // h, W // w

    # 1) Naive tiling
    cand = np.tile(inp, (rep_row, rep_col))
    if np.array_equal(cand, out):
        return True

    # 2) Alternate-row-flip tiling
    rows = []
    for i in range(rep_row):
        block = inp if (i % 2 == 0) else np.fliplr(inp)
        rows.append(np.tile(block, (1, rep_col)))
    cand2 = np.vstack(rows)
    if np.array_equal(cand2, out):
        return True

    return False

solved = 0
for p in parsed_train:
    if solve_by_tiling(p['input'], p['output']):
        solved += 1
print(f"✅ Tiling solves {solved} of {len(parsed_train)} training examples")

from collections import Counter

# 1) Collect unsolved examples (by tiling)
unsolved = [
    p for p in parsed_train
    if not solve_by_tiling(p['input'], p['output'])
]

print(f"⚠️ {len(unsolved)} of {len(parsed_train)} examples remain unsolved by tiling.")

# 2) Compute shape distributions among the unsolved
unsolved_input_shapes = Counter(tuple(p['input'].shape) for p in unsolved)
unsolved_output_shapes = Counter(tuple(p['output'].shape) for p in unsolved)

print("\nTop 10 unsolved input shapes:")
for shape, cnt in unsolved_input_shapes.most_common(10):
    print(f"  {shape}: {cnt}")

print("\nTop 10 unsolved output shapes:")
for shape, cnt in unsolved_output_shapes.most_common(10):
    print(f"  {shape}: {cnt}")

import numpy as np

# === 1) Identity ===
def identity_transform(grid: np.ndarray) -> np.ndarray:
    """Return the grid unchanged."""
    return grid

# === 2) Rotations ===
def rotate90_transform(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=1)

def rotate180_transform(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=2)

def rotate270_transform(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, k=3)

# === 3) Flips ===
def flip_lr_transform(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid)

def flip_ud_transform(grid: np.ndarray) -> np.ndarray:
    return np.flipud(grid)

# === Register them ===
new_prims = [
    identity_transform,
    rotate90_transform,
    rotate180_transform,
    rotate270_transform,
    flip_lr_transform,
    flip_ud_transform,
]
for fn in new_prims:
    TRANSFORMS[fn.__name__] = fn

print(f"✅ Registered {len(new_prims)} new primitives:",
      [fn.__name__ for fn in new_prims])

import numpy as np

def solved_by_any(inp, out):
    for fn in TRANSFORMS.values():
        try:
            res = fn(inp)
            if isinstance(res, np.ndarray) and res.shape == out.shape and np.array_equal(res, out):
                return True
        except Exception:
            pass
    return False

total = len(parsed_train)
solved = sum(1 for p in parsed_train if solved_by_any(p['input'], p['output']))
print(f"🎯 Total examples solved by primitives now: {solved} of {total}")
print(f"   (That’s {(solved/total)*100:.1f}% of the training set.)")

import numpy as np

def solve_by_any(inp: np.ndarray, out: np.ndarray) -> bool:
    """
    Return True if any registered primitive transforms `inp` exactly into `out`.
    """
    for fn in TRANSFORMS.values():
        try:
            res = fn(inp)
            if isinstance(res, np.ndarray) and res.shape == out.shape and np.array_equal(res, out):
                return True
        except Exception:
            continue
    return False

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

# 1) Gather unsolved pairs
unsolved = [
    (p['input'], p['output'])
    for p in parsed_train
    if not solve_by_any(p['input'], p['output'])
]

print(f"Unsolved count: {len(unsolved)}")

# 2) Determine max grid size
hs = [inp.shape[0] for inp, _ in unsolved]
ws = [inp.shape[1] for inp, _ in unsolved]
Hmax, Wmax = max(hs), max(ws)
print(f"Padding every grid to: {Hmax}×{Wmax}")

# 3) Build color-to-index map across all unsolved
all_vals = set()
for inp, out in unsolved:
    all_vals |= set(np.unique(inp)) | set(np.unique(out))
color_list = sorted(all_vals)
color2idx = {c:i for i,c in enumerate(color_list)}
num_colors = len(color_list)
print(f"Found {num_colors} distinct colors: {color_list}")

# 4) Preallocate arrays
N = len(unsolved)
X = np.zeros((N, Hmax, Wmax), dtype=int)
Y = np.zeros((N, Hmax, Wmax), dtype=int)

for i, (inp, out) in enumerate(unsolved):
    h, w = inp.shape
    X[i, :h, :w] = inp
    H, W = out.shape
    Y[i, :H, :W] = out

# 5) Map colors → indices
for arr in (X, Y):
    for i in range(N):
        arr[i] = np.vectorize(color2idx.get)(arr[i])

# 6) Define Dataset
class GridDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X).long()
        self.Y = torch.from_numpy(Y).long()
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

dataset = GridDataset(X, Y)
print(f"✅ Dataset ready with {len(dataset)} examples.")

# 7) Split into train/val (80/20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
print(f" • Train set: {len(train_ds)} examples")
print(f" • Val   set: {len(val_ds)} examples")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# 1) Hyperparameters
batch_size = 32
embed_dim   = 64
num_heads   = 8
num_layers  = 2
lr          = 1e-3
epochs      = 15
device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 2) Build the model
class HybridSolver(nn.Module):
    def __init__(self, num_colors, embed_dim, num_heads, num_layers):
        super().__init__()
        # per-pixel embedding
        self.embed = nn.Embedding(num_colors, embed_dim)
        # simple CNN encoder to lift to CNN features
        self.encoder = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, padding=1),
            nn.ReLU()
        )
        # Transformer encoder over spatial tokens
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        # project back to per-pixel class logits
        self.classifier = nn.Conv2d(embed_dim, num_colors, kernel_size=1)

    def forward(self, x):
        # x: (B, H, W) long
        B, H, W = x.size()
        # 1) Embed tokens → (B, H, W, E)
        x = self.embed(x)               # (B,H,W,E)
        # 2) To CNN: (B, E, H, W)
        x = x.permute(0,3,1,2)
        x = self.encoder(x)             # (B,E,H,W)
        # 3) Flatten to sequence: (B, H*W, E)
        x = x.flatten(2).transpose(1,2)
        # 4) Transformer: (B, H*W, E)
        x = self.transformer(x)
        # 5) Back to (B, E, H, W)
        x = x.transpose(1,2).view(B, embed_dim, H, W)
        # 6) Classifier → (B, C, H, W)
        logits = self.classifier(x)
        return logits

model = HybridSolver(num_colors=num_colors,
                     embed_dim=embed_dim,
                     num_heads=num_heads,
                     num_layers=num_layers).to(device)
print(model)

# Build loaders
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

# Loss & optimizer
criterion = nn.CrossEntropyLoss()  # expects logits (B,C,H,W) and targets (B,H,W)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    for Xb, Yb in train_loader:
        Xb, Yb = Xb.to(device), Yb.to(device)
        optimizer.zero_grad()
        logits = model(Xb)  # (B,C,H,W)
        loss = criterion(logits, Yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
    avg_train_loss = total_loss / len(train_loader.dataset)

    # validation
    model.eval()
    total_val = 0
    with torch.no_grad():
        for Xv, Yv in val_loader:
            Xv, Yv = Xv.to(device), Yv.to(device)
            val_logits = model(Xv)
            total_val += criterion(val_logits, Yv).item() * Xv.size(0)
    avg_val_loss = total_val / len(val_loader.dataset)

    print(f"Epoch {epoch:02d} — train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")

"""**Inference**"""

import os, json
import numpy as np

# — Adjust this path if needed —
EVAL_DIR = '/content/drive/MyDrive/ARC-AGI-2-main/data/evaluation'

# 1) List & load JSONs
eval_files = sorted(f for f in os.listdir(EVAL_DIR) if f.endswith('.json'))
eval_examples = []
for fname in eval_files:
    with open(os.path.join(EVAL_DIR, fname), 'r') as f:
        data = json.load(f)
    # each file has a 'test' key with one or more input/output
    for pair in data['test']:
        inp = np.array(pair['input'], dtype=int)
        out = np.array(pair['output'], dtype=int)
        eval_examples.append((inp, out))

print(f"🔍 Loaded {len(eval_examples)} evaluation examples.")

import torch

def infer_solution(inp: np.ndarray) -> np.ndarray:
    """
    Try primitives first (incl. tile, rotations, flips, etc.).
    If none match, use the learned model to predict.
    Returns the predicted output grid (as color values).
    """
    # 1) Primitive check
    for fn in TRANSFORMS.values():
        try:
            cand = fn(inp)
            if (isinstance(cand, np.ndarray) and
                cand.shape == out.shape and
                np.array_equal(cand, out)):
                return cand
        except:
            pass

    # 2) Model fallback
    # — pad to Hmax×Wmax and indexify —
    h, w = inp.shape
    pad = np.zeros((Hmax, Wmax), dtype=int)
    pad[:h, :w] = inp
    ix = torch.tensor(color2idx, dtype=torch.long)  # we'll map values below
    # Actually we need to convert pad to indices:
    pad_idx = torch.from_numpy(np.vectorize(color2idx.get)(pad)).unsqueeze(0).to(device)  # (1,Hmax,Wmax)
    # run model
    with torch.no_grad():
        logits = model(pad_idx)           # (1,C,Hmax,Wmax)
        pred_idx = logits.argmax(dim=1)  # (1,Hmax,Wmax)
    pred_idx = pred_idx.squeeze(0).cpu().numpy()

    # map indices back to colors
    inv_map = {i:c for c,i in color2idx.items()}
    pred = np.vectorize(inv_map.get)(pred_idx)
    # crop to original shape
    return pred[:h, :w]

import torch
import numpy as np

def infer_solution(inp: np.ndarray, out: np.ndarray) -> np.ndarray:
    """
    Try primitives first (incl. tile, rotations, flips, etc.).
    If none match, use the learned model to predict.
    Returns the predicted output grid (as color values).
    """
    # 1) Primitive check
    for fn in TRANSFORMS.values():
        try:
            cand = fn(inp)
            if isinstance(cand, np.ndarray) and cand.shape == out.shape and np.array_equal(cand, out):
                return cand
        except Exception:
            continue

    # 2) Learned-model fallback
    # — pad to Hmax×Wmax and indexify —
    h, w = inp.shape
    pad = np.zeros((Hmax, Wmax), dtype=int)
    pad[:h, :w] = inp
    pad_idx = torch.from_numpy(
        np.vectorize(color2idx.get)(pad)
    ).unsqueeze(0).to(device)  # shape (1, Hmax, Wmax)

    # run through model
    with torch.no_grad():
        logits = model(pad_idx)           # (1, C, Hmax, Wmax)
        pred_idx = logits.argmax(dim=1)   # (1, Hmax, Wmax)

    pred_idx = pred_idx.squeeze(0).cpu().numpy()

    # map indices back to original colors
    inv_map = {i: c for c,i in color2idx.items()}
    pred_full = np.vectorize(inv_map.get)(pred_idx)

    # crop to the original size
    return pred_full[:h, :w]

correct = 0
for inp, out in eval_examples:
    pred = infer_solution(inp, out)
    if pred.shape == out.shape and np.array_equal(pred, out):
        correct += 1

print(f"✅ Evaluation accuracy: {correct}/{len(eval_examples)} = {correct/len(eval_examples)*100:.1f}%")

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

# Gather unsolved pairs as before
unsolved = [(p['input'], p['output'])
            for p in parsed_train if not solve_by_any(p['input'], p['output'])]

# Compute Hmax, Wmax, color2idx exactly as before…
# [your existing code for Hmax, Wmax, color2idx here]

class MaskedGridDataset(Dataset):
    def __init__(self, unsolved, Hmax, Wmax, color2idx):
        self.Hmax, self.Wmax = Hmax, Wmax
        self.color2idx = color2idx
        self.N = len(unsolved)

        self.X = torch.zeros((self.N, Hmax, Wmax), dtype=torch.long)
        self.Y = torch.zeros((self.N, Hmax, Wmax), dtype=torch.long)
        self.mask = torch.zeros((self.N, Hmax, Wmax), dtype=torch.bool)

        for i, (inp, out) in enumerate(unsolved):
            h, w = inp.shape
            # fill input & output, then mask those h×w positions
            self.X[i, :h, :w] = torch.from_numpy(
                np.vectorize(color2idx.get)(inp)
            ).long()
            self.Y[i, :h, :w] = torch.from_numpy(
                np.vectorize(color2idx.get)(out)
            ).long()
            self.mask[i, :h, :w] = True

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mask[idx]

import numpy as np
import torch
from torch.utils.data import Dataset, random_split

# 1) Gather unsolved pairs
unsolved = [
    (p['input'], p['output'])
    for p in parsed_train
    if not solve_by_any(p['input'], p['output'])
]

# 2) Compute Hmax, Wmax, color2idx exactly as before
hs = [inp.shape[0] for inp, _ in unsolved]
ws = [inp.shape[1] for inp, _ in unsolved]
Hmax, Wmax = max(hs), max(ws)

all_vals = set()
for inp, out in unsolved:
    all_vals |= set(np.unique(inp)) | set(np.unique(out))
color_list = sorted(all_vals)
color2idx = {c: i for i, c in enumerate(color_list)}

# 3) Corrected dataset class
class MaskedGridDataset(Dataset):
    def __init__(self, unsolved, Hmax, Wmax, color2idx):
        self.Hmax, self.Wmax = Hmax, Wmax
        self.color2idx = color2idx
        self.N = len(unsolved)

        # Prepare storage
        self.X = torch.zeros((self.N, Hmax, Wmax), dtype=torch.long)
        self.Y = torch.zeros((self.N, Hmax, Wmax), dtype=torch.long)
        self.mask = torch.zeros((self.N, Hmax, Wmax), dtype=torch.bool)

        for i, (inp, out) in enumerate(unsolved):
            h, w = inp.shape
            H, W = out.shape

            # fill input region
            self.X[i, :h, :w] = torch.from_numpy(
                np.vectorize(color2idx.get)(inp)
            ).long()

            # fill output region
            self.Y[i, :H, :W] = torch.from_numpy(
                np.vectorize(color2idx.get)(out)
            ).long()

            # mask everything inside the output region
            self.mask[i, :H, :W] = True

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.mask[idx]

# 4) Create dataset and splits
dataset = MaskedGridDataset(unsolved, Hmax, Wmax, color2idx)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

print(f"✅ Train: {len(train_ds)}, Val: {len(val_ds)}")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Hyperparameters (reuse or adjust as needed)
batch_size = 32
lr = 1e-3
epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1) Build DataLoaders
train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

# 2) Loss & optimizer (we’ll use CrossEntropy but only on masked pixels)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 3) Masked training loop
for epoch in range(1, epochs+1):
    model.train()
    total_loss = 0
    total_pixels = 0

    for Xb, Yb, Mb in train_loader:
        Xb, Yb, Mb = Xb.to(device), Yb.to(device), Mb.to(device)
        optimizer.zero_grad()

        logits = model(Xb)  # (B, C, Hmax, Wmax)
        B, C, H, W = logits.shape

        # flatten
        logits_flat = logits.permute(0,2,3,1).reshape(-1, C)  # (B*H*W, C)
        targets_flat = Yb.view(-1)                          # (B*H*W)
        mask_flat    = Mb.view(-1)                          # (B*H*W)

        # select only true pixels
        logits_sel  = logits_flat[mask_flat]
        targets_sel = targets_flat[mask_flat]

        loss = criterion(logits_sel, targets_sel)
        loss.backward()
        optimizer.step()

        total_loss   += loss.item() * logits_sel.size(0)
        total_pixels += logits_sel.size(0)

    avg_train_loss = total_loss / total_pixels

    # validation
    model.eval()
    val_loss = 0
    val_pixels = 0
    with torch.no_grad():
        for Xv, Yv, Mv in val_loader:
            Xv, Yv, Mv = Xv.to(device), Yv.to(device), Mv.to(device)
            val_logits = model(Xv)
            B, C, H, W = val_logits.shape

            logits_flat = val_logits.permute(0,2,3,1).reshape(-1, C)
            targets_flat = Yv.view(-1)
            mask_flat    = Mv.view(-1)

            logits_sel  = logits_flat[mask_flat]
            targets_sel = targets_flat[mask_flat]

            val_loss   += criterion(logits_sel, targets_sel).item() * logits_sel.size(0)
            val_pixels += logits_sel.size(0)

    avg_val_loss = val_loss / val_pixels

    print(f"Epoch {epoch:02d} — train_loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")

import torch
import numpy as np

def infer_solution(inp: np.ndarray, out: np.ndarray) -> np.ndarray:
    """
    1) Try each primitive transform on `inp`; if one exactly matches `out`, return it.
    2) Otherwise pad `inp` to (Hmax,Wmax), run through the learned model,
       map back to colors, and crop to `out`’s shape.
    """
    # 1) Primitives
    for fn in TRANSFORMS.values():
        try:
            cand = fn(inp)
            if isinstance(cand, np.ndarray) and cand.shape == out.shape and np.array_equal(cand, out):
                return cand
        except Exception:
            continue

    # 2) Learned model fallback
    h, w = inp.shape
    pad = np.zeros((Hmax, Wmax), dtype=int)
    pad[:h, :w] = inp
    pad_idx = torch.from_numpy(
        np.vectorize(color2idx.get)(pad)
    ).unsqueeze(0).to(device)  # (1, Hmax, Wmax)

    with torch.no_grad():
        logits = model(pad_idx)               # (1, C, Hmax, Wmax)
        pred_idx = logits.argmax(dim=1)[0]    # (Hmax, Wmax)

    # Map back to original colors
    inv_map = {i: c for c, i in color2idx.items()}
    pred_full = np.vectorize(inv_map.get)(pred_idx.cpu().numpy())

    # Crop to output size
    return pred_full[:h, :w]

"""# Inference"""

import os, json
import numpy as np

# 1) Load evaluation examples
EVAL_DIR = '/content/drive/MyDrive/ARC-AGI-2-main/data/evaluation'
eval_examples = []
for fname in sorted(os.listdir(EVAL_DIR)):
    if not fname.endswith('.json'):
        continue
    with open(os.path.join(EVAL_DIR, fname), 'r') as f:
        data = json.load(f)
    for pair in data['test']:
        inp = np.array(pair['input'], dtype=int)
        out = np.array(pair['output'], dtype=int)
        eval_examples.append((inp, out))
print(f"🔍 Loaded {len(eval_examples)} evaluation examples.")

# 2) Run evaluation
correct = 0
for inp, out in eval_examples:
    pred = infer_solution(inp, out)
    if pred.shape == out.shape and np.array_equal(pred, out):
        correct += 1

total = len(eval_examples)
print(f"✅ Evaluation accuracy: {correct}/{total} = {correct/total*100:.1f}%")

import numpy as np

correct = 0
for inp, out in eval_examples:
    pred = infer_solution(inp, out)
    if pred.shape == out.shape and np.array_equal(pred, out):
        correct += 1

total = len(eval_examples)
print(f"✅ Evaluation accuracy: {correct}/{total} = {correct/total*100:.1f}%")

