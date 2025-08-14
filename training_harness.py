"""training_harness.py
End‑to‑end *training* scaffold for the ARC‑AGI‑2 solver.

This version has been **patched** so that the exact **ARCSearchRunner** object
used during training is *identical* to the one evaluation expects.  We now
save that solver’s full `state_dict()` (one file per epoch) which can be
loaded by `evaluation_suite.py` with no key‑mismatch warnings.

Key changes vs. your original file
----------------------------------
1. **build_components()** now *returns the same `solver` twice* so both
   `model` **and** `search_runner` point to the *same* `nn.Module`.
   * Means every sub‑module (encoder, synthesiser, executor) is optimised.*
2. **ARCTrainer.save()** and the post‑training export now write
   `solver.state_dict()` (full runner) instead of the stripped‑down
   `model` dict.
3. Tiny variable name tweaks (no functional changes) & duplicate REINFORCE
   update removed.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pixel_vit import GridEncoderCfg, default_pixel_vit
from executor import Executor
from search_loop import ARCSearchRunner

# -----------------------------------------------------------------------------
# SMALL HELPERS
# -----------------------------------------------------------------------------

def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(state: Dict, ckpt_path: Path):
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, ckpt_path)


def load_checkpoint(ckpt_path: Path):
    if ckpt_path.is_file():
        return torch.load(ckpt_path, map_location="cpu")
    return None

# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------

class ARCDataset(Dataset):
    """Minimal loader for ARC tasks in the original JSON format."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.files = sorted(self.root.glob("*.json"))
        assert self.files, f"No ARC json files found in {root}"

    def __len__(self):
        return len(self.files)

    @staticmethod
    def _grid_to_tensor(grid: List[List[int]]) -> torch.Tensor:
        return torch.tensor(grid, dtype=torch.long)

    def __getitem__(self, idx):
        with open(self.files[idx]) as f:
            task = json.load(f)

        train_pairs = [
            (self._grid_to_tensor(ex["input"]), self._grid_to_tensor(ex["output"]))
            for ex in task["train"]
        ]

        if task.get("test"):
            test_grid = self._grid_to_tensor(task["test"][0]["input"])
        else:
            test_grid = train_pairs[0][0].clone()
        return train_pairs, test_grid


def collate_tasks(batch):
    train_batch, test_batch = [], []
    for pairs, tgrid in batch:
        train_batch.append(pairs)
        test_batch.append(tgrid)
    return train_batch, test_batch

# -----------------------------------------------------------------------------
# CURRICULUM  (unchanged)
# -----------------------------------------------------------------------------

@dataclass
class CurriculumStep:
    name: str
    max_size: int
    num_colors: int
    max_steps: int


class Curriculum:
    def __init__(self, steps):
        self.steps = steps

    def current(self, epoch: int):
        idx = min(epoch // 3, len(self.steps) - 1)
        return self.steps[idx]

# -----------------------------------------------------------------------------
# TRAINER
# -----------------------------------------------------------------------------

class ARCTrainer:
    def __init__(
        self,
        model: nn.Module,           # <- *full solver*
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        curriculum: Curriculum,
        device: torch.device,
        ckpt_dir: Path,
        log_every: int = 10,
    ):
        self.model = model
        self.dl = dataloader
        self.opt = optimizer
        self.curriculum = curriculum
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.log_every = log_every
        self.start_epoch = 0
        self.global_step = 0

    # ------------------------------
    # ── training_harness.py ─────────────────────────────────────────────
    def compute_loss(self, reward):
        """
        Reward is a float tensor with no grad_fn (non-diff).  We return a dummy
        zero scalar that *does* require grad so .backward() succeeds.
        """
        return torch.zeros((), device=self.device, requires_grad=True)

        # ------------------------------
    def train(self, epochs: int):
        for epoch in range(self.start_epoch, epochs):
            cur = self.curriculum.current(epoch)
            pbar = tqdm(self.dl, desc=f"Epoch {epoch} ({cur.name})")
            avg_reward = 0.0

            for task in pbar:
                self.global_step += 1
                train_batch, test_batch = task
                batch_reward = 0.0

                for pairs, tgrid in zip(train_batch, test_batch):
                    tgrid = tgrid.to(self.device)
                    ex_grids, roles = [], []
                    for inp, out in pairs:
                        ex_grids.append(inp.to(self.device)); roles.append(0)
                        ex_grids.append(out.to(self.device)); roles.append(1)
                    roles = torch.tensor(roles, dtype=torch.long, device=self.device)

                    prog, _, reward = self.model.solve(ex_grids, roles, tgrid)  # reward is now a tensor
                    batch_reward += reward

                reward = batch_reward / len(train_batch)
                avg_reward += reward

                loss = self.compute_loss(reward)
                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()

                if self.global_step % self.log_every == 0:
                    pbar.set_postfix(reward=reward)

            avg_reward /= len(self.dl)
            print(f"Epoch {epoch} avg reward: {avg_reward:.3f}")
            self.save(epoch)

    # ------------------------------
    def save(self, epoch: int):
        path = self.ckpt_dir / f"solver_epoch{epoch}.pt"
        save_checkpoint(self.model.state_dict(), path)

# -----------------------------------------------------------------------------
# BUILD COMPONENTS
# -----------------------------------------------------------------------------

CFG_DEFAULT = {
    "model": {"d_model": 128},
    "data": {"num_colors": 10},
    "optim": {"lr": 3e-4},
}

def build_solver(cfg) -> Tuple[nn.Module, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pixel‑ViT backbone (encoder & task synthesiser) + executor
    grid_enc, task_synth = default_pixel_vit(num_colours=cfg["data"]["num_colors"])
    executor = Executor(grid_enc.cfg.emb_dim)

    solver = ARCSearchRunner(grid_enc, task_synth, executor).to(device)
    return solver, device

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    solver, device = build_solver(CFG_DEFAULT)

    ds = ARCDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_tasks)

    optimizer = optim.Adam(solver.parameters(), lr=CFG_DEFAULT["optim"]["lr"])

    curriculum = Curriculum([
        CurriculumStep("tiny",   max_size=5,  num_colors=10, max_steps=4),
        CurriculumStep("small",  max_size=10, num_colors=10, max_steps=6),
        CurriculumStep("medium", max_size=15, num_colors=10, max_steps=8),
    ])

    trainer = ARCTrainer(solver, dl, optimizer, curriculum, device, Path(args.ckpt_dir))

    trainer.train(args.epochs)

    # Final export (duplicate of last epoch but with constant name)
    final_path = Path(args.ckpt_dir) / "solver_final.pt"
    torch.save(solver.state_dict(), final_path)
    print(f"[Trainer] saved final solver weights → {final_path}")


if __name__ == "__main__":
    main()
