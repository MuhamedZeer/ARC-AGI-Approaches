"""training_harness.py
End‑to‑end *training* scaffold for the ARC‑AGI‑2 solver.

It couples:
    • GridEncoder + TaskSynthesiser  (pixel_vit)
    • Executor                        (executor)
    • BeamSearchLoop                  (search_loop)
implements a curriculum‑driven RL loop with checkpointing, logging and
resume support.  The file stays self‑contained (no external lib except
PyTorch + tqdm + ruamel.yaml for config).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pixel_vit import GridEncoder, TaskSynthesiser, GridEncoderCfg
from executor  import Executor
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
    """Minimal loader for ARC tasks in the original JSON format.

    Each item returns: (examples: List[Tuple[in_grid, out_grid]])
    where `in_grid` / `out_grid` are `torch.LongTensor` of shape (H, W).
    """

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
        with open(self.files[idx], "r") as f:
            task = json.load(f)
        pairs = []
        for ex in task["train"]:
            inp = self._grid_to_tensor(ex["input"])
            out = self._grid_to_tensor(ex["output"])
            pairs.append((inp, out))
        return pairs  # List[(in, out)] length == 3 for ARC


def collate_tasks(batch):
    """Keeps original variable sizes; search runner handles padding."""
    return batch  # DataLoader will give List[task] where task is List[pairs]


# -----------------------------------------------------------------------------
# CURRICULUM SCHEDULER
# -----------------------------------------------------------------------------

@dataclass
class CurriculumStep:
    name: str
    max_size: int  # maximum H/W allowed
    num_colors: int
    max_steps: int  # max program length


class Curriculum:
    """Simple size‑based curriculum that grows complexity every N epochs."""

    def __init__(self, steps: List[CurriculumStep]):
        self.steps = steps

    def current(self, epoch: int) -> CurriculumStep:
        idx = min(epoch // 3, len(self.steps) - 1)  # change every 3 epochs
        return self.steps[idx]


# -----------------------------------------------------------------------------
# TRAINER
# -----------------------------------------------------------------------------

class ARCTrainer:
    def __init__(
        self,
        model: nn.Module,
        search_runner,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        curriculum: Curriculum,
        device: torch.device,
        ckpt_dir: Path,
        log_every: int = 10,
    ):
        self.model = model
        self.search_runner = search_runner
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.curriculum = curriculum
        self.device = device
        self.ckpt_dir = ckpt_dir
        self.log_every = log_every
        self.start_epoch = 0
        self.global_step = 0

    # ------------------------------------------------------------------
    # LOSS  (negative reward)
    # ------------------------------------------------------------------

    def compute_loss(self, reward: float) -> torch.Tensor:
        return torch.tensor(-reward, requires_grad=True, device=self.device)

    # ------------------------------------------------------------------
    # TRAIN LOOP
    # ------------------------------------------------------------------

    # ── training_harness.py ───────────────────────────────────────────────
    def train(self, num_epochs: int):
        for epoch in range(self.start_epoch, num_epochs):
            cur = self.curriculum.current(epoch)
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch} ({cur.name})")
            avg_reward = 0.0

            # ---------------------------------- inside train() -----------------------------
            for i, task in enumerate(pbar):
                self.global_step += 1

                # ------------------------------------------------------------------
                # Unpack the batch --------------------------------------------------
                # collate_tasks returns TWO lists of length == batch_size:
                #   • train_batch : list[list[tuple[input, output]]]
                #   • test_batch  : list[LongTensor]   (one test-input grid per task)
                # ------------------------------------------------------------------
                train_batch, test_batch = task  # both are Python lists
                batch_size = len(train_batch)

                batch_reward = 0.0

                for train_pairs, test_grid in zip(train_batch, test_batch):
                    # Move test grid to device
                    test_grid = test_grid.to(self.device)

                    # ---- build ex_grids + roles tensor ---------------------------------
                    ex_grids = []
                    roles = []
                    for inp, out in train_pairs:
                        ex_grids.append(inp.to(self.device));
                        roles.append(0)  # input
                        ex_grids.append(out.to(self.device));
                        roles.append(1)  # output
                    roles = torch.tensor(roles, dtype=torch.long, device=self.device)

                    # ---- call the new solver API --------------------------------------
                    reward, _ = self.search_runner.solve(ex_grids, roles, test_grid)
                    batch_reward += reward

                # Average reward over the tasks in this batch
                reward = batch_reward / batch_size
                avg_reward += reward

                # ---------------- REINFORCE update (unchanged) ----------------------
                loss = self.compute_loss(reward)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                if self.global_step % self.log_every == 0:
                    pbar.set_postfix(reward=reward)
                # -------------------------------------------------------------------------------

                # REINFORCE update (unchanged)
                loss = self.compute_loss(reward)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                if self.global_step % self.log_every == 0:
                    pbar.set_postfix(reward=reward)

            avg_reward /= len(self.dataloader)
            print(f"Epoch {epoch} avg reward: {avg_reward:.3f}")
            self.save(epoch)

    # ──────────────────────────────────────────────────────────────────────

    # ------------------------------------------------------------------
    # CHECKPOINTING
    # ------------------------------------------------------------------

    def save(self, epoch: int):
        state = {
            "epoch": epoch + 1,
            "global_step": self.global_step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }
        save_checkpoint(state, self.ckpt_dir / f"ckpt_{epoch}.pt")

    def load(self, ckpt_path: Path):
        state = load_checkpoint(ckpt_path)
        if state is None:
            print("No checkpoint found – starting fresh.")
            return
        self.start_epoch = state["epoch"]
        self.global_step = state["global_step"]
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        print(f"Resumed from {ckpt_path} (epoch {self.start_epoch})")


# -----------------------------------------------------------------------------
# MAIN ENTRY
# -----------------------------------------------------------------------------

def build_components(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dim = cfg["model"]["d_model"]

    # GridEncoder and TaskSynthesiser expect dim as *first positional* arg
    grid_cfg = GridEncoderCfg(
        emb_dim=dim,  # <- same “d_model” as before
        num_colours=cfg["data"]["num_colors"],  # <- British spelling matches the dataclass
        # depth, num_heads, mlp_ratio, dropout, max_grid_size all keep their default values
    )
    encoder = GridEncoder(grid_cfg).to(device)
    synthesiser = TaskSynthesiser(dim).to(device)
    # -- OR, if you want to be explicit --
    # synthesiser = TaskSynthesiser(dim, depth=2).to(device)

    model = nn.ModuleDict({
        "encoder": encoder,
        "synth": synthesiser,
    })

    executor = Executor(d_model=dim).to(device)  # max_steps keeps its default (10)

    solver = ARCSearchRunner(
        encoder,
        synthesiser,
        executor,
        beam_width=cfg.get("search", {}).get("beam_width", 10),
        max_depth=cfg.get("search", {}).get("max_depth", 8),
    ).to(device)

    return model, solver, device




def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt_dir", default="checkpoints")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


CFG_DEFAULT = {
    "model": {
        "d_model": 128,
    },
    "data": {
        "num_colors": 10,
    },
}


def main():
    args = parse_args()
    set_seed(args.seed)

    model, search_runner, device = build_components(CFG_DEFAULT)

    ds = ARCDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_tasks)

    optim_ = optim.Adam(model.parameters(), lr=3e-4)

    curriculum = Curriculum([
        CurriculumStep("tiny", max_size=5, num_colors=10, max_steps=4),
        CurriculumStep("small", max_size=10, num_colors=10, max_steps=6),
        CurriculumStep("medium", max_size=15, num_colors=10, max_steps=8),
    ])

    trainer = ARCTrainer(model, search_runner, dl, optim_, curriculum, device, Path(args.ckpt_dir))

    # Auto‑resume if a checkpoint exists
    latest_ckpt = sorted(Path(args.ckpt_dir).glob("ckpt_*.pt"))
    if latest_ckpt:
        trainer.load(latest_ckpt[-1])

    trainer.train(args.epochs)


# -----------------------------------------------------------------------------
# UNIT TESTS
# -----------------------------------------------------------------------------

# To keep the file self‑contained, tests create a *tiny* synthetic dataset
# with random grids.

def _fake_arc_task(size=(3, 3), num_colors=10):
    h, w = size
    grid = torch.randint(0, num_colors, (h, w)).tolist()
    ex = {"input": grid, "output": grid}
    return {"train": [ex, ex, ex]}


def _make_synthetic_dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("arc_fake")
    for i in range(5):
        task = _fake_arc_task()
        with open(root / f"{i}.json", "w") as f:
            json.dump(task, f)
    return str(root)


def test_dataset_loading(tmp_path_factory):
    root = _make_synthetic_dataset(tmp_path_factory)
    ds = ARCDataset(root)
    assert len(ds) == 5
    pairs = ds[0]
    assert len(pairs) == 3 and pairs[0][0].shape == pairs[0][1].shape


if __name__ == "__main__":
    # Smoke test: train for 1 epoch on synthetic data
    import tempfile

    tmpdir = tempfile.mkdtemp()
    synthetic_root = Path(tmpdir) / "data"
    synthetic_root.mkdir()
    # Create 3 synthetic tasks
    for i in range(3):
        task = _fake_arc_task(size=(5, 5))
        with open(synthetic_root / f"{i}.json", "w") as f:
            json.dump(task, f)

    class Args:
        data_root = str(synthetic_root)
        ckpt_dir = str(Path(tmpdir) / "ckpt")
        epochs = 1
        batch_size = 2
        seed = 0

    args = Args()
    CFG_DEFAULT["data"]["num_colors"] = 10

    set_seed(args.seed)
    model, runner, device = build_components(CFG_DEFAULT)
    ds = ARCDataset(args.data_root)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_tasks)
    optim_ = optim.Adam(model.parameters(), lr=1e-3)
    curriculum = Curriculum([
        CurriculumStep("tiny", max_size=5, num_colors=10, max_steps=4),
    ])
    trainer = ARCTrainer(model, runner, dl, optim_, curriculum, device, Path(args.ckpt_dir))
    trainer.train(args.epochs)
    print("Smoke test finished OK")
