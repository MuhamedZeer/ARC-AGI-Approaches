"""evaluation_suite.py
Self‑contained **evaluation harness** for the ARC‑AGI‑2 solver.

It loads ARC JSON tasks (the official Kaggle format), spins‑up the full model
stack (``GridEncoder`` + ``TaskSynthesiser`` + ``Executor`` + ``BeamSearchLoop``),
runs inference on the *test* grids, and writes a leaderboard‑ready report.

CLI usage (after installing requirements:: ``python evaluation_suite.py --help``)::

    python evaluation_suite.py \
        --tasks_dir data/ARC \
        --ckpt_path checkpoints/epoch_10.pt \
        --out_csv results.csv \
        --out_json results.json \
        --device cuda:0

Everything lives in a **single file** and depends only on the Python stdlib,
*PyTorch* and *tqdm*.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

# Local imports – assume all four core modules sit in the same folder
from pixel_vit import default_pixel_vit
from executor import Executor
from search_loop import ARCSearchRunner

__all__ = [
    "evaluate_task",
    "evaluate_directory",
]

# -----------------------------------------------------------------------------
# Utility – ARC I/O helpers
# -----------------------------------------------------------------------------


def load_arc_json(path: Path) -> Dict:
    """Load a single ARC task JSON file (Kaggle format)."""
    with path.open("r") as fh:
        return json.load(fh)


# -----------------------------------------------------------------------------
# Evaluation logic
# -----------------------------------------------------------------------------


def build_solver(ckpt_path: Path | None = None, device: str = "cpu") -> ARCSearchRunner:
    """Instantiate the *inference‑only* pipeline and load weights if provided."""

    grid_encoder, task_synth = default_pixel_vit(num_colours=10)
    executor = Executor(grid_encoder.cfg.emb_dim)
    solver = ARCSearchRunner(grid_encoder, task_synth, executor).to(device)

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)

        missing, unexpected = solver.load_state_dict(state, strict=False)

        if missing:
            print("[Eval] Warning: missing keys:", missing)
        if unexpected:
            print("[Eval] Warning: unexpected keys:", unexpected)
    solver.eval()
    return solver


@torch.inference_mode()
def evaluate_task(
    task_path: Path,
    solver: ARCSearchRunner,
    beam_width: int = 5,
    device: str = "cpu",
) -> Dict:
    """Run *one* ARC task and return a result dictionary."""

    task_json = load_arc_json(task_path)

    # ARC format: "train" is a list of (input, output) pairs; "test" is list of inputs
    train_pairs = task_json["train"]
    test_pairs = task_json["test"]

    # --- Encode the six example grids -------------------------------------------------
    ex_grids: List[torch.Tensor] = []
    roles: List[int] = []
    for pair in train_pairs:
        ex_grids.append(torch.tensor(pair["input"], dtype=torch.long, device=device))
        roles.append(0)  # input
        ex_grids.append(torch.tensor(pair["output"], dtype=torch.long, device=device))
        roles.append(1)  # output
    roles_t = torch.tensor(roles, dtype=torch.long, device=device)

    # --- Solve each test input --------------------------------------------------------
    preds: List[Tuple[List[List[int]], bool, int]] = []
    start_t = time.perf_counter()
    for sample in test_pairs:
        inp_grid = torch.tensor(sample["input"], dtype=torch.long, device=device)
        best_prog, pred_grid, _ = solver.solve(ex_grids, roles_t, inp_grid, beam_width=beam_width)

        gt_grid = torch.tensor(sample["output"], dtype=torch.long, device=device)

        # handle shape mismatch
        if pred_grid.shape != gt_grid.shape:
            pix_err = 1.0
        else:
            pix_err = (pred_grid != gt_grid).float().mean().item()

        exact = pix_err == 0.0
        preds.append((pred_grid.cpu().tolist(), exact, pix_err))

    dt = time.perf_counter() - start_t

    # Aggregate metrics (multiple test inputs uncommon but spec‑compliant)
    exact_match = all(p[1] for p in preds)
    pixel_error = sum(p[2] for p in preds)

    return {
        "task_id": task_path.stem,
        "exact_match": int(exact_match),
        "pixel_error": int(pixel_error),
        "n_test": len(test_pairs),
        "runtime_sec": dt,
    }


def evaluate_directory(
    tasks_dir: Path,
    ckpt_path: Path | None,
    out_csv: Path | None,
    out_json: Path | None,
    beam_width: int = 5,
    device: str = "cpu",
):
    """Evaluate *all* JSON files in a directory and save results to disk."""

    solver = build_solver(ckpt_path, device)
    tasks = sorted(tasks_dir.glob("*.json"))
    results: List[Dict] = []

    pbar = tqdm(tasks, desc="Evaluating", unit="task")
    for task_path in pbar:
        res = evaluate_task(task_path, solver, beam_width, device)
        results.append(res)
        pbar.set_postfix({"exact": res["exact_match"], "pix_err": res["pixel_error"]})

    # Write CSV / JSON
    if out_csv is not None:
        keys = results[0].keys()
        with out_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(results)
    if out_json is not None:
        with out_json.open("w") as fh:
            json.dump(results, fh, indent=2)

    # Print summary
    exact_total = sum(r["exact_match"] for r in results)
    print(f"\nFinished {len(results)} tasks · exact‑match {exact_total}/{len(results)}")


# -----------------------------------------------------------------------------
# CLI entry‑point
# -----------------------------------------------------------------------------


def _cli():
    parser = argparse.ArgumentParser(description="ARC‑AGI‑2 evaluation suite")
    parser.add_argument("--tasks_dir", type=Path, required=True, help="Folder of ARC JSON tasks")
    parser.add_argument("--ckpt_path", type=Path, default=None, help="Model checkpoint (optional)")
    parser.add_argument("--out_csv", type=Path, default=None, help="CSV summary out path")
    parser.add_argument("--out_json", type=Path, default=None, help="JSON summary out path")
    parser.add_argument("--beam_width", type=int, default=5, help="Beam width during search")
    parser.add_argument("--device", type=str, default="cpu", help="cuda:0, cpu, etc.")
    args = parser.parse_args()

    evaluate_directory(
        args.tasks_dir,
        args.ckpt_path,
        args.out_csv,
        args.out_json,
        args.beam_width,
        args.device,
    )


# -----------------------------------------------------------------------------
# Pytest – very lightweight sanity checks (run `pytest evaluation_suite.py -q`)
# -----------------------------------------------------------------------------


def test_fake_task(tmp_path: Path):
    """Create a trivial copy task and ensure exact‑match == 1."""
    task = {
        "train": [{"input": [[1]], "output": [[1]]}],
        "test": [{"input": [[1]]}],
    }
    path = tmp_path / "copy.json"
    path.write_text(json.dumps(task))

    solver = build_solver(None)
    res = evaluate_task(path, solver)
    assert res["exact_match"] == 1
    assert res["pixel_error"] == 0


# -----------------------------------------------------------------------------
# Smoke‑test
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    _cli()
