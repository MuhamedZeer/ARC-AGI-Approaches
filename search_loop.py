"""search_loop.py
End‑to‑end **search + scoring** orchestrator for the ARC‑AGI‑2 stack.

This module now exposes two public classes:

* **SearchLoop** – original wrapper that expects a list of ``Example``
  dataclass objects.  Useful during training where we already have explicit
  (input, output) pairs.
* **ARCSearchRunner** – thin façade expected by ``evaluation_suite.py``.
  It accepts the *flat* ``ex_grids`` + ``roles`` tensors produced by the
  evaluation code, converts them into ``Example`` objects, and delegates to
  the underlying ``SearchLoop``.  This keeps both call‑styles working without
  breaking older code.

Running ``python search_loop.py`` still triggers the smoke‑test and now also
prints confirmation that the *ARCSearchRunner* alias behaves the same.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Sequence

import torch
import torch.nn.functional as F
import torch.nn as nn

# Local imports – assume pixel_vit.py and executor.py live on the same module path
from pixel_vit import GridEncoder, TaskSynthesiser
from executor import Executor

__all__ = [
    "Example",
    "SearchLoop",
    "ARCSearchRunner",  # alias expected by evaluation_suite.py
]

# -----------------------------------------------------------------------------
# Utility data‑classes
# -----------------------------------------------------------------------------


@dataclass
class Example:
    """A single (input, output) training example for an ARC task."""

    x: torch.LongTensor  # [H, W]
    y: torch.LongTensor  # [H, W]


# -----------------------------------------------------------------------------
# Scoring helpers
# -----------------------------------------------------------------------------


def grid_score(pred: torch.LongTensor, target: torch.LongTensor) -> torch.Tensor:
    """Pixel-wise error (0 = perfect match, 1 = completely wrong).
    Returns a *scalar tensor* so gradients can flow."""
    if pred.shape != target.shape:
        return torch.ones((), device=pred.device)      # max error
    return (pred != target).float().mean()             # scalar tensor



# -----------------------------------------------------------------------------
# Beam search driver
# -----------------------------------------------------------------------------


class BeamSearcher:
    """Breadth-wise search over generated programs.

    We keep the `beam_width` best partial programs at each depth.
    """

    def __init__(
        self,
        executor: Executor,
        beam_width: int = 10,
        max_depth: int = 8,
        temperature: float = 1.0,
    ) -> None:
        self.executor    = executor
        self.beam_width  = beam_width
        self.max_depth   = max_depth
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------
    def search(
        self,
        examples: Sequence[Example],
        test_grid: torch.LongTensor,
        device: torch.device | str = "cpu",
    ) -> Tuple[List[int], torch.LongTensor, torch.Tensor]:
        """Beam-search; returns (tokens, predicted_grid, error_tensor)."""

        self.executor.to(device)
        test_grid = test_grid.to(device)

        grids_in  = [ex.x for ex in examples]
        grids_out = [ex.y for ex in examples]
        me        = self.executor.meta_encode(grids_in, grids_out)

        # ------------------------------------------------------------------
        # 1. Seed the beam with the identity program  (copy-input)
        # ------------------------------------------------------------------
        id_prog  = [self.executor.tok_noop]           # single-token NOOP
        with torch.no_grad():
            errs = [
                grid_score(
                    self.executor.run_program(id_prog, x.clone()), y
                )
                for x, y in zip(grids_in, grids_out)
            ]
        id_score = torch.stack(errs).mean()           # scalar tensor (0 ≤ s ≤ 1)
        id_grid  = self.executor.run_program(id_prog, test_grid.clone())

        BeamItem = Tuple[List[int], torch.LongTensor, torch.Tensor]
        beams: List[BeamItem] = [(id_prog, id_grid, id_score)]
        best: BeamItem | None = None

        # ------------------------------------------------------------------
        # 2. Expand the beam depth-by-depth
        # ------------------------------------------------------------------
        for depth in range(self.max_depth):
            candidates: List[BeamItem] = []

            for prog_tokens, grid_state, _ in beams:
                logits = self.executor.sample_step(
                    me, prog_tokens, temperature=self.temperature
                )
                for tok in torch.topk(logits, self.beam_width).indices.tolist():
                    new_prog = prog_tokens + [tok]

                    # ── If program is finished, score it on *all* examples ──
                    if tok == self.executor.tok_stop or depth == self.max_depth - 1:
                        errs = [
                            grid_score(
                                self.executor.run_program(new_prog, x.clone()), y
                            )
                            for x, y in zip(grids_in, grids_out)
                        ]
                        s = torch.stack(errs).mean()          # scalar tensor
                        new_grid = self.executor.run_program(
                            new_prog, test_grid.clone()
                        )
                        candidates.append((new_prog, new_grid, s))
                    else:
                        # Partial program: keep current grid_state, max error
                        candidates.append(
                            (new_prog, grid_state, torch.ones((), device=device))
                        )

            # Keep lowest-error `beam_width` candidates
            beams = sorted(candidates, key=lambda t: t[2].item())[: self.beam_width]
            best  = beams[0]

            if best[2].item() == 0.0:     # perfect match → early exit
                break

        assert best is not None
        return best       # (program_tokens, predicted_grid, pixel_error)

# -----------------------------------------------------------------------------
# Glue class – SearchLoop
# -----------------------------------------------------------------------------


class SearchLoop(nn.Module):
    """High‑level convenience wrapper used by training / inference scripts."""

    def __init__(
        self,
        encoder: GridEncoder,
        synthesiser: TaskSynthesiser,
        executor: Executor,
        beam_width: int = 10,
        max_depth: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.synthesiser = synthesiser
        self.executor = executor
        self.beam_width = beam_width
        self.max_depth = max_depth
        self._attach_encoder_to_executor()

    def _attach_encoder_to_executor(self) -> None:
        # Let Executor reuse the same encoder/synthesiser so gradients
        # flow end‑to‑end (useful during training)
        self.executor.attach_encoder(self.encoder, self.synthesiser)


    def solve_examples(
        self,
        examples: Sequence[Example],
        test_grid: torch.LongTensor,
        beam_width: int | None = None,
        max_depth: int | None = None,
    ) -> Tuple[List[int], torch.LongTensor, float]:
        searcher = BeamSearcher(
            self.executor,
            beam_width=beam_width or self.beam_width,
            max_depth=max_depth or self.max_depth,
        )
        return searcher.search(examples, test_grid)


# -----------------------------------------------------------------------------
# Compatibility façade – ARCSearchRunner
# -----------------------------------------------------------------------------


class ARCSearchRunner(SearchLoop):
    """Thin wrapper providing the call‑signature expected by *evaluation_suite.py*.

    It converts the *flat* (ex_grids, roles) representation into the (input,
    output) pairs consumed by the parent ``SearchLoop``.
    """

    def __init__(
        self,
        encoder: GridEncoder,
        synthesiser: TaskSynthesiser,
        executor: Executor,
        beam_width: int = 10,
        max_depth: int = 8,
    ) -> None:
        super().__init__(encoder, synthesiser, executor, beam_width, max_depth)

    def solve(
        self,
        ex_grids: Sequence[torch.LongTensor],
        roles: torch.LongTensor,
        test_grid: torch.LongTensor,
        beam_width: int = 5,
    ) -> Tuple[List[int], torch.LongTensor, float]:
        # Split into inputs/outputs using roles (0=input, 1=output)
        grids_in: List[torch.LongTensor] = []
        grids_out: List[torch.LongTensor] = []
        for g, r in zip(ex_grids, roles.tolist()):
            if r == 0:
                grids_in.append(g)
            else:
                grids_out.append(g)

        # Pair them
        examples = [Example(x, y) for x, y in zip(grids_in, grids_out)]
        return self.solve_examples(examples, test_grid, beam_width=beam_width)


# -----------------------------------------------------------------------------
# Unit tests (pytest ‑q search_loop.py)
# -----------------------------------------------------------------------------

import pytest


@pytest.fixture(scope="module")
def dummy_grids():
    x = torch.zeros(5, 5, dtype=torch.long)
    y = torch.zeros(5, 5, dtype=torch.long)
    return [Example(x, y)]


def test_grid_score_perfect():
    g = torch.tensor([[1, 2], [3, 4]])
    assert grid_score(g, g) == 0.0


def test_grid_score_mismatch():
    a = torch.tensor([[1, 0]])
    b = torch.tensor([[1, 1]])
    assert grid_score(a, b) == 0.5


# -----------------------------------------------------------------------------
# Smoke test – ``python search_loop.py``
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running smoke test…")
    # Build minimal viable network sizes to keep CPU test super fast
    encoder = GridEncoder(d_model=32, num_layers=2)  # type: ignore[arg‑type]
    synthesiser = TaskSynthesiser(d_model=32, num_layers=2)  # type: ignore[arg‑type]
    executor = Executor(d_model=32, max_steps=6)  # type: ignore[arg‑type]

    # Original API
    loop = SearchLoop(encoder, synthesiser, executor, beam_width=4, max_depth=5)

    # Create toy examples: input == output (identity task)
    ex_in = torch.randint(0, 5, (5, 5))
    ex_out = ex_in.clone()
    example = Example(ex_in, ex_out)

    test_grid = ex_in.clone()

    best_prog, best_grid, score = loop.solve_examples([example], test_grid)
    print("[SearchLoop] Program:", best_prog, "score:", score)

    # ARCSearchRunner API (roles vector)
    arc_runner = ARCSearchRunner(encoder, synthesiser, executor, beam_width=4, max_depth=5)
    roles_vec = torch.tensor([0, 1])  # input, output
    best_prog2, best_grid2, score2 = arc_runner.solve([ex_in, ex_out], roles_vec, test_grid)
    print("[ARCSearchRunner] Program:", best_prog2, "score:", score2)
