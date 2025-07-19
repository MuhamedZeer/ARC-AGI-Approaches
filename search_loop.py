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


def grid_score(pred: torch.LongTensor, target: torch.LongTensor) -> float:
    """Pixel‑wise accuracy (0 = perfect, 1 = completely wrong)."""
    if pred.shape != target.shape:
        return 1.0  # maximal error if sizes disagree
    total = target.numel()
    wrong = (pred != target).float().sum().item()
    return wrong / total


# -----------------------------------------------------------------------------
# Beam search driver
# -----------------------------------------------------------------------------


class BeamSearcher:
    """Simple breadth‑wise search over generated programs.

    At each depth we keep the *beam_width* best partial programs (lowest loss).
    The Executor exposes *sample_step* which produces the next token(s)
    conditioned on the previous ones, so we can explore multiple branches.
    """

    def __init__(
        self,
        executor: Executor,
        beam_width: int = 10,
        max_depth: int = 8,
        temperature: float = 1.0,
    ) -> None:
        self.executor = executor
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.temperature = temperature

    @torch.no_grad()
    def search(
        self,
        examples: Sequence[Example],
        test_grid: torch.LongTensor,
        device: torch.device | str = "cpu",
    ) -> Tuple[List[int], torch.LongTensor, float]:
        """Run beam search – returns (best_program_tokens, best_grid, score)."""
        self.executor.to(device)
        test_grid = test_grid.to(device)

        # Pre‑compute the task embedding once – no need to redo at each beam step
        grids_in = [ex.x for ex in examples]
        grids_out = [ex.y for ex in examples]
        me = self.executor.meta_encode(grids_in, grids_out)  # shape [1, d]

        # Each beam item: (program_tokens[List[int]], grid, score)
        BeamItem = Tuple[List[int], torch.LongTensor, float]
        beams: List[BeamItem] = [([], test_grid, 1.0)]
        best: BeamItem | None = None

        for depth in range(self.max_depth):
            candidates: List[BeamItem] = []
            for prog_tokens, grid_state, _ in beams:
                logits = self.executor.sample_step(me, prog_tokens, temperature=self.temperature)
                # Sample top‑k tokens (includes STOP)
                topk = torch.topk(logits, self.beam_width).indices.tolist()
                for tok in topk:
                    new_prog = prog_tokens + [tok]
                    # Execute only when STOP or last depth – else defer to save compute
                    if tok == self.executor.tokenizer.tok_stop or depth == self.max_depth - 1:
                        new_grid = self.executor.run_program(new_prog, grid_state.clone())
                        s = grid_score(new_grid, grids_out[0])  # score vs first example output
                        candidates.append((new_prog, new_grid, s))
                    else:
                        # For non‑terminal partial program we carry forward same grid (no op yet)
                        candidates.append((new_prog, grid_state, 1.0))

            # Keep lowest *beam_width* by score
            beams = sorted(candidates, key=lambda t: t[2])[: self.beam_width]
            best = beams[0]
            # Early exit if perfect match
            if best[2] == 0.0:
                break

        assert best is not None, "Beam search produced no result!?"
        return best  # (program_tokens, grid, score)


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

    @torch.no_grad()
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

    @torch.no_grad()
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
