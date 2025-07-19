"""executor.py
Program‑generation & sandbox executor for ARC‑AGI‑2.

Patch v1.1 — adds ``attach_encoder`` so it integrates with ``search_loop.py``
and ``evaluation_suite.py``.  No other behaviour changes.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# DSL & Tokeniser
# -----------------------------------------------------------------------------

class DSL:
    """Minimal grid‑mutation DSL."""

    OPS = [
        "PAINT",      # x, y, colour
        "FILL",       # colour_old, colour_new
        "MIRROR_X",   # -
        "MIRROR_Y",   # -
        "ROT90",      # -
        "ROT180",     # -
        "COPY",       # x0, y0, w, h, dx, dy
        "STOP",       # end program
    ]

    @classmethod
    def op2id(cls, name: str) -> int:  # noqa: D401  (simple function)
        return cls.OPS.index(name)

    @classmethod
    def id2op(cls, idx: int) -> str:
        return cls.OPS[idx]


class ProgramTokenizer(nn.Module):
    """Embeds DSL op‑codes + integer args."""

    def __init__(self, d_model: int, vocab_extra: int = 256):
        super().__init__()
        self.tok_stop = DSL.op2id("STOP")
        self.vocab_size = len(DSL.OPS) + vocab_extra
        self.emb = nn.Embedding(self.vocab_size, d_model)

    def forward(self, toks: torch.LongTensor) -> torch.Tensor:  # [B, T]
        return self.emb(toks)


# -----------------------------------------------------------------------------
# Program synthesiser (decoder)
# -----------------------------------------------------------------------------

class ProgramSynthesiser(nn.Module):
    def __init__(self, d_model: int, max_steps: int = 10, n_heads: int = 8):
        super().__init__()
        self.token_emb = ProgramTokenizer(d_model)
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.mem_proj = nn.Linear(d_model, d_model)  # task embedding → memory key
        self.lm_head = nn.Linear(d_model, self.token_emb.vocab_size)
        self.max_steps = max_steps

        self.register_buffer("start_tok", torch.tensor([DSL.op2id("STOP")]))  # start symbol

    def forward(self, task_emb: torch.Tensor, prev_toks: torch.LongTensor) -> torch.Tensor:
        # task_emb: [B, d]  prev_toks: [B, T]
        B = task_emb.size(0)
        if prev_toks.numel() == 0:
            prev_toks = self.start_tok.expand(B, 1)
        tok_emb = self.token_emb(prev_toks)  # [B, T, d]
        tgt = tok_emb.transpose(0, 1)  # [T, B, d]
        mem = self.mem_proj(task_emb).unsqueeze(0)  # [1, B, d]
        out = self.decoder(tgt, mem)
        logits = self.lm_head(out[-1])  # last token
        return logits  # [B, V]


# -----------------------------------------------------------------------------
# Grid sandbox
# -----------------------------------------------------------------------------

def _mirror_x(grid: torch.LongTensor) -> torch.LongTensor:
    return grid.flip(1)

def _mirror_y(grid: torch.LongTensor) -> torch.LongTensor:
    return grid.flip(0)

def _rot90(grid: torch.LongTensor) -> torch.LongTensor:
    return grid.transpose(0, 1).flip(1)

def _rot180(grid: torch.LongTensor) -> torch.LongTensor:
    return grid.flip(0, 1)


class GridSandbox:
    """Applies DSL ops to a grid (in place where safe)."""

    @staticmethod
    def step(grid: torch.LongTensor, tok: int, args: List[int]) -> torch.LongTensor:
        op = DSL.id2op(tok)
        g = grid
        if op == "PAINT":
            x, y, c = args
            if 0 <= y < g.shape[0] and 0 <= x < g.shape[1]:
                g[y, x] = c
        elif op == "FILL":
            cold, cnew = args
            g[g == cold] = cnew
        elif op == "MIRROR_X":
            g = _mirror_x(g)
        elif op == "MIRROR_Y":
            g = _mirror_y(g)
        elif op == "ROT90":
            g = _rot90(g)
        elif op == "ROT180":
            g = _rot180(g)
        elif op == "COPY":
            x0, y0, w, h, dx, dy = args
            xs = slice(y0, y0 + h)
            ys = slice(x0, x0 + w)
            patch = g[xs, ys].clone()
            tgt_x = x0 + dx
            tgt_y = y0 + dy
            if 0 <= tgt_y < g.shape[0] and 0 <= tgt_x < g.shape[1]:
                g[tgt_y : tgt_y + h, tgt_x : tgt_x + w] = patch
        elif op == "STOP":
            pass
        return g


# -----------------------------------------------------------------------------
# Executor
# -----------------------------------------------------------------------------

class Executor(nn.Module):
    """Generates a program then executes it on an input grid."""

    def __init__(self, d_model: int = 128, max_steps: int = 10):
        super().__init__()
        self.synth = ProgramSynthesiser(d_model, max_steps=max_steps)
        self.tokenizer = self.synth.token_emb  # convenience
        self._have_encoder = False  # flag after attach

    # ------------------------------------------------------------------
    # Plumbing to attach encoder/synthesiser (used by SearchLoop)
    # ------------------------------------------------------------------
    def attach_encoder(self, grid_encoder, task_synthesiser):
        """Attach external encoder + task synthesiser so *meta_encode* works."""
        self.grid_encoder = grid_encoder
        self.task_synthesiser = task_synthesiser
        self._have_encoder = True

    # ------------------------------------------------------------------
    # Meta‑encoding helpers
    # ------------------------------------------------------------------
    def meta_encode(self, grids_in: List[torch.Tensor], grids_out: List[torch.Tensor]) -> torch.Tensor:
        if not self._have_encoder:
            raise RuntimeError("Executor: call attach_encoder() first.")
        batch = grids_in + grids_out  # list of 6 tensors
        roles = torch.tensor([0] * len(grids_in) + [1] * len(grids_out))
        enc = self.grid_encoder
        synth = self.task_synthesiser
        with torch.no_grad():
            grid_vecs = enc(batch)  # (6, d)
            task_emb = synth(grid_vecs, roles)  # (d,)
        return task_emb

    # ------------------------------------------------------------------
    # Sampling & execution
    # ------------------------------------------------------------------
    def sample_step(self, task_emb: torch.Tensor, prev_toks: List[int], temperature: float = 1.0) -> torch.Tensor:
        prev = torch.tensor(prev_toks, dtype=torch.long).unsqueeze(0)  # [1, T]
        logits = self.synth(task_emb, prev)  # [1, V]
        if temperature != 1.0:
            logits = logits / temperature
        return logits.squeeze(0)  # [V]

    def run_program(self, program: List[int], grid: torch.LongTensor) -> torch.LongTensor:
        """
        Execute a flat token list on a copy of *grid*.

        Tokens are read in chunks of four: [opcode, arg1, arg2, arg3].
        • Execution stops if we hit the explicit STOP token **or**
          if the opcode ID is outside the valid DSL range (guard-rail).
        • Missing arguments are padded with zeros.
        """
        g = grid.clone()
        i = 0
        while i < len(program):
            tok = program[i]

            # ── Halt on STOP or invalid opcode ────────────────────────────────
            if tok == self.tokenizer.tok_stop or tok >= len(DSL.OPS):
                break

            # ── Fetch up to three arguments, pad if fewer are present ────────
            args = program[i + 1 : i + 4]
            args = args + [0] * (3 - len(args))

            g = GridSandbox.step(g, tok, args)
            i += 4  # advance to next opcode slot

        return g




# -----------------------------------------------------------------------------
# __all__
# -----------------------------------------------------------------------------

__all__ = [
    "Executor",
    "ProgramSynthesiser",
    "ProgramTokenizer",
    "DSL",
]


# -----------------------------------------------------------------------------
# Smoke test – ``python executor.py``
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running executor smoke test…")
    ex = Executor(d_model=32)
    dummy_grid = torch.arange(25).view(5, 5) % 10
    # attach fake encoder/synth so meta_encode raises friendly error if used
    ex.attach_encoder(lambda *x: torch.zeros(1, 32), lambda x: torch.zeros(1, 32))  # type: ignore[arg-type]
    prog = [DSL.op2id("MIRROR_X"), 0, 0, 0, DSL.op2id("STOP")]
    out = ex.run_program(prog, dummy_grid)
    print("Input:\n", dummy_grid)
    print("Output:\n", out)
