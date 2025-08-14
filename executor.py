"""executor.py
Program‑generation & sandbox executor for ARC‑AGI‑2.

Patch v1.1 — adds ``attach_encoder`` so it integrates with ``search_loop.py``
and ``evaluation_suite.py``.  No other behaviour changes.
"""
from __future__ import annotations

import math
import random
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# DSL & Tokeniser
# -----------------------------------------------------------------------------

from types import SimpleNamespace

class DSL:
    """Minimal grid-mutation DSL with explicit arg counts."""

    # ── Opcode table --------------------------------------------------------
    # name        n_args   comments
    OPCODES = [
        ("PAINT"   , 3),   # x, y, colour
        ("FILL"    , 2),   # colour_old, colour_new
        ("MIRROR_X", 0),   # no args
        ("MIRROR_Y", 0),   # no args
        ("ROT90"   , 0),   # no args
        ("ROT180"  , 0),   # no args
        ("COPY"    , 6),   # x0, y0, w, h, dx, dy
        ("STOP"    , 0),   # end program
    ]

    # Build helper lists so we can index by token ID
    OPS       = [name for name, _ in OPCODES]                       # ['PAINT', …]
    N_ARGS    = [n for _, n in OPCODES]                             # [3, 2, 0, …]

    @classmethod
    def op2id(cls, name: str) -> int:
        """Return integer ID for an opcode name."""
        return cls.OPS.index(name)

    @classmethod
    def id2op(cls, idx: int) -> str:
        """Return opcode name for an integer ID."""
        return cls.OPS[idx]

    @classmethod
    def nargs(cls, idx: int) -> int:
        """Number of arguments that opcode *idx* expects."""
        return cls.N_ARGS[idx]




class ProgramTokenizer(nn.Module):
    """
    Lightweight token-to-vector embedder for program tokens.

    * `vocab_size` can be passed explicitly (recommended) — otherwise it defaults
      to `len(DSL.OPS) + 1`, which was the original size (all DSL opcodes + STOP).
    * `d_model` is the embedding dimension shared with the rest of the network.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: Optional[int] = None,
        pad_id: Optional[int] = None,
    ):
        super().__init__()

        if vocab_size is None:
            vocab_size = len(DSL.OPS) + 1          # original: opcodes + STOP

        self.vocab_size = vocab_size
        self.pad_id = pad_id if pad_id is not None else 0

        self.emb = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_id)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, tok_ids: torch.LongTensor) -> torch.Tensor:
        """
        Args
        ----
        tok_ids : LongTensor of shape [B, T]

        Returns
        -------
        Tensor of shape [B, T, d_model]
        """
        return self.emb(tok_ids)

    # ------------------------------------------------------------------
    # Optional: grow the vocabulary after initialisation
    # ------------------------------------------------------------------
    def expand_vocab(self, extra_tokens: int = 1):
        """
        Increase the embedding matrix by `extra_tokens` rows.
        Useful if you decide to add new opcodes or special tokens late.

        NOTE: Call *before* you load any pretrained weights that depend on
        the original embedding size.
        """
        if extra_tokens <= 0:
            return  # nothing to do

        with torch.no_grad():
            old_weight = self.emb.weight.data
            B, d = old_weight.shape
            new_rows = torch.randn(extra_tokens, d, device=old_weight.device) * 0.02
            new_weight = torch.cat([old_weight, new_rows], dim=0)
            self.emb = nn.Embedding.from_pretrained(new_weight, padding_idx=self.pad_id)
            self.vocab_size += extra_tokens


# -----------------------------------------------------------------------------
# Program synthesiser (decoder)
# -----------------------------------------------------------------------------

class ProgramSynthesiser(nn.Module):
    """
    Takes a task embedding + a sequence of previous tokens and returns logits
    for the next token.  Uses a 2-layer Transformer decoder.
    """

    def __init__(
        self,
        d_model: int,
        max_steps: int = 10,
        n_heads: int = 8,
        vocab_size: Optional[int] = None,
    ):
        """
        Args
        ----
        d_model     : hidden size
        max_steps   : maximum program length we’ll sample
        n_heads     : transformer heads
        vocab_size  : total number of token IDs (defaults to len(DSL)+1)
                      We pass `len(DSL.OPS) + 2` so we have room for NOOP + STOP.
        """
        super().__init__()

        if vocab_size is None:
            vocab_size = len(DSL.OPS) + 1                # original behaviour

        # --- Token embedding -------------------------------------------------
        self.token_emb = ProgramTokenizer(d_model, vocab_size=vocab_size)
        self.vocab_size = vocab_size                     # save for reference

        # --- 2-layer Transformer decoder -------------------------------------
        decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        # --- Project task embedding into decoder “memory” --------------------
        self.mem_proj = nn.Linear(d_model, d_model)

        # --- Language-model head ---------------------------------------------
        self.lm_head = nn.Linear(d_model, vocab_size)

        self.max_steps = max_steps

        # Use STOP as the BOS token (simplest): [STOP] means “start decoding”
        self.register_buffer(
            "start_tok",
            torch.tensor([DSL.op2id("STOP")], dtype=torch.long)
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        task_emb: torch.Tensor,          # shape [B, d]
        prev_toks: torch.LongTensor,     # shape [B, T]
    ) -> torch.Tensor:                   # returns logits [B, vocab_size]
        B = task_emb.size(0)

        # If no previous tokens, feed the BOS (= STOP) token
        if prev_toks.numel() == 0:
            prev_toks = self.start_tok.to(task_emb.device).expand(B, 1)

        # --- Embed tokens & project task memory -----------------------------
        tok_emb = self.token_emb(prev_toks)          # [B, T, d]
        tgt     = tok_emb.transpose(0, 1)            # [T, B, d]

        mem     = self.mem_proj(task_emb).unsqueeze(0)  # [1, B, d]

        # --- Transformer decoding -------------------------------------------
        out = self.decoder(tgt, mem)                 # [T, B, d]

        # Return logits for the last generated position
        logits = self.lm_head(out[-1])               # [B, vocab_size]
        return logits



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

    # ------------------------------------------------------------------
    # Token IDs (keep them consistent everywhere)
    # ------------------------------------------------------------------
    tok_noop: int = 0                                 # new: identity op
    tok_stop: int = len(DSL.OPS) + 1                  # stop / EOS token

    def __init__(self, d_model: int = 128, max_steps: int = 10):
        super().__init__()

        #  We need one extra vocab slot for NOOP
        vocab_size = len(DSL.OPS) + 2                 # +1 STOP, +1 NOOP

        self.synth = ProgramSynthesiser(
            d_model,
            max_steps=max_steps,
            vocab_size=vocab_size,                    # <-- enlarged
        )
        self.tokenizer = self.synth.token_emb     # <-- define this first


        self.tokenizer.tok_noop = self.tok_noop  # alias: 0
        self.tokenizer.tok_stop = self.tok_stop  # alias: len(DSL.OPS)+1

        self._have_encoder = False


    # ------------------------------------------------------------------
    # Plumbing to attach encoder / synthesiser (used by SearchLoop)
    # ------------------------------------------------------------------
    def attach_encoder(self, grid_encoder, task_synthesiser):
        """Attach external encoder + task synthesiser so *meta_encode* works."""
        self.grid_encoder = grid_encoder
        self.task_synthesiser = task_synthesiser
        self._have_encoder = True

    # ------------------------------------------------------------------
    # Meta-encoding helpers
    # ------------------------------------------------------------------
    def meta_encode(
        self,
        grids_in: List[torch.Tensor],
        grids_out: List[torch.Tensor],
    ) -> torch.Tensor:
        if not self._have_encoder:
            raise RuntimeError("Executor: call attach_encoder() first.")

        batch = grids_in + grids_out                 # list of 6 tensors
        roles = torch.tensor([0] * len(grids_in) + [1] * len(grids_out))

        with torch.no_grad():
            grid_vecs = self.grid_encoder(batch)     # (6, d)
            task_emb  = self.task_synthesiser(grid_vecs, roles)  # (d,)

        return task_emb

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_step(
        self,
        task_emb: torch.Tensor,
        prev_toks: List[int],
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Return logits over the *full* vocabulary (including NOOP + STOP)."""
        prev = torch.tensor(prev_toks, dtype=torch.long).unsqueeze(0)  # [1, T]
        logits = self.synth(task_emb, prev)                            # [1, V]
        if temperature != 1.0:
            logits = logits / temperature
        return logits.squeeze(0)                                       # [V]

    # ------------------------------------------------------------------
    # Program execution
    # ------------------------------------------------------------------
    def run_program(
        self,
        program: List[int],
        grid: torch.LongTensor,
    ) -> torch.LongTensor:
        """
        Execute *program* on a clone of *grid*.

        • Identity if program is [], [STOP], or [NOOP].
        • Each opcode knows how many arguments it needs: DSL.OPS[tok].n_args.
        • We slice exactly that many from the token stream and pad with zeros
          if the list runs short.
        """
        # ── Identity shortcuts ─────────────────────────────────────────────
        if (
            len(program) == 0
            or program == [self.tok_stop]
            or program == [self.tok_noop]
        ):
            return grid.clone()

        g = grid.clone()
        i = 0
        while i < len(program):
            tok = program[i]

            # ── Halt on STOP / NOOP / invalid opcode ───────────────────────
            if tok in (self.tok_stop, self.tok_noop) or tok >= len(DSL.OPS):
                break

            # ── How many args does this opcode need? -----------------------
            try:
                n_args = DSL.nargs(tok)  # returns 0, 2, or 6 for this opcode
            except AttributeError:
                # Fallback: assume 0-arg op if metadata missing
                n_args = 0

            # ── Slice exactly that many, pad with zeros if fewer present ───
            args = program[i + 1 : i + 1 + n_args]
            args = args + [0] * (n_args - len(args))

            g = GridSandbox.step(g, tok, args)
            i += 1 + n_args                              # advance pointer

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
