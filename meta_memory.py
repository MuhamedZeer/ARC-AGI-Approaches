"""meta_memory.py -- for now, it is only an idea
 **Meta‑Memory Cache** for ARC‑AGI‑2.

Purpose
-------
Speed up inference by storing task embeddings + solved programs so that
future tasks with *similar* embeddings can skip search and directly apply the
cached program.

Key components
--------------
1. **EmbeddingStore** – Faiss‑like cosine index (pure Torch for portability).
2. **ProgramCache**   – LRU dict mapping task_id → (program_tokens, score).
3. **MetaMemory**     – Combines the two; on `query(embedding)` returns the
   best cached program within a similarity threshold, else `None`.
4. Unit tests & smoke test.

Usage
-----
::
    from meta_memory import MetaMemory
    memory = MetaMemory(capacity=1000, thresh=0.95)
    prog = memory.query(task_emb)
    if prog is None:
        prog = beam_search(...)
        memory.add(task_emb, prog)
"""
from __future__ import annotations

import math
import torch
from collections import OrderedDict
from typing import List, Tuple, Optional


class EmbeddingStore:
    """Naïve cosine‑similarity index implemented in Torch (no Faiss dep)."""

    def __init__(self, dim: int, device: str | torch.device = "cpu") -> None:
        self.dim = dim
        self.device = device
        self.embeddings: List[torch.Tensor] = []  # each [dim]

    def add(self, emb: torch.Tensor) -> None:
        self.embeddings.append(emb.detach().to(self.device))

    def query(self, emb: torch.Tensor, thresh: float = 0.95) -> Optional[int]:
        if not self.embeddings:
            return None
        mat = torch.stack(self.embeddings)  # [N, d]
        sim = F.cosine_similarity(mat, emb.unsqueeze(0), dim=1)  # [N]
        best_val, idx = sim.max(dim=0)
        return idx.item() if best_val.item() >= thresh else None


class ProgramCache(OrderedDict):
    """LRU cache of solved programs."""

    def __init__(self, capacity: int = 1000):
        super().__init__()
        self.capacity = capacity

    def __setitem__(self, key, value):  # type: ignore[override]
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.capacity:
            self.popitem(last=False)


class MetaMemory:
    def __init__(self, dim: int = 128, capacity: int = 1000, thresh: float = 0.95):
        self.store = EmbeddingStore(dim)
        self.cache = ProgramCache(capacity)
        self.thresh = thresh

    def add(self, emb: torch.Tensor, program: List[int], score: float) -> None:
        key = len(self.store.embeddings)
        self.store.add(emb)
        self.cache[key] = (program, score)

    def query(self, emb: torch.Tensor) -> Optional[Tuple[List[int], float]]:
        idx = self.store.query(emb, self.thresh)
        return self.cache.get(idx) if idx is not None else None


# ----------------------- Unit test -----------------------
if __name__ == "__main__":
    import random
    random.seed(0)
    torch.manual_seed(0)

    memory = MetaMemory(dim=4, capacity=3, thresh=0.9)
    for i in range(5):
        e = torch.randn(4)
        p = [random.randint(0, 10) for _ in range(3)]
        memory.add(e, p, 0.0)
    q = memory.store.embeddings[-1]  # query with exact vector
    prog = memory.query(q)
    print("Retrieved:", prog)
