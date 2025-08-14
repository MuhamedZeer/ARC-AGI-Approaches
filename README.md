# MergedAgent — Unified Vision Transformer + Tree-of-Thought Agent

We combined a **Vision Transformer (ViT)** perception module with a **Tree-of-Thought (ToT)** reasoning controller into a single agent.  
The ViT handles visual understanding; ToT explores multiple reasoning paths and selects the best one to produce robust decisions/answers.

---

## Why this merge?
- **ViT** provides strong visual features (patch embeddings, global context via self-attention).
- **ToT** improves reliability by **branching**, **evaluating**, and **pruning** intermediate thoughts instead of committing to a single chain.
