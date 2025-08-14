from pathlib import Path
import torch
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

# === Project Base Directory ===
BASE_DIR = Path(__file__).resolve().parent

# === Enhanced Configuration ===
@dataclass
class ViTConfig:
    emb_dim: int = 256  # Increased for better representation
    depth: int = 6      # Deeper transformer
    num_heads: int = 8
    mlp_ratio: int = 4
    dropout: float = 0.1
    num_colours: int = 10
    max_grid_size: int = 30  # Explicitly set for ARC tasks
    device: Optional[str] = None  # Will inherit from parent

@dataclass
class TaskSynthesiserConfig:
    emb_dim: int = 256  # Match ViT dimension
    depth: int = 2
    num_heads: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    device: Optional[str] = None

@dataclass
class ProgramSynthesiserConfig:
    d_model: int = 256  # Increased capacity
    max_steps: int = 16  # Longer programs
    n_heads: int = 8
    vocab_size: int = 128  # Larger vocabulary
    device: Optional[str] = None

@dataclass
class HybridConfig:
    # Device handling - now properly initialized
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Sub-configs with device inheritance
    vit: ViTConfig = field(default_factory=lambda: ViTConfig(device="cuda" if torch.cuda.is_available() else "cpu"))
    task_synthesiser: TaskSynthesiserConfig = field(default_factory=lambda: TaskSynthesiserConfig(device="cuda" if torch.cuda.is_available() else "cpu"))
    program_synthesiser: ProgramSynthesiserConfig = field(default_factory=lambda: ProgramSynthesiserConfig(device="cuda" if torch.cuda.is_available() else "cpu"))
    
    # Execution parameters
    beam_width: int = 5
    max_depth: int = 8
    temperature: float = 1.0
    symbolic_first: bool = True
    fallback_to_neural: bool = True
    hybrid_threshold: float = 0.6
    
    # Resource management
    max_grid_size: int = 30  # Now at top level for easy access
    meta_program_cache_size: int = 1000
    feature_cache_size: int = 500
    max_time_per_task: int = 3600
    
    # Training (if applicable)
    learning_rate: float = 1e-4
    batch_size: int = 32
    max_epochs: int = 100
    
    # Debugging
    debug_mode: bool = False

# === Initialization Helpers ===
def get_default_config() -> HybridConfig:
    """Create properly initialized config with device inheritance"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return HybridConfig(
        device=device,
        vit=ViTConfig(device=device),
        task_synthesiser=TaskSynthesiserConfig(device=device),
        program_synthesiser=ProgramSynthesiserConfig(device=device),
        max_grid_size=30
    )

# Constants (unchanged)
DSL_OPS = [
    ("PAINT", 3), ("FILL", 2), ("MIRROR_X", 0), ("MIRROR_Y", 0),
    ("ROT90", 0), ("ROT180", 0), ("COPY", 6), ("STOP", 0)
]
EXTENDED_DSL_OPS = DSL_OPS + [
    ("IDENTITY", 0), ("CROP", 0), ("PAD", 2), ("FILL_GAPS_H", 0),
    ("FILL_GAPS_V", 0), ("DILATE", 1), ("ERODE", 1), ("MEDIAN_FILTER", 0),
    ("INVERT", 0), ("COLOR_MAP", 2), ("SCALE", 1)
]

MAX_CONST = 30
CONST_OFFSET = len(EXTENDED_DSL_OPS) + 2
CNN_FEATURE_DIMS = 256
SYMBOLIC_FEATURE_DIMS = 128
COMBINED_FEATURE_DIMS = CNN_FEATURE_DIMS + SYMBOLIC_FEATURE_DIMS

DATA_DIR = BASE_DIR / "Data" / "training"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

METRICS = [
    "accuracy", "reasoning_interpretability", "search_budget_used",
    "dsl_coverage", "reasoning_depth_score", "generalization_index"
]