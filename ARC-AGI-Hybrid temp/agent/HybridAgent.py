"""
Hybrid ARC Agent
Combines Tree-of-Thought style symbolic reasoning with neural program synthesis.
This file includes small but important fixes and quality-of-life improvements:
- Robust config handling (dicts OR dataclasses)
- Safer tensor ops (fliplr/flipud fallback)
- No-grad evaluation & device handling
- Deterministic seeding (optional)
- Cleaner program synthesiser config usage
"""

# agent/HybridAgent.py

import json
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# If your project exposes these under agent.*, keep; otherwise adjust import paths.
try:
    from agent import TaskSynthesiser, GridExecutor  # noqa
except Exception:  # optional components; we'll guard usage
    TaskSynthesiser = None
    GridExecutor = None

# Pull your config dataclass from config.py (do NOT redefine it here)
from config import HybridConfig as ProjectHybridConfig  # noqa

logger = logging.getLogger(__name__)


# -------------------------------
# Utility helpers
# -------------------------------

def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move tensor to device if needed."""
    return x.to(device) if x.device != device else x


def _crop_to_max(grid: torch.Tensor, max_h: int, max_w: int) -> torch.Tensor:
    """Center-crop a grid to (<=max_h, <=max_w) if it exceeds max dims."""
    h, w = grid.shape
    if h <= max_h and w <= max_w:
        return grid

    new_h = min(h, max_h)
    new_w = min(w, max_w)

    top = max(0, (h - new_h) // 2)
    left = max(0, (w - new_w) // 2)
    return grid[top:top + new_h, left:left + new_w]


# -------------------------------
# Minimal, shape-agnostic encoder
# -------------------------------

class HybridGridEncoder(nn.Module):
    """
    A lightweight encoder that:
      - embeds color IDs (0..9)
      - adds 2D positional encoding
      - mean-pools over spatial dims
    Works for any H×W (no fixed size).
    """
    def __init__(self, d_model: int = 256, num_colours: int = 10) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_colours = num_colours
        self.color_emb = nn.Embedding(num_colours, d_model)
        self.pos_proj = nn.Linear(2, d_model)

    def forward(self, grid_bchw: torch.Tensor) -> torch.Tensor:
        """
        Args
        ----
        grid_bchw: (B, 1, H, W) long tensor with values in [0, num_colours-1]

        Returns
        -------
        (B, d_model) feature vectors.
        """
        if grid_bchw.dim() != 4 or grid_bchw.size(1) != 1:
            raise ValueError("HybridGridEncoder expects input of shape (B, 1, H, W)")

        B, _, H, W = grid_bchw.shape
        device = grid_bchw.device

        # (B, H, W, d_model)
        colors = self.color_emb(grid_bchw.squeeze(1).clamp(min=0, max=self.num_colours - 1))

        # (H, W, 2) -> (H, W, d_model)
        xs = torch.linspace(0, 1, W, device=device).repeat(H, 1)
        ys = torch.linspace(0, 1, H, device=device).unsqueeze(1).repeat(1, W)
        pos_2d = torch.stack([xs, ys], dim=-1)  # (H, W, 2)
        pos = self.pos_proj(pos_2d)  # (H, W, d_model)
        pos = pos.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, d_model)

        feat = colors + pos  # (B, H, W, d_model)
        feat = feat.mean(dim=(1, 2))  # (B, d_model)
        return feat


class NeuralSynthesiser(nn.Module):
    """Tiny head to turn an aggregated context vector into a 'program' embedding."""
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, d_model)
        return self.fc(x)


# -------------------------------
# HybridAgent
# -------------------------------

class HybridAgent(nn.Module):
    """
    End-to-end agent wrapper that:
      - preprocesses JSON grids to tensors,
      - encodes train/test grids on the *same device*,
      - synthesises a program embedding,
      - (optionally) executes a program (stubbed/mocked here),
      - returns predictions and scores.
    """

    def __init__(self, config: Union[ProjectHybridConfig, Dict[str, Any], None] = None, seed: int = 123) -> None:
        super().__init__()

        # Normalize config to a dict for flexible access
        if config is None:
            # Create default from project config dataclass
            self.config = asdict(ProjectHybridConfig()) if is_dataclass(ProjectHybridConfig()) else ProjectHybridConfig().__dict__
        elif is_dataclass(config):
            self.config = asdict(config)
        elif isinstance(config, dict):
            self.config = dict(config)
        else:
            # Fallback: treat as object with attributes
            self.config = config.__dict__

        # Core hyper-params with safe defaults
        self.device = torch.device(self.config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        self.d_model = int(self.config.get("d_model", 256))
        self.max_grid_size = int(self.config.get("max_grid_size", 64))
        self.beam_width = int(self.config.get("beam_width", 5))
        self.exec_timeout = float(self.config.get("exec_timeout", 5.0))

        # Seed
        torch.manual_seed(seed)

        # Models on the SAME device
        self.encoder = HybridGridEncoder(d_model=self.d_model).to(self.device)
        self.neural_synthesiser = NeuralSynthesiser(d_model=self.d_model).to(self.device)

        # Optional modules (guarded)
        self.task_synthesiser = None
        if TaskSynthesiser is not None and "task_synthesiser" in self.config:
            try:
                self.task_synthesiser = TaskSynthesiser(self.config["task_synthesiser"]).to(self.device)
            except Exception as e:
                logger.warning(f"TaskSynthesiser init failed (continuing without it): {e}")

        self.executor = None
        if GridExecutor is not None:
            try:
                self.executor = GridExecutor(max_grid_size=self.max_grid_size)  # executor typically fine on CPU
            except Exception as e:
                logger.warning(f"GridExecutor init failed (continuing without it): {e}")

        self.eval()

    # -------- I/O --------

    def _prep_grid_tensor(self, grid: List[List[int]]) -> torch.Tensor:
        """
        Convert a nested list grid to (1,1,H,W) long tensor on self.device.
        Grids larger than (max_grid_size, max_grid_size) are center-cropped.
        """
        t = torch.tensor(grid, dtype=torch.long)
        t = _crop_to_max(t, self.max_grid_size, self.max_grid_size)
        # (1,1,H,W) and move to device
        return _to_device(t.unsqueeze(0).unsqueeze(0), self.device)

    def load_task(self, json_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load a single ARC task file and prepare tensors.
        """
        p = Path(json_path)
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        def proc_pair(pair):
            return (
                self._prep_grid_tensor(pair["input"]),
                self._prep_grid_tensor(pair["output"]),
            )

        train = [proc_pair(tp) for tp in data.get("train", [])]
        test = []
        for pair in data.get("test", []):
            inp = self._prep_grid_tensor(pair["input"])
            out = self._prep_grid_tensor(pair["output"]) if "output" in pair else None
            test.append((inp, out))

        return {"train": train, "test": test, "filename": p.name}

    # -------- Core pipeline --------

    @torch.no_grad()
    def _encode_pair(self, inp_bchw: torch.Tensor, out_bchw: torch.Tensor) -> torch.Tensor:
        """
        Encode a train pair into a single (1, d_model) vector on self.device.
        We average input & output features to keep dimensionality = d_model.
        """
        inp_bchw = _to_device(inp_bchw, self.device)
        out_bchw = _to_device(out_bchw, self.device)

        f_in = self.encoder(inp_bchw)   # (1, d_model)
        f_out = self.encoder(out_bchw)  # (1, d_model)
        return (f_in + f_out) * 0.5     # (1, d_model)

    @torch.no_grad()
    def _context_from_train(self, train_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Aggregate train pairs into a single (1, d_model) context vector.
        """
        if not train_pairs:
            # empty context
            return torch.zeros(1, self.d_model, device=self.device)

        feats = [self._encode_pair(inp, out) for inp, out in train_pairs]  # list of (1, d_model)
        ctx = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)            # (1, d_model)
        return ctx

    @torch.no_grad()
    def solve_task(
        self,
        train_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
        test_input: torch.Tensor,
        target_output: Optional[torch.Tensor] = None,
        beam_width: Optional[int] = None,
        exec_time_budget_s: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        End-to-end single-pair solve.
        Ensures EVERYTHING lives on self.device before compute.
        """
        beam_width = beam_width or self.beam_width
        exec_time_budget_s = exec_time_budget_s or self.exec_timeout

        # 1) Encode context + test input (all on same device)
        ctx = self._context_from_train(train_pairs)                  # (1, d_model) on self.device
        test_input = _to_device(test_input, self.device)
        test_feat = self.encoder(test_input)                         # (1, d_model)

        # 2) Simple fusion: average context and test feature -> (1, d_model)
        fused = 0.5 * (ctx + test_feat)

        # 3) Neural program synthesiser (on same device)
        prog_emb = self.neural_synthesiser(fused)                    # (1, d_model)

        # 4) Execute (stub): here we just return input as prediction.
        #    If you call a CPU executor, move tensors appropriately.
        prediction = test_input.clone()                              # still on self.device

        # 5) Score (keep on same device during compute; detach for Python)
        score: float
        if target_output is None:
            score = 0.0
        else:
            target_output = _to_device(target_output, self.device)
            # Use a simple integer equality score as example; replace with your metric
            same = (prediction == target_output).float().mean()
            score = float(same.item())

        return {
            "prediction": prediction,        # (1,1,H,W) on self.device
            "program": prog_emb.detach().cpu().tolist(),  # exportable
            "score": score,
        }

    # -------- Batch API --------

    @torch.no_grad()
    def batch_solve(self, task_dir: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Evaluate all tasks in a directory.
        """
        p = Path(task_dir)
        files = sorted(p.glob("*.json"))
        results: List[Dict[str, Any]] = []

        for f in files:
            try:
                task = self.load_task(f)
                train_pairs = task["train"]
                task_details = []

                for (test_inp, tgt) in task["test"]:
                    out = self.solve_task(
                        train_pairs=train_pairs,
                        test_input=test_inp,
                        target_output=tgt,
                        beam_width=self.beam_width,
                        exec_time_budget_s=self.exec_timeout,
                    )
                    # Save a small, JSON-serializable view
                    detail = {
                        "program": out["program"],
                        "score": out["score"],
                        "success": bool(
                            tgt is not None
                            and out["prediction"].shape == tgt.shape
                            and torch.equal(out["prediction"], tgt)
                        ),
                    }
                    task_details.append(detail)

                results.append(
                    {
                        "task": task["filename"],
                        "num_pairs": len(task_details),
                        "details": task_details,
                    }
                )
            except Exception as e:
                logger.error(f"Failed {f.name}: {e}")
                results.append({"task": f.name, "error": str(e)})

        return results


"""
Hybrid ARC Agent
Combines Tree-of-Thought symbolic reasoning with neural program synthesis
"""
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter, defaultdict
import heapq
import time
import hashlib
import logging

from config import DEFAULT_CONFIG, ToTConfig
from .encoder import HybridEncoder
from .symbolic import symbolic_ops
from .dsl import DSL, NeuralProgramSynthesiser, DSLInterpreter, ProgramGenerator
from .memory import MetaProgramMemory, SymbolicTrace, DSLProgram
from .utils import grid_score, extract_features

logger = logging.getLogger(__name__)

# === Tree-of-Thought Node ===

@dataclass(order=True)
class ThoughtNode:
    """Node in the Tree-of-Thought search."""
    score: float
    grid: torch.Tensor = field(compare=False)
    steps: List[str] = field(default_factory=list, compare=False)
    features: torch.Tensor = field(default=None, compare=False)
    confidence: float = field(default=0.0, compare=False)
    grid_hash: str = field(default="", compare=False)

    def __post_init__(self):
        if not self.grid_hash:
            self.grid_hash = self._compute_grid_hash()

    def _compute_grid_hash(self) -> str:
        """Compute hash of grid for loop detection."""
        return hashlib.md5(self.grid.cpu().numpy().tobytes()).hexdigest()

# === Pattern Analysis ===

class PatternAnalyzer:
    """Analyzes grid patterns for symbolic reasoning."""

    @staticmethod
    def analyze_grid_properties(grid: torch.Tensor) -> Dict[str, Any]:
        """Analyze comprehensive grid properties."""
        props = {
            'shape': tuple(grid.shape),
            'unique_colors': len(torch.unique(grid)),
            'color_distribution': dict(zip(*torch.unique(grid, return_counts=True))),
            'has_symmetry_h': torch.allclose(grid, torch.fliplr(grid)),
            'has_symmetry_v': torch.allclose(grid, torch.flipud(grid)),
            'has_rotational_symmetry': torch.allclose(grid, torch.rot90(grid, 2)),
            'sparsity': (grid == 0).float().mean().item(),
            'bounding_box': PatternAnalyzer._get_bounding_box(grid),
            'dominant_color': PatternAnalyzer._get_dominant_color(grid),
        }
        return props

    @staticmethod
    def _get_bounding_box(grid: torch.Tensor) -> Tuple[int, int, int, int]:
        """Get bounding box of non-zero elements."""
        non_zero = grid != 0
        if not non_zero.any():
            return (0, 0, 0, 0)

        rows = torch.any(non_zero, dim=1)
        cols = torch.any(non_zero, dim=0)

        r_indices = torch.where(rows)[0]
        c_indices = torch.where(cols)[0]

        rmin, rmax = r_indices[0].item(), r_indices[-1].item()
        cmin, cmax = c_indices[0].item(), c_indices[-1].item()

        return (rmin, rmax, cmin, cmax)

    @staticmethod
    def _get_dominant_color(grid: torch.Tensor) -> int:
        """Get most frequent non-zero color."""
        non_zero = grid[grid != 0]
        if non_zero.numel() == 0:
            return 0

        colors, counts = torch.unique(non_zero, return_counts=True)
        return colors[counts.argmax()].item()

    @staticmethod
    def detect_transformation_type(input_grid: torch.Tensor, output_grid: torch.Tensor) -> str:
        """Detect type of transformation between input and output."""
        if torch.allclose(input_grid, output_grid):
            return "identity"

        # Check geometric transformations
        if torch.allclose(output_grid, torch.rot90(input_grid, 1)):
            return "rotate90"
        if torch.allclose(output_grid, torch.rot90(input_grid, 2)):
            return "rotate180"
        if torch.allclose(output_grid, torch.rot90(input_grid, 3)):
            return "rotate270"
        if torch.allclose(output_grid, torch.fliplr(input_grid)):
            return "flip_horizontal"
        if torch.allclose(output_grid, torch.flipud(input_grid)):
            return "flip_vertical"
        if torch.allclose(output_grid, input_grid.t()):
            return "transpose"

        # Check color transformations
        input_colors = torch.unique(input_grid)
        output_colors = torch.unique(output_grid)

        if len(input_colors) == len(output_colors):
            # Check if it's a color mapping
            color_map = {}
            for i, input_color in enumerate(input_colors):
                if i < len(output_colors):
                    color_map[input_color.item()] = output_colors[i].item()

            # Test the mapping
            mapped = input_grid.clone()
            for old_color, new_color in color_map.items():
                mapped[input_grid == old_color] = new_color

            if torch.allclose(mapped, output_grid):
                return "color_mapping"

        return "complex"

# === Symbolic Reasoner ===

class SymbolicReasoner:
    """Performs symbolic reasoning using Tree-of-Thought search."""

    def __init__(self, config: ToTConfig):
        self.config = config
        self.analyzer = PatternAnalyzer()
        self.visited_hashes: Set[str] = set()

    def reason_symbolically(self, input_grid: torch.Tensor,
                          target_grid: torch.Tensor,
                          train_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Optional[Tuple[torch.Tensor, List[str], float]]:
        """Perform symbolic reasoning to find transformation."""
        self.visited_hashes.clear()

        # Initialize search
        initial_node = ThoughtNode(
            score=0.0,
            grid=input_grid,
            steps=[],
            features=extract_features(input_grid),
            confidence=0.0
        )

        # Priority queue for beam search
        queue = [initial_node]
        best_result = None
        best_score = float('inf')

        for depth in range(self.config.max_depth):
            if not queue:
                break

            # Get top nodes for this depth
            current_nodes = heapq.nsmallest(self.config.max_beam_width, queue)
            queue = []

            for node in current_nodes:
                # Check if we've reached the target
                current_score = grid_score(node.grid, target_grid)
                if current_score < best_score:
                    best_score = current_score
                    best_result = (node.grid, node.steps, node.confidence)

                # If we're close enough, return
                if current_score < 0.1:  # 90% accuracy threshold
                    return best_result

                # Generate next steps
                next_nodes = self._generate_next_steps(node, target_grid, train_pairs)

                for next_node in next_nodes:
                    if next_node.grid_hash not in self.visited_hashes:
                        self.visited_hashes.add(next_node.grid_hash)
                        queue.append(next_node)

        return best_result

    def _generate_next_steps(self, node: ThoughtNode,
                           target_grid: torch.Tensor,
                           train_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[ThoughtNode]:
        """Generate next possible steps from current node."""
        next_nodes = []

        # Get applicable operations
        operations = self._get_applicable_operations(node.grid, target_grid, train_pairs)

        for op_name in operations:
            try:
                # Apply operation
                if op_name in symbolic_ops.operations:
                    op_func = symbolic_ops.get_operation(op_name)
                    result_grid = op_func(node.grid)
                else:
                    continue

                # Skip if result is same as input
                if torch.allclose(result_grid, node.grid):
                    continue

                # Compute score
                score = grid_score(result_grid, target_grid)

                # Compute confidence based on training pairs
                confidence = self._compute_confidence(result_grid, train_pairs)

                # Create new node
                new_steps = node.steps + [op_name]
                new_features = extract_features(result_grid)

                next_node = ThoughtNode(
                    score=score,
                    grid=result_grid,
                    steps=new_steps,
                    features=new_features,
                    confidence=confidence
                )

                next_nodes.append(next_node)

            except Exception as e:
                logger.debug(f"Error applying operation {op_name}: {e}")
                continue

        return next_nodes

    def _get_applicable_operations(self, grid: torch.Tensor,
                                 target_grid: torch.Tensor,
                                 train_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> List[str]:
        """Get operations that are likely to be useful."""
        # Analyze grid properties
        props = self.analyzer.analyze_grid_properties(grid)
        target_props = self.analyzer.analyze_grid_properties(target_grid)

        # Detect transformation type
        transform_type = self.analyzer.detect_transformation_type(grid, target_grid)

        # Select operations based on transformation type
        if transform_type in ["rotate90", "rotate180", "rotate270"]:
            return ["rotate90", "rotate180", "rotate270"]
        elif transform_type in ["flip_horizontal", "flip_vertical"]:
            return ["flip_horizontal", "flip_vertical"]
        elif transform_type == "transpose":
            return ["transpose"]
        elif transform_type == "color_mapping":
            return ["apply_color_mapping", "fill"]
        else:
            # Complex transformation - try pattern operations
            operations = []

            # Add geometric operations
            operations.extend(["rotate90", "rotate180", "flip_horizontal", "flip_vertical"])

            # Add pattern operations based on grid properties
            if props['sparsity'] > 0.5:
                operations.extend(["fill_horizontal_gap", "fill_vertical_gap"])

            if props['unique_colors'] > 3:
                operations.extend(["color_cluster_simplify", "invert"])

            # Add morphological operations
            operations.extend(["cv2_dilate", "cv2_erode", "median_filter"])

            return operations[:10]  # Limit to top 10 operations

    def _compute_confidence(self, grid: torch.Tensor,
                          train_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Compute confidence based on similarity to training patterns."""
        if not train_pairs:
            return 0.5

        similarities = []
        for train_input, train_output in train_pairs:
            # Compare with both input and output patterns
            input_sim = 1.0 - grid_score(grid, train_input)
            output_sim = 1.0 - grid_score(grid, train_output)
            similarities.append(max(input_sim, output_sim))

        return np.mean(similarities) if similarities else 0.5

# === Neural Program Synthesiser ===

class NeuralSynthesiser:
    """Neural program synthesis using task embeddings."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        ps_cfg = config['program_synthesiser']  # This is a dataclass, not a dict

        self.synthesiser = NeuralProgramSynthesiser(
            d_model=ps_cfg.d_model,
            max_steps=ps_cfg.max_steps,
            n_heads=ps_cfg.n_heads,
            vocab_size=ps_cfg.vocab_size
)

        self.generator = ProgramGenerator(self.synthesiser)
        self.interpreter = DSLInterpreter()

    def synthesize_program(self, task_embedding: torch.Tensor,
                          beam_width: int = 5) -> List[Tuple[List[int], float]]:
        """Synthesize DSL programs using beam search."""
        programs = self.generator.generate_program_beam_search(
            task_embedding, beam_width, self.config['program_synthesiser']['max_steps']
        )

        # Score programs
        scored_programs = []
        for program in programs:
            # Validate and repair program
            valid_program = self.generator.validate_and_repair_program(program)

            # Simple scoring based on program length and validity
            score = 1.0 / (len(valid_program) + 1)  # Shorter programs get higher scores

            scored_programs.append((valid_program, score))

        return scored_programs

# === Hybrid Agent ===

class HybridAgent:
    """Main hybrid agent combining symbolic and neural reasoning."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DEFAULT_CONFIG.__dict__

        # Initialize components
        self.encoder = HybridEncoder(
            self.config['vit'],
            self.config['task_synthesiser']
        )

        self.symbolic_reasoner = SymbolicReasoner(self.config['tot'])
        self.neural_synthesiser = NeuralSynthesiser(self.config)
        self.memory = MetaProgramMemory(self.config)

        # Statistics
        self.stats = {
            'symbolic_success': 0,
            'neural_success': 0,
            'hybrid_success': 0,
            'total_attempts': 0,
            'symbolic_time': 0.0,
            'neural_time': 0.0,
        }

    def solve_task(self, train_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
                  test_input: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Solve an ARC task using hybrid reasoning."""
        self.stats['total_attempts'] += 1

        # Convert numpy arrays to tensors if needed
        train_pairs = [(torch.tensor(inp) if not isinstance(inp, torch.Tensor) else inp,
                       torch.tensor(out) if not isinstance(out, torch.Tensor) else out)
                      for inp, out in train_pairs]
        test_input = torch.tensor(test_input) if not isinstance(test_input, torch.Tensor) else test_input

        # Encode task
        all_grids = [inp for inp, _ in train_pairs] + [out for _, out in train_pairs] + [test_input]
        roles = torch.tensor([0] * len(train_pairs) + [1] * len(train_pairs) + [0])  # 0=input, 1=output

        with torch.no_grad():
            grid_embeddings = self.encoder(all_grids)
            task_embedding = self.encoder.task_synthesiser(grid_embeddings, roles)

        # Try memory lookup first
        memory_result = self._try_memory_lookup(task_embedding, test_input)
        if memory_result is not None:
            return memory_result, {'method': 'memory', 'confidence': 0.9}

        # Try symbolic reasoning first (if enabled)
        if self.config.get('symbolic_first', True):
            symbolic_result = self._try_symbolic_reasoning(test_input, train_pairs)
            if symbolic_result is not None:
                result_grid, steps, confidence = symbolic_result

                # Store in memory
                self._store_symbolic_trace(test_input, result_grid, steps, confidence, True)

                self.stats['symbolic_success'] += 1
                return result_grid, {
                    'method': 'symbolic',
                    'steps': steps,
                    'confidence': confidence
                }

        # Try neural synthesis
        neural_result = self._try_neural_synthesis(task_embedding, test_input, train_pairs)
        if neural_result is not None:
            result_grid, program, confidence = neural_result

            # Store in memory
            self._store_dsl_program(task_embedding, program, confidence, True)

            self.stats['neural_success'] += 1
            return result_grid, {
                'method': 'neural',
                'program': program,
                'confidence': confidence
            }

        # Fallback: try hybrid approach
        hybrid_result = self._try_hybrid_approach(test_input, task_embedding, train_pairs)
        if hybrid_result is not None:
            result_grid, method_info = hybrid_result

            self.stats['hybrid_success'] += 1
            return result_grid, method_info

        # Final fallback: return input unchanged
        logger.warning("All reasoning methods failed, returning input unchanged")
        return test_input, {'method': 'fallback', 'confidence': 0.0}

    def _try_memory_lookup(self, task_embedding: torch.Tensor,
                          test_input: torch.Tensor) -> Optional[torch.Tensor]:
        """Try to find solution in memory."""
        # Look for similar patterns
        similar_patterns = self.memory.find_similar_patterns(task_embedding, threshold=0.8)

        if similar_patterns:
            best_pattern, similarity = similar_patterns[0]

            # Try DSL programs from this pattern
            for dsl_program in best_pattern.dsl_programs:
                if dsl_program.success:
                    try:
                        result = self.neural_synthesiser.interpreter.execute_program(
                            dsl_program.program, test_input
                        )

                        # Check if result is reasonable
                        if not torch.allclose(result, test_input):
                            return result
                    except Exception:
                        continue

        return None

    def _try_symbolic_reasoning(self, test_input: torch.Tensor,
                              train_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Optional[Tuple[torch.Tensor, List[str], float]]:
        """Try symbolic reasoning."""
        start_time = time.time()

        try:
            # Use first training pair as target for symbolic reasoning
            if train_pairs:
                target_grid = train_pairs[0][1]
                result = self.symbolic_reasoner.reason_symbolically(
                    test_input, target_grid, train_pairs
                )

                self.stats['symbolic_time'] += time.time() - start_time
                return result
        except Exception as e:
            logger.debug(f"Symbolic reasoning failed: {e}")

        return None

    def _try_neural_synthesis(self, task_embedding: torch.Tensor,
                            test_input: torch.Tensor,
                            train_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Optional[Tuple[torch.Tensor, List[int], float]]:
        """Try neural program synthesis."""
        start_time = time.time()

        try:
            # Generate programs
            programs = self.neural_synthesiser.synthesize_program(task_embedding, beam_width=5)

            # Test each program
            for program, score in programs:
                try:
                    result = self.neural_synthesiser.interpreter.execute_program(program, test_input)

                    # Check if result is reasonable
                    if not torch.allclose(result, test_input):
                        # Validate against training pairs
                        confidence = self._validate_against_training(result, train_pairs)
                        if confidence > 0.5:
                            self.stats['neural_time'] += time.time() - start_time
                            return result, program, confidence
                except Exception:
                    continue
        except Exception as e:
            logger.debug(f"Neural synthesis failed: {e}")

        return None

    def _try_hybrid_approach(self, test_input: torch.Tensor,
                           task_embedding: torch.Tensor,
                           train_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Optional[Tuple[torch.Tensor, Dict[str, Any]]]:
        """Try hybrid approach combining symbolic and neural."""
        # Get similar symbolic traces
        similar_traces = self.memory.find_similar_traces(
            extract_features(test_input), threshold=0.7
        )

        if similar_traces:
            best_trace, similarity = similar_traces[0]

            # Apply symbolic operations from trace
            current_grid = test_input
            for operation in best_trace.operations:
                try:
                    if operation in symbolic_ops.operations:
                        op_func = symbolic_ops.get_operation(operation)
                        current_grid = op_func(current_grid)
                except Exception:
                    continue

            # Validate result
            confidence = self._validate_against_training(current_grid, train_pairs)
            if confidence > 0.3:
                return current_grid, {
                    'method': 'hybrid_symbolic',
                    'trace_id': best_trace.task_id,
                    'operations': best_trace.operations,
                    'confidence': confidence
                }

        # Try neural with symbolic hints
        similar_programs = self.memory.find_similar_programs(task_embedding, threshold=0.7)

        if similar_programs:
            best_program, similarity = similar_programs[0]

            try:
                result = self.neural_synthesiser.interpreter.execute_program(
                    best_program.program, test_input
                )

                confidence = self._validate_against_training(result, train_pairs)
                if confidence > 0.3:
                    return result, {
                        'method': 'hybrid_neural',
                        'program_id': best_program.task_id,
                        'confidence': confidence
                    }
            except Exception:
                pass

        return None

    def _validate_against_training(self, result: torch.Tensor,
                                 train_pairs: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
        """Validate result against training pairs."""
        if not train_pairs:
            return 0.5

        similarities = []
        for _, target in train_pairs:
            similarity = 1.0 - grid_score(result, target)
            similarities.append(similarity)

        return np.mean(similarities) if similarities else 0.5

    def _store_symbolic_trace(self, input_grid: torch.Tensor, output_grid: torch.Tensor,
                            operations: List[str], confidence: float, success: bool):
        """Store symbolic trace in memory."""
        trace = SymbolicTrace(
            task_id=f"symbolic_{len(self.memory.symbolic_traces)}",
            input_features=extract_features(input_grid),
            output_features=extract_features(output_grid),
            operations=operations,
            confidence=confidence,
            success=success
        )
        self.memory.add_symbolic_trace(trace)

    def _store_dsl_program(self, task_embedding: torch.Tensor, program: List[int],
                          confidence: float, success: bool):
        """Store DSL program in memory."""
        program_entry = DSLProgram(
            task_id=f"dsl_{len(self.memory.dsl_programs)}",
            task_embedding=task_embedding,
            program=program,
            confidence=confidence,
            success=success,
            execution_time=0.0  # Could track actual execution time
        )
        self.memory.add_dsl_program(program_entry)

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        total_attempts = self.stats['total_attempts']
        if total_attempts == 0:
            return self.stats

        return {
            **self.stats,
            'symbolic_success_rate': self.stats['symbolic_success'] / total_attempts,
            'neural_success_rate': self.stats['neural_success'] / total_attempts,
            'hybrid_success_rate': self.stats['hybrid_success'] / total_attempts,
            'avg_symbolic_time': self.stats['symbolic_time'] / max(1, self.stats['symbolic_success']),
            'avg_neural_time': self.stats['neural_time'] / max(1, self.stats['neural_success']),
            'memory_stats': self.memory.get_memory_stats()
        }

    def save_agent(self, filepath: str):
        """Save agent state."""
        state = {
            'config': self.config,
            'stats': self.stats,
            'encoder_state': self.encoder.state_dict(),
            'synthesiser_state': self.neural_synthesiser.synthesiser.state_dict(),
        }
        torch.save(state, filepath)

        # Save memory separately
        memory_filepath = filepath.replace('.pt', '_memory.pkl')
        self.memory.save_memory(memory_filepath)

    def load_agent(self, filepath: str):
        """Load agent state."""
        state = torch.load(filepath, map_location='cpu')

        self.config = state['config']
        self.stats = state['stats']
        self.encoder.load_state_dict(state['encoder_state'])
        self.neural_synthesiser.synthesiser.load_state_dict(state['synthesiser_state'])

        # Load memory
        memory_filepath = filepath.replace('.pt', '_memory.pkl')
        try:
            self.memory.load_memory(memory_filepath)
        except FileNotFoundError:
            logger.warning(f"Memory file not found: {memory_filepath}")

    def train(self, train_data: List[Tuple[List[Tuple[torch.Tensor, torch.Tensor]], torch.Tensor, torch.Tensor]],
             epochs: int = 10, learning_rate: float = 1e-4):
        """Train the neural components of the agent."""
        # This is a simplified training loop
        # In practice, you'd want more sophisticated training with proper loss functions

        optimizer = torch.optim.Adam([
            {'params': self.encoder.parameters()},
            {'params': self.neural_synthesiser.synthesiser.parameters()}
        ], lr=learning_rate)

        for epoch in range(epochs):
            total_loss = 0.0

            for train_pairs, test_input, target in train_data:
                optimizer.zero_grad()

                # Forward pass
                all_grids = [inp for inp, _ in train_pairs] + [out for _, out in train_pairs] + [test_input]
                roles = torch.tensor([0] * len(train_pairs) + [1] * len(train_pairs) + [0])

                grid_embeddings = self.encoder(all_grids)
                task_embedding = self.encoder.task_synthesiser(grid_embeddings, roles)

                # Simple reconstruction loss
                loss = F.mse_loss(grid_embeddings.mean(dim=0), task_embedding)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_data):.4f}")


'''