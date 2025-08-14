"""
Utility functions for ARC grid operations, feature extraction, and visualization
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple, Dict, Any, Optional
import json
import os
from datetime import datetime
import hashlib
import logging
# === Grid Operations ===

def grid_score(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Pixel-wise error (0 = perfect match, 1 = completely wrong)."""
    if pred.shape != target.shape:
        return torch.ones((), device=pred.device)  # max error
    return (pred != target).float().mean()  # scalar tensor

def extract_features(grid: torch.Tensor) -> torch.Tensor:
    """Extract basic features from grid."""
    features = []
    
    # Basic statistics
    features.append(torch.tensor([grid.shape[0], grid.shape[1]], dtype=torch.float))
    features.append(torch.tensor([len(torch.unique(grid))], dtype=torch.float))
    features.append(torch.tensor([(grid == 0).float().mean()], dtype=torch.float))  # sparsity
    
    # Color distribution (first 10 colors)
    color_dist = torch.zeros(10, dtype=torch.float)
    unique_colors, counts = torch.unique(grid, return_counts=True)
    for color, count in zip(unique_colors, counts):
        if color < 10:
            color_dist[color] = count.float()
    color_dist = color_dist / color_dist.sum()
    features.append(color_dist)
    
    # Geometric features
    non_zero = grid != 0
    if non_zero.any():
        rows = torch.any(non_zero, dim=1)
        cols = torch.any(non_zero, dim=0)
        r_indices = torch.where(rows)[0]
        c_indices = torch.where(cols)[0]
        
        if len(r_indices) > 0 and len(c_indices) > 0:
            bbox_area = (r_indices[-1] - r_indices[0] + 1) * (c_indices[-1] - c_indices[0] + 1)
            bbox_ratio = bbox_area / (grid.shape[0] * grid.shape[1])
        else:
            bbox_ratio = 0.0
    else:
        bbox_ratio = 0.0
    
    features.append(torch.tensor([bbox_ratio], dtype=torch.float))
    
    # Symmetry features
    symmetry_h = torch.allclose(grid, torch.fliplr(grid))
    symmetry_v = torch.allclose(grid, torch.flipud(grid))
    symmetry_r = torch.allclose(grid, torch.rot90(grid, 2))
    features.append(torch.tensor([symmetry_h, symmetry_v, symmetry_r], dtype=torch.float))
    
    return torch.cat(features)

def normalize_grid(grid: torch.Tensor) -> torch.Tensor:
    """Normalize grid to [0, 1] range."""
    if grid.dtype == torch.long:
        grid = grid.float()
    
    if grid.max() > 1:
        grid = grid / grid.max()
    
    return grid

def pad_grid(grid: torch.Tensor, target_shape: Tuple[int, int], pad_value: int = 0) -> torch.Tensor:
    """Pad grid to target shape."""
    h, w = grid.shape
    th, tw = target_shape
    
    if h > th or w > tw:
        # Crop if too large
        grid = grid[:min(h, th), :min(w, tw)]
        h, w = grid.shape
    
    # Create padded grid
    padded = torch.full(target_shape, pad_value, dtype=grid.dtype, device=grid.device)
    
    # Center the grid
    start_h = (th - h) // 2
    start_w = (tw - w) // 2
    padded[start_h:start_h + h, start_w:start_w + w] = grid
    
    return padded

def resize_grid(grid: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """Resize grid to target shape using interpolation."""
    if grid.shape == target_shape:
        return grid
    
    # Convert to float for interpolation
    grid_float = grid.float()
    
    # Add batch and channel dimensions
    grid_4d = grid_float.unsqueeze(0).unsqueeze(0)
    
    # Resize
    resized = F.interpolate(grid_4d, size=target_shape, mode='nearest')
    
    # Remove extra dimensions and convert back to integer
    return resized.squeeze().long()

# === Visualization ===

def plot_grid(grid: torch.Tensor, title: str = "", ax: Optional[plt.Axes] = None, 
             show_values: bool = False, color_map: Optional[Dict[int, str]] = None):
    """Plot a grid with optional color mapping."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Convert to numpy if needed
    if isinstance(grid, torch.Tensor):
        grid_np = grid.cpu().numpy()
    else:
        grid_np = grid
    
    # Default color map
    if color_map is None:
        colors = ['white', 'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray']
        color_map = {i: colors[i % len(colors)] for i in range(10)}
    
    # Create colored grid
    colored_grid = np.zeros((*grid_np.shape, 4))  # RGBA
    for i in range(grid_np.shape[0]):
        for j in range(grid_np.shape[1]):
            color = color_map.get(grid_np[i, j], 'black')
            if color == 'white':
                colored_grid[i, j] = [1, 1, 1, 1]
            elif color == 'red':
                colored_grid[i, j] = [1, 0, 0, 1]
            elif color == 'blue':
                colored_grid[i, j] = [0, 0, 1, 1]
            elif color == 'green':
                colored_grid[i, j] = [0, 1, 0, 1]
            elif color == 'yellow':
                colored_grid[i, j] = [1, 1, 0, 1]
            elif color == 'purple':
                colored_grid[i, j] = [0.5, 0, 0.5, 1]
            elif color == 'orange':
                colored_grid[i, j] = [1, 0.5, 0, 1]
            elif color == 'pink':
                colored_grid[i, j] = [1, 0.75, 0.8, 1]
            elif color == 'brown':
                colored_grid[i, j] = [0.6, 0.4, 0.2, 1]
            elif color == 'gray':
                colored_grid[i, j] = [0.5, 0.5, 0.5, 1]
            else:
                colored_grid[i, j] = [0, 0, 0, 1]
    
    # Plot
    ax.imshow(colored_grid)
    ax.set_title(title)
    ax.set_xticks(range(grid_np.shape[1]))
    ax.set_yticks(range(grid_np.shape[0]))
    ax.grid(True, which='both', color='black', linewidth=0.5)
    
    # Show values if requested
    if show_values:
        for i in range(grid_np.shape[0]):
            for j in range(grid_np.shape[1]):
                ax.text(j, i, str(grid_np[i, j]), ha='center', va='center', fontsize=8)

def plot_task(train_pairs: List[Tuple[torch.Tensor, torch.Tensor]], 
              test_input: torch.Tensor, 
              test_output: Optional[torch.Tensor] = None,
              prediction: Optional[torch.Tensor] = None):
    """Plot a complete ARC task."""
    n_train = len(train_pairs)
    n_cols = max(3, n_train + 1)  # At least 3 columns for train pairs + test
    
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 8))
    
    # Plot training pairs
    for i, (train_input, train_output) in enumerate(train_pairs):
        plot_grid(train_input, f"Train Input {i+1}", axes[0, i])
        plot_grid(train_output, f"Train Output {i+1}", axes[1, i])
    
    # Plot test input
    plot_grid(test_input, "Test Input", axes[0, n_train])
    
    # Plot test output or prediction
    if test_output is not None:
        plot_grid(test_output, "Test Output", axes[1, n_train])
    elif prediction is not None:
        plot_grid(prediction, "Prediction", axes[1, n_train])
    
    # Hide unused subplots
    for i in range(n_train + 1, n_cols):
        axes[0, i].set_visible(False)
        axes[1, i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_comparison(original: torch.Tensor, predicted: torch.Tensor, target: torch.Tensor):
    """Plot comparison between original, predicted, and target grids."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    plot_grid(original, "Original", axes[0])
    plot_grid(predicted, "Predicted", axes[1])
    plot_grid(target, "Target", axes[2])
    
    # Compute accuracy
    accuracy = 1.0 - grid_score(predicted, target).item()
    fig.suptitle(f"Accuracy: {accuracy:.2%}")
    
    plt.tight_layout()
    return fig

# === Data Loading ===

def load_arc_task(filepath: str) -> Dict[str, Any]:
    """Load an ARC task from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert to tensors
    train_pairs = []
    for pair in data.get('train', []):
        input_grid = torch.tensor(pair['input'])
        output_grid = torch.tensor(pair['output'])
        train_pairs.append((input_grid, output_grid))
    
    test_pairs = []
    for pair in data.get('test', []):
        input_grid = torch.tensor(pair['input'])
        output_grid = torch.tensor(pair['output'])
        test_pairs.append((input_grid, output_grid))
    
    return {
        'train': train_pairs,
        'test': test_pairs,
        'task_id': data.get('task_id', 'unknown')
    }

def save_results(results: Dict[str, Any], filepath: str):
    """Save evaluation results to JSON file."""
    # Convert tensors to lists for JSON serialization
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_tensors(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)

# === Evaluation Metrics ===

def compute_accuracy(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """Compute pixel-wise accuracy."""
    return 1.0 - grid_score(predicted, target).item()

def compute_metrics(predictions: List[torch.Tensor], 
                   targets: List[torch.Tensor],
                   methods: List[str]) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics."""
    metrics = {
        'accuracy': [],
        'method_breakdown': {},
        'reasoning_interpretability': 0,
        'search_budget_used': 0,
        'dsl_coverage': set(),
        'reasoning_depth_score': 0,
    }
    
    for pred, target, method in zip(predictions, targets, methods):
        accuracy = compute_accuracy(pred, target)
        metrics['accuracy'].append(accuracy)
        
        # Track method performance
        if method not in metrics['method_breakdown']:
            metrics['method_breakdown'][method] = {'count': 0, 'accuracy': []}
        
        metrics['method_breakdown'][method]['count'] += 1
        metrics['method_breakdown'][method]['accuracy'].append(accuracy)
    
    # Compute overall statistics
    metrics['overall_accuracy'] = np.mean(metrics['accuracy'])
    metrics['std_accuracy'] = np.std(metrics['accuracy'])
    
    # Compute method-specific statistics
    for method, data in metrics['method_breakdown'].items():
        data['avg_accuracy'] = np.mean(data['accuracy'])
        data['success_rate'] = sum(1 for acc in data['accuracy'] if acc > 0.9) / len(data['accuracy'])
    
    return metrics

# === Logging ===

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration."""
    import logging
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_task_results(task_id: str, method: str, accuracy: float, 
                    confidence: float, steps: Optional[List[str]] = None,
                    program: Optional[List[int]] = None):
    """Log results for a single task."""
    logger = logging.getLogger(__name__)
    
    log_msg = f"Task {task_id}: {method} (accuracy: {accuracy:.3f}, confidence: {confidence:.3f})"
    
    if steps:
        log_msg += f" - Steps: {steps}"
    
    if program:
        log_msg += f" - Program length: {len(program)}"
    
    logger.info(log_msg)

# === Performance Monitoring ===

class PerformanceMonitor:
    """Monitor performance metrics during evaluation."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {
            'total_tasks': 0,
            'successful_tasks': 0,
            'total_time': 0.0,
            'method_times': {},
            'method_successes': {},
        }
    
    def start_task(self):
        """Start timing a task."""
        self.start_time = datetime.now()
        self.metrics['total_tasks'] += 1
    
    def end_task(self, method: str, success: bool):
        """End timing a task."""
        if self.start_time is None:
            return
        
        duration = (datetime.now() - self.start_time).total_seconds()
        self.metrics['total_time'] += duration
        
        # Track method performance
        if method not in self.metrics['method_times']:
            self.metrics['method_times'][method] = []
            self.metrics['method_successes'][method] = 0
        
        self.metrics['method_times'][method].append(duration)
        if success:
            self.metrics['successful_tasks'] += 1
            self.metrics['method_successes'][method] += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'total_tasks': self.metrics['total_tasks'],
            'success_rate': self.metrics['successful_tasks'] / self.metrics['total_tasks'] if self.metrics['total_tasks'] > 0 else 0,
            'avg_time_per_task': self.metrics['total_time'] / self.metrics['total_tasks'] if self.metrics['total_tasks'] > 0 else 0,
            'method_breakdown': {}
        }
        
        for method in self.metrics['method_times']:
            method_times = self.metrics['method_times'][method]
            method_successes = self.metrics['method_successes'][method]
            method_total = len(method_times)
            
            summary['method_breakdown'][method] = {
                'count': method_total,
                'success_rate': method_successes / method_total if method_total > 0 else 0,
                'avg_time': np.mean(method_times) if method_times else 0,
                'std_time': np.std(method_times) if method_times else 0,
            }
        
        return summary

# === Configuration ===

def load_config(config_file: str) -> Dict[str, Any]:
    """Load configuration from file."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    return config

def save_config(config: Dict[str, Any], config_file: str):
    """Save configuration to file."""
    os.makedirs(os.path.dirname(config_file), exist_ok=True)
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

# === Utility Functions ===

def create_task_id(grid: torch.Tensor) -> str:
    """Create a unique task ID based on grid content."""
    # Simple hash of grid content
    grid_bytes = grid.cpu().numpy().tobytes()
    return hashlib.md5(grid_bytes).hexdigest()[:8]

def is_valid_grid(grid: torch.Tensor) -> bool:
    """Check if grid is valid (non-empty, reasonable size)."""
    if grid is None:
        return False
    
    if not isinstance(grid, torch.Tensor):
        return False
    
    if grid.dim() != 2:
        return False
    
    if grid.shape[0] == 0 or grid.shape[1] == 0:
        return False
    
    if grid.shape[0] > 100 or grid.shape[1] > 100:  # Reasonable size limit
        return False
    
    return True

def get_grid_statistics(grid: torch.Tensor) -> Dict[str, Any]:
    """Get comprehensive statistics about a grid."""
    if not is_valid_grid(grid):
        return {}
    
    stats = {
        'shape': tuple(grid.shape),
        'dtype': str(grid.dtype),
        'min_value': grid.min().item(),
        'max_value': grid.max().item(),
        'unique_values': len(torch.unique(grid)),
        'sparsity': (grid == 0).float().mean().item(),
        'total_pixels': grid.numel(),
        'non_zero_pixels': (grid != 0).sum().item(),
    }
    
    # Color distribution
    unique_colors, counts = torch.unique(grid, return_counts=True)
    color_dist = dict(zip(unique_colors.tolist(), counts.tolist()))
    stats['color_distribution'] = color_dist
    
    # Dominant color
    if len(unique_colors) > 0:
        dominant_color = unique_colors[counts.argmax()].item()
        stats['dominant_color'] = dominant_color
    
    return stats 