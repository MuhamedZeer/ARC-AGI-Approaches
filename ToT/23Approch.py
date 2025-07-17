# Enhanced ARC Agent with Advanced Tree-of-Thought (ToT) and Meta-Programming
# Integrated with comprehensive transformation functions from Functions.py
# Features: Dynamic beam width, tree pruning, loop detection, advanced CNN embeddings

import os, json, numpy as np
from typing import List, Callable, Dict, Any, Tuple, Optional, Set
import tensorflow as tf
from tensorflow import keras
from dataclasses import dataclass, field
import itertools
from collections import Counter, deque
import logging
import cv2
import hashlib
from scipy.ndimage import convolve
from joblib import Parallel, delayed
import argparse
import time
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

layers = keras.layers
models = keras.models

# === Enhanced Data Structures ===
@dataclass
class TransformationStep:
    operation: str
    parameters: Dict[str, Any]
    confidence: float
    reasoning: str

@dataclass
class ProblemAnalysis:
    input_patterns: Dict[str, Any]
    output_patterns: Dict[str, Any]
    transformation_type: str
    complexity_score: float
    size_invariant: bool
    color_mapping: Dict[int, int]

# === Advanced Tree-of-Thought Node with Loop Detection ===
@dataclass(order=True)
class ThoughtNode:
    score: float
    grid: np.ndarray = field(compare=False)
    steps: List[str] = field(default_factory=list, compare=False)
    features: np.ndarray = field(default=None, compare=False)
    confidence: float = field(default=0.0, compare=False)
    grid_hash: str = field(default="", compare=False)  # For loop detection
    
    def __post_init__(self):
        if not self.grid_hash:
            self.grid_hash = self._compute_grid_hash()
    
    def _compute_grid_hash(self) -> str:
        """Compute hash of grid for loop detection"""
        return hashlib.md5(self.grid.tobytes()).hexdigest()

# === Comprehensive Transformation Functions from Functions.py ===
def identity(grid: np.ndarray) -> np.ndarray:
    return grid.copy()

def rotate90(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 1)

def rotate180(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 2)

def rotate270(grid: np.ndarray) -> np.ndarray:
    return np.rot90(grid, 3)

def flip_horizontal(grid: np.ndarray) -> np.ndarray:
    return np.fliplr(grid)

def flip_vertical(grid: np.ndarray) -> np.ndarray:
    return np.flipud(grid)

def transpose(grid: np.ndarray) -> np.ndarray:
    return np.transpose(grid)

def fill_horizontal_gap(grid: np.ndarray) -> np.ndarray:
    """Fill horizontal gaps between same-colored pixels"""
    new_grid = grid.copy()
    rows, cols = new_grid.shape
    for i in range(rows):
        row = new_grid[i].copy()
        start_idx = None
        for j in range(cols):
            if row[j] != 0:
                if start_idx is None:
                    start_idx = j
                else:
                    if row[j] == row[start_idx] and (j - start_idx) > 1:
                        row[start_idx+1:j] = row[start_idx]
                    start_idx = j
        new_grid[i] = row
    return new_grid

def fill_vertical_gap(grid: np.ndarray) -> np.ndarray:
    """Fill vertical gaps between same-colored pixels"""
    new_grid = grid.copy()
    rows, cols = new_grid.shape
    for j in range(cols):
        col = new_grid[:, j].copy()
        start_idx = None
        for i in range(rows):
            if col[i] != 0:
                if start_idx is None:
                    start_idx = i
                else:
                    if col[i] == col[start_idx] and (i - start_idx) > 1:
                        col[start_idx+1:i] = col[start_idx]
                    start_idx = i
        new_grid[:, j] = col
    return new_grid

def crop_nonzero(grid: np.ndarray) -> np.ndarray:
    """Crop grid to non-zero content"""
    nz = np.nonzero(grid)
    if nz[0].size == 0 or nz[1].size == 0:
        return grid
    rmin, rmax = nz[0].min(), nz[0].max()
    cmin, cmax = nz[1].min(), nz[1].max()
    return grid[rmin:rmax+1, cmin:cmax+1]

def pad_to_shape(grid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Pad grid to target shape"""
    r, c = grid.shape
    tr, tc = target_shape
    
    if r > tr or c > tc:
        grid = grid[:min(r, tr), :min(c, tc)]
        r, c = grid.shape
    
    padded = np.zeros(target_shape, dtype=grid.dtype)
    r_start = (tr - r) // 2
    c_start = (tc - c) // 2
    padded[r_start:r_start + r, c_start:c_start + c] = grid
    return padded

def mirror_extend(g, direction='right'):
    """Mirror and extend grid"""
    if direction == 'right':
        return np.hstack([g, np.fliplr(g)])
    elif direction == 'down':
        return np.vstack([g, np.flipud(g)])
    elif direction == 'both':
        temp = np.hstack([g, np.fliplr(g)])
        return np.vstack([temp, np.flipud(temp)])
    return g

def extract_connected_components(g):
    """Extract connected components of the same color"""
    components = []
    visited = np.zeros_like(g, dtype=bool)
    
    def flood_fill(start_r, start_c, color):
        stack = [(start_r, start_c)]
        component = []
        while stack:
            r, c = stack.pop()
            if (r < 0 or r >= g.shape[0] or c < 0 or c >= g.shape[1] or 
                visited[r, c] or g[r, c] != color):
                continue
            visited[r, c] = True
            component.append((r, c))
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                stack.append((r+dr, c+dc))
        return component
    
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if not visited[r, c] and g[r, c] != 0:
                comp = flood_fill(r, c, g[r, c])
                if comp:
                    components.append(comp)
    return components

def apply_color_mapping(g, color_map):
    """Apply color mapping transformation"""
    result = g.copy()
    for old_color, new_color in color_map.items():
        result[g == old_color] = new_color
    return result

def scale_grid(g, factor):
    """Scale grid by integer factor"""
    if factor == 1:
        return g.copy()
    return np.kron(g, np.ones((factor, factor), dtype=g.dtype))

# Advanced transformations from Functions.py
def cv2_resize(grid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(grid.astype(np.uint8), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST)

def cv2_dilate(grid: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(grid.astype(np.uint8), kernel, iterations=iterations)

def cv2_erode(grid: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(grid.astype(np.uint8), kernel, iterations=iterations)

def median_filter_transform(grid: np.ndarray) -> np.ndarray:
    return cv2.medianBlur(grid.astype(np.uint8), 3)

def inversion_transform(grid: np.ndarray) -> np.ndarray:
    max_val = grid.max()
    if max_val == 0:
        return grid.copy()
    return max_val - grid

def fill_with_mode_transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    nonzero = grid[grid != 0]
    if nonzero.size == 0:
        return out
    vals, counts = np.unique(nonzero, return_counts=True)
    mode_val = vals[np.argmax(counts)]
    out[grid != 0] = mode_val
    return out

def region_fill_transform(grid: np.ndarray) -> np.ndarray:
    grid_uint8 = grid.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(grid_uint8, connectivity=8)
    if num_labels <= 1:
        return grid_uint8
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_label = 1 + np.argmax(areas)
    mask = (labels == max_label).astype(np.uint8) * 255
    vals, counts = np.unique(grid_uint8[mask == 255], return_counts=True)
    mode_val = vals[np.argmax(counts)]
    filled = np.where(mask == 255, mode_val, grid_uint8)
    return filled

def row_mode_fill_transform(grid: np.ndarray, threshold: float = 0.6) -> np.ndarray:
    out = grid.copy()
    for i in range(out.shape[0]):
        row = out[i]
        nonzero = row[row != 0]
        if nonzero.size > 0:
            vals, counts = np.unique(nonzero, return_counts=True)
            mode_val = vals[np.argmax(counts)]
            ratio = np.max(counts) / nonzero.size
            if ratio >= threshold:
                out[i] = np.where(row != 0, mode_val, 0)
    return out

def diagonal_propagate_transform(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    rows, cols = grid.shape
    # Top-left propagation pass
    for i in range(1, rows):
        for j in range(1, cols):
            if out[i, j] == 0 and out[i-1, j-1] != 0:
                out[i, j] = out[i-1, j-1]
    # Top-right propagation pass
    for i in range(1, rows):
        for j in range(cols-2, -1, -1):
            if out[i, j] == 0 and out[i-1, j+1] != 0:
                out[i, j] = out[i-1, j+1]
    return out

# Task-specific transformations (key ones from Functions.py)
def transform_31d5ba1a(grid: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    mid = H // 2
    top_presence = grid[:mid] != 0
    bottom_presence = grid[mid:] != 0
    xor_mask = top_presence ^ bottom_presence
    return np.where(xor_mask, 6, 0).astype(grid.dtype)

def transform_c8b7cc0f(grid: np.ndarray) -> np.ndarray:
    colors, counts = np.unique(grid, return_counts=True)
    mask = (colors != 0) & (colors != 1)
    C = colors[mask][0]
    H, W = grid.shape
    bh, bw = H // 3, W // 3
    out = np.zeros((3, 3), dtype=grid.dtype)
    for i in range(3):
        for j in range(3):
            block = grid[i*bh:(i+1)*bh, j*bw:(j+1)*bw]
            if np.any(block == C):
                out[i, j] = C
    for i in range(3):
        row = out[i]
        idx = np.where(row == C)[0]
        if idx.size > 1:
            start, end = idx.min(), idx.max()
            row[start : end + 1] = C
    return out

def transform_custom(grid: np.ndarray) -> np.ndarray:
    g = grid.copy()
    H, W = g.shape
    vals, counts = np.unique(g[g != 0], return_counts=True)
    anchor_val = vals[counts == 1][0]
    r0, c0 = np.argwhere(g == anchor_val)[0]
    
    def scan_and_fill(dr, dc, paint=False):
        r, c = r0 + dr, c0 + dc
        while 0 <= r < H and 0 <= c < W and g[r, c] != 0:
            r += dr
            c += dc
        length = 0
        while 0 <= r < H and 0 <= c < W and g[r, c] == 0:
            if paint:
                g[r, c] = anchor_val
            length += 1
            r += dr
            c += dc
        return length
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    best_dir = max(directions, key=lambda d: scan_and_fill(*d))
    scan_and_fill(*best_dir, paint=True)
    return g
def connected_component_labels(grid: np.ndarray) -> np.ndarray:
    """
    Label connected components with unique integers.
    Different colors get separate labels even if touching.
    Returns labeled grid.
    """
    from scipy.ndimage import label
    structure = np.ones((3,3), dtype=int)  # 8-connectivity
    labeled, num_features = label(grid > 0, structure=structure)
    return labeled

def boundary_outline(grid: np.ndarray) -> np.ndarray:
    """
    Extract the boundary outline of shapes in the grid.
    Uses morphological gradient (dilation - erosion).
    """
    kernel = np.ones((3,3), dtype=np.uint8)
    dilated = cv2.dilate(grid.astype(np.uint8), kernel, iterations=1)
    eroded = cv2.erode(grid.astype(np.uint8), kernel, iterations=1)
    outline = dilated - eroded
    return outline

def color_cluster_simplify(grid: np.ndarray, n_clusters: int = 3) -> np.ndarray:
    """
    Simplify colors by clustering pixels into n_clusters based on color value.
    Uses KMeans clustering from sklearn.
    """
    from sklearn.cluster import KMeans
    pixels = grid.flatten().reshape(-1, 1)
    nonzero_mask = pixels != 0
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    clustered = pixels.copy()
    clustered[nonzero_mask] = kmeans.fit_predict(pixels[nonzero_mask])
    return clustered.reshape(grid.shape)

def fill_holes(grid: np.ndarray) -> np.ndarray:
    """
    Fill holes inside connected components.
    Uses binary fill holes from scipy.ndimage.
    """
    from scipy.ndimage import binary_fill_holes
    mask = grid > 0
    filled = binary_fill_holes(mask)
    result = grid.copy()
    result[filled & ~mask] = np.max(grid)  # fill holes with max color
    return result

def skeletonize_transform(grid: np.ndarray) -> np.ndarray:
    """
    Generate skeleton (thinned) representation of shapes.
    Uses skeletonize function from skimage.
    """
    from skimage.morphology import skeletonize
    binary = grid > 0
    skeleton = skeletonize(binary).astype(np.uint8) * np.max(grid)
    return skeleton

# === Comprehensive Primitives Dictionary ===
prims: Dict[str, Callable] = {
    # Basic geometric transformations
    'identity': identity,
    'rotate90': rotate90,
    'rotate180': rotate180,
    'rotate270': rotate270,
    'flip_h': flip_horizontal,
    'flip_v': flip_vertical,
    'transpose': transpose,
    
    # Advanced transformations
    'crop_to_content': crop_nonzero,
    'scale_2x': lambda g: cv2_resize(g, (g.shape[0]*2, g.shape[1]*2)),
    'scale_3x': lambda g: cv2_resize(g, (g.shape[0]*3, g.shape[1]*3)),
    'mirror_right': lambda g: mirror_extend(g, 'right'),
    'mirror_down': lambda g: mirror_extend(g, 'down'),
    'mirror_both': lambda g: mirror_extend(g, 'both'),
    'rotate90_then_flip': lambda g: flip_horizontal(rotate90(g)),
    'rotate270_then_flip': lambda g: flip_horizontal(rotate270(g)),
    'double_transpose': lambda g: transpose(transpose(g)),
    
    # Gap filling operations
    'fill_horizontal_gap': fill_horizontal_gap,
    'fill_vertical_gap': fill_vertical_gap,
    
    # Color and region operations
    'crop_nonzero': crop_nonzero,
    'pad_to_shape': lambda g: pad_to_shape(g, (g.shape[0]+2, g.shape[1]+2)),
    
    # Computer vision operations
    'cv2_dilate': lambda g: cv2_dilate(g, 3, 1),
    'cv2_erode': lambda g: cv2_erode(g, 3, 1),
    'median_filter_transform': median_filter_transform,
    'inversion_transform': inversion_transform,
    'fill_with_mode_transform': fill_with_mode_transform,
    'region_fill_transform': region_fill_transform,
    'row_mode_fill_transform': lambda g: row_mode_fill_transform(g, 0.6),
    
    # Advanced operations
    'diagonal_propagate_transform': diagonal_propagate_transform,
    
    # Task-specific transformations
    'transform_31d5ba1a': transform_31d5ba1a,
    'transform_c8b7cc0f': transform_c8b7cc0f,
    'transform_custom': transform_custom,

    'connected_component_labels': connected_component_labels,
    'boundary_outline': boundary_outline,
    'color_cluster_simplify': color_cluster_simplify,
    'fill_holes': fill_holes,
    'skeletonize_transform': skeletonize_transform,
}

# === Advanced Pattern Analysis ===
class PatternAnalyzer:
    @staticmethod
    def analyze_grid_properties(grid: np.ndarray) -> Dict[str, Any]:
        """Analyze comprehensive grid properties"""
        props = {
            'shape': grid.shape,
            'unique_colors': len(np.unique(grid)),
            'color_distribution': dict(zip(*np.unique(grid, return_counts=True))),
            'has_symmetry_h': np.array_equal(grid, np.fliplr(grid)),
            'has_symmetry_v': np.array_equal(grid, np.flipud(grid)),
            'has_rotational_symmetry': np.array_equal(grid, np.rot90(grid, 2)),
            'sparsity': np.sum(grid == 0) / grid.size,
            'connectivity': len(extract_connected_components(grid)),
            'bounding_box': PatternAnalyzer._get_bounding_box(grid),
            'dominant_color': PatternAnalyzer._get_dominant_color(grid),
        }
        return props
    
    @staticmethod
    def _get_bounding_box(grid: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of non-zero elements"""
        rows = np.any(grid != 0, axis=1)
        cols = np.any(grid != 0, axis=0)
        if not np.any(rows) or not np.any(cols):
            return (0, 0, 0, 0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return (rmin, rmax, cmin, cmax)
    
    @staticmethod
    def _get_dominant_color(grid: np.ndarray) -> int:
        """Get most frequent non-zero color"""
        colors, counts = np.unique(grid[grid != 0], return_counts=True)
        return colors[np.argmax(counts)] if len(colors) > 0 else 0
    
    @staticmethod
    def detect_transformation_type(input_grid: np.ndarray, output_grid: np.ndarray) -> str:
        """Detect type of transformation between input and output"""
        if np.array_equal(input_grid, output_grid):
            return "identity"
        
        if input_grid.shape != output_grid.shape:
            if output_grid.shape[0] * output_grid.shape[1] > input_grid.shape[0] * input_grid.shape[1]:
                return "expansion"
            else:
                return "contraction"
        
        # Check for geometric transformations
        if np.array_equal(np.rot90(input_grid, 1), output_grid):
            return "rotate90"
        if np.array_equal(np.rot90(input_grid, 2), output_grid):
            return "rotate180"
        if np.array_equal(np.rot90(input_grid, 3), output_grid):
            return "rotate270"
        if np.array_equal(np.fliplr(input_grid), output_grid):
            return "flip_horizontal"
        if np.array_equal(np.flipud(input_grid), output_grid):
            return "flip_vertical"
        
        # Check for color transformations
        unique_in = set(input_grid.flatten())
        unique_out = set(output_grid.flatten())
        if len(unique_in) == len(unique_out) and input_grid.shape == output_grid.shape:
            return "color_mapping"
        
        return "complex"

# === Advanced CNN Feature Extractor with Multi-Scale Features ===
def build_advanced_cnn(input_shape=(30, 30, 1), feat_dim=256):
    """Build sophisticated CNN with multi-scale feature extraction"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Multi-scale feature extraction
        # Scale 1: Fine details
        layers.Conv2D(64, 3, activation='relu', padding='same', name='conv1_3x3'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPool2D(2),
        layers.Dropout(0.1),
        
        # Scale 2: Medium patterns
        layers.Conv2D(128, 5, activation='relu', padding='same', name='conv2_5x5'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 5, activation='relu', padding='same'),
        layers.MaxPool2D(2),
        layers.Dropout(0.1),
        
        # Scale 3: Global patterns
        layers.Conv2D(256, 7, activation='relu', padding='same', name='conv3_7x7'),
        layers.BatchNormalization(),
        layers.Conv2D(256, 7, activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),
        
        # Attention mechanism
        layers.Dense(feat_dim, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(feat_dim // 2, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(feat_dim // 4, activation='relu'),
    ])
    
    return model

# Initialize advanced CNN
advanced_cnn = build_advanced_cnn()

def extract_advanced_features(grid: np.ndarray) -> np.ndarray:
    """Extract advanced features using multi-scale CNN"""
    H, W = grid.shape
    pad = np.zeros((30, 30), dtype=grid.dtype)
    pad[:min(H, 30), :min(W, 30)] = grid[:min(H, 30), :min(W, 30)]
    
    # Normalize
    pad_norm = pad.astype(np.float32) / 10.0
    
    pred = advanced_cnn(pad_norm.reshape(1, 30, 30, 1))
    return pred.numpy().flatten()

def extract_handcrafted_features(grid: np.ndarray) -> np.ndarray:
    """Extract handcrafted features for additional context"""
    features = []
    
    # Basic statistics
    features.extend([
        grid.shape[0], grid.shape[1],  # Dimensions
        len(np.unique(grid)),  # Unique colors
        np.sum(grid != 0) / grid.size,  # Sparsity
        np.std(grid),  # Standard deviation
    ])
    
    # Color distribution
    colors, counts = np.unique(grid, return_counts=True)
    color_dist = dict(zip(colors, counts))
    for i in range(10):  # Assume max 10 colors
        features.append(color_dist.get(i, 0))
    
    # Symmetry features
    features.extend([
        float(np.array_equal(grid, np.fliplr(grid))),  # Horizontal symmetry
        float(np.array_equal(grid, np.flipud(grid))),  # Vertical symmetry
        float(np.array_equal(grid, np.rot90(grid, 2))),  # Rotational symmetry
    ])
    
    # Connectivity features
    components = extract_connected_components(grid)
    features.extend([
        len(components),  # Number of components
        max([len(comp) for comp in components]) if components else 0,  # Largest component
    ])
    
    return np.array(features, dtype=np.float32)

def extract_combined_features(grid: np.ndarray) -> np.ndarray:
    """Combine CNN and handcrafted features"""
    cnn_features = extract_advanced_features(grid)
    handcrafted_features = extract_handcrafted_features(grid)
    return np.concatenate([cnn_features, handcrafted_features])

# === Enhanced Meta-Programming Engine ===
class MetaProgrammer:
    def __init__(self):
        self.program_templates = {
            "simple_geometric": ["apply_single_transform"],
            "color_mapping": ["detect_color_mapping", "apply_color_mapping"],
            "multi_step": ["analyze_structure", "apply_transforms", "validate_result"],
            "adaptive": ["try_multiple_strategies", "select_best_result"],
            "pattern_based": ["extract_pattern", "apply_pattern", "refine_result"],
            "hierarchical": ["coarse_transform", "fine_tune", "validate"]
        }
        
        # Operation categories for intelligent selection
        self.operation_categories = {
            'geometric': ['rotate90', 'rotate180', 'rotate270', 'flip_h', 'flip_v', 'transpose'],
            'scaling': ['scale_2x', 'scale_3x', 'crop_to_content'],
            'filling': ['fill_horizontal_gap', 'fill_vertical_gap', 'fill_with_mode_transform'],
            'morphological': ['cv2_dilate', 'cv2_erode', 'median_filter_transform'],
            'color': ['inversion_transform', 'region_fill_transform'],
            'advanced': ['diagonal_propagate_transform', 'transform_custom', 'transform_31d5ba1a']
        }
    
    def generate_programs(self, operations: List[str], complexity_limit: int = 4) -> List[List[str]]:
        """Generate program variations with complexity control"""
        if len(operations) <= 1:
            return [operations]
        
        programs = []
        for length in range(1, min(len(operations) + 1, complexity_limit + 1)):
            for combo in itertools.combinations(operations, length):
                for perm in itertools.permutations(combo):
                    programs.append(list(perm))
        
        return programs[:20]  # Limit to prevent explosion
    
    def synthesize_program_from_pairs(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[str]:
        """Synthesize program from training pairs with pattern analysis"""
        transformations = []
        for inp, out in train_pairs:
            transform_type = PatternAnalyzer.detect_transformation_type(inp, out)
            transformations.append(transform_type)
        
        # Find most common transformation
        most_common = Counter(transformations).most_common(1)[0][0]
        
        # Enhanced mapping with pattern analysis
        transform_map = {
            'rotate90': ['rotate90'],
            'rotate180': ['rotate180'],
            'rotate270': ['rotate270'],
            'flip_horizontal': ['flip_h'],
            'flip_vertical': ['flip_v'],
            'identity': ['identity'],
            'color_mapping': ['fill_with_mode_transform'],
            'expansion': ['scale_2x'],
            'contraction': ['crop_to_content'],
            'complex': self._generate_complex_program(train_pairs)
        }
        
        return transform_map.get(most_common, ['identity'])
    
    def _generate_complex_program(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[str]:
        """Generate complex multi-step program based on training patterns"""
        # Analyze patterns in training data
        patterns = []
        for inp, out in train_pairs:
            inp_props = PatternAnalyzer.analyze_grid_properties(inp)
            out_props = PatternAnalyzer.analyze_grid_properties(out)
            patterns.append((inp_props, out_props))
        
        # Determine appropriate operations based on pattern analysis
        operations = []
        
        # Check for size changes
        size_changes = [out_props['shape'] != inp_props['shape'] for inp_props, out_props in patterns]
        if any(size_changes):
            operations.extend(['crop_to_content', 'scale_2x'])
        
        # Check for color changes
        color_changes = [out_props['unique_colors'] != inp_props['unique_colors'] for inp_props, out_props in patterns]
        if any(color_changes):
            operations.extend(['fill_with_mode_transform', 'inversion_transform'])
        
        # Check for symmetry changes
        symmetry_changes = [out_props['has_symmetry_h'] != inp_props['has_symmetry_h'] for inp_props, out_props in patterns]
        if any(symmetry_changes):
            operations.extend(['flip_h', 'flip_v'])
        
        # Add advanced operations for complex patterns
        operations.extend(['transform_custom', 'diagonal_propagate_transform'])
        
        return operations[:3]  # Limit complexity
    
    def select_operations_by_category(self, category: str, max_ops: int = 3) -> List[str]:
        """Select operations from specific category"""
        if category in self.operation_categories:
            return self.operation_categories[category][:max_ops]
        return []

# === Enhanced Tree-of-Thought ARC Agent ===
class ToTArcAgent:
    def __init__(self):
        self.meta_programmer = MetaProgrammer()
        self.pattern_analyzer = PatternAnalyzer()
        self.max_depth = 5
        self.base_beam_width = 8
        self.max_beam_width = 15
        self.success_history = []
        self.visited_states: Set[str] = set()  # For loop detection
        
    def make_predictions(self, problem) -> List[np.ndarray]:
        """Enhanced prediction method using advanced Tree-of-Thought"""
        try:
            # Extract training data
            train_pairs = []
            for arc_set in problem.training_set():
                inp = arc_set.get_input_data().data()
                out = arc_set.get_output_data().data()
                train_pairs.append((inp, out))
            
            test_input = problem.test_set().get_input_data().data()
            
            # Clear visited states for new problem
            self.visited_states.clear()
            
            logger.info(f"Starting advanced ToT search for {problem.problem_name()}")
            
            # Use enhanced ToT search
            result = self._enhanced_tot_search(test_input, train_pairs)
            if result is not None:
                return [result]
            
            # Fallback to heuristic-based approach
            logger.warning(f"Enhanced ToT search failed for {problem.problem_name()}, using heuristics")
            return [self._heuristic_prediction(test_input, train_pairs)]
            
        except Exception as e:
            logger.error(f"Error in prediction for {problem.problem_name()}: {e}")
            return [test_input.copy()]
    
    def _enhanced_tot_search(self, test_input: np.ndarray, 
                           train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> Optional[np.ndarray]:
        """Enhanced Tree-of-Thought search with dynamic beam width and pruning"""
        
        # Generate candidate programs from training pairs
        candidate_ops = self.meta_programmer.synthesize_program_from_pairs(train_pairs)
        
        # Initialize search with root node
        root = ThoughtNode(
            score=0.0, 
            grid=test_input.copy(), 
            steps=[], 
            features=extract_combined_features(test_input),
            confidence=1.0
        )
        
        frontier = [root]
        best_solutions = []
        current_beam_width = self.base_beam_width
        
        # Validate candidate operations on training data first
        validated_ops = self._validate_operations_on_training(candidate_ops, train_pairs)
        
        for depth in range(self.max_depth):
            new_candidates = []
            
            # Dynamic beam width adjustment
            if depth > 0 and len(best_solutions) == 0:
                current_beam_width = min(current_beam_width + 2, self.max_beam_width)
                logger.info(f"Increasing beam width to {current_beam_width} at depth {depth}")
            
            for node in frontier:
                # Try validated operations first, then all operations
                ops_to_try = validated_ops if depth == 0 else list(prims.keys())
                
                for op_name in ops_to_try:
                    if op_name in prims:
                        try:
                            new_grid = prims[op_name](node.grid)
                            new_features = extract_combined_features(new_grid)
                            
                            # Loop detection
                            new_hash = hashlib.md5(new_grid.tobytes()).hexdigest()
                            if new_hash in self.visited_states:
                                continue
                            self.visited_states.add(new_hash)
                            
                            # Tree pruning by feature distance
                            if not self._passes_feature_threshold(new_features, train_pairs):
                                continue
                            
                            # Score based on training pattern similarity
                            score = self._score_against_training_patterns(
                                new_grid, new_features, train_pairs, node.steps + [op_name]
                            )
                            
                            confidence = self._calculate_confidence(node.steps + [op_name], train_pairs)
                            
                            new_candidates.append(ThoughtNode(
                                score=-score,  # Negative for sorting (higher is better)
                                grid=new_grid,
                                steps=node.steps + [op_name],
                                features=new_features,
                                confidence=confidence
                            ))
                            
                        except Exception as e:
                            continue
            
            # Select best candidates for next iteration
            frontier = sorted(new_candidates)[:current_beam_width]
            
            # Check if we found good solutions
            for node in frontier:
                if node.confidence > 0.8:  # High confidence threshold
                    best_solutions.append(node)
            
            if not frontier:  # No valid candidates
                break
        
        # Return best solution if found
        if best_solutions:
            best_solution = max(best_solutions, key=lambda x: x.confidence)
            logger.info(f"Found high-confidence solution: {best_solution.steps}")
            return best_solution.grid
        
        # Return best frontier candidate
        if frontier:
            logger.info(f"Returning best candidate: {frontier[0].steps}")
            return frontier[0].grid
        
        return None
    
    def _passes_feature_threshold(self, features: np.ndarray, 
                                train_pairs: List[Tuple[np.ndarray, np.ndarray]], 
                                threshold: float = 0.3) -> bool:
        """Check if features are similar enough to training outputs"""
        training_output_features = []
        for _, out in train_pairs:
            training_output_features.append(extract_combined_features(out))
        
        # Calculate similarity to training outputs
        similarities = []
        for train_feat in training_output_features:
            sim = -np.linalg.norm(features - train_feat)  # Negative distance
            similarities.append(sim)
        
        # Check if any similarity is above threshold
        max_sim = max(similarities) if similarities else float('-inf')
        return max_sim > threshold
    
    def _validate_operations_on_training(self, operations: List[str], 
                                       train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[str]:
        """Validate which operations work consistently on training data"""
        validated = []
        
        for op_name in operations:
            if op_name in prims:
                works_count = 0
                for inp, expected_out in train_pairs:
                    try:
                        result = prims[op_name](inp)
                        if np.array_equal(result, expected_out):
                            works_count += 1
                    except:
                        continue
                
                # If operation works on majority of training examples
                if works_count >= len(train_pairs) * 0.8:
                    validated.append(op_name)
        
        return validated
    
    def _score_against_training_patterns(self, grid: np.ndarray, features: np.ndarray,
                                       train_pairs: List[Tuple[np.ndarray, np.ndarray]],
                                       steps: List[str]) -> float:
        """Enhanced scoring against training patterns"""
        # Extract patterns from training outputs
        training_output_features = []
        for _, out in train_pairs:
            training_output_features.append(extract_combined_features(out))
        
        # Calculate similarity to training outputs
        similarities = []
        for train_feat in training_output_features:
            sim = -np.linalg.norm(features - train_feat)  # Negative distance
            similarities.append(sim)
        
        # Base score is max similarity to training outputs
        base_score = max(similarities) if similarities else 0.0
        
        # Bonus for validated operation sequences
        sequence_bonus = 0.0
        if len(steps) == 1 and steps[0] in self._get_validated_single_ops(train_pairs):
            sequence_bonus = 0.5
        
        # Penalty for complexity
        complexity_penalty = len(steps) * 0.1
        
        # Pattern consistency bonus
        pattern_bonus = self._calculate_pattern_consistency(grid, train_pairs)
        
        return base_score + sequence_bonus + pattern_bonus - complexity_penalty
    
    def _calculate_pattern_consistency(self, grid: np.ndarray, 
                                     train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Calculate how consistent the grid is with training patterns"""
        grid_props = PatternAnalyzer.analyze_grid_properties(grid)
        
        consistency_scores = []
        for _, out in train_pairs:
            out_props = PatternAnalyzer.analyze_grid_properties(out)
            
            # Compare properties
            color_similarity = 1.0 - abs(grid_props['unique_colors'] - out_props['unique_colors']) / 10.0
            sparsity_similarity = 1.0 - abs(grid_props['sparsity'] - out_props['sparsity'])
            symmetry_similarity = (
                (grid_props['has_symmetry_h'] == out_props['has_symmetry_h']) +
                (grid_props['has_symmetry_v'] == out_props['has_symmetry_v']) +
                (grid_props['has_rotational_symmetry'] == out_props['has_rotational_symmetry'])
            ) / 3.0
            
            consistency_scores.append((color_similarity + sparsity_similarity + symmetry_similarity) / 3.0)
        
        return max(consistency_scores) if consistency_scores else 0.0
    
    def _get_validated_single_ops(self, train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> List[str]:
        """Get single operations that work on training data"""
        validated = []
        for op_name, op_func in prims.items():
            works_for_all = True
            for inp, out in train_pairs:
                try:
                    if not np.array_equal(op_func(inp), out):
                        works_for_all = False
                        break
                except:
                    works_for_all = False
                    break
            if works_for_all:
                validated.append(op_name)
        return validated
    
    def _calculate_confidence(self, steps: List[str], 
                            train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> float:
        """Calculate confidence score for a sequence of steps"""
        # Test sequence on training data
        success_count = 0
        for inp, expected_out in train_pairs:
            try:
                result = inp.copy()
                for step in steps:
                    if step in prims:
                        result = prims[step](result)
                    else:
                        break
                
                if np.array_equal(result, expected_out):
                    success_count += 1
            except:
                continue
        
        # Confidence based on success rate
        success_rate = success_count / len(train_pairs) if train_pairs else 0.0
        
        # Adjust for sequence complexity
        complexity_factor = 1.0 / (1.0 + 0.2 * len(steps))
        
        return success_rate * complexity_factor
    
    def _heuristic_prediction(self, test_input: np.ndarray, 
                            train_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """Enhanced fallback heuristic prediction"""
        # Try simple operations that work on training data
        validated_ops = self._get_validated_single_ops(train_pairs)
        
        if validated_ops:
            # Use first validated operation
            op_name = validated_ops[0]
            return prims[op_name](test_input)
        
        # Try common transformations
        for op_name in ['identity', 'rotate90', 'flip_h', 'flip_v', 'crop_to_content']:
            if op_name in prims:
                works_for_all = True
                for inp, out in train_pairs:
                    try:
                        if not np.array_equal(prims[op_name](inp), out):
                            works_for_all = False
                            break
                    except:
                        works_for_all = False
                        break
                
                if works_for_all:
                    return prims[op_name](test_input)
        
        # Last resort: return input
        return test_input.copy()

# === Task Loading (Enhanced) ===
def load_tasks_from_dir(path: str) -> List[Dict[str, Any]]:
    """Enhanced task loading with error handling"""
    tasks = []
    for fname in sorted(os.listdir(path)):
        if fname.endswith('.json'):
            try:
                with open(os.path.join(path, fname), 'r') as f:
                    task = json.load(f)
                    task['filename'] = fname
                    tasks.append(task)
            except Exception as e:
                logger.error(f"Failed to load {fname}: {e}")
    return tasks

# === Enhanced Solver ===
def solve_task_enhanced(task: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Enhanced solver with advanced ToT approach"""
    try:
        # Create mock problem object for compatibility
        class MockProblem:
            def __init__(self, task_data):
                self.task_data = task_data
                self._name = task_data.get('filename', 'unknown')
            
            def problem_name(self):
                return self._name
            
            def training_set(self):
                return [MockArcSet(p) for p in self.task_data['train']]
            
            def test_set(self):
                return MockArcSet(self.task_data['test'][0])
        
        class MockArcSet:
            def __init__(self, data):
                self.data = data
            
            def get_input_data(self):
                return MockData(np.array(self.data['input']))
            
            def get_output_data(self):
                return MockData(np.array(self.data['output']))
        
        class MockData:
            def __init__(self, array):
                self.array = array
            
            def data(self):
                return self.array
        
        problem = MockProblem(task)
        agent = ToTArcAgent()
        
        # Get prediction using enhanced ToT
        predictions = agent.make_predictions(problem)
        prediction = predictions[0]
        
        # Get ground truth
        test_output = np.array(task['test'][0]['output'])
        
        # Evaluate
        correct = np.array_equal(prediction, test_output)
        
        analysis = {
            'correct': correct,
            'prediction_shape': prediction.shape,
            'expected_shape': test_output.shape,
            'task_name': task.get('filename', 'unknown')
        }
        
        return correct, analysis
        
    except Exception as e:
        logger.error(f"Task solving failed: {e}")
        return False, {'error': str(e)}

# === Enhanced Runner ===
def evaluate_all_enhanced(eval_dir: str) -> Dict[str, Any]:
    """Enhanced evaluation with detailed reporting"""
    tasks = load_tasks_from_dir(eval_dir)
    total = len(tasks)
    correct = 0
    results = []
    
    logger.info(f"Evaluating {total} tasks from {eval_dir}")
    
    for i, task in enumerate(tasks):
        logger.info(f"Processing task {i+1}/{total}: {task.get('filename', 'unknown')}")
        
        is_correct, analysis = solve_task_enhanced(task)
        if is_correct:
            correct += 1
        
        results.append(analysis)
        
        # Progress reporting
        if (i + 1) % 10 == 0:
            current_acc = correct / (i + 1)
            logger.info(f"Progress: {i+1}/{total}, Current accuracy: {current_acc:.2%}")
    
    final_accuracy = correct / total if total > 0 else 0.0
    
    summary = {
        'total_tasks': total,
        'correct_tasks': correct,
        'accuracy': final_accuracy,
        'results': results
    }
    
    logger.info(f"Final Results: {correct}/{total} = {final_accuracy:.2%}")
    
    return summary

# === Main Execution ===
if __name__ == "__main__":
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Enhanced ARC-AGI Agent with Tree-of-Thought')
    parser.add_argument('--eval_path', type=str, 
                       default=r"C:\Users\razan\Desktop\ARC-AGI\data\training1",
                       help='Path to evaluation data directory')
    parser.add_argument('--output', type=str, default='tot_results.json',
                       help='Output file for results')
    parser.add_argument('--max_tasks', type=int, default=None,
                       help='Maximum number of tasks to evaluate (for testing)')
    args = parser.parse_args()
    
    try:
        # Check if evaluation path exists
        if not os.path.exists(args.eval_path):
            logger.error(f"Evaluation path does not exist: {args.eval_path}")
            logger.info("Please provide a valid path using --eval_path argument")
            sys.exit(1)
        
        # Check if it's a directory
        if not os.path.isdir(args.eval_path):
            logger.error(f"Path is not a directory: {args.eval_path}")
            sys.exit(1)
        
        logger.info("=" * 60)
        logger.info("Enhanced ARC-AGI Agent with Tree-of-Thought")
        logger.info("=" * 60)
        logger.info(f"Evaluation path: {args.eval_path}")
        logger.info(f"Output file: {args.output}")
        if args.max_tasks:
            logger.info(f"Max tasks to evaluate: {args.max_tasks}")
        logger.info("=" * 60)
        
        # Start timing
        start_time = time.time()
        
        # Run evaluation
        logger.info("Starting enhanced ARC-AGI evaluation with Tree-of-Thought...")
        results = evaluate_all_enhanced(args.eval_path)
        
        # Calculate timing
        elapsed_time = time.time() - start_time
        
        # Save results
        logger.info(f"Saving results to {args.output}...")
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("EVALUATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total tasks evaluated: {results['total_tasks']}")
        logger.info(f"Correct predictions: {results['correct_tasks']}")
        logger.info(f"Accuracy: {results['accuracy']:.2%}")
        logger.info(f"Total time: {elapsed_time:.2f} seconds")
        if results['total_tasks'] > 0:
            logger.info(f"Average time per task: {elapsed_time/results['total_tasks']:.2f} seconds")
        logger.info(f"Results saved to: {args.output}")
        logger.info("=" * 60)
        
        # Exit with appropriate code
        if results['accuracy'] > 0.5:
            logger.info(" Good performance achieved!")
            sys.exit(0)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        logger.info("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during evaluation: {e}")
        logger.error("Please check your input data and try again")
        sys.exit(1)
