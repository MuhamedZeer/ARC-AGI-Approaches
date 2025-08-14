"""
Symbolic Operations Module
Combines symbolic transformations from Model A with DSL operations from Model B
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Callable, Any, Tuple, List, Optional
import cv2
from collections import Counter

# === Basic Geometric Transformations ===

def identity(grid: torch.Tensor) -> torch.Tensor:
    """Identity transformation."""
    return grid.clone()

def rotate90(grid: torch.Tensor) -> torch.Tensor:
    """Rotate grid 90 degrees clockwise."""
    return torch.rot90(grid, k=1, dims=[0, 1])

def rotate180(grid: torch.Tensor) -> torch.Tensor:
    """Rotate grid 180 degrees."""
    return torch.rot90(grid, k=2, dims=[0, 1])

def rotate270(grid: torch.Tensor) -> torch.Tensor:
    """Rotate grid 270 degrees clockwise (90 degrees counter-clockwise)."""
    return torch.rot90(grid, k=3, dims=[0, 1])

def flip_horizontal(grid: torch.Tensor) -> torch.Tensor:
    """Flip grid horizontally."""
    return torch.fliplr(grid)

def flip_vertical(grid: torch.Tensor) -> torch.Tensor:
    """Flip grid vertically."""
    return torch.flipud(grid)

def transpose(grid: torch.Tensor) -> torch.Tensor:
    """Transpose grid."""
    return grid.t()

# === Gap Filling Operations ===

def fill_horizontal_gap(grid: torch.Tensor) -> torch.Tensor:
    """Fill horizontal gaps between same-colored pixels."""
    new_grid = grid.clone()
    rows, cols = new_grid.shape
    
    for i in range(rows):
        row = new_grid[i].clone()
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

def fill_vertical_gap(grid: torch.Tensor) -> torch.Tensor:
    """Fill vertical gaps between same-colored pixels."""
    new_grid = grid.clone()
    rows, cols = new_grid.shape
    
    for j in range(cols):
        col = new_grid[:, j].clone()
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

# === Cropping and Padding Operations ===

def crop_nonzero(grid: torch.Tensor) -> torch.Tensor:
    """Crop grid to non-zero content."""
    nz = torch.nonzero(grid)
    if nz.size(0) == 0:
        return grid
    
    rmin, rmax = nz[:, 0].min(), nz[:, 0].max()
    cmin, cmax = nz[:, 1].min(), nz[:, 1].max()
    
    return grid[rmin:rmax+1, cmin:cmax+1]

def pad_to_shape(grid: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """Pad grid to target shape."""
    r, c = grid.shape
    tr, tc = target_shape
    
    if r > tr or c > tc:
        grid = grid[:min(r, tr), :min(c, tc)]
        r, c = grid.shape
    
    padded = torch.zeros(target_shape, dtype=grid.dtype, device=grid.device)
    r_start = (tr - r) // 2
    c_start = (tc - c) // 2
    padded[r_start:r_start+r, c_start:c_start+c] = grid
    
    return padded

def mirror_extend(grid: torch.Tensor, direction: str = 'right') -> torch.Tensor:
    """Extend grid by mirroring in specified direction."""
    if direction == 'right':
        return torch.cat([grid, torch.fliplr(grid)], dim=1)
    elif direction == 'left':
        return torch.cat([torch.fliplr(grid), grid], dim=1)
    elif direction == 'down':
        return torch.cat([grid, torch.flipud(grid)], dim=0)
    elif direction == 'up':
        return torch.cat([torch.flipud(grid), grid], dim=0)
    else:
        return grid

# === Color Operations ===

def apply_color_mapping(grid: torch.Tensor, color_map: Dict[int, int]) -> torch.Tensor:
    """Apply color mapping to grid."""
    new_grid = grid.clone()
    for old_color, new_color in color_map.items():
        new_grid[grid == old_color] = new_color
    return new_grid

def scale_grid(grid: torch.Tensor, factor: float) -> torch.Tensor:
    """Scale grid by factor using interpolation."""
    if factor == 1.0:
        return grid
    
    h, w = grid.shape
    new_h, new_w = int(h * factor), int(w * factor)
    
    # Convert to float for interpolation
    grid_float = grid.float()
    
    # Resize using bilinear interpolation
    resized = F.interpolate(
        grid_float.unsqueeze(0).unsqueeze(0),
        size=(new_h, new_w),
        mode='bilinear',
        align_corners=False
    )
    
    # Convert back to integer
    return resized.squeeze().round().long()

# === OpenCV-based Operations ===

def cv2_resize(grid: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
    """Resize grid using OpenCV."""
    grid_np = grid.cpu().numpy().astype(np.uint8)
    resized = cv2.resize(grid_np, target_shape[::-1], interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(resized).to(grid.device)

def cv2_dilate(grid: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """Dilate grid using OpenCV."""
    grid_np = grid.cpu().numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(grid_np, kernel, iterations=iterations)
    return torch.from_numpy(dilated).to(grid.device)

def cv2_erode(grid: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """Erode grid using OpenCV."""
    grid_np = grid.cpu().numpy().astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(grid_np, kernel, iterations=iterations)
    return torch.from_numpy(eroded).to(grid.device)

# === Filter Operations ===

def median_filter_transform(grid: torch.Tensor) -> torch.Tensor:
    """Apply median filter to grid."""
    grid_np = grid.cpu().numpy().astype(np.uint8)
    filtered = cv2.medianBlur(grid_np, 3)
    return torch.from_numpy(filtered).to(grid.device)

def inversion_transform(grid: torch.Tensor) -> torch.Tensor:
    """Invert colors in grid (0 becomes 9, 1 becomes 8, etc.)."""
    inverted = torch.zeros_like(grid)
    for i in range(10):  # Assuming 10 colors
        inverted[grid == i] = 9 - i
    return inverted

def fill_with_mode_transform(grid: torch.Tensor) -> torch.Tensor:
    """Fill empty cells with mode of surrounding cells."""
    new_grid = grid.clone()
    rows, cols = grid.shape
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:  # Empty cell
                # Get surrounding cells
                neighbors = []
                for di in [-1, 0, 1]:
                    for dj in [-1, 0, 1]:
                        ni, nj = i + di, j + dj
                        if (0 <= ni < rows and 0 <= nj < cols and 
                            grid[ni, nj] != 0):
                            neighbors.append(grid[ni, nj].item())
                
                if neighbors:
                    # Find mode
                    counter = Counter(neighbors)
                    mode = counter.most_common(1)[0][0]
                    new_grid[i, j] = mode
    
    return new_grid

# === Advanced Pattern Operations ===

def region_fill_transform(grid: torch.Tensor) -> torch.Tensor:
    """Fill regions based on connectivity."""
    new_grid = grid.clone()
    rows, cols = grid.shape
    
    def flood_fill(start_i: int, start_j: int, target_color: int, replacement_color: int):
        """Flood fill from starting position."""
        if (start_i < 0 or start_i >= rows or start_j < 0 or start_j >= cols or
            new_grid[start_i, start_j] != target_color):
            return
        
        new_grid[start_i, start_j] = replacement_color
        
        # Recursively fill neighbors
        flood_fill(start_i + 1, start_j, target_color, replacement_color)
        flood_fill(start_i - 1, start_j, target_color, replacement_color)
        flood_fill(start_i, start_j + 1, target_color, replacement_color)
        flood_fill(start_i, start_j - 1, target_color, replacement_color)
    
    # Find connected components and fill
    visited = torch.zeros_like(grid, dtype=torch.bool)
    
    for i in range(rows):
        for j in range(cols):
            if not visited[i, j] and grid[i, j] != 0:
                # Start flood fill
                flood_fill(i, j, grid[i, j], grid[i, j])
                visited[i, j] = True
    
    return new_grid

def row_mode_fill_transform(grid: torch.Tensor, threshold: float = 0.6) -> torch.Tensor:
    """Fill rows with mode if threshold is met."""
    new_grid = grid.clone()
    rows, cols = grid.shape
    
    for i in range(rows):
        row = grid[i]
        non_zero = row[row != 0]
        
        if len(non_zero) > threshold * cols:
            # Find mode of non-zero elements
            counter = Counter(non_zero.tolist())
            mode = counter.most_common(1)[0][0]
            
            # Fill empty cells with mode
            new_grid[i, row == 0] = mode
    
    return new_grid

def diagonal_propagate_transform(grid: torch.Tensor) -> torch.Tensor:
    """Propagate colors along diagonals."""
    new_grid = grid.clone()
    rows, cols = grid.shape
    
    # Propagate along main diagonal
    for i in range(min(rows, cols)):
        if grid[i, i] != 0:
            # Propagate to neighbors
            for di, dj in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                ni, nj = i + di, i + dj
                if (0 <= ni < rows and 0 <= nj < cols and new_grid[ni, nj] == 0):
                    new_grid[ni, nj] = grid[i, i]
    
    return new_grid

# === DSL-Compatible Operations ===

def paint_pixel(grid: torch.Tensor, x: int, y: int, color: int) -> torch.Tensor:
    """Paint a single pixel."""
    new_grid = grid.clone()
    if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0]:
        new_grid[y, x] = color
    return new_grid

def fill_color(grid: torch.Tensor, old_color: int, new_color: int) -> torch.Tensor:
    """Fill all pixels of old_color with new_color."""
    new_grid = grid.clone()
    new_grid[grid == old_color] = new_color
    return new_grid

def copy_region(grid: torch.Tensor, x0: int, y0: int, w: int, h: int, dx: int, dy: int) -> torch.Tensor:
    """Copy a region from (x0, y0) to (x0+dx, y0+dy)."""
    new_grid = grid.clone()
    rows, cols = grid.shape
    
    # Source region
    src_x0, src_y0 = max(0, x0), max(0, y0)
    src_x1, src_y1 = min(cols, x0 + w), min(rows, y0 + h)
    
    # Destination region
    dst_x0, dst_y0 = max(0, x0 + dx), max(0, y0 + dy)
    dst_x1, dst_y1 = min(cols, x0 + dx + w), min(rows, y0 + dy + h)
    
    # Copy region
    src_region = grid[src_y0:src_y1, src_x0:src_x1]
    dst_h, dst_w = dst_y1 - dst_y0, dst_x1 - dst_x0
    
    if dst_h > 0 and dst_w > 0:
        # Resize if necessary
        if src_region.shape != (dst_h, dst_w):
            src_region = cv2_resize(src_region, (dst_h, dst_w))
        new_grid[dst_y0:dst_y1, dst_x0:dst_x1] = src_region
    
    return new_grid

# === Operation Registry ===

class SymbolicOperations:
    """Registry of all symbolic operations."""
    
    def __init__(self):
        self.operations = {
            # Basic geometric
            'identity': identity,
            'rotate90': rotate90,
            'rotate180': rotate180,
            'rotate270': rotate270,
            'flip_horizontal': flip_horizontal,
            'flip_vertical': flip_vertical,
            'transpose': transpose,
            
            # Gap filling
            'fill_horizontal_gap': fill_horizontal_gap,
            'fill_vertical_gap': fill_vertical_gap,
            
            # Cropping and padding
            'crop_nonzero': crop_nonzero,
            'pad_to_shape': pad_to_shape,
            'mirror_extend': mirror_extend,
            
            # Color operations
            'apply_color_mapping': apply_color_mapping,
            'scale_grid': scale_grid,
            
            # OpenCV operations
            'cv2_resize': cv2_resize,
            'cv2_dilate': cv2_dilate,
            'cv2_erode': cv2_erode,
            'median_filter': median_filter_transform,
            'invert': inversion_transform,
            'fill_with_mode': fill_with_mode_transform,
            
            # Advanced patterns
            'region_fill': region_fill_transform,
            'row_mode_fill': row_mode_fill_transform,
            'diagonal_propagate': diagonal_propagate_transform,
            
            # DSL operations
            'paint': paint_pixel,
            'fill': fill_color,
            'copy': copy_region,
        }
        
        # Operation metadata
        self.metadata = {
            'identity': {'args': 0, 'category': 'geometric'},
            'rotate90': {'args': 0, 'category': 'geometric'},
            'rotate180': {'args': 0, 'category': 'geometric'},
            'rotate270': {'args': 0, 'category': 'geometric'},
            'flip_horizontal': {'args': 0, 'category': 'geometric'},
            'flip_vertical': {'args': 0, 'category': 'geometric'},
            'transpose': {'args': 0, 'category': 'geometric'},
            
            'fill_horizontal_gap': {'args': 0, 'category': 'pattern'},
            'fill_vertical_gap': {'args': 0, 'category': 'pattern'},
            
            'crop_nonzero': {'args': 0, 'category': 'geometric'},
            'pad_to_shape': {'args': 2, 'category': 'geometric'},
            'mirror_extend': {'args': 1, 'category': 'geometric'},
            
            'apply_color_mapping': {'args': 1, 'category': 'color'},
            'scale_grid': {'args': 1, 'category': 'geometric'},
            
            'cv2_resize': {'args': 2, 'category': 'geometric'},
            'cv2_dilate': {'args': 2, 'category': 'morphological'},
            'cv2_erode': {'args': 2, 'category': 'morphological'},
            'median_filter': {'args': 0, 'category': 'filter'},
            'invert': {'args': 0, 'category': 'color'},
            'fill_with_mode': {'args': 0, 'category': 'pattern'},
            
            'region_fill': {'args': 0, 'category': 'pattern'},
            'row_mode_fill': {'args': 1, 'category': 'pattern'},
            'diagonal_propagate': {'args': 0, 'category': 'pattern'},
            
            'paint': {'args': 3, 'category': 'pixel'},
            'fill': {'args': 2, 'category': 'color'},
            'copy': {'args': 6, 'category': 'geometric'},
        }
    
    def get_operation(self, name: str) -> Callable:
        """Get operation by name."""
        return self.operations.get(name)
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """Get operation metadata."""
        return self.metadata.get(name, {})
    
    def list_operations(self, category: Optional[str] = None) -> List[str]:
        """List operations, optionally filtered by category."""
        if category is None:
            return list(self.operations.keys())
        else:
            return [name for name, meta in self.metadata.items() 
                   if meta.get('category') == category]
    
    def apply_operation(self, name: str, grid: torch.Tensor, *args) -> torch.Tensor:
        """Apply operation to grid."""
        op = self.get_operation(name)
        if op is None:
            raise ValueError(f"Unknown operation: {name}")
        
        return op(grid, *args)
    
    def get_operation_signature(self, name: str) -> Tuple[int, str]:
        """Get operation signature (num_args, category)."""
        meta = self.get_metadata(name)
        return meta.get('args', 0), meta.get('category', 'unknown')

# Global instance
symbolic_ops = SymbolicOperations() 