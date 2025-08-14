"""
Grid Executor Module
Executes both DSL programs and symbolic plans on ARC grids
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass
import time

from .symbolic import symbolic_ops
from .dsl import DSL, DSLInterpreter

# === Execution Context ===

@dataclass
class ExecutionContext:
    """Context for grid execution."""
    grid: torch.Tensor
    step: int = 0
    max_steps: int = 100
    timeout: float = 30.0  # seconds
    start_time: Optional[float] = None
    
    def __post_init__(self):
        if self.start_time is None:
            self.start_time = time.time()
    
    def check_timeout(self) -> bool:
        """Check if execution has timed out."""
        return time.time() - self.start_time > self.timeout
    
    def check_max_steps(self) -> bool:
        """Check if maximum steps reached."""
        return self.step >= self.max_steps

@dataclass
class ExecutionResult:
    """Result of grid execution."""
    final_grid: torch.Tensor
    success: bool
    steps_executed: int
    execution_time: float
    error_message: Optional[str] = None
    intermediate_states: Optional[List[torch.Tensor]] = None

# === Grid Executor ===

class GridExecutor:
    def __init__(self, max_grid_size: int = 30, max_colors: int = 10):
        self.max_grid_size = max_grid_size
        self.max_colors = max_colors
        self.dsl_interpreter = DSLInterpreter()
    
    def execute_symbolic_plan(self, grid: torch.Tensor, 
                            operations: List[str],
                            max_steps: int = 50,
                            timeout: float = 30.0) -> ExecutionResult:
        """Execute a sequence of symbolic operations."""
        context = ExecutionContext(
            grid=grid.clone(),
            max_steps=max_steps,
            timeout=timeout
        )
        
        intermediate_states = [context.grid.clone()]
        
        try:
            for i, operation in enumerate(operations):
                if context.check_timeout():
                    return ExecutionResult(
                        final_grid=context.grid,
                        success=False,
                        steps_executed=i,
                        execution_time=time.time() - context.start_time,
                        error_message="Execution timeout",
                        intermediate_states=intermediate_states
                    )
                
                if context.check_max_steps():
                    return ExecutionResult(
                        final_grid=context.grid,
                        success=False,
                        steps_executed=i,
                        execution_time=time.time() - context.start_time,
                        error_message="Maximum steps reached",
                        intermediate_states=intermediate_states
                    )
                
                # Validate grid before operation
                if not self._validate_grid(context.grid):
                    return ExecutionResult(
                        final_grid=context.grid,
                        success=False,
                        steps_executed=i,
                        execution_time=time.time() - context.start_time,
                        error_message="Invalid grid state",
                        intermediate_states=intermediate_states
                    )
                
                # Execute operation
                try:
                    if operation in symbolic_ops.operations:
                        op_func = symbolic_ops.get_operation(operation)
                        context.grid = op_func(context.grid)
                    else:
                        # Try DSL operation
                        context.grid = self._execute_dsl_operation(context.grid, operation)
                    
                    # Validate result
                    if not self._validate_grid(context.grid):
                        return ExecutionResult(
                            final_grid=context.grid,
                            success=False,
                            steps_executed=i + 1,
                            execution_time=time.time() - context.start_time,
                            error_message="Invalid result from operation",
                            intermediate_states=intermediate_states
                        )
                    
                    intermediate_states.append(context.grid.clone())
                    context.step += 1
                    
                except Exception as e:
                    return ExecutionResult(
                        final_grid=context.grid,
                        success=False,
                        steps_executed=i,
                        execution_time=time.time() - context.start_time,
                        error_message=f"Operation failed: {str(e)}",
                        intermediate_states=intermediate_states
                    )
            
            return ExecutionResult(
                final_grid=context.grid,
                success=True,
                steps_executed=len(operations),
                execution_time=time.time() - context.start_time,
                intermediate_states=intermediate_states
            )
            
        except Exception as e:
            return ExecutionResult(
                final_grid=context.grid,
                success=False,
                steps_executed=context.step,
                execution_time=time.time() - context.start_time,
                error_message=f"Execution error: {str(e)}",
                intermediate_states=intermediate_states
            )
    
    def execute_dsl_program(self, grid: torch.Tensor,
                          program: List[int],
                          max_steps: int = 50,
                          timeout: float = 30.0) -> ExecutionResult:
        """Execute a DSL program on a grid."""
        context = ExecutionContext(
            grid=grid.clone(),
            max_steps=max_steps,
            timeout=timeout
        )
        
        intermediate_states = [context.grid.clone()]
        
        try:
            # Execute program using DSL interpreter
            result_grid = self.dsl_interpreter.execute_program(program, context.grid)
            
            # Validate result
            if not self._validate_grid(result_grid):
                return ExecutionResult(
                    final_grid=result_grid,
                    success=False,
                    steps_executed=0,
                    execution_time=time.time() - context.start_time,
                    error_message="Invalid result from DSL program",
                    intermediate_states=intermediate_states
                )
            
            return ExecutionResult(
                final_grid=result_grid,
                success=True,
                steps_executed=len(program),
                execution_time=time.time() - context.start_time,
                intermediate_states=intermediate_states
            )
            
        except Exception as e:
            return ExecutionResult(
                final_grid=context.grid,
                success=False,
                steps_executed=0,
                execution_time=time.time() - context.start_time,
                error_message=f"DSL execution error: {str(e)}",
                intermediate_states=intermediate_states
            )
    
    def execute_hybrid_plan(self, grid: torch.Tensor,
                          symbolic_ops: List[str],
                          dsl_program: Optional[List[int]] = None,
                          max_steps: int = 50,
                          timeout: float = 30.0) -> ExecutionResult:
        """Execute a hybrid plan combining symbolic and DSL operations."""
        context = ExecutionContext(
            grid=grid.clone(),
            max_steps=max_steps,
            timeout=timeout
        )
        
        intermediate_states = [context.grid.clone()]
        
        try:
            # Execute symbolic operations first
            if symbolic_ops:
                symbolic_result = self.execute_symbolic_plan(
                    context.grid, symbolic_ops, max_steps, timeout
                )
                
                if not symbolic_result.success:
                    return symbolic_result
                
                context.grid = symbolic_result.final_grid
                intermediate_states.extend(symbolic_result.intermediate_states[1:])
                context.step += symbolic_result.steps_executed
            
            # Execute DSL program if provided
            if dsl_program:
                dsl_result = self.execute_dsl_program(
                    context.grid, dsl_program, max_steps - context.step, timeout
                )
                
                if not dsl_result.success:
                    return dsl_result
                
                context.grid = dsl_result.final_grid
                intermediate_states.extend(dsl_result.intermediate_states[1:])
                context.step += dsl_result.steps_executed
            
            return ExecutionResult(
                final_grid=context.grid,
                success=True,
                steps_executed=context.step,
                execution_time=time.time() - context.start_time,
                intermediate_states=intermediate_states
            )
            
        except Exception as e:
            return ExecutionResult(
                final_grid=context.grid,
                success=False,
                steps_executed=context.step,
                execution_time=time.time() - context.start_time,
                error_message=f"Hybrid execution error: {str(e)}",
                intermediate_states=intermediate_states
            )
    
    def _execute_dsl_operation(self, grid: torch.Tensor, operation: str) -> torch.Tensor:
        """Execute a single DSL operation."""
        # Convert operation string to DSL program
        if operation == "PAINT":
            # Simple paint operation (center pixel)
            h, w = grid.shape
            center_h, center_w = h // 2, w // 2
            program = [DSL.op2id("PAINT"), DSL.op2id("STOP")]
            # Add coordinates and color (simplified)
            program.insert(1, center_w)  # x
            program.insert(2, center_h)  # y
            program.insert(3, 1)  # color
        elif operation == "ROT90":
            program = [DSL.op2id("ROT90"), DSL.op2id("STOP")]
        elif operation == "FLIP_H":
            program = [DSL.op2id("MIRROR_X"), DSL.op2id("STOP")]
        elif operation == "FLIP_V":
            program = [DSL.op2id("MIRROR_Y"), DSL.op2id("STOP")]
        else:
            # Default to identity
            program = [DSL.op2id("STOP")]
        
        return self.dsl_interpreter.execute_program(program, grid)
    
    def _validate_grid(self, grid: torch.Tensor) -> bool:
        """Validate grid for safety and correctness."""
        if grid is None:
            return False
        
        if not isinstance(grid, torch.Tensor):
            return False
        
        if grid.dim() != 2:
            return False
        
        # Check size limits
        if grid.shape[0] > self.max_grid_size or grid.shape[1] > self.max_grid_size:
            return False
        
        if grid.shape[0] == 0 or grid.shape[1] == 0:
            return False
        
        # Check color limits
        if grid.max() >= self.max_colors or grid.min() < 0:
            return False
        
        # Check for NaN or infinite values
        if torch.isnan(grid).any() or torch.isinf(grid).any():
            return False
        
        return True

# === Sandbox Environment ===

class GridSandbox:
    """Sandbox environment for safe grid execution."""
    
    def __init__(self, max_grid_size: int = 100, max_colors: int = 10):
        self.executor = GridExecutor(max_grid_size, max_colors)
        self.execution_history = []
    
    def execute_plan(self, grid: torch.Tensor, 
                    plan: Union[List[str], List[int], Dict[str, Any]],
                    plan_type: str = "symbolic") -> ExecutionResult:
        """Execute a plan in the sandbox environment."""
        start_time = time.time()
        
        if plan_type == "symbolic":
            result = self.executor.execute_symbolic_plan(grid, plan)
        elif plan_type == "dsl":
            result = self.executor.execute_dsl_program(grid, plan)
        elif plan_type == "hybrid":
            symbolic_ops = plan.get("symbolic_ops", [])
            dsl_program = plan.get("dsl_program")
            result = self.executor.execute_hybrid_plan(grid, symbolic_ops, dsl_program)
        else:
            result = ExecutionResult(
                final_grid=grid,
                success=False,
                steps_executed=0,
                execution_time=time.time() - start_time,
                error_message=f"Unknown plan type: {plan_type}"
            )
        
        # Record execution
        self.execution_history.append({
            'plan_type': plan_type,
            'plan': plan,
            'result': result,
            'timestamp': time.time()
        })
        
        return result
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics."""
        if not self.execution_history:
            return {}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for h in self.execution_history if h['result'].success)
        
        plan_types = {}
        for history in self.execution_history:
            plan_type = history['plan_type']
            if plan_type not in plan_types:
                plan_types[plan_type] = {'total': 0, 'successful': 0}
            
            plan_types[plan_type]['total'] += 1
            if history['result'].success:
                plan_types[plan_type]['successful'] += 1
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': successful_executions / total_executions if total_executions > 0 else 0,
            'plan_type_breakdown': plan_types,
            'avg_execution_time': np.mean([h['result'].execution_time for h in self.execution_history])
        }
    
    def clear_history(self):
        """Clear execution history."""
        self.execution_history.clear()

# === Utility Functions ===

def validate_operation(operation: str, grid: torch.Tensor) -> bool:
    """Validate if an operation can be applied to a grid."""
    if operation not in symbolic_ops.operations:
        return False
    
    # Check operation-specific constraints
    if operation in ["pad_to_shape", "cv2_resize"]:
        # These operations require additional parameters
        return False
    
    return True

def get_operation_complexity(operation: str) -> int:
    """Get complexity score for an operation."""
    complexity_scores = {
        'identity': 1,
        'rotate90': 2,
        'rotate180': 2,
        'rotate270': 2,
        'flip_horizontal': 2,
        'flip_vertical': 2,
        'transpose': 3,
        'fill_horizontal_gap': 4,
        'fill_vertical_gap': 4,
        'crop_nonzero': 3,
        'apply_color_mapping': 3,
        'cv2_dilate': 5,
        'cv2_erode': 5,
        'median_filter': 6,
        'region_fill': 7,
        'diagonal_propagate': 8,
    }
    
    return complexity_scores.get(operation, 5)

def estimate_execution_time(operations: List[str]) -> float:
    """Estimate execution time for a sequence of operations."""
    base_time = 0.01  # Base time per operation
    total_complexity = sum(get_operation_complexity(op) for op in operations)
    return base_time * total_complexity

def optimize_operation_sequence(operations: List[str]) -> List[str]:
    """Optimize a sequence of operations by removing redundant ones."""
    if not operations:
        return operations
    
    optimized = []
    prev_op = None
    
    for op in operations:
        # Skip redundant operations
        if op == prev_op and op in ['rotate90', 'rotate180', 'rotate270']:
            continue
        
        # Combine consecutive rotations
        if op == 'rotate90' and prev_op == 'rotate270':
            optimized[-1] = 'rotate180'
            continue
        elif op == 'rotate270' and prev_op == 'rotate90':
            optimized[-1] = 'rotate180'
            continue
        
        optimized.append(op)
        prev_op = op
    
    return optimized 