"""
Hybrid ARC Agent Package
Combines Tree-of-Thought symbolic reasoning with neural program synthesis
"""

from .HybridAgent import HybridAgent
from .encoder import HybridEncoder, GridEncoder, TaskSynthesiser
from .symbolic import symbolic_ops, SymbolicOperations
from .dsl import DSL, NeuralProgramSynthesiser, DSLInterpreter, ProgramGenerator
from .memory import MetaProgramMemory, SymbolicTrace, DSLProgram, MetaProgram
from .executor import GridExecutor, GridSandbox, ExecutionResult
from .utils import (
    grid_score, extract_features, plot_grid, plot_task, plot_comparison,
    load_arc_task, save_results, setup_logging, compute_metrics,
    PerformanceMonitor
)

__all__ = [
    # Main agent
    'HybridAgent',
    
    # Encoder components
    'HybridEncoder',
    'GridEncoder', 
    'TaskSynthesiser',
    
    # Symbolic reasoning
    'symbolic_ops',
    'SymbolicOperations',
    
    # DSL components
    'DSL',
    'NeuralProgramSynthesiser',
    'DSLInterpreter',
    'ProgramGenerator',
    
    # Memory system
    'MetaProgramMemory',
    'SymbolicTrace',
    'DSLProgram',
    'MetaProgram',
    
    # Execution
    'GridExecutor',
    'GridSandbox',
    'ExecutionResult',
    
    # Utilities
    'grid_score',
    'extract_features',
    'plot_grid',
    'plot_task',
    'plot_comparison',
    'load_arc_task',
    'save_results',
    'setup_logging',
    'compute_metrics',
    'PerformanceMonitor',
]

__version__ = "1.0.0"
__author__ = "Hybrid ARC Agent Team"
__description__ = "Hybrid AI agent combining symbolic reasoning with neural program synthesis for ARC tasks" 