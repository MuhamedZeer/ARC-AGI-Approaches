"""
Hybrid ARC Agent
A hybrid artificial intelligence agent that combines Tree-of-Thought symbolic reasoning 
with neural program synthesis to solve ARC (Abstraction and Reasoning Corpus) tasks.
"""

from .agent import HybridAgent
from .config import DEFAULT_CONFIG, HybridConfig

__version__ = "1.0.0"
__author__ = "Hybrid ARC Agent Team"
__description__ = "Hybrid AI agent combining symbolic reasoning with neural program synthesis for ARC tasks"

__all__ = [
    'HybridAgent',
    'DEFAULT_CONFIG',
    'HybridConfig',
] 