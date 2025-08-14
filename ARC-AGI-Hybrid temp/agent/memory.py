"""
Meta-Program Memory Module
Stores and retrieves successful symbolic traces and DSL programs for reuse
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import hashlib
import json
import pickle
from datetime import datetime

from config import DEFAULT_CONFIG

# === Memory Entry Types ===

@dataclass
class SymbolicTrace:
    """Represents a symbolic reasoning trace."""
    task_id: str
    input_features: torch.Tensor
    output_features: torch.Tensor
    operations: List[str]
    confidence: float
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'input_features': self.input_features.cpu().numpy().tolist(),
            'output_features': self.output_features.cpu().numpy().tolist(),
            'operations': self.operations,
            'confidence': self.confidence,
            'success': self.success,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SymbolicTrace':
        """Create from dictionary."""
        return cls(
            task_id=data['task_id'],
            input_features=torch.tensor(data['input_features']),
            output_features=torch.tensor(data['output_features']),
            operations=data['operations'],
            confidence=data['confidence'],
            success=data['success'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

@dataclass
class DSLProgram:
    """Represents a DSL program with metadata."""
    task_id: str
    task_embedding: torch.Tensor
    program: List[int]
    confidence: float
    success: bool
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'task_id': self.task_id,
            'task_embedding': self.task_embedding.cpu().numpy().tolist(),
            'program': self.program,
            'confidence': self.confidence,
            'success': self.success,
            'execution_time': self.execution_time,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DSLProgram':
        """Create from dictionary."""
        return cls(
            task_id=data['task_id'],
            task_embedding=torch.tensor(data['task_embedding']),
            program=data['program'],
            confidence=data['confidence'],
            success=data['success'],
            execution_time=data['execution_time'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )

@dataclass
class MetaProgram:
    """Represents a meta-program that can be applied to similar tasks."""
    pattern_id: str
    symbolic_traces: List[SymbolicTrace]
    dsl_programs: List[DSLProgram]
    pattern_features: torch.Tensor
    success_rate: float
    avg_confidence: float
    usage_count: int = 0
    last_used: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'pattern_id': self.pattern_id,
            'symbolic_traces': [trace.to_dict() for trace in self.symbolic_traces],
            'dsl_programs': [prog.to_dict() for prog in self.dsl_programs],
            'pattern_features': self.pattern_features.cpu().numpy().tolist(),
            'success_rate': self.success_rate,
            'avg_confidence': self.avg_confidence,
            'usage_count': self.usage_count,
            'last_used': self.last_used.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MetaProgram':
        """Create from dictionary."""
        return cls(
            pattern_id=data['pattern_id'],
            symbolic_traces=[SymbolicTrace.from_dict(t) for t in data['symbolic_traces']],
            dsl_programs=[DSLProgram.from_dict(p) for p in data['dsl_programs']],
            pattern_features=torch.tensor(data['pattern_features']),
            success_rate=data['success_rate'],
            avg_confidence=data['avg_confidence'],
            usage_count=data['usage_count'],
            last_used=datetime.fromisoformat(data['last_used'])
        )

# === Feature Similarity ===

class FeatureSimilarity:
    """Computes similarity between feature vectors."""
    
    @staticmethod
    def cosine_similarity(features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Compute cosine similarity between feature vectors."""
        if features1.dim() == 1:
            features1 = features1.unsqueeze(0)
        if features2.dim() == 1:
            features2 = features2.unsqueeze(0)
        
        # Normalize vectors
        features1_norm = F.normalize(features1, p=2, dim=1)
        features2_norm = F.normalize(features2, p=2, dim=1)
        
        # Compute cosine similarity
        similarity = torch.mm(features1_norm, features2_norm.t())
        return similarity.item()
    
    @staticmethod
    def euclidean_distance(features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Compute Euclidean distance between feature vectors."""
        if features1.dim() == 1:
            features1 = features1.unsqueeze(0)
        if features2.dim() == 1:
            features2 = features2.unsqueeze(0)
        
        distance = torch.cdist(features1, features2, p=2)
        return distance.item()
    
    @staticmethod
    def manhattan_distance(features1: torch.Tensor, features2: torch.Tensor) -> float:
        """Compute Manhattan distance between feature vectors."""
        if features1.dim() == 1:
            features1 = features1.unsqueeze(0)
        if features2.dim() == 1:
            features2 = features2.unsqueeze(0)
        
        distance = torch.cdist(features1, features2, p=1)
        return distance.item()

# === Pattern Clustering ===

class PatternCluster:
    """Clusters similar patterns together."""
    
    def __init__(self, cluster_id: str, centroid: torch.Tensor):
        self.cluster_id = cluster_id
        self.centroid = centroid
        self.members: List[MetaProgram] = []
        self.updated_at = datetime.now()
    
    def add_member(self, meta_program: MetaProgram):
        """Add a meta-program to this cluster."""
        self.members.append(meta_program)
        self._update_centroid()
        self.updated_at = datetime.now()
    
    def _update_centroid(self):
        """Update cluster centroid based on member features."""
        if not self.members:
            return
        
        features = torch.stack([mp.pattern_features for mp in self.members])
        self.centroid = features.mean(dim=0)
    
    def get_similarity(self, features: torch.Tensor) -> float:
        """Get similarity to cluster centroid."""
        return FeatureSimilarity.cosine_similarity(features, self.centroid)
    
    def get_best_programs(self, n: int = 5) -> List[Tuple[DSLProgram, float]]:
        """Get best DSL programs from this cluster."""
        all_programs = []
        for mp in self.members:
            for prog in mp.dsl_programs:
                if prog.success:
                    all_programs.append((prog, prog.confidence))
        
        # Sort by confidence and return top n
        all_programs.sort(key=lambda x: x[1], reverse=True)
        return all_programs[:n]
    
    def get_best_traces(self, n: int = 5) -> List[Tuple[SymbolicTrace, float]]:
        """Get best symbolic traces from this cluster."""
        all_traces = []
        for mp in self.members:
            for trace in mp.symbolic_traces:
                if trace.success:
                    all_traces.append((trace, trace.confidence))
        
        # Sort by confidence and return top n
        all_traces.sort(key=lambda x: x[1], reverse=True)
        return all_traces[:n]

# === Meta-Program Memory ===

class MetaProgramMemory:
    """Main memory system for storing and retrieving meta-programs."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or DEFAULT_CONFIG.__dict__
        
        # Storage
        self.symbolic_traces: Dict[str, SymbolicTrace] = {}
        self.dsl_programs: Dict[str, DSLProgram] = {}
        self.meta_programs: Dict[str, MetaProgram] = {}
        self.pattern_clusters: Dict[str, PatternCluster] = {}
        
        # Indexing
        self.feature_index: Dict[str, torch.Tensor] = {}
        self.operation_index: Dict[str, Set[str]] = defaultdict(set)
        self.success_index: Dict[str, List[str]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            'total_traces': 0,
            'total_programs': 0,
            'total_meta_programs': 0,
            'total_clusters': 0,
            'cache_hits': 0,
            'cache_misses': 0,
        }
    
    def add_symbolic_trace(self, trace: SymbolicTrace):
        """Add a symbolic trace to memory."""
        self.symbolic_traces[trace.task_id] = trace
        self.feature_index[trace.task_id] = trace.input_features
        
        # Index by operations
        for op in trace.operations:
            self.operation_index[op].add(trace.task_id)
        
        # Index by success
        if trace.success:
            self.success_index['symbolic'].append(trace.task_id)
        
        self.stats['total_traces'] += 1
    
    def add_dsl_program(self, program: DSLProgram):
        """Add a DSL program to memory."""
        self.dsl_programs[program.task_id] = program
        self.feature_index[program.task_id] = program.task_embedding
        
        # Index by success
        if program.success:
            self.success_index['dsl'].append(program.task_id)
        
        self.stats['total_programs'] += 1
    
    def create_meta_program(self, pattern_id: str, traces: List[SymbolicTrace], 
                          programs: List[DSLProgram]) -> MetaProgram:
        """Create a meta-program from similar traces and programs."""
        if not traces and not programs:
            raise ValueError("Cannot create meta-program without traces or programs")
        
        # Compute pattern features (average of all features)
        all_features = []
        if traces:
            all_features.extend([trace.input_features for trace in traces])
        if programs:
            all_features.extend([prog.task_embedding for prog in programs])
        
        pattern_features = torch.stack(all_features).mean(dim=0)
        
        # Compute success rate and average confidence
        total_items = len(traces) + len(programs)
        successful_items = sum(1 for t in traces if t.success) + sum(1 for p in programs if p.success)
        success_rate = successful_items / total_items if total_items > 0 else 0.0
        
        avg_confidence = (
            sum(t.confidence for t in traces) + sum(p.confidence for p in programs)
        ) / total_items if total_items > 0 else 0.0
        
        meta_program = MetaProgram(
            pattern_id=pattern_id,
            symbolic_traces=traces,
            dsl_programs=programs,
            pattern_features=pattern_features,
            success_rate=success_rate,
            avg_confidence=avg_confidence
        )
        
        self.meta_programs[pattern_id] = meta_program
        self.stats['total_meta_programs'] += 1
        
        return meta_program
    
    def find_similar_patterns(self, features: torch.Tensor, 
                            threshold: float = 0.7, 
                            max_results: int = 10) -> List[Tuple[MetaProgram, float]]:
        """Find similar patterns based on feature similarity."""
        similarities = []
        
        for pattern_id, meta_program in self.meta_programs.items():
            similarity = FeatureSimilarity.cosine_similarity(
                features, meta_program.pattern_features
            )
            if similarity >= threshold:
                similarities.append((meta_program, similarity))
        
        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def find_similar_traces(self, features: torch.Tensor, 
                          threshold: float = 0.7,
                          max_results: int = 10) -> List[Tuple[SymbolicTrace, float]]:
        """Find similar symbolic traces."""
        similarities = []
        
        for task_id, trace in self.symbolic_traces.items():
            similarity = FeatureSimilarity.cosine_similarity(
                features, trace.input_features
            )
            if similarity >= threshold:
                similarities.append((trace, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def find_similar_programs(self, task_embedding: torch.Tensor,
                            threshold: float = 0.7,
                            max_results: int = 10) -> List[Tuple[DSLProgram, float]]:
        """Find similar DSL programs."""
        similarities = []
        
        for task_id, program in self.dsl_programs.items():
            similarity = FeatureSimilarity.cosine_similarity(
                task_embedding, program.task_embedding
            )
            if similarity >= threshold:
                similarities.append((program, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:max_results]
    
    def get_best_programs_by_operation(self, operation: str, 
                                     max_results: int = 5) -> List[DSLProgram]:
        """Get best programs that use a specific operation."""
        task_ids = self.operation_index.get(operation, set())
        programs = []
        
        for task_id in task_ids:
            if task_id in self.dsl_programs:
                program = self.dsl_programs[task_id]
                if program.success:
                    programs.append(program)
        
        # Sort by confidence
        programs.sort(key=lambda p: p.confidence, reverse=True)
        return programs[:max_results]
    
    def get_successful_patterns(self, min_success_rate: float = 0.8) -> List[MetaProgram]:
        """Get patterns with high success rates."""
        successful = []
        
        for meta_program in self.meta_programs.values():
            if meta_program.success_rate >= min_success_rate:
                successful.append(meta_program)
        
        # Sort by success rate
        successful.sort(key=lambda mp: mp.success_rate, reverse=True)
        return successful
    
    def cluster_patterns(self, similarity_threshold: float = 0.8):
        """Cluster similar patterns together."""
        # Clear existing clusters
        self.pattern_clusters.clear()
        
        # Get all meta-programs
        meta_programs = list(self.meta_programs.values())
        if not meta_programs:
            return
        
        # Simple clustering: assign to nearest cluster or create new one
        for i, meta_program in enumerate(meta_programs):
            best_cluster = None
            best_similarity = 0.0
            
            # Find best matching cluster
            for cluster in self.pattern_clusters.values():
                similarity = cluster.get_similarity(meta_program.pattern_features)
                if similarity > best_similarity and similarity >= similarity_threshold:
                    best_similarity = similarity
                    best_cluster = cluster
            
            # Assign to best cluster or create new one
            if best_cluster is not None:
                best_cluster.add_member(meta_program)
            else:
                cluster_id = f"cluster_{len(self.pattern_clusters)}"
                new_cluster = PatternCluster(cluster_id, meta_program.pattern_features)
                new_cluster.add_member(meta_program)
                self.pattern_clusters[cluster_id] = new_cluster
        
        self.stats['total_clusters'] = len(self.pattern_clusters)
    
    def get_cluster_programs(self, features: torch.Tensor, 
                           max_results: int = 5) -> List[DSLProgram]:
        """Get programs from the most similar cluster."""
        if not self.pattern_clusters:
            return []
        
        # Find most similar cluster
        best_cluster = None
        best_similarity = 0.0
        
        for cluster in self.pattern_clusters.values():
            similarity = cluster.get_similarity(features)
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster
        
        if best_cluster is None:
            return []
        
        # Get best programs from cluster
        best_programs = best_cluster.get_best_programs(max_results)
        return [prog for prog, _ in best_programs]
    
    def cleanup_old_entries(self, max_age_days: int = 30):
        """Remove old entries to prevent memory bloat."""
        cutoff_date = datetime.now().replace(day=datetime.now().day - max_age_days)
        
        # Clean up symbolic traces
        old_traces = [task_id for task_id, trace in self.symbolic_traces.items()
                     if trace.timestamp < cutoff_date]
        for task_id in old_traces:
            del self.symbolic_traces[task_id]
            if task_id in self.feature_index:
                del self.feature_index[task_id]
        
        # Clean up DSL programs
        old_programs = [task_id for task_id, program in self.dsl_programs.items()
                       if program.timestamp < cutoff_date]
        for task_id in old_programs:
            del self.dsl_programs[task_id]
            if task_id in self.feature_index:
                del self.feature_index[task_id]
        
        # Rebuild indexes
        self._rebuild_indexes()
    
    def _rebuild_indexes(self):
        """Rebuild all indexes after cleanup."""
        # Clear indexes
        self.operation_index.clear()
        self.success_index.clear()
        
        # Rebuild operation index
        for trace in self.symbolic_traces.values():
            for op in trace.operations:
                self.operation_index[op].add(trace.task_id)
        
        # Rebuild success index
        for trace in self.symbolic_traces.values():
            if trace.success:
                self.success_index['symbolic'].append(trace.task_id)
        
        for program in self.dsl_programs.values():
            if program.success:
                self.success_index['dsl'].append(program.task_id)
    
    def save_memory(self, filepath: str):
        """Save memory to file."""
        data = {
            'symbolic_traces': {k: v.to_dict() for k, v in self.symbolic_traces.items()},
            'dsl_programs': {k: v.to_dict() for k, v in self.dsl_programs.items()},
            'meta_programs': {k: v.to_dict() for k, v in self.meta_programs.items()},
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_memory(self, filepath: str):
        """Load memory from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Load symbolic traces
        self.symbolic_traces = {
            k: SymbolicTrace.from_dict(v) for k, v in data['symbolic_traces'].items()
        }
        
        # Load DSL programs
        self.dsl_programs = {
            k: DSLProgram.from_dict(v) for k, v in data['dsl_programs'].items()
        }
        
        # Load meta programs
        self.meta_programs = {
            k: MetaProgram.from_dict(v) for k, v in data['meta_programs'].items()
        }
        
        # Load stats
        self.stats = data['stats']
        
        # Rebuild indexes
        self._rebuild_indexes()
        
        # Rebuild clusters
        self.cluster_patterns()
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            **self.stats,
            'memory_size_mb': self._estimate_memory_size(),
            'feature_dimensions': len(self.feature_index) if self.feature_index else 0,
            'unique_operations': len(self.operation_index),
            'successful_symbolic': len(self.success_index['symbolic']),
            'successful_dsl': len(self.success_index['dsl'])
        }
    
    def _estimate_memory_size(self) -> float:
        """Estimate memory size in MB."""
        total_size = 0
        
        # Estimate tensor sizes
        for features in self.feature_index.values():
            total_size += features.numel() * 4  # 4 bytes per float32
        
        # Estimate other data structures
        total_size += len(self.symbolic_traces) * 1024  # ~1KB per trace
        total_size += len(self.dsl_programs) * 512      # ~512B per program
        total_size += len(self.meta_programs) * 2048    # ~2KB per meta-program
        
        return total_size / (1024 * 1024)  # Convert to MB 