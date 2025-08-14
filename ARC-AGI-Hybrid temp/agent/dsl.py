"""
Domain Specific Language (DSL) Module
Extends basic DSL with symbolic operations for hybrid reasoning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

from config import EXTENDED_DSL_OPS, MAX_CONST, CONST_OFFSET
from .symbolic import symbolic_ops

# === DSL Definition ===

class DSL:
    """Extended DSL with symbolic operations."""
    
    # Basic operations from Model B
    BASIC_OPS = [
        ("PAINT", 3),      # x, y, colour
        ("FILL", 2),       # colour_old, colour_new
        ("MIRROR_X", 0),   # no args
        ("MIRROR_Y", 0),   # no args
        ("ROT90", 0),      # no args
        ("ROT180", 0),     # no args
        ("COPY", 6),       # x0, y0, w, h, dx, dy
        ("STOP", 0),       # end program
    ]
    
    # Extended operations with symbolic transformations
    EXTENDED_OPS = EXTENDED_DSL_OPS
    
    # Build helper lists
    OPS = [name for name, _ in EXTENDED_OPS]
    N_ARGS = [n for _, n in EXTENDED_OPS]
    
    @classmethod
    def op2id(cls, name: str) -> int:
        """Return integer ID for an opcode name."""
        return cls.OPS.index(name)
    
    @classmethod
    def id2op(cls, idx: int) -> str:
        """Return opcode name for an integer ID."""
        return cls.OPS[idx]
    
    @classmethod
    def nargs(cls, idx: int) -> int:
        """Number of arguments that opcode idx expects."""
        return cls.N_ARGS[idx]
    
    @classmethod
    def is_valid_op(cls, name: str) -> bool:
        """Check if operation name is valid."""
        return name in cls.OPS
    
    @classmethod
    def get_op_category(cls, name: str) -> str:
        """Get category of operation."""
        if name in symbolic_ops.operations:
            return symbolic_ops.get_metadata(name).get('category', 'unknown')
        return 'basic'

# === Tokenizer ===

class ProgramTokenizer(nn.Module):
    """Token-to-vector embedder for program tokens."""
    
    def __init__(
        self,
        d_model: int,
        vocab_size: Optional[int] = None,
        pad_id: Optional[int] = None,
    ):
        super().__init__()
        
        if vocab_size is None:
            vocab_size = len(DSL.OPS) + MAX_CONST + 2  # ops + constants + NOOP + STOP
        
        self.vocab_size = vocab_size
        self.pad_id = pad_id if pad_id is not None else 0
        self.emb = nn.Embedding(self.vocab_size, d_model, padding_idx=self.pad_id)
    
    def forward(self, tok_ids: torch.LongTensor) -> torch.Tensor:
        """Args:
            tok_ids: LongTensor of shape [B, T]
        Returns:
            embeddings: [B, T, d_model] tensor
        """
        return self.emb(tok_ids)
    
    def expand_vocab(self, extra_tokens: int = 1):
        """Expand vocabulary size."""
        old_emb = self.emb
        new_vocab_size = self.vocab_size + extra_tokens
        
        self.vocab_size = new_vocab_size
        self.emb = nn.Embedding(new_vocab_size, old_emb.embedding_dim, padding_idx=self.pad_id)
        
        # Copy old embeddings
        with torch.no_grad():
            self.emb.weight[:old_emb.num_embeddings] = old_emb.weight

# === Program Synthesiser ===

class NeuralProgramSynthesiser(nn.Module):
    """Neural program synthesiser that generates DSL programs from task embeddings."""
    
    def __init__(
        self,
        d_model: int,
        max_steps: int = 10,
        n_heads: int = 8,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        
        if vocab_size is None:
            vocab_size = len(DSL.OPS) + MAX_CONST + 2
        
        self.d_model = d_model
        self.max_steps = max_steps
        self.vocab_size = vocab_size
        
        # Task embedding projection
        self.task_proj = nn.Linear(d_model, d_model)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # Tokenizer
        self.tokenizer = ProgramTokenizer(d_model, vocab_size)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Positional encoding
        self.pos_emb = nn.Parameter(torch.randn(1, max_steps, d_model))
    
    def forward(
        self,
        task_emb: torch.Tensor,          # shape [B, d]
        prev_toks: torch.LongTensor,     # shape [B, T]
    ) -> torch.Tensor:                   # returns logits [B, vocab_size]
        """Generate next token given task embedding and previous tokens."""
        batch_size = task_emb.size(0)
        
        # Project task embedding
        task_emb = self.task_proj(task_emb).unsqueeze(1)  # [B, 1, d]
        
        # Token embeddings
        tok_emb = self.tokenizer(prev_toks)  # [B, T, d]
        
        # Add positional encoding
        seq_len = tok_emb.size(1)
        pos_emb = self.pos_emb[:, :seq_len, :]
        tok_emb = tok_emb + pos_emb
        
        # Decode
        memory = task_emb
        decoded = self.decoder(tok_emb, memory)
        
        # Project to vocabulary
        logits = self.output_proj(decoded)  # [B, T, vocab_size]
        
        return logits
    
    def sample_step(
        self,
        task_emb: torch.Tensor,
        prev_toks: List[int],
        temperature: float = 1.0,
    ) -> int:
        """Sample next token."""
        if not prev_toks:
            prev_toks = [len(DSL.OPS)]  # NOOP token
        
        prev_tensor = torch.tensor([prev_toks], device=task_emb.device)
        
        with torch.no_grad():
            logits = self.forward(task_emb.unsqueeze(0), prev_tensor)
            logits = logits[0, -1] / temperature  # Last token logits
            
            # Apply softmax and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
        
        return next_token

# === DSL Parser ===

@dataclass
class DSLInstruction:
    """Represents a single DSL instruction."""
    opcode: str
    args: List[int]
    line_number: int

class DSLParser:
    """Parser for DSL programs."""
    
    def __init__(self):
        self.instructions = []
        self.current_line = 0
    
    def parse_program(self, program: List[int]) -> List[DSLInstruction]:
        """Parse a program from token IDs."""
        self.instructions = []
        self.current_line = 0
        i = 0
        
        while i < len(program):
            token = program[i]
            
            # Check if it's a constant
            if token >= CONST_OFFSET:
                # This should be an argument, not an opcode
                raise ValueError(f"Unexpected constant token {token} at position {i}")
            
            # Get opcode
            if token >= len(DSL.OPS):
                raise ValueError(f"Invalid opcode token {token}")
            
            opcode = DSL.id2op(token)
            n_args = DSL.nargs(token)
            
            # Get arguments
            args = []
            for j in range(n_args):
                arg_idx = i + 1 + j
                if arg_idx >= len(program):
                    raise ValueError(f"Missing argument for {opcode}")
                
                arg_token = program[arg_idx]
                if arg_token < CONST_OFFSET:
                    raise ValueError(f"Expected constant argument, got opcode {arg_token}")
                
                arg_value = arg_token - CONST_OFFSET
                args.append(arg_value)
            
            # Create instruction
            instruction = DSLInstruction(
                opcode=opcode,
                args=args,
                line_number=self.current_line
            )
            self.instructions.append(instruction)
            
            # Move to next instruction
            i += 1 + n_args
            self.current_line += 1
        
        return self.instructions
    
    def validate_program(self, program: List[int]) -> bool:
        """Validate program structure."""
        try:
            self.parse_program(program)
            return True
        except (ValueError, IndexError):
            return False

# === DSL Interpreter ===

class DSLInterpreter:
    """Interpreter for DSL programs."""
    
    def __init__(self):
        self.parser = DSLParser()
    
    def execute_program(self, program: List[int], grid: torch.Tensor) -> torch.Tensor:
        """Execute a DSL program on a grid."""
        try:
            instructions = self.parser.parse_program(program)
        except ValueError as e:
            # Invalid program, return original grid
            return grid
        
        current_grid = grid.clone()
        
        for instruction in instructions:
            try:
                current_grid = self._execute_instruction(instruction, current_grid)
            except Exception:
                # Skip invalid instructions
                continue
        
        return current_grid
    
    def _execute_instruction(self, instruction: DSLInstruction, grid: torch.Tensor) -> torch.Tensor:
        """Execute a single instruction."""
        opcode = instruction.opcode
        args = instruction.args
        
        # Check if it's a symbolic operation
        if opcode in symbolic_ops.operations:
            return symbolic_ops.apply_operation(opcode, grid, *args)
        
        # Handle basic DSL operations
        if opcode == "PAINT":
            x, y, color = args
            return symbolic_ops.apply_operation("paint", grid, x, y, color)
        
        elif opcode == "FILL":
            old_color, new_color = args
            return symbolic_ops.apply_operation("fill", grid, old_color, new_color)
        
        elif opcode == "MIRROR_X":
            return symbolic_ops.apply_operation("flip_horizontal", grid)
        
        elif opcode == "MIRROR_Y":
            return symbolic_ops.apply_operation("flip_vertical", grid)
        
        elif opcode == "ROT90":
            return symbolic_ops.apply_operation("rotate90", grid)
        
        elif opcode == "ROT180":
            return symbolic_ops.apply_operation("rotate180", grid)
        
        elif opcode == "COPY":
            x0, y0, w, h, dx, dy = args
            return symbolic_ops.apply_operation("copy", grid, x0, y0, w, h, dx, dy)
        
        elif opcode == "STOP":
            return grid
        
        else:
            # Unknown opcode, return grid unchanged
            return grid
    
    def execute_step(self, program: List[int], grid: torch.Tensor, step: int) -> torch.Tensor:
        """Execute program up to a specific step."""
        try:
            instructions = self.parser.parse_program(program)
        except ValueError:
            return grid
        
        current_grid = grid.clone()
        
        for i, instruction in enumerate(instructions):
            if i >= step:
                break
            
            try:
                current_grid = self._execute_instruction(instruction, current_grid)
            except Exception:
                continue
        
        return current_grid

# === Program Generator ===

class ProgramGenerator:
    """Generates DSL programs using various strategies."""
    
    def __init__(self, synthesiser: NeuralProgramSynthesiser):
        self.synthesiser = synthesiser
        self.interpreter = DSLInterpreter()
    
    def generate_program(
        self,
        task_emb: torch.Tensor,
        max_steps: int = 10,
        temperature: float = 1.0,
    ) -> List[int]:
        """Generate a program using the neural synthesiser."""
        program = []
        
        for step in range(max_steps):
            next_token = self.synthesiser.sample_step(
                task_emb, program, temperature
            )
            
            program.append(next_token)
            
            # Check if we should stop
            if next_token == DSL.op2id("STOP"):
                break
        
        return program
    
    def generate_program_beam_search(
        self,
        task_emb: torch.Tensor,
        beam_width: int = 5,
        max_steps: int = 10,
    ) -> List[List[int]]:
        """Generate programs using beam search."""
        beams = [([], 0.0)]  # (program, score)
        
        for step in range(max_steps):
            new_beams = []
            
            for program, score in beams:
                # Skip completed programs
                if program and program[-1] == DSL.op2id("STOP"):
                    new_beams.append((program, score))
                    continue
                
                # Get next token probabilities
                with torch.no_grad():
                    logits = self.synthesiser.forward(
                        task_emb.unsqueeze(0),
                        torch.tensor([program], device=task_emb.device)
                    )
                    logits = logits[0, -1]  # Last token
                    probs = F.softmax(logits, dim=-1)
                
                # Get top-k tokens
                top_probs, top_tokens = torch.topk(probs, beam_width)
                
                for prob, token in zip(top_probs, top_tokens):
                    new_program = program + [token.item()]
                    new_score = score + torch.log(prob).item()
                    new_beams.append((new_program, new_score))
            
            # Keep top beam_width programs
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]
        
        return [program for program, _ in beams]
    
    def validate_and_repair_program(self, program: List[int]) -> List[int]:
        """Validate and repair a program if needed."""
        parser = DSLParser()
        
        if parser.validate_program(program):
            return program
        
        # Simple repair: truncate at first invalid token
        repaired = []
        i = 0
        
        while i < len(program):
            token = program[i]
            
            if token >= len(DSL.OPS):
                break
            
            opcode = DSL.id2op(token)
            n_args = DSL.nargs(token)
            
            # Check if we have enough tokens for arguments
            if i + n_args >= len(program):
                break
            
            # Check if arguments are valid constants
            valid_args = True
            for j in range(n_args):
                arg_token = program[i + 1 + j]
                if arg_token < CONST_OFFSET:
                    valid_args = False
                    break
            
            if not valid_args:
                break
            
            # Add instruction and arguments
            repaired.extend(program[i:i + 1 + n_args])
            i += 1 + n_args
        
        # Add STOP if not present
        if not repaired or repaired[-1] != DSL.op2id("STOP"):
            repaired.append(DSL.op2id("STOP"))
        
        return repaired

# === Utility Functions ===

def token_to_int(token: int) -> int:
    """Convert token ID to integer value."""
    if token < CONST_OFFSET:
        return token
    return token - CONST_OFFSET

def int_to_token(value: int) -> int:
    """Convert integer value to token ID."""
    if value < 0 or value >= MAX_CONST:
        raise ValueError(f"Value {value} out of range [0, {MAX_CONST-1}]")
    return value + CONST_OFFSET

def program_to_string(program: List[int]) -> str:
    """Convert program to human-readable string."""
    parser = DSLParser()
    try:
        instructions = parser.parse_program(program)
        lines = []
        
        for instr in instructions:
            if instr.args:
                args_str = ", ".join(map(str, instr.args))
                lines.append(f"{instr.opcode}({args_str})")
            else:
                lines.append(instr.opcode)
        
        return "\n".join(lines)
    
    except ValueError:
        return f"INVALID_PROGRAM: {program}"

def string_to_program(program_str: str) -> List[int]:
    """Convert human-readable string to program."""
    lines = program_str.strip().split('\n')
    program = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Parse instruction
        if '(' in line:
            opcode = line[:line.index('(')]
            args_str = line[line.index('(')+1:line.rindex(')')]
            args = [int(arg.strip()) for arg in args_str.split(',')]
        else:
            opcode = line
            args = []
        
        # Add opcode
        if not DSL.is_valid_op(opcode):
            raise ValueError(f"Invalid opcode: {opcode}")
        
        program.append(DSL.op2id(opcode))
        
        # Add arguments
        for arg in args:
            program.append(int_to_token(arg))
    
    return program 