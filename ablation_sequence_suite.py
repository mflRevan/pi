#!/usr/bin/env python3
"""
Comprehensive Sequence Learning Test Suite
==========================================

Tests HolographicTransformer vs SwiGLU baseline on various sequence tasks
with configurable length generalization.

Tasks:
1. Modular Arithmetic - (a + b) mod p
2. Sorting - Sort a sequence of numbers
3. Positional Addition - Add numbers at specific positions
4. Sequence Reversal - Reverse input sequence
5. Bitwise Addition - Binary addition with carry
6. Needle in Haystack - Retrieve specific token from context

Each task supports:
- Configurable train/test sequence lengths
- Length generalization (train short, test long)
- Accuracy and loss metrics
- Plotting and analysis
"""

import os
import sys
import json
import time
import math
import random
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import matplotlib.pyplot as plt

# Import our models
from rin import HolographicTransformer, SwiGLUTransformer
from rin.kernels import TRITON_AVAILABLE


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TaskConfig:
    """Configuration for a single task."""
    name: str
    train_length: int
    test_lengths: List[int]  # Multiple test lengths for generalization
    vocab_size: int = 128
    num_train: int = 10000
    num_test: int = 1000
    epochs: int = 1000
    batch_size: int = 32
    
    # Task-specific parameters
    modular_p: int = 97  # Prime for modular arithmetic
    num_bits: int = 16   # Bits for bitwise addition
    haystack_size: int = 100  # Context size for needle task


@dataclass
class ModelConfig:
    """Model configuration."""
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 8
    n_phase: int = 32
    expansion: int = 3
    dropout: float = 0.0
    max_seq_len: int = 512
    
    # Holographic specific
    gate_mode: str = 'omniware'
    use_triton: bool = True
    log_grad: bool = True


@dataclass
class BenchmarkConfig:
    """Full benchmark configuration."""
    tasks: List[str] = field(default_factory=lambda: [
        'modular_arithmetic',
        'sorting',
        'positional_addition',
        'sequence_reversal',
        'bitwise_addition',
        'needle_haystack',
    ])
    
    # Length generalization settings
    length_generalization: bool = True
    
    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    output_dir: str = 'results/benchmark_suite'


# =============================================================================
# TASK DATASETS
# =============================================================================

class ModularArithmeticDataset(Dataset):
    """
    Task: Compute (a + b) mod p
    Input: [a, b, =]
    Output: [result]
    """
    
    def __init__(self, num_samples: int, p: int = 97, max_val: int = None):
        self.p = p
        self.max_val = max_val or p - 1
        self.samples = []
        
        for _ in range(num_samples):
            a = random.randint(0, self.max_val)
            b = random.randint(0, self.max_val)
            result = (a + b) % p
            
            # Encode: [a, b, EQUALS_TOKEN, result, EOS]
            # Using offset tokens: 0-96 for numbers, 97=EQUALS, 98=EOS
            self.samples.append({
                'input': [a, b, p],  # EQUALS token = p
                'target': result,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        # Input sequence: [a, b, =]
        input_seq = torch.tensor(sample['input'], dtype=torch.long)
        # Target: predict result after =
        target = torch.tensor(sample['target'], dtype=torch.long)
        return input_seq, target


class SortingDataset(Dataset):
    """
    Task: Sort a sequence of numbers
    Input: [n1, n2, ..., nk, SEP]
    Output: [sorted_n1, sorted_n2, ..., sorted_nk]
    """
    
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int = 100):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.sep_token = vocab_size  # Separator token
        self.samples = []
        
        for _ in range(num_samples):
            # Generate random sequence
            seq = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
            sorted_seq = sorted(seq)
            
            self.samples.append({
                'input': seq + [self.sep_token],
                'target': sorted_seq,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.tensor(sample['input'], dtype=torch.long)
        target_seq = torch.tensor(sample['target'], dtype=torch.long)
        return input_seq, target_seq


class PositionalAdditionDataset(Dataset):
    """
    Task: Add numbers at specified positions
    Input: [n1, n2, ..., nk, POS_i, POS_j, =]
    Output: [n_i + n_j]
    """
    
    def __init__(self, num_samples: int, seq_len: int, max_val: int = 50, max_pos: int = 256):
        self.seq_len = seq_len
        self.max_val = max_val
        self.max_pos = max_pos
        self.samples = []
        
        # Fixed token scheme to handle variable lengths:
        # 0 to max_val-1: numbers
        # max_val to max_val+max_pos-1: positions  
        # max_val+max_pos: EQUALS token
        self.pos_offset = max_val
        self.equals_token = max_val + max_pos  # Fixed, not dependent on seq_len
        
        for _ in range(num_samples):
            seq = [random.randint(0, max_val - 1) for _ in range(seq_len)]
            pos_i = random.randint(0, seq_len - 1)
            pos_j = random.randint(0, seq_len - 1)
            result = (seq[pos_i] + seq[pos_j]) % max_val
            
            self.samples.append({
                'input': seq + [self.pos_offset + pos_i, self.pos_offset + pos_j, self.equals_token],
                'target': result,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.tensor(sample['input'], dtype=torch.long)
        target = torch.tensor(sample['target'], dtype=torch.long)
        return input_seq, target


class SequenceReversalDataset(Dataset):
    """
    Task: Reverse input sequence
    Input: [n1, n2, ..., nk, SEP]
    Output: [nk, ..., n2, n1]
    """
    
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int = 100):
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.sep_token = vocab_size
        self.samples = []
        
        for _ in range(num_samples):
            seq = [random.randint(0, vocab_size - 1) for _ in range(seq_len)]
            reversed_seq = seq[::-1]
            
            self.samples.append({
                'input': seq + [self.sep_token],
                'target': reversed_seq,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.tensor(sample['input'], dtype=torch.long)
        target_seq = torch.tensor(sample['target'], dtype=torch.long)
        return input_seq, target_seq


class BitwiseAdditionDataset(Dataset):
    """
    Task: Binary addition with carry
    Input: [bit1_a, bit1_b, bit2_a, bit2_b, ..., SEP]
    Output: [sum_bit1, sum_bit2, ..., carry]
    
    Numbers represented LSB first for easier carry propagation learning.
    """
    
    def __init__(self, num_samples: int, num_bits: int):
        self.num_bits = num_bits
        self.sep_token = 2  # 0, 1 for bits, 2 for separator
        self.samples = []
        
        max_val = 2 ** num_bits - 1
        
        for _ in range(num_samples):
            a = random.randint(0, max_val)
            b = random.randint(0, max_val)
            result = a + b
            
            # Convert to binary (LSB first)
            a_bits = [(a >> i) & 1 for i in range(num_bits)]
            b_bits = [(b >> i) & 1 for i in range(num_bits)]
            result_bits = [(result >> i) & 1 for i in range(num_bits + 1)]  # +1 for carry
            
            # Interleave: [a0, b0, a1, b1, ...]
            input_seq = []
            for i in range(num_bits):
                input_seq.extend([a_bits[i], b_bits[i]])
            input_seq.append(self.sep_token)
            
            self.samples.append({
                'input': input_seq,
                'target': result_bits,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.tensor(sample['input'], dtype=torch.long)
        target_seq = torch.tensor(sample['target'], dtype=torch.long)
        return input_seq, target_seq


class NeedleHaystackDataset(Dataset):
    """
    Task: Retrieve a specific token from context
    Input: [KEY, value, ..., QUERY_KEY, =]
    Output: [value]
    
    Multiple key-value pairs, need to find the right one.
    """
    
    def __init__(self, num_samples: int, haystack_size: int, num_keys: int = 10, vocab_size: int = 100):
        self.haystack_size = haystack_size
        self.num_keys = num_keys
        self.vocab_size = vocab_size
        
        # Tokens: 0-99 values, 100-109 keys, 110=QUERY, 111=EQUALS
        self.key_offset = vocab_size
        self.query_token = vocab_size + num_keys
        self.equals_token = vocab_size + num_keys + 1
        
        self.samples = []
        
        for _ in range(num_samples):
            # Generate key-value pairs
            keys = list(range(num_keys))
            random.shuffle(keys)
            values = [random.randint(0, vocab_size - 1) for _ in range(num_keys)]
            
            # Build haystack with padding
            haystack = []
            for k, v in zip(keys, values):
                haystack.extend([self.key_offset + k, v])
            
            # Add padding/noise to reach haystack_size
            while len(haystack) < haystack_size:
                haystack.append(random.randint(0, vocab_size - 1))
            
            # Random query
            query_idx = random.randint(0, num_keys - 1)
            query_key = keys[query_idx]
            target_value = values[query_idx]
            
            # Input: haystack + [QUERY, key, =]
            input_seq = haystack[:haystack_size] + [self.query_token, self.key_offset + query_key, self.equals_token]
            
            self.samples.append({
                'input': input_seq,
                'target': target_value,
            })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        input_seq = torch.tensor(sample['input'], dtype=torch.long)
        target = torch.tensor(sample['target'], dtype=torch.long)
        return input_seq, target


# =============================================================================
# TASK FACTORY
# =============================================================================

def get_task_config(task_name: str, length_gen: bool = True, quick: bool = False) -> TaskConfig:
    """Get task configuration with length generalization settings."""
    
    # Reduce for quick mode
    num_train = 2000 if quick else 10000
    num_test = 200 if quick else 1000
    epoch_mult = 0.3 if quick else 1.0
    
    configs = {
        'modular_arithmetic': TaskConfig(
            name='modular_arithmetic',
            train_length=3,  # Fixed length for this task
            test_lengths=[3],  # Test with larger primes instead
            vocab_size=128,
            modular_p=97,
            num_train=num_train,
            num_test=num_test,
            epochs=max(500, int(15 * epoch_mult)),
            batch_size=32,
        ),
        'sorting': TaskConfig(
            name='sorting',
            train_length=15,
            test_lengths=[15, 25, 50, 100] if length_gen else [15],
            vocab_size=101,  # 0-99 + separator
            num_train=num_train,
            num_test=num_test,
            epochs=max(500, int(20 * epoch_mult)),
            batch_size=32,
        ),
        'positional_addition': TaskConfig(
            name='positional_addition',
            train_length=20,
            test_lengths=[20, 50, 100, 200] if length_gen else [20],
            vocab_size=307,  # 50 (numbers) + 256 (positions) + 1 (equals)
            num_train=num_train,
            num_test=num_test,
            epochs=max(500, int(15 * epoch_mult)),
            batch_size=32,
        ),
        'sequence_reversal': TaskConfig(
            name='sequence_reversal',
            train_length=20,
            test_lengths=[20, 50, 100, 200] if length_gen else [20],
            vocab_size=101,
            num_train=num_train,
            num_test=num_test,
            epochs=max(500, int(15 * epoch_mult)),
            batch_size=32,
        ),
        'bitwise_addition': TaskConfig(
            name='bitwise_addition',
            train_length=16,  # bits
            test_lengths=[16, 24, 32, 48] if length_gen else [16],
            vocab_size=4,  # 0, 1, separator, padding
            num_bits=16,
            num_train=num_train,
            num_test=num_test,
            epochs=max(500, int(20 * epoch_mult)),
            batch_size=32,
        ),
        'needle_haystack': TaskConfig(
            name='needle_haystack',
            train_length=50,  # haystack size
            test_lengths=[50, 100, 200, 500] if length_gen else [50],
            vocab_size=112,  # 100 (values) + 10 (keys) + 1 (query) + 1 (equals)
            haystack_size=50,
            num_train=num_train,
            num_test=num_test,
            epochs=max(500, int(15 * epoch_mult)),
            batch_size=32,
        ),
    }
    
    return configs.get(task_name)


def create_dataset(task_name: str, config: TaskConfig, length: int, num_samples: int) -> Dataset:
    """Create dataset for a specific task and length."""
    
    if task_name == 'modular_arithmetic':
        return ModularArithmeticDataset(num_samples, config.modular_p)
    
    elif task_name == 'sorting':
        return SortingDataset(num_samples, length, config.vocab_size - 1)
    
    elif task_name == 'positional_addition':
        # max_pos=256 allows positions up to 255, vocab_size = 50 + 256 + 1 = 307
        return PositionalAdditionDataset(num_samples, length, max_val=50, max_pos=256)
    
    elif task_name == 'sequence_reversal':
        return SequenceReversalDataset(num_samples, length, config.vocab_size - 1)
    
    elif task_name == 'bitwise_addition':
        return BitwiseAdditionDataset(num_samples, length)
    
    elif task_name == 'needle_haystack':
        return NeedleHaystackDataset(num_samples, length)
    
    else:
        raise ValueError(f"Unknown task: {task_name}")


# =============================================================================
# MODEL CREATION
# =============================================================================

def create_model(model_type: str, model_cfg: ModelConfig, vocab_size: int, max_seq_len: int) -> nn.Module:
    """Create model for the task."""
    
    if model_type == 'holographic':
        model = HolographicTransformer(
            vocab_size=vocab_size,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            n_layers=model_cfg.n_layers,
            n_phase=model_cfg.n_phase,
            expansion=model_cfg.expansion,
            dropout=model_cfg.dropout,
            gate_mode=model_cfg.gate_mode,
            use_triton=model_cfg.use_triton,
            log_grad=model_cfg.log_grad,
            causal=True,
            max_seq_len=max_seq_len,
        )
    elif model_type == 'swiglu':
        model = SwiGLUTransformer(
            vocab_size=vocab_size,
            d_model=model_cfg.d_model,
            n_heads=model_cfg.n_heads,
            num_layers=model_cfg.n_layers,
            d_ff=model_cfg.d_model * model_cfg.expansion,
            dropout=model_cfg.dropout,
            max_seq_len=max_seq_len,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


# =============================================================================
# TRAINING AND EVALUATION
# =============================================================================

def collate_fn(batch):
    """Custom collate to handle variable output lengths."""
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Pad inputs
    max_input_len = max(len(x) for x in inputs)
    padded_inputs = torch.zeros(len(inputs), max_input_len, dtype=torch.long)
    for i, x in enumerate(inputs):
        padded_inputs[i, :len(x)] = x
    
    # Handle targets (could be single value or sequence)
    if targets[0].dim() == 0:
        # Single target value
        targets = torch.stack(targets)
        return padded_inputs, targets, 'single'
    else:
        # Sequence targets - create full sequence for causal LM training
        # Input: [in1, in2, ..., SEP, out1, out2, ...]
        # We'll train to predict the output part
        full_seqs = []
        target_masks = []
        
        for inp, tgt in zip(inputs, targets):
            # Concatenate input and target
            full_seq = torch.cat([inp, tgt])
            full_seqs.append(full_seq)
            
            # Mask: 0 for input positions, 1 for output positions
            mask = torch.cat([
                torch.zeros(len(inp), dtype=torch.bool),
                torch.ones(len(tgt), dtype=torch.bool)
            ])
            target_masks.append(mask)
        
        # Pad full sequences
        max_len = max(len(s) for s in full_seqs)
        padded_seqs = torch.zeros(len(full_seqs), max_len, dtype=torch.long)
        padded_masks = torch.zeros(len(full_seqs), max_len, dtype=torch.bool)
        
        for i, (seq, mask) in enumerate(zip(full_seqs, target_masks)):
            padded_seqs[i, :len(seq)] = seq
            padded_masks[i, :len(mask)] = mask
        
        return padded_seqs, padded_masks, 'sequence'


def train_epoch(model, dataloader, optimizer, scheduler, device, task_type='single'):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch in dataloader:
        if len(batch) == 3:
            inputs, targets_or_mask, batch_type = batch
        else:
            inputs, targets_or_mask = batch
            batch_type = task_type
        
        inputs = inputs.to(device)
        targets_or_mask = targets_or_mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = model(inputs)
        
        if batch_type == 'single':
            # Single output prediction (last position)
            pred_logits = logits[:, -1, :]
            loss = F.cross_entropy(pred_logits, targets_or_mask)
            preds = pred_logits.argmax(dim=-1)
            total_correct += (preds == targets_or_mask).sum().item()
            total_samples += targets_or_mask.size(0)
        else:
            # Sequence prediction using causal LM objective
            # Shift: predict next token at each position
            # Only compute loss on output positions (where mask is True)
            mask = targets_or_mask  # Boolean mask
            
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            
            # Compute loss only on masked (output) positions
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            loss = loss.view(shift_logits.size(0), -1)
            loss = (loss * shift_mask.float()).sum() / shift_mask.sum()
            
            # Accuracy on output positions
            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_labels) & shift_mask
            total_correct += correct.sum().item()
            total_samples += shift_mask.sum().item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': total_correct / total_samples if total_samples > 0 else 0,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, task_type='single'):
    """Evaluate model."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_seq_correct = 0  # Full sequence accuracy
    total_seqs = 0
    
    for batch in dataloader:
        if len(batch) == 3:
            inputs, targets_or_mask, batch_type = batch
        else:
            inputs, targets_or_mask = batch
            batch_type = task_type
        
        inputs = inputs.to(device)
        targets_or_mask = targets_or_mask.to(device)
        
        logits = model(inputs)
        
        if batch_type == 'single':
            pred_logits = logits[:, -1, :]
            loss = F.cross_entropy(pred_logits, targets_or_mask)
            preds = pred_logits.argmax(dim=-1)
            total_correct += (preds == targets_or_mask).sum().item()
            total_samples += targets_or_mask.size(0)
            total_seq_correct += (preds == targets_or_mask).sum().item()
            total_seqs += targets_or_mask.size(0)
        else:
            # Sequence prediction
            mask = targets_or_mask
            
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:].contiguous()
            shift_mask = mask[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            )
            loss = loss.view(shift_logits.size(0), -1)
            loss = (loss * shift_mask.float()).sum() / shift_mask.sum().clamp(min=1)
            
            preds = shift_logits.argmax(dim=-1)
            correct = (preds == shift_labels) & shift_mask
            total_correct += correct.sum().item()
            total_samples += shift_mask.sum().item()
            
            # Full sequence accuracy (all output tokens correct)
            seq_correct = (correct | ~shift_mask).all(dim=1)
            total_seq_correct += seq_correct.sum().item()
            total_seqs += inputs.size(0)
        
        total_loss += loss.item()
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': total_correct / total_samples if total_samples > 0 else 0,
        'sequence_accuracy': total_seq_correct / total_seqs if total_seqs > 0 else 0,
    }


# =============================================================================
# BENCHMARK RUNNER
# =============================================================================

def run_task_benchmark(
    task_name: str,
    model_cfg: ModelConfig,
    bench_cfg: BenchmarkConfig,
    quick: bool = False,
) -> Dict[str, Any]:
    """Run benchmark for a single task."""
    
    print(f"\n{'='*60}")
    print(f"TASK: {task_name.upper()}")
    print(f"{'='*60}")
    
    task_cfg = get_task_config(task_name, bench_cfg.length_generalization, quick=quick)
    if task_cfg is None:
        print(f"Unknown task: {task_name}")
        return {}
    
    # Determine task type
    task_type = 'single' if task_name in ['modular_arithmetic', 'positional_addition', 'needle_haystack'] else 'sequence'
    
    # Determine max sequence length needed
    max_len = max(task_cfg.test_lengths) * 3 + 10  # Buffer for input + output
    if task_name == 'bitwise_addition':
        max_len = max(task_cfg.test_lengths) * 4 + 10
    
    results = {
        'task': task_name,
        'task_config': asdict(task_cfg),
        'models': {},
    }
    
    # Run for both model types
    for model_type in ['holographic', 'swiglu']:
        print(f"\n--- {model_type.upper()} ---")
        
        # Create model
        model = create_model(model_type, model_cfg, task_cfg.vocab_size, max_len)
        model = model.to(bench_cfg.device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {num_params:,}")
        
        # Create training data
        train_dataset = create_dataset(task_name, task_cfg, task_cfg.train_length, task_cfg.num_train)
        train_loader = DataLoader(train_dataset, batch_size=task_cfg.batch_size, shuffle=True, collate_fn=collate_fn)
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=bench_cfg.learning_rate,
            weight_decay=bench_cfg.weight_decay,
        )
        total_steps = len(train_loader) * task_cfg.epochs
        warmup_steps = int(total_steps * bench_cfg.warmup_ratio)
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        # Training
        print(f"Training for {task_cfg.epochs} epochs on length {task_cfg.train_length}...")
        train_history = []
        
        for epoch in range(1, task_cfg.epochs + 1):
            metrics = train_epoch(model, train_loader, optimizer, scheduler, bench_cfg.device, task_type)
            train_history.append(metrics)
            
            if epoch % 5 == 0 or epoch == task_cfg.epochs:
                print(f"  Epoch {epoch}: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.4f}")
        
        # Evaluate on all test lengths
        print(f"\nEvaluating length generalization...")
        test_results = {}
        
        for test_len in task_cfg.test_lengths:
            test_dataset = create_dataset(task_name, task_cfg, test_len, task_cfg.num_test)
            test_loader = DataLoader(test_dataset, batch_size=task_cfg.batch_size, shuffle=False, collate_fn=collate_fn)
            
            metrics = evaluate(model, test_loader, bench_cfg.device, task_type)
            test_results[test_len] = metrics
            
            gen_label = " (OOD)" if test_len > task_cfg.train_length else ""
            print(f"  Length {test_len}{gen_label}: Acc={metrics['accuracy']:.4f}, SeqAcc={metrics['sequence_accuracy']:.4f}")
        
        results['models'][model_type] = {
            'num_params': num_params,
            'train_history': train_history,
            'test_results': test_results,
            'final_train_acc': train_history[-1]['accuracy'],
        }
        
        # Clean up
        del model
        torch.cuda.empty_cache()
    
    return results


def plot_results(all_results: Dict[str, Any], output_dir: Path):
    """Generate plots for benchmark results."""
    
    # Create figure with subplots for each task
    num_tasks = len(all_results)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    task_names = list(all_results.keys())
    
    for idx, task_name in enumerate(task_names):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        results = all_results[task_name]
        
        if 'models' not in results:
            continue
        
        # Plot length generalization
        for model_type, model_results in results['models'].items():
            test_results = model_results['test_results']
            lengths = sorted([int(k) for k in test_results.keys()])
            accuracies = [test_results[l]['accuracy'] for l in lengths]
            
            color = '#2ecc71' if model_type == 'holographic' else '#3498db'
            label = 'Holographic' if model_type == 'holographic' else 'SwiGLU'
            ax.plot(lengths, accuracies, 'o-', color=color, label=label, linewidth=2, markersize=8)
        
        # Mark training length
        train_len = results['task_config']['train_length']
        ax.axvline(x=train_len, color='gray', linestyle='--', alpha=0.5, label='Train Length')
        
        ax.set_xlabel('Sequence Length')
        ax.set_ylabel('Accuracy')
        ax.set_title(task_name.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    # Hide unused subplots
    for idx in range(len(task_names), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Length Generalization: Holographic vs SwiGLU', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'length_generalization.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Training curves
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, task_name in enumerate(task_names):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        results = all_results[task_name]
        
        if 'models' not in results:
            continue
        
        for model_type, model_results in results['models'].items():
            history = model_results['train_history']
            epochs = list(range(1, len(history) + 1))
            accuracies = [h['accuracy'] for h in history]
            
            color = '#2ecc71' if model_type == 'holographic' else '#3498db'
            label = 'Holographic' if model_type == 'holographic' else 'SwiGLU'
            ax.plot(epochs, accuracies, '-', color=color, label=label, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Accuracy')
        ax.set_title(task_name.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    for idx in range(len(task_names), len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Training Curves: Holographic vs SwiGLU', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Plots saved to {output_dir}")


def print_summary(all_results: Dict[str, Any]):
    """Print summary table."""
    
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Header
    print(f"\n{'Task':<25} {'Train Len':<10} {'Model':<12} {'Train Acc':<12} {'OOD Acc':<12}")
    print("-" * 80)
    
    for task_name, results in all_results.items():
        if 'models' not in results:
            continue
            
        train_len = results['task_config']['train_length']
        test_lengths = results['task_config']['test_lengths']
        ood_len = max(test_lengths) if len(test_lengths) > 1 else test_lengths[0]
        
        for model_type, model_results in results['models'].items():
            train_acc = model_results['final_train_acc']
            ood_acc = model_results['test_results'].get(ood_len, {}).get('accuracy', 0)
            
            print(f"{task_name:<25} {train_len:<10} {model_type:<12} {train_acc:<12.4f} {ood_acc:<12.4f}")
    
    # Compute averages
    print("-" * 80)
    
    for model_type in ['holographic', 'swiglu']:
        train_accs = []
        ood_accs = []
        
        for task_name, results in all_results.items():
            if 'models' not in results or model_type not in results['models']:
                continue
            
            model_results = results['models'][model_type]
            train_accs.append(model_results['final_train_acc'])
            
            test_lengths = results['task_config']['test_lengths']
            ood_len = max(test_lengths)
            ood_accs.append(model_results['test_results'].get(ood_len, {}).get('accuracy', 0))
        
        if train_accs:
            print(f"{'AVERAGE':<25} {'':<10} {model_type:<12} {np.mean(train_accs):<12.4f} {np.mean(ood_accs):<12.4f}")


def main():
    parser = argparse.ArgumentParser(description='Sequence Learning Benchmark Suite')
    parser.add_argument('--tasks', type=str, default='all',
                        help='Comma-separated tasks or "all"')
    parser.add_argument('--no-length-gen', action='store_true',
                        help='Disable length generalization testing')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with smaller model, fewer samples and epochs')
    parser.add_argument('--no-triton', action='store_true',
                        help='Disable Triton kernels (use PyTorch)')
    parser.add_argument('--output-dir', type=str, default='results/benchmark_suite',
                        help='Output directory')
    args = parser.parse_args()
    
    # Configuration
    use_triton = not args.no_triton
    
    if args.quick:
        # Smaller model for quick testing
        model_cfg = ModelConfig(
            d_model=128,
            n_heads=4,
            n_layers=3,
            n_phase=32,  # divisible by n_heads
            expansion=4,
            use_triton=use_triton,
        )
        print(f"Running in QUICK mode (smaller model, reduced samples, triton={use_triton})")
    else:
        model_cfg = ModelConfig(use_triton=use_triton)
    
    bench_cfg = BenchmarkConfig(
        length_generalization=not args.no_length_gen,
        output_dir=args.output_dir,
    )
    
    # Select tasks
    if args.tasks == 'all':
        tasks = bench_cfg.tasks
    else:
        tasks = [t.strip() for t in args.tasks.split(',')]
    
    # Create output directory
    output_dir = Path(bench_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print configuration
    print("="*60)
    print("SEQUENCE LEARNING BENCHMARK SUITE")
    print("="*60)
    print(f"Device: {bench_cfg.device}")
    print(f"Triton Available: {TRITON_AVAILABLE}")
    print(f"Tasks: {tasks}")
    print(f"Length Generalization: {bench_cfg.length_generalization}")
    print(f"\nModel Config:")
    print(f"  d_model: {model_cfg.d_model}")
    print(f"  n_heads: {model_cfg.n_heads}")
    print(f"  n_layers: {model_cfg.n_layers}")
    print(f"  n_phase: {model_cfg.n_phase}")
    print(f"  gate_mode: {model_cfg.gate_mode}")
    print(f"  use_triton: {model_cfg.use_triton}")
    
    # Set seed
    torch.manual_seed(bench_cfg.seed)
    random.seed(bench_cfg.seed)
    np.random.seed(bench_cfg.seed)
    
    # Run benchmarks
    all_results = {}
    
    for task_name in tasks:
        try:
            results = run_task_benchmark(task_name, model_cfg, bench_cfg, quick=args.quick)
            all_results[task_name] = results
        except Exception as e:
            print(f"Error in task {task_name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[task_name] = {'error': str(e)}
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = output_dir / f'benchmark_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to {results_path}")
    
    # Generate plots
    try:
        plot_results(all_results, output_dir)
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    # Print summary
    print_summary(all_results)


if __name__ == '__main__':
    main()
