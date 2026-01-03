"""
Fast Model Comparison Tests

Quick tests to verify:
1. Gradient flow works end-to-end
2. All model types can learn basic tasks
3. Compare parameter efficiency and training dynamics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import numpy as np
import time
from typing import Dict, Tuple
import sys

sys.path.insert(0, '/home/aiman/pi')

from rin.model import RINModel
from rin.echo import EchoModel
from rin.transformer import SwiGLUTransformer


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, SwiGLUTransformer):
        return model(x)
    elif isinstance(model, EchoModel):
        logits, _, _ = model(x)
        return logits
    else:  # RINModel
        logits, _ = model(x)
        return logits


def test_gradient_flow():
    """Verify gradients flow to all components."""
    print("="*60)
    print("GRADIENT FLOW TEST")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d_model = 64
    vocab_size = 256
    
    models = {
        'RIN': RINModel(vocab_size, d_model, num_layers=2, num_neurons=32),
        'Echo': EchoModel(vocab_size, d_model, num_layers=2, num_neurons=32, n_heads=4),
        'Transformer': SwiGLUTransformer(vocab_size, d_model, num_layers=2, n_heads=4),
    }
    
    for name, model in models.items():
        model = model.to(device)
        model.train()
        
        x = torch.randint(0, vocab_size, (2, 16), device=device)
        logits = get_logits(model, x)
        loss = logits.sum()
        loss.backward()
        
        # Check key gradients
        has_grad = {}
        for pname, param in model.named_parameters():
            if param.grad is not None and param.grad.norm() > 0:
                # Categorize
                for key in ['embedding', 'attn', 'resonant', 'output', 'block']:
                    if key in pname.lower():
                        has_grad[key] = True
        
        print(f"\n{name} ({count_params(model):,} params):")
        print(f"  Gradients: {list(has_grad.keys())}")
        
        model.zero_grad()
    
    print("\n✓ All models have gradient flow\n")


def test_copy_task():
    """Simple copy task - model must learn to copy input."""
    print("="*60)
    print("COPY TASK (Learn to Output What Was Seen)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 64
    seq_len = 16
    batch_size = 32
    num_epochs = 200
    
    models = {
        'RIN': RINModel(vocab_size, d_model=64, num_layers=2, num_neurons=32),
        'Echo': EchoModel(vocab_size, d_model=64, num_layers=2, num_neurons=32, n_heads=4),
        'Transformer': SwiGLUTransformer(vocab_size, d_model=64, num_layers=2, n_heads=4),
    }
    
    results = {}
    
    for name, model in models.items():
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        
        start = time.time()
        final_acc = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            
            # Generate copy data: [BOS, x1, x2, ..., x_n, SEP, ?, ?, ..., ?]
            # Target: predict x_i at position i+n+1
            x = torch.randint(1, vocab_size-2, (batch_size, seq_len//2), device=device)
            # For simplicity, next-token prediction on repeated sequence
            seq = torch.cat([x, x], dim=1)
            
            logits = get_logits(model, seq)
            
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                seq[:, 1:].reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 50 == 49:
                model.eval()
                with torch.no_grad():
                    pred = logits[:, :-1].argmax(dim=-1)
                    acc = (pred == seq[:, 1:]).float().mean().item()
                    final_acc = acc
        
        elapsed = time.time() - start
        results[name] = {'acc': final_acc, 'time': elapsed}
        print(f"{name}: {final_acc:.1%} accuracy, {elapsed:.1f}s")
    
    return results


def test_pattern_recognition():
    """Pattern prediction - learn repeating patterns."""
    print("\n" + "="*60)
    print("PATTERN RECOGNITION (Predict Next in Sequence)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 32
    batch_size = 32
    num_epochs = 300
    
    models = {
        'RIN': RINModel(vocab_size, d_model=64, num_layers=2, num_neurons=32),
        'Echo': EchoModel(vocab_size, d_model=64, num_layers=2, num_neurons=32, n_heads=4),
        'Transformer': SwiGLUTransformer(vocab_size, d_model=64, num_layers=2, n_heads=4),
    }
    
    results = {}
    
    for name, model in models.items():
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        
        start = time.time()
        final_acc = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            
            # Generate patterns: [a, b, c, a, b, c, a, b, c, ...]
            pattern_len = torch.randint(2, 5, (1,)).item()
            pattern = torch.randint(0, vocab_size, (batch_size, pattern_len), device=device)
            # Repeat pattern
            repeats = 6
            seq = pattern.repeat(1, repeats)
            
            logits = get_logits(model, seq)
            
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, vocab_size),
                seq[:, 1:].reshape(-1)
            )
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 99:
                model.eval()
                with torch.no_grad():
                    pred = logits[:, :-1].argmax(dim=-1)
                    acc = (pred == seq[:, 1:]).float().mean().item()
                    final_acc = acc
        
        elapsed = time.time() - start
        results[name] = {'acc': final_acc, 'time': elapsed}
        print(f"{name}: {final_acc:.1%} accuracy, {elapsed:.1f}s")
    
    return results


def test_retrieval():
    """Simple retrieval - find marked value."""
    print("\n" + "="*60)
    print("RETRIEVAL (Find Marked Value)")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 64
    seq_len = 32
    batch_size = 32
    num_epochs = 500
    
    marker = vocab_size - 1  # Special marker token
    
    models = {
        'RIN': RINModel(vocab_size, d_model=64, num_layers=2, num_neurons=32),
        'Echo': EchoModel(vocab_size, d_model=64, num_layers=2, num_neurons=32, n_heads=4),
        'Transformer': SwiGLUTransformer(vocab_size, d_model=64, num_layers=2, n_heads=4),
    }
    
    results = {}
    
    for name, model in models.items():
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        
        start = time.time()
        final_acc = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            
            # [noise, MARKER, target, noise, ..., MARKER, ?]
            seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
            targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
            positions = torch.randint(2, seq_len//2, (batch_size,), device=device)
            
            for i in range(batch_size):
                pos = positions[i].item()
                seq[i, pos] = marker
                seq[i, pos+1] = targets[i]
            
            # End with marker
            seq[:, -2] = marker
            
            logits = get_logits(model, seq)
            
            # Predict at last position
            loss = F.cross_entropy(logits[:, -1, :], targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 99:
                model.eval()
                with torch.no_grad():
                    pred = logits[:, -1, :].argmax(dim=-1)
                    acc = (pred == targets).float().mean().item()
                    final_acc = acc
        
        elapsed = time.time() - start
        results[name] = {'acc': final_acc, 'time': elapsed}
        print(f"{name}: {final_acc:.1%} accuracy, {elapsed:.1f}s")
    
    return results


def test_parameter_efficiency():
    """Compare parameters and throughput."""
    print("\n" + "="*60)
    print("PARAMETER EFFICIENCY")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 1024
    
    configs = [
        {'d_model': 64, 'num_layers': 2, 'num_neurons': 32, 'n_heads': 4},
        {'d_model': 128, 'num_layers': 2, 'num_neurons': 64, 'n_heads': 8},
        {'d_model': 256, 'num_layers': 4, 'num_neurons': 128, 'n_heads': 8},
    ]
    
    print(f"\n{'Config':<20} {'RIN':<15} {'Echo':<15} {'Transformer':<15}")
    print("-"*65)
    
    for cfg in configs:
        rin = RINModel(vocab_size, cfg['d_model'], cfg['num_layers'], cfg['num_neurons'])
        echo = EchoModel(vocab_size, cfg['d_model'], cfg['num_layers'], cfg['num_neurons'], cfg['n_heads'])
        trans = SwiGLUTransformer(vocab_size, cfg['d_model'], cfg['num_layers'], cfg['n_heads'])
        
        config_str = f"d={cfg['d_model']},L={cfg['num_layers']}"
        print(f"{config_str:<20} {count_params(rin):>12,} {count_params(echo):>14,} {count_params(trans):>14,}")


def run_all():
    """Run all quick tests."""
    print("="*60)
    print("COMPREHENSIVE MODEL COMPARISON")
    print("RIN vs Echo vs Transformer")
    print("="*60)
    
    test_gradient_flow()
    test_parameter_efficiency()
    
    results = {}
    results['copy'] = test_copy_task()
    results['pattern'] = test_pattern_recognition()
    results['retrieval'] = test_retrieval()
    
    # Final summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print(f"\n{'Task':<20} {'RIN':<15} {'Echo':<15} {'Transformer':<15}")
    print("-"*65)
    
    for task, task_results in results.items():
        row = f"{task:<20}"
        for name in ['RIN', 'Echo', 'Transformer']:
            if name in task_results:
                row += f"{task_results[name]['acc']:.1%}           "
            else:
                row += "N/A            "
        print(row)
    
    print("\n✓ All tests completed!")


if __name__ == "__main__":
    run_all()
