"""
Fast Comparison: Original RIN, Echo V2 (parallel), vs Transformer

Tests whether the parallel Echo architecture is faster and learns better.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
import time
import sys

sys.path.insert(0, '/home/aiman/pi')

from rin.model import RINModel
from rin.echo_v2 import EchoModelV2
from rin.transformer import SwiGLUTransformer


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(model, SwiGLUTransformer):
        return model(x)
    elif isinstance(model, EchoModelV2):
        return model(x)
    else:  # RINModel
        logits, _ = model(x)
        return logits


def test_forward_speed():
    """Test how fast each model can process batches."""
    print("="*60)
    print("FORWARD PASS SPEED TEST")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 512
    batch_size = 32
    seq_len = 64
    
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    
    models = {
        'RIN (recurrent)': RINModel(vocab_size, d_model=64, num_layers=2, num_neurons=32),
        'Echo V2 (parallel)': EchoModelV2(vocab_size, d_model=64, num_layers=2, num_neurons=32, n_heads=4),
        'Transformer': SwiGLUTransformer(vocab_size, d_model=64, num_layers=2, n_heads=4),
    }
    
    print(f"\nBatch: {batch_size}, Seq: {seq_len}")
    print(f"{'Model':<20} {'Params':<12} {'Time (ms)':<12} {'Throughput':<12}")
    print("-"*60)
    
    for name, model in models.items():
        model = model.to(device)
        model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = get_logits(model, x)
        
        # Timing
        torch.cuda.synchronize() if device == "cuda" else None
        start = time.time()
        
        num_runs = 20
        with torch.no_grad():
            for _ in range(num_runs):
                _ = get_logits(model, x)
        
        torch.cuda.synchronize() if device == "cuda" else None
        elapsed = (time.time() - start) * 1000 / num_runs
        
        params = count_params(model)
        tokens_per_sec = (batch_size * seq_len) / (elapsed / 1000)
        
        print(f"{name:<20} {params:>10,} {elapsed:>10.1f} {tokens_per_sec:>11.0f}")


def test_copy_task():
    """Learn to copy input to output."""
    print("\n" + "="*60)
    print("COPY TASK")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 64
    seq_len = 16
    batch_size = 32
    num_epochs = 200
    
    models = {
        'RIN': RINModel(vocab_size, d_model=64, num_layers=2, num_neurons=32),
        'Echo V2': EchoModelV2(vocab_size, d_model=64, num_layers=2, num_neurons=32, n_heads=4),
        'Transformer': SwiGLUTransformer(vocab_size, d_model=64, num_layers=2, n_heads=4),
    }
    
    print(f"\nEpochs: {num_epochs}")
    print(f"{'Model':<15} {'Final Acc':<12} {'Train Time':<12} {'Params':<12}")
    print("-"*60)
    
    for name, model in models.items():
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        
        start_time = time.time()
        final_acc = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            
            # Copy task: repeat sequence
            x = torch.randint(1, vocab_size-1, (batch_size, seq_len//2), device=device)
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
                    final_acc = (pred == seq[:, 1:]).float().mean().item()
        
        elapsed = time.time() - start_time
        params = count_params(model)
        
        print(f"{name:<15} {final_acc:>10.1%} {elapsed:>10.1f}s {params:>10,}")


def test_retrieval_task():
    """Find marked value in sequence."""
    print("\n" + "="*60)
    print("RETRIEVAL TASK")
    print("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 64
    seq_len = 32
    batch_size = 32
    num_epochs = 300
    
    marker = vocab_size - 1
    
    models = {
        'RIN': RINModel(vocab_size, d_model=64, num_layers=2, num_neurons=32),
        'Echo V2': EchoModelV2(vocab_size, d_model=64, num_layers=2, num_neurons=32, n_heads=4),
        'Transformer': SwiGLUTransformer(vocab_size, d_model=64, num_layers=2, n_heads=4),
    }
    
    print(f"\nEpochs: {num_epochs}")
    print(f"{'Model':<15} {'Final Acc':<12} {'Train Time':<12} {'Params':<12}")
    print("-"*60)
    
    for name, model in models.items():
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        
        start_time = time.time()
        final_acc = 0.0
        
        for epoch in range(num_epochs):
            model.train()
            
            # [noise, MARKER, target, noise, ...]
            seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
            targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
            
            # Place marker and target
            for i in range(batch_size):
                pos = torch.randint(2, seq_len//2, (1,)).item()
                seq[i, pos] = marker
                seq[i, pos+1] = targets[i]
            
            # Query at end
            seq[:, -2] = marker
            
            logits = get_logits(model, seq)
            
            loss = F.cross_entropy(logits[:, -1, :], targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 100 == 99:
                model.eval()
                with torch.no_grad():
                    pred = logits[:, -1, :].argmax(dim=-1)
                    final_acc = (pred == targets).float().mean().item()
        
        elapsed = time.time() - start_time
        params = count_params(model)
        
        print(f"{name:<15} {final_acc:>10.1%} {elapsed:>10.1f}s {params:>10,}")


def test_parameter_efficiency():
    """Check parameter counts across scales."""
    print("\n" + "="*60)
    print("PARAMETER EFFICIENCY")
    print("="*60)
    
    vocab_size = 1024
    
    configs = [
        {'d_model': 64, 'num_layers': 2, 'num_neurons': 32, 'n_heads': 4},
        {'d_model': 128, 'num_layers': 2, 'num_neurons': 64, 'n_heads': 8},
    ]
    
    print(f"\n{'Config':<25} {'RIN':<15} {'Echo V2':<15} {'Transformer':<15}")
    print("-"*70)
    
    for cfg in configs:
        rin = RINModel(vocab_size, cfg['d_model'], cfg['num_layers'], cfg['num_neurons'])
        echo = EchoModelV2(vocab_size, cfg['d_model'], cfg['num_layers'], cfg['num_neurons'], cfg['n_heads'])
        trans = SwiGLUTransformer(vocab_size, cfg['d_model'], cfg['num_layers'], cfg['n_heads'])
        
        config_str = f"d={cfg['d_model']}, L={cfg['num_layers']}"
        print(f"{config_str:<25} {count_params(rin):>13,} {count_params(echo):>14,} {count_params(trans):>14,}")


def run_all():
    print("="*60)
    print("FAST ECHO ARCHITECTURE COMPARISON")
    print("="*60)
    
    test_parameter_efficiency()
    test_forward_speed()
    test_copy_task()
    test_retrieval_task()
    
    print("\n" + "="*60)
    print("OBSERVATIONS")
    print("="*60)
    print("""
✓ Echo V2 (parallel) should be much faster than RIN (recurrent)
✓ Echo V2 should have fewer parameters than Transformer
✓ Performance will likely still be below Transformer, but with better speed
  
Key trade-off:
  - RIN/Echo: Specialized interference-based processing
  - Transformer: Parallel attention, proven on all tasks
  
The parallel Echo V2 removes the recurrent bottleneck while keeping
the resonant interference patterns for diversity.
    """)


if __name__ == "__main__":
    run_all()
