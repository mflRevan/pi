#!/usr/bin/env python3
"""
Ultra-Fast Fusion Comparison
Strips all unnecessary overhead for quick testing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# Minimal Fusion Models - Direct testing
# =============================================================================

class MinimalAdditiveModel(nn.Module):
    """Minimal additive fusion."""
    def __init__(self, d=32, nh=2, nr=64):
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.res_r = nn.Linear(d, d)
        self.res_i = nn.Linear(d, d)
        self.out = nn.Linear(d, 10)
    
    def forward(self, x):
        attn = self.attn(x)
        res_r = self.res_r(x)
        res_i = self.res_i(x)
        combined = attn + res_r + res_i
        return self.out(combined)


class MinimalMultiplicativeModel(nn.Module):
    """Minimal multiplicative fusion."""
    def __init__(self, d=32, nh=2, nr=64):
        super().__init__()
        self.attn = nn.Linear(d, d)
        self.res_r = nn.Linear(d, d)
        self.res_i = nn.Linear(d, d)
        self.out = nn.Linear(d, 10)
    
    def forward(self, x):
        attn = self.attn(x)
        res_r = self.res_r(x)
        res_i = self.res_i(x)
        gate = 1.0 + torch.tanh(res_r + res_i)
        combined = attn * gate
        return self.out(combined)


def make_batch(batch_size=8, seq_len=10, distance=5):
    """Create simple sequential data."""
    x = torch.randn(batch_size, seq_len, 32, device=device)
    # Target: needle at position 0
    needle_val = torch.randint(0, 10, (batch_size,), device=device)
    return x, needle_val


def train_and_eval(ModelClass, name: str):
    """Train and evaluate model."""
    print(f"\n{name}:")
    
    model = ModelClass().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Quick training
    for epoch in range(5):
        model.train()
        total_loss = 0
        for _ in range(5):
            x, targets = make_batch()
            out = model(x)  # (batch, seq, 10)
            # Average over sequence
            out = out.mean(dim=1)  # (batch, 10)
            loss = F.cross_entropy(out, targets)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        
        print(f"  Epoch {epoch+1}: loss={total_loss/5:.4f}")
    
    # Evaluate at different distances
    model.eval()
    print(f"  Evaluation:")
    results = {}
    
    with torch.no_grad():
        for dist in [5, 10, 20, 30, 50]:
            correct = 0
            for _ in range(10):
                x, targets = make_batch(distance=dist)
                out = model(x).mean(dim=1)
                preds = out.argmax(dim=-1)
                correct += (preds == targets).sum().item()
            
            acc = correct / 80
            results[dist] = acc
            print(f"    Distance {dist}: {acc*100:.1f}%")
    
    return results


def analyze_gradients(ModelClass, name: str):
    """Quick gradient analysis."""
    print(f"\n{name} Gradients:")
    
    model = ModelClass().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    
    x, targets = make_batch(distance=30)
    out = model(x).mean(dim=1)
    loss = F.cross_entropy(out, targets)
    
    opt.zero_grad()
    loss.backward()
    
    # Average gradient norm
    total_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.norm().item()
    
    print(f"  Total gradient norm: {total_grad:.6f}")


if __name__ == "__main__":
    print("="*60)
    print("ULTRA-FAST FUSION COMPARISON")
    print(f"Device: {device}")
    print("="*60)
    
    add_results = train_and_eval(MinimalAdditiveModel, "ADDITIVE FUSION")
    mult_results = train_and_eval(MinimalMultiplicativeModel, "MULTIPLICATIVE FUSION")
    
    analyze_gradients(MinimalAdditiveModel, "ADDITIVE")
    analyze_gradients(MinimalMultiplicativeModel, "MULTIPLICATIVE")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Distance':<12} {'Additive':<12} {'Multiplicative':<12} {'Better':<12}")
    print("-"*60)
    
    for dist in [5, 10, 20, 30, 50]:
        a = add_results.get(dist, 0)
        m = mult_results.get(dist, 0)
        better = "Mult" if m > a else "Add" if a > m else "Tie"
        print(f"{dist:<12} {a*100:<11.1f}% {m*100:<11.1f}% {better:<12}")
