#!/usr/bin/env python3
"""
Fast Fusion Comparison - Minimal Overhead

Simplified fusion test without torch overhead.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import List, Dict
import sys
from datetime import datetime

sys.path.insert(0, '/home/aiman/pi')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PHI = (1 + math.sqrt(5)) / 2


# =============================================================================
# Minimal Euler Attention Head
# =============================================================================

class EulerAttentionHead(nn.Module):
    """Phase-based attention using Euler transform."""
    
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.d_model = d_model
        self.head_dim = head_dim
        
        # Wavelengths for phase computation
        self.wavelengths = nn.Parameter(
            torch.arange(1, head_dim + 1, dtype=torch.float32) * 2 * math.pi / head_dim,
            requires_grad=False
        )
        self.phase_bias = nn.Parameter(torch.zeros(head_dim))
        
        # Linear projections
        self.W_q = nn.Linear(d_model, head_dim)
        self.W_k = nn.Linear(d_model, head_dim)
        self.W_v = nn.Linear(d_model, head_dim)
        self.W_o = nn.Linear(head_dim, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model) -> (batch, seq, d_model)"""
        batch, seq, _ = x.shape
        
        # Project to head dimension
        Q = self.W_q(x)  # (batch, seq, head_dim)
        K = self.W_k(x)
        V = self.W_v(x)
        
        # Euler transform: θ = x/wavelength + bias
        theta_q = Q / (self.wavelengths.unsqueeze(0).unsqueeze(0) + 1e-8) + self.phase_bias
        theta_k = K / (self.wavelengths.unsqueeze(0).unsqueeze(0) + 1e-8) + self.phase_bias
        
        # Project to phase space
        q_real = torch.cos(theta_q)
        q_imag = torch.sin(theta_q)
        k_real = torch.cos(theta_k)
        k_imag = torch.sin(theta_k)
        
        # Attention: cos(θ_q - θ_k) = cos(θ_q)cos(θ_k) + sin(θ_q)sin(θ_k)
        # (batch, seq, head_dim, 1) @ (batch, 1, head_dim, seq)
        sim = torch.matmul(q_real, k_real.transpose(-2, -1)) + torch.matmul(q_imag, k_imag.transpose(-2, -1))
        
        # Attention weights
        weights = F.softmax(sim / math.sqrt(self.head_dim), dim=-1)
        
        # Apply to values
        out = torch.matmul(weights, V)
        out = self.W_o(out)
        
        return out


# =============================================================================
# Minimal Resonant Layer
# =============================================================================

class ResonantLayer(nn.Module):
    """Simple per-neuron interference layer."""
    
    def __init__(self, d_model: int, num_neurons: int):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        self.fc_real = nn.Linear(d_model, num_neurons)
        self.fc_imag = nn.Linear(d_model, num_neurons)
        
        # Per-neuron wavelengths
        self.wavelengths = nn.Parameter(
            torch.arange(1, num_neurons + 1, dtype=torch.float32) * 2 * math.pi / num_neurons,
            requires_grad=False
        )
        self.phase_bias = nn.Parameter(torch.zeros(num_neurons))
        
        self.fc_out_real = nn.Linear(num_neurons, d_model)
        self.fc_out_imag = nn.Linear(num_neurons, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model) -> (batch, seq, d_model) with phase processing"""
        real = self.fc_real(x)
        imag = self.fc_imag(x)
        
        # Phase computation
        theta_real = real / (self.wavelengths.unsqueeze(0).unsqueeze(0) + 1e-8) + self.phase_bias
        theta_imag = imag / (self.wavelengths.unsqueeze(0).unsqueeze(0) + 1e-8) + self.phase_bias
        
        # Interference (sum of phase projections)
        proj_real = torch.cos(theta_real)
        proj_imag = torch.sin(theta_imag)
        
        interf = proj_real.sum(dim=-1, keepdim=True) + proj_imag.sum(dim=-1, keepdim=True)
        
        # Project back
        out_real = self.fc_out_real(proj_real)
        out_imag = self.fc_out_imag(proj_imag)
        
        return out_real, out_imag


# =============================================================================
# Fusion Models
# =============================================================================

class FusionModel(nn.Module):
    """Base model with attention + resonant fusion."""
    
    def __init__(self, vocab_size: int, d_model: int = 48, n_heads: int = 3, num_neurons: int = 96):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Multi-head attention
        self.heads = nn.ModuleList([EulerAttentionHead(d_model, d_model // n_heads) for _ in range(n_heads)])
        
        # Resonant layer
        self.resonant = ResonantLayer(d_model, num_neurons)
        
        # Output projection
        self.proj = nn.Linear(d_model, vocab_size)
    
    def fuse_outputs(self, attn_out: torch.Tensor, res_real: torch.Tensor, res_imag: torch.Tensor):
        """Fusion strategy - to be overridden."""
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq) -> (batch, seq, vocab_size)"""
        x = self.embed(x)
        
        # Attention heads
        attn_out = torch.cat([head(x) for head in self.heads], dim=-1)
        
        # Resonant layer
        res_real, res_imag = self.resonant(x)
        
        # Fuse
        out = self.fuse_outputs(attn_out, res_real, res_imag)
        
        # Project to vocab
        logits = self.proj(out)
        return logits


class AdditiveFusion(FusionModel):
    """x_out = attention + resonant"""
    
    def fuse_outputs(self, attn_out, res_real, res_imag):
        return attn_out + res_real + res_imag


class MultiplicativeFusion(FusionModel):
    """x_out = attention * resonant (GLU-style)"""
    
    def fuse_outputs(self, attn_out, res_real, res_imag):
        # Gate by resonant: attention * (1 + resonant)
        gate = 1.0 + F.tanh(res_real + res_imag)
        return attn_out * gate


# =============================================================================
# Needle Task
# =============================================================================

def make_needle_batch(batch_size: int, distance: int):
    """Create needle-in-haystack batches."""
    seqs, targets = [], []
    for _ in range(batch_size):
        needle = random.randint(1, 10)
        haystack = [random.randint(50, 99) for _ in range(distance)]
        seq = [needle] + haystack + [0]
        seqs.append(seq)
        targets.append(needle)
    
    return torch.tensor(seqs, dtype=torch.long, device=device), torch.tensor(targets, dtype=torch.long, device=device)


def train_model(ModelClass, name: str, epochs: int = 15, steps_per_epoch: int = 8):
    """Train and evaluate model."""
    print(f"\n{'='*70}")
    print(f"Training: {name}")
    print(f"{'='*70}")
    
    model = ModelClass().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Curriculum learning
    distances = [5, 10, 20, 30, 50]
    for epoch in range(epochs):
        model.train()
        max_dist = distances[min(epoch // 3, len(distances) - 1)]
        
        loss_sum = 0
        for step in range(steps_per_epoch):
            seqs, targets = make_needle_batch(16, max_dist)
            logits = model(seqs)
            
            # Loss on last token
            pred_logits = logits[:, -1, :]
            loss = F.cross_entropy(pred_logits, targets)
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            loss_sum += loss.item()
        
        print(f"  Epoch {epoch+1}: max_dist={max_dist}, loss={loss_sum/steps_per_epoch:.4f}")
    
    # Evaluate
    print(f"\nFinal Evaluation ({name}):")
    model.eval()
    results = {}
    
    with torch.no_grad():
        for dist in distances:
            correct, total = 0, 0
            for _ in range(20):
                seqs, targets = make_needle_batch(16, dist)
                logits = model(seqs)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += seqs.size(0)
            
            acc = correct / total
            results[dist] = acc
            print(f"  Distance {dist:2d}: {acc*100:5.1f}%")
    
    return results


def analyze_gradient_flow(ModelClass, name: str):
    """Analyze gradient magnitudes."""
    print(f"\n{'='*70}")
    print(f"Gradient Analysis: {name}")
    print(f"{'='*70}")
    
    model = ModelClass().to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # Forward pass
    seqs, targets = make_needle_batch(8, 30)
    logits = model(seqs)
    loss = F.cross_entropy(logits[:, -1, :], targets)
    
    # Backward
    optim.zero_grad()
    loss.backward()
    
    # Analyze gradients by layer type
    attn_grads = []
    res_grads = []
    
    for name_param, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if 'heads' in name_param or 'W_q' in name_param or 'W_k' in name_param:
                attn_grads.append(grad_norm)
            elif 'resonant' in name_param:
                res_grads.append(grad_norm)
    
    attn_mean = np.mean(attn_grads) if attn_grads else 0
    res_mean = np.mean(res_grads) if res_grads else 0
    
    print(f"  Attention gradient norm (mean): {attn_mean:.6f}")
    print(f"  Resonant gradient norm (mean): {res_mean:.6f}")
    print(f"  Ratio (attention/resonant): {attn_mean / (res_mean + 1e-8):.3f}x")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ATTENTION-RESONANT FUSION COMPARISON (FAST)")
    print(f"Device: {device}")
    print("="*70)
    
    # Test both strategies
    results_additive = train_model(AdditiveFusion, "Additive Fusion", epochs=15, steps_per_epoch=8)
    results_mult = train_model(MultiplicativeFusion, "Multiplicative Fusion", epochs=15, steps_per_epoch=8)
    
    # Gradient analysis
    analyze_gradient_flow(AdditiveFusion, "Additive Fusion")
    analyze_gradient_flow(MultiplicativeFusion, "Multiplicative Fusion")
    
    # Comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Distance':<15} {'Additive':<15} {'Multiplicative':<15} {'Winner':<15}")
    print("-" * 60)
    
    for dist in [5, 10, 20, 30, 50]:
        add_acc = results_additive.get(dist, 0)
        mult_acc = results_mult.get(dist, 0)
        winner = "Multiplicative" if mult_acc > add_acc else "Additive" if add_acc > mult_acc else "Tie"
        print(f"{dist:<15} {add_acc*100:<14.1f}% {mult_acc*100:<14.1f}% {winner:<15}")
