#!/usr/bin/env python3
"""
Attention-Resonant Fusion: Euler Transform + Resonant Layers

Architecture:
- Euler transform for attention (phase-based query/key matching)
- Resonant layer (per-dimension interference) 
- Two fusion strategies: Additive vs. Multiplicative (GLU-style)
- Test on needle-in-haystack task with distances 5-50 tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import List, Dict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PHI = (1 + math.sqrt(5)) / 2


# =============================================================================
# Euler Transform Attention
# =============================================================================

class EulerAttention(nn.Module):
    """Phase-based attention using Euler transform."""
    
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        
        # Wavelengths for phase computation
        self.register_buffer(
            'wavelengths', 
            torch.arange(1, d_model + 1, dtype=torch.float32) * (2 * math.pi / d_model)
        )
        self.phase_bias = nn.Parameter(torch.zeros(d_model))
        
        # Projections
        self.Q = nn.Linear(d_model, d_model)
        self.K = nn.Linear(d_model, d_model)
        self.V = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model) -> (batch, seq, d_model)"""
        batch, seq, dim = x.shape
        
        # Project to query/key space
        Q = self.Q(x)  # (batch, seq, d)
        K = self.K(x)
        V = self.V(x)
        
        # Euler transform: θ = x/wavelength + bias
        theta_q = Q / (self.wavelengths + 1e-8) + self.phase_bias
        theta_k = K / (self.wavelengths + 1e-8) + self.phase_bias
        
        # Project to phase space [cos(θ), sin(θ)]
        q_real = torch.cos(theta_q)
        q_imag = torch.sin(theta_q)
        k_real = torch.cos(theta_k)
        k_imag = torch.sin(theta_k)
        
        # Phase similarity: cos(θ_q - θ_k) = cos(θ_q)cos(θ_k) + sin(θ_q)sin(θ_k)
        sim = torch.matmul(q_real, k_real.transpose(-2, -1)) + \
              torch.matmul(q_imag, k_imag.transpose(-2, -1))
        
        # Attention weights
        weights = F.softmax(sim / math.sqrt(dim), dim=-1)
        
        # Apply to values
        attn_out = torch.matmul(weights, V)
        return attn_out


# =============================================================================
# Resonant Layer
# =============================================================================

class ResonantLayer(nn.Module):
    """Per-dimension interference layer using Euler transform."""
    
    def __init__(self, d_model: int, num_neurons: int = None):
        super().__init__()
        if num_neurons is None:
            num_neurons = d_model * 2
        
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        # Wavelengths for each neuron
        self.register_buffer(
            'wavelengths',
            torch.arange(1, num_neurons + 1, dtype=torch.float32) * (2 * math.pi / num_neurons)
        )
        self.phase_bias = nn.Parameter(torch.zeros(num_neurons))
        
        # Projections
        self.fc_real = nn.Linear(d_model, num_neurons)
        self.fc_imag = nn.Linear(d_model, num_neurons)
        self.fc_out = nn.Linear(num_neurons * 2, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq, d_model) -> (batch, seq, d_model)"""
        batch, seq, dim = x.shape
        
        # Project to neuron space
        real_in = self.fc_real(x)  # (batch, seq, num_neurons)
        imag_in = self.fc_imag(x)
        
        # Euler transform: compute phase for each neuron
        theta_real = real_in / (self.wavelengths + 1e-8) + self.phase_bias
        theta_imag = imag_in / (self.wavelengths + 1e-8) + self.phase_bias
        
        # Project to phase space
        proj_real = torch.cos(theta_real)  # (batch, seq, num_neurons)
        proj_imag = torch.sin(theta_imag)
        
        # Interference: combine the two phase projections
        combined = torch.cat([proj_real, proj_imag], dim=-1)  # (batch, seq, 2*num_neurons)
        
        # Project back to model dimension
        res_out = self.fc_out(combined)
        return res_out


# =============================================================================
# Fusion Models
# =============================================================================

class FusionBase(nn.Module):
    """Base model with attention + resonant layers."""
    
    def __init__(self, vocab_size: int, d_model: int = 48):
        super().__init__()
        self.d_model = d_model
        
        self.embed = nn.Embedding(vocab_size, d_model)
        self.attn = EulerAttention(d_model)
        self.resonant = ResonantLayer(d_model, d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)
    
    def fuse_outputs(self, attn_out: torch.Tensor, res_out: torch.Tensor) -> torch.Tensor:
        """Combine attention and resonant outputs. Subclass override this."""
        raise NotImplementedError
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq) -> (batch, seq, vocab_size)"""
        # Embed
        x_embed = self.embed(x)
        
        # Attention output
        attn_out = self.attn(x_embed)
        
        # Resonant output
        res_out = self.resonant(x_embed)
        
        # Fuse
        fused = self.fuse_outputs(attn_out, res_out)
        
        # Project to vocabulary
        logits = self.out_proj(fused)
        return logits


class AdditiveFusionModel(FusionBase):
    """Additive fusion: output = attention + resonant"""
    
    def fuse_outputs(self, attn_out, res_out):
        return attn_out + res_out


class MultiplicativeFusionModel(FusionBase):
    """Multiplicative fusion: output = attention * (1 + gate(resonant))"""
    
    def fuse_outputs(self, attn_out, res_out):
        # GLU-style gating with resonant as gate
        gate = 1.0 + torch.tanh(res_out)
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
        seq = [needle] + haystack + [0]  # 0 = trigger token
        seqs.append(seq)
        targets.append(needle)
    
    seqs_t = torch.tensor(seqs, dtype=torch.long, device=device)
    targets_t = torch.tensor(targets, dtype=torch.long, device=device)
    return seqs_t, targets_t


def train_model(ModelClass, name: str, epochs: int = 20, steps_per_epoch: int = 10):
    """Train and evaluate model."""
    print(f"\n{'='*70}")
    print(f"Training: {name}")
    print(f"{'='*70}")
    
    model = ModelClass(vocab_size=100, d_model=48).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, epochs)
    
    # Training with curriculum
    distances = [5, 10, 20, 30, 50]
    
    for epoch in range(epochs):
        model.train()
        
        # Curriculum: increase max distance over time
        max_dist = distances[min(epoch // 4, len(distances) - 1)]
        
        loss_sum = 0
        for step in range(steps_per_epoch):
            seqs, targets = make_needle_batch(batch_size=16, distance=max_dist)
            
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            
            loss_sum += loss.item()
        
        scheduler.step()
        avg_loss = loss_sum / steps_per_epoch
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:2d}: max_dist={max_dist:2d}, loss={avg_loss:.4f}")
    
    # Evaluation
    print(f"\nFinal Evaluation:")
    model.eval()
    results = {}
    
    with torch.no_grad():
        for dist in distances:
            correct, total = 0, 0
            for _ in range(20):
                seqs, targets = make_needle_batch(batch_size=16, distance=dist)
                logits = model(seqs)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += seqs.size(0)
            
            acc = correct / total
            results[dist] = acc
            status = "✓" if acc > 0.7 else "◐" if acc > 0.3 else "✗"
            print(f"  {status} Distance {dist:2d}: {acc*100:5.1f}%")
    
    return results


def analyze_gradients(ModelClass, name: str):
    """Analyze gradient magnitudes by component."""
    print(f"\n{'='*70}")
    print(f"Gradient Analysis: {name}")
    print(f"{'='*70}")
    
    model = ModelClass(vocab_size=100, d_model=48).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Forward pass on longer sequence
    seqs, targets = make_needle_batch(batch_size=8, distance=30)
    logits = model(seqs)
    loss = F.cross_entropy(logits[:, -1, :], targets)
    
    # Backward
    optim.zero_grad()
    loss.backward()
    
    # Collect gradients by component
    grad_stats = {
        'embed': [],
        'attn': [],
        'resonant': [],
        'out': []
    }
    
    for pname, param in model.named_parameters():
        if param.grad is not None:
            g_norm = param.grad.norm().item()
            if 'embed' in pname:
                grad_stats['embed'].append(g_norm)
            elif 'attn' in pname:
                grad_stats['attn'].append(g_norm)
            elif 'resonant' in pname:
                grad_stats['resonant'].append(g_norm)
            elif 'out' in pname:
                grad_stats['out'].append(g_norm)
    
    # Print statistics
    for component, norms in grad_stats.items():
        if norms:
            mean_norm = np.mean(norms)
            max_norm = np.max(norms)
            print(f"  {component:12s}: mean={mean_norm:.6f}, max={max_norm:.6f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("ATTENTION-RESONANT FUSION COMPARISON")
    print(f"Device: {device}")
    print("="*70)
    
    # Train both models
    add_results = train_model(AdditiveFusionModel, "Additive Fusion", epochs=20, steps_per_epoch=10)
    mult_results = train_model(MultiplicativeFusionModel, "Multiplicative Fusion", epochs=20, steps_per_epoch=10)
    
    # Gradient analysis
    analyze_gradients(AdditiveFusionModel, "Additive Fusion")
    analyze_gradients(MultiplicativeFusionModel, "Multiplicative Fusion")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n{'Distance':<12} {'Additive':<15} {'Multiplicative':<15} {'Better':<15}")
    print("-" * 60)
    
    for dist in [5, 10, 20, 30, 50]:
        add_acc = add_results.get(dist, 0)
        mult_acc = mult_results.get(dist, 0)
        
        if mult_acc > add_acc:
            better = "Multiplicative"
        elif add_acc > mult_acc:
            better = "Additive"
        else:
            better = "Tie"
        
        diff = abs(mult_acc - add_acc) * 100
        print(f"{dist:<12} {add_acc*100:<14.1f}% {mult_acc*100:<14.1f}% {better:<15}")
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    
    add_avg = np.mean(list(add_results.values()))
    mult_avg = np.mean(list(mult_results.values()))
    
    print(f"\nAverage accuracy:")
    print(f"  Additive:       {add_avg*100:.1f}%")
    print(f"  Multiplicative: {mult_avg*100:.1f}%")
    print(f"  Winner: {'Multiplicative' if mult_avg > add_avg else 'Additive'}")
    
    # Long-range performance (distances >= 30)
    long_range_add = np.mean([add_results.get(d, 0) for d in [30, 50]])
    long_range_mult = np.mean([mult_results.get(d, 0) for d in [30, 50]])
    
    print(f"\nLong-range (dist >= 30):")
    print(f"  Additive:       {long_range_add*100:.1f}%")
    print(f"  Multiplicative: {long_range_mult*100:.1f}%")
    print(f"  Winner: {'Multiplicative' if long_range_mult > long_range_add else 'Additive'}")
