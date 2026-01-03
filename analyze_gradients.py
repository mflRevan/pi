#!/usr/bin/env python3
"""
Gradient and Learning Dynamics Analysis: Collapsed vs Separated Euler Transform

This script analyzes the fundamental differences in gradient flow between:
1. COLLAPSED: h_combined = h_real + h_imag → single theta → single (sin, cos)
2. SEPARATED: theta_real, theta_imag → complex multiplication → preserves both paths

Key metrics analyzed:
- Gradient magnitudes through h_real vs h_imag paths
- Gradient correlation/independence between paths
- Effective rank of gradient covariance
- Learning specialization (do real/imag learn different things?)
"""

import argparse
import math
import random
from typing import Dict, List, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rin import get_global_lut, PHI, wrap_time_periodic


class ModularAdditionDataset(Dataset):
    """Dataset for (a + b) mod p"""
    
    def __init__(self, p: int, split: str = "train", train_frac: float = 0.5, seed: int = 42):
        self.p = p
        self.data = []
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        random.seed(seed)
        random.shuffle(all_pairs)
        split_idx = int(len(all_pairs) * train_frac)
        pairs = all_pairs[:split_idx] if split == "train" else all_pairs[split_idx:]
        for a, b in pairs:
            self.data.append((torch.tensor([a, b, p]), (a + b) % p))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class AnalyzableModularRIN(nn.Module):
    """RIN with hooks for gradient analysis."""
    
    def __init__(self, vocab_size, d_model=48, num_layers=2, num_neurons=96, 
                 use_swish=True, wrap_time=True, separate_theta=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.wrap_time = wrap_time
        self.separate_theta = separate_theta
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        self.layers = nn.ModuleList([
            AnalyzableResonantLayer(d_model, num_neurons, use_swish=use_swish)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.5)
        
        self._lut = None
        
        # Storage for gradient analysis
        self.grad_h_real_history = []
        self.grad_h_imag_history = []
        self.grad_theta_real_history = []
        self.grad_theta_imag_history = []
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, input_ids):
        lut = self._get_lut(input_ids.device)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) * PHI
        
        for t in range(seq_len):
            t_val = t_indices[t]
            t_val_use = wrap_time_periodic(t_val) if self.wrap_time else t_val
            wavelength = 1.0 + w_emb[:, t, :].abs()
            
            if self.separate_theta:
                # Separated approach - preserves complex plane
                theta_real = h_real / wavelength + b_emb[:, t, :] + t_val_use
                theta_imag = h_imag / wavelength + b_emb[:, t, :] + t_val_use
                
                # Register hooks for gradient capture on last timestep
                if t == seq_len - 1 and self.training:
                    theta_real.register_hook(lambda g: self.grad_theta_real_history.append(g.detach().clone()))
                    theta_imag.register_hook(lambda g: self.grad_theta_imag_history.append(g.detach().clone()))
                
                sin_real, cos_real = lut.lookup_sin_cos(theta_real)
                sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
                
                # Complex multiplication
                h_real = cos_real * cos_imag - sin_real * sin_imag
                h_imag = cos_real * sin_imag + sin_real * cos_imag
            else:
                # Collapsed approach - loses information
                h_combined = h_real + h_imag
                theta = h_combined / wavelength + b_emb[:, t, :] + t_val_use
                
                if t == seq_len - 1 and self.training:
                    theta.register_hook(lambda g: self.grad_theta_real_history.append(g.detach().clone()))
                
                h_imag, h_real = lut.lookup_sin_cos(theta)
            
            # Capture h gradients on last timestep
            if t == seq_len - 1 and self.training:
                h_real.register_hook(lambda g: self.grad_h_real_history.append(g.detach().clone()))
                h_imag.register_hook(lambda g: self.grad_h_imag_history.append(g.detach().clone()))
            
            h = h_real + h_imag
            for layer in self.layers:
                h = h + layer(h, t_val_use)
        
        return self.output_proj(h)
    
    def clear_grad_history(self):
        self.grad_h_real_history.clear()
        self.grad_h_imag_history.clear()
        self.grad_theta_real_history.clear()
        self.grad_theta_imag_history.clear()


class AnalyzableResonantLayer(nn.Module):
    def __init__(self, d_model, num_neurons, use_swish=True):
        super().__init__()
        self.use_swish = use_swish
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        self.proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x, t):
        lut = self._get_lut(x.device)
        theta = x @ self.W.T + self.bias + t
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        out = self.proj_real(cos_theta) + self.proj_imag(sin_theta)
        return F.silu(out) if self.use_swish else out


def compute_gradient_metrics(grad_real: torch.Tensor, grad_imag: torch.Tensor) -> Dict[str, float]:
    """Compute key gradient metrics."""
    # Flatten gradients
    g_r = grad_real.view(-1)
    g_i = grad_imag.view(-1)
    
    # Magnitudes
    mag_real = g_r.norm().item()
    mag_imag = g_i.norm().item()
    
    # Correlation (how similar are the gradients?)
    if mag_real > 1e-8 and mag_imag > 1e-8:
        correlation = (g_r @ g_i / (mag_real * mag_imag)).item()
    else:
        correlation = 0.0
    
    # Angle between gradients (in degrees)
    angle = math.acos(max(-1, min(1, correlation))) * 180 / math.pi
    
    # Ratio of magnitudes
    ratio = mag_real / (mag_imag + 1e-8)
    
    # Independence metric: 1 - |correlation| (higher = more independent)
    independence = 1 - abs(correlation)
    
    return {
        'mag_real': mag_real,
        'mag_imag': mag_imag,
        'correlation': correlation,
        'angle_deg': angle,
        'mag_ratio': ratio,
        'independence': independence,
    }


def analyze_model(args, separate_theta: bool) -> Dict[str, List[float]]:
    """Train a model and collect gradient statistics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_ds = ModularAdditionDataset(args.p, "train", args.train_frac)
    test_ds = ModularAdditionDataset(args.p, "test", args.train_frac)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    
    vocab_size = args.p + 1
    model = AnalyzableModularRIN(
        vocab_size, d_model=args.d_model, num_layers=args.num_layers,
        num_neurons=args.num_neurons, wrap_time=True, separate_theta=separate_theta
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Metrics storage
    metrics = {
        'epoch': [],
        'train_acc': [],
        'test_acc': [],
        'loss': [],
        'grad_mag_real': [],
        'grad_mag_imag': [],
        'grad_correlation': [],
        'grad_angle': [],
        'grad_independence': [],
        'theta_grad_mag': [],
    }
    
    for epoch in range(args.epochs):
        model.train()
        model.clear_grad_history()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
        
        scheduler.step()
        train_acc = correct / total
        
        # Test accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        
        # Analyze gradients (from last batch of training)
        if len(model.grad_h_real_history) > 0 and len(model.grad_h_imag_history) > 0:
            grad_real = model.grad_h_real_history[-1]
            grad_imag = model.grad_h_imag_history[-1]
            grad_metrics = compute_gradient_metrics(grad_real, grad_imag)
            
            theta_grad_mag = 0.0
            if len(model.grad_theta_real_history) > 0:
                theta_grad_mag = model.grad_theta_real_history[-1].norm().item()
                if separate_theta and len(model.grad_theta_imag_history) > 0:
                    theta_grad_mag += model.grad_theta_imag_history[-1].norm().item()
        else:
            grad_metrics = {k: 0.0 for k in ['mag_real', 'mag_imag', 'correlation', 'angle_deg', 'independence']}
            theta_grad_mag = 0.0
        
        # Store metrics
        metrics['epoch'].append(epoch)
        metrics['train_acc'].append(train_acc)
        metrics['test_acc'].append(test_acc)
        metrics['loss'].append(total_loss / len(train_loader))
        metrics['grad_mag_real'].append(grad_metrics['mag_real'])
        metrics['grad_mag_imag'].append(grad_metrics['mag_imag'])
        metrics['grad_correlation'].append(grad_metrics['correlation'])
        metrics['grad_angle'].append(grad_metrics['angle_deg'])
        metrics['grad_independence'].append(grad_metrics['independence'])
        metrics['theta_grad_mag'].append(theta_grad_mag)
        
        # Log periodically
        if epoch < 10 or epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"Epoch {epoch+1:3d} | Loss {metrics['loss'][-1]:.4f} | "
                  f"Train {train_acc*100:5.1f}% | Test {test_acc*100:5.1f}% | "
                  f"∇h_r {grad_metrics['mag_real']:.4f} | ∇h_i {grad_metrics['mag_imag']:.4f} | "
                  f"corr {grad_metrics['correlation']:+.3f} | indep {grad_metrics['independence']:.3f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Gradient Analysis: Collapsed vs Separated")
    parser.add_argument("--p", type=int, default=97)
    parser.add_argument("--train_frac", type=float, default=0.5)
    parser.add_argument("--d_model", type=int, default=48)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neurons", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1.0)
    args = parser.parse_args()
    
    print("=" * 80)
    print("GRADIENT ANALYSIS: Collapsed vs Separated Euler Transform")
    print("=" * 80)
    
    # Analyze collapsed approach
    print("\n" + "=" * 80)
    print("COLLAPSED: h_combined = h_real + h_imag → single theta")
    print("=" * 80)
    metrics_collapsed = analyze_model(args, separate_theta=False)
    
    # Analyze separated approach
    print("\n" + "=" * 80)
    print("SEPARATED: theta_real, theta_imag → complex multiplication")
    print("=" * 80)
    metrics_separated = analyze_model(args, separate_theta=True)
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("GRADIENT DYNAMICS COMPARISON SUMMARY")
    print("=" * 80)
    
    # Average metrics over training
    def avg(lst, start=10):
        return sum(lst[start:]) / len(lst[start:]) if len(lst) > start else sum(lst) / len(lst)
    
    print(f"\n{'Metric':<35} {'Collapsed':>15} {'Separated':>15}")
    print("-" * 65)
    print(f"{'Final Test Accuracy':<35} {metrics_collapsed['test_acc'][-1]*100:>14.2f}% {metrics_separated['test_acc'][-1]*100:>14.2f}%")
    print(f"{'Best Test Accuracy':<35} {max(metrics_collapsed['test_acc'])*100:>14.2f}% {max(metrics_separated['test_acc'])*100:>14.2f}%")
    print(f"{'Avg ∇h_real magnitude':<35} {avg(metrics_collapsed['grad_mag_real']):>15.4f} {avg(metrics_separated['grad_mag_real']):>15.4f}")
    print(f"{'Avg ∇h_imag magnitude':<35} {avg(metrics_collapsed['grad_mag_imag']):>15.4f} {avg(metrics_separated['grad_mag_imag']):>15.4f}")
    print(f"{'Avg gradient correlation':<35} {avg(metrics_collapsed['grad_correlation']):>+14.3f} {avg(metrics_separated['grad_correlation']):>+14.3f}")
    print(f"{'Avg gradient angle (deg)':<35} {avg(metrics_collapsed['grad_angle']):>15.1f} {avg(metrics_separated['grad_angle']):>15.1f}")
    print(f"{'Avg gradient independence':<35} {avg(metrics_collapsed['grad_independence']):>15.3f} {avg(metrics_separated['grad_independence']):>15.3f}")
    print(f"{'Avg theta gradient magnitude':<35} {avg(metrics_collapsed['theta_grad_mag']):>15.4f} {avg(metrics_separated['theta_grad_mag']):>15.4f}")
    
    print("\n" + "-" * 65)
    print("\nKEY INSIGHTS:")
    
    # Analyze independence
    ind_coll = avg(metrics_collapsed['grad_independence'])
    ind_sep = avg(metrics_separated['grad_independence'])
    print(f"\n1. GRADIENT INDEPENDENCE:")
    print(f"   Collapsed: {ind_coll:.3f} | Separated: {ind_sep:.3f}")
    if ind_sep > ind_coll:
        print(f"   → Separated has {(ind_sep/ind_coll - 1)*100:.1f}% more independent gradients")
        print(f"   → Real and imag channels learn MORE DIFFERENT features")
    else:
        print(f"   → Collapsed has more independent gradients (unexpected)")
    
    # Analyze magnitude balance
    ratio_coll = avg(metrics_collapsed['grad_mag_real']) / (avg(metrics_collapsed['grad_mag_imag']) + 1e-8)
    ratio_sep = avg(metrics_separated['grad_mag_real']) / (avg(metrics_separated['grad_mag_imag']) + 1e-8)
    print(f"\n2. GRADIENT MAGNITUDE BALANCE (real/imag ratio):")
    print(f"   Collapsed: {ratio_coll:.3f} | Separated: {ratio_sep:.3f}")
    print(f"   → Ratio closer to 1.0 = more balanced learning between channels")
    
    # Analyze angle
    angle_coll = avg(metrics_collapsed['grad_angle'])
    angle_sep = avg(metrics_separated['grad_angle'])
    print(f"\n3. GRADIENT ANGLE (degrees between ∇h_real and ∇h_imag):")
    print(f"   Collapsed: {angle_coll:.1f}° | Separated: {angle_sep:.1f}°")
    print(f"   → 90° = orthogonal (maximum diversity)")
    print(f"   → 0° or 180° = parallel (redundant)")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
