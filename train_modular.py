#!/usr/bin/env python3
"""
Train RIN on Modular Arithmetic (Grokking Task)

The classic test for generalization: (a + b) mod p

This task famously exhibits "grokking" - sudden generalization after
prolonged training. RIN with Euler's formula achieves:
- 100% test accuracy
- 0 stability dips  
- Grokking in ~60 epochs

Usage:
    python train_modular.py
    python train_modular.py --p 113 --epochs 500
    python train_modular.py --wrap_time  # Test with t mod 2π
    python train_modular.py --compare    # Compare both approaches
"""

import argparse
import math
import random
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rin import RINModel, get_global_lut, PHI, wrap_time_periodic


class ModularAdditionDataset(Dataset):
    """Dataset for (a + b) mod p"""
    
    def __init__(self, p: int, split: str = "train", train_frac: float = 0.5, seed: int = 42):
        self.p = p
        self.data = []
        
        # Generate all pairs
        all_pairs = [(a, b) for a in range(p) for b in range(p)]
        random.seed(seed)
        random.shuffle(all_pairs)
        
        split_idx = int(len(all_pairs) * train_frac)
        pairs = all_pairs[:split_idx] if split == "train" else all_pairs[split_idx:]
        
        for a, b in pairs:
            # Input: [a, b, =]  Output: (a+b) mod p
            self.data.append((torch.tensor([a, b, p]), (a + b) % p))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class ModularRIN(nn.Module):
    """RIN adapted for modular arithmetic classification."""
    
    def __init__(self, vocab_size, d_model=48, num_layers=2, num_neurons=96, use_swish=True, wrap_time=False, separate_theta=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.wrap_time = wrap_time
        self.separate_theta = separate_theta  # New: use separate theta_real/theta_imag
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        self.layers = nn.ModuleList([
            ModularResonantLayer(d_model, num_neurons, use_swish=use_swish, wrap_time=wrap_time)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.5)
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, input_ids):
        lut = self._get_lut(input_ids.device)
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Euler hidden state
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        # Pre-compute timestep tensors for torch.compile compatibility
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) * PHI
        
        for t in range(seq_len):
            t_val = t_indices[t]  # Scalar tensor
            
            # Wrap time to [0, 2π) if enabled (detached modulo for gradient flow)
            t_val_use = wrap_time_periodic(t_val) if self.wrap_time else t_val
            
            wavelength = 1.0 + w_emb[:, t, :].abs()
            
            if self.separate_theta:
                # NEW: Separate theta computation preserves real/imag distinction
                theta_real = h_real / wavelength + b_emb[:, t, :] + t_val_use
                theta_imag = h_imag / wavelength + b_emb[:, t, :] + t_val_use
                
                # Euler decomposition for each component
                sin_real, cos_real = lut.lookup_sin_cos(theta_real)
                sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
                
                # Complex multiplication: (cos_r + i·sin_r) × (cos_i + i·sin_i)
                # Preserves BOTH gradient paths through h_real and h_imag
                h_real = cos_real * cos_imag - sin_real * sin_imag
                h_imag = cos_real * sin_imag + sin_real * cos_imag
            else:
                # OLD: Collapsed theta (loses information!)
                h_combined = h_real + h_imag
                theta = h_combined / wavelength + b_emb[:, t, :] + t_val_use
                h_imag, h_real = lut.lookup_sin_cos(theta)
            
            # Process through layers
            h = h_real + h_imag
            for layer in self.layers:
                h = h + layer(h, t_val_use)
        
        return self.output_proj(h)


class ModularResonantLayer(nn.Module):
    """
    Resonant layer optimized for modular arithmetic.
    
    Uses matrix-multiply formulation where each neuron is tuned to specific
    input patterns (like Fourier basis functions). This is more appropriate
    for the modular arithmetic task where we need discrete representations.
    
    θ_n = x · W[n] + b[n] + t  (one theta per neuron)
    out = proj_real(cos(θ)) + proj_imag(sin(θ))
    
    This is equivalent to the original RIN layer - each neuron resonates
    with a specific weighted combination of inputs.
    """
    
    def __init__(self, d_model, num_neurons, use_swish=True, wrap_time=False):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.use_swish = use_swish
        self.wrap_time = wrap_time
        
        # Pattern matching weights: each neuron has a pattern vector
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
        """
        Args:
            x: Input (batch, d_model)
            t: Timestep scalar or tensor
        """
        lut = self._get_lut(x.device)
        
        # Each neuron computes weighted sum of inputs
        # θ_n = x · W[n] + b[n] + t
        theta = x @ self.W.T + self.bias  # (batch, num_neurons)
        
        # Add time
        if isinstance(t, (int, float)):
            theta = theta + t
        elif t.dim() == 0:
            theta = theta + t
        else:
            theta = theta + t.unsqueeze(-1) if t.dim() == 1 else theta + t
        
        # Euler decomposition
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # Project back to d_model
        out = self.proj_real(cos_theta) + self.proj_imag(sin_theta)
        return F.silu(out) if self.use_swish else out


def get_log_interval(epoch, total_epochs):
    """Dynamic log interval: frequent at start/end, sparse in middle."""
    if epoch < 20:
        return 1
    elif epoch < 50:
        return 5
    elif epoch < 100:
        return 10
    elif epoch > total_epochs - 50:
        return 10
    elif epoch > total_epochs - 20:
        return 5
    return 25


def train(args, wrap_time=None, separate_theta=None):
    """
    Train the model. 
    
    Args:
        args: Command line arguments
        wrap_time: Override for wrap_time flag (used in comparison mode)
        separate_theta: Override for separate_theta flag (used in euler comparison mode)
    """
    # Determine settings
    use_wrap_time = wrap_time if wrap_time is not None else args.wrap_time
    use_separate_theta = separate_theta if separate_theta is not None else True  # Default to new behavior
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Golden ratio φ = {PHI:.6f}")
    print(f"Task: (a + b) mod {args.p}")
    print(f"Time wrapping: {'t mod 2π (detached)' if use_wrap_time else 'absolute t'}")
    print(f"Euler transform: {'separate θ_real/θ_imag (new)' if use_separate_theta else 'collapsed h_combined (old)'}")
    
    # Data
    train_ds = ModularAdditionDataset(args.p, "train", args.train_frac)
    test_ds = ModularAdditionDataset(args.p, "test", args.train_frac)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, pin_memory=True)
    
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    
    # Model
    vocab_size = args.p + 1
    model = ModularRIN(
        vocab_size, 
        d_model=args.d_model, 
        num_layers=args.num_layers, 
        num_neurons=args.num_neurons,
        use_swish=args.use_swish,
        wrap_time=use_wrap_time,
        separate_theta=use_separate_theta,
    ).to(device)
    
    if hasattr(torch, 'compile') and args.compile:
        model = torch.compile(model)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Training loop
    best_test_acc = 0
    stability_dips = 0
    prev_test_acc = 0
    last_log = -1
    
    print(f"\n{'Epoch':>6} | {'Loss':>8} | {'Train':>7} | {'Test':>7}")
    print("-" * 40)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            correct += (logits.argmax(-1) == y).sum().item()
            total += y.size(0)
        
        scheduler.step()
        train_acc = correct / total
        
        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                correct += (logits.argmax(-1) == y).sum().item()
                total += y.size(0)
        
        test_acc = correct / total
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        if test_acc < prev_test_acc - 0.05:
            stability_dips += 1
        prev_test_acc = test_acc
        
        # Dynamic logging
        interval = get_log_interval(epoch, args.epochs)
        if (epoch + 1) - last_log >= interval or epoch == args.epochs - 1:
            print(f"{epoch+1:6d} | {total_loss/len(train_loader):8.4f} | {train_acc*100:6.1f}% | {test_acc*100:6.1f}%")
            last_log = epoch + 1
    
    print("-" * 40)
    print(f"Best test accuracy: {best_test_acc*100:.2f}%")
    print(f"Stability dips: {stability_dips}")
    
    return {"best_test_acc": best_test_acc, "stability_dips": stability_dips}


def main():
    parser = argparse.ArgumentParser(description="Train RIN on modular arithmetic")
    parser.add_argument("--p", type=int, default=97, help="Modulus (prime)")
    parser.add_argument("--train_frac", type=float, default=0.5, help="Training fraction")
    parser.add_argument("--d_model", type=int, default=48, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--num_neurons", type=int, default=96, help="Neurons per layer")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1.0, help="Weight decay")
    parser.add_argument("--use_swish", action="store_true", default=True, help="Use swish activation")
    parser.add_argument("--no_swish", action="store_false", dest="use_swish", help="Disable swish")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--wrap_time", action="store_true", help="Use t mod 2π with detached modulo")
    parser.add_argument("--compare", action="store_true", help="Compare absolute t vs wrapped t")
    parser.add_argument("--compare_euler", action="store_true", help="Compare old vs new euler_transform")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("RIN - Modular Arithmetic (Grokking)")
    print("=" * 50)
    
    if args.compare_euler:
        # Compare old collapsed euler vs new separated euler
        print("\n" + "=" * 60)
        print("EULER COMPARISON: Collapsed vs Separated theta propagation")
        print("=" * 60)
        
        print("\n" + "-" * 60)
        print("TEST 1: OLD - Collapsed h_combined = h_real + h_imag")
        print("-" * 60)
        result_collapsed = train(args, wrap_time=True, separate_theta=False)
        
        print("\n" + "-" * 60)
        print("TEST 2: NEW - Separate θ_real and θ_imag propagation")
        print("-" * 60)
        result_separated = train(args, wrap_time=True, separate_theta=True)
        
        # Summary
        print("\n" + "=" * 60)
        print("EULER TRANSFORM COMPARISON SUMMARY")
        print("=" * 60)
        print(f"{'Metric':<25} {'Collapsed (old)':>18} {'Separated (new)':>18}")
        print("-" * 62)
        print(f"{'Best Test Accuracy':<25} {result_collapsed['best_test_acc']*100:>17.2f}% {result_separated['best_test_acc']*100:>17.2f}%")
        print(f"{'Stability Dips':<25} {result_collapsed['stability_dips']:>18} {result_separated['stability_dips']:>18}")
        
        # Verdict
        print("\n" + "-" * 62)
        improvement = result_separated['best_test_acc'] - result_collapsed['best_test_acc']
        if improvement > 0.001:
            print(f"✓ Separated theta improves accuracy by {improvement*100:.2f}%!")
        elif improvement >= 0:
            print("~ Results are similar")
        else:
            print(f"✗ Collapsed theta performed better by {-improvement*100:.2f}%")
            
    elif args.compare:
        # Run comparison between both approaches
        print("\n" + "=" * 50)
        print("COMPARISON MODE: Testing both time representations")
        print("=" * 50)
        
        print("\n" + "-" * 50)
        print("TEST 1: Absolute time (current implementation)")
        print("-" * 50)
        result_absolute = train(args, wrap_time=False)
        
        print("\n" + "-" * 50)
        print("TEST 2: Wrapped time (t mod 2π with detached modulo)")
        print("-" * 50)
        result_wrapped = train(args, wrap_time=True)
        
        # Summary
        print("\n" + "=" * 50)
        print("COMPARISON SUMMARY")
        print("=" * 50)
        print(f"{'Metric':<25} {'Absolute t':>15} {'t mod 2π':>15}")
        print("-" * 55)
        print(f"{'Best Test Accuracy':<25} {result_absolute['best_test_acc']*100:>14.2f}% {result_wrapped['best_test_acc']*100:>14.2f}%")
        print(f"{'Stability Dips':<25} {result_absolute['stability_dips']:>15} {result_wrapped['stability_dips']:>15}")
        
        # Verdict
        print("\n" + "-" * 55)
        if result_wrapped['best_test_acc'] >= result_absolute['best_test_acc']:
            if result_wrapped['stability_dips'] <= result_absolute['stability_dips']:
                print("✓ Wrapped time (t mod 2π) performs at least as well!")
            else:
                print("~ Mixed results: wrapped time has better accuracy but more dips")
        else:
            print("✗ Absolute time performs better on this run")
    else:
        train(args)


if __name__ == "__main__":
    main()