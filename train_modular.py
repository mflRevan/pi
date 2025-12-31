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
"""

import argparse
import math
import random
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rin import RINModel, get_global_lut, PHI


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
    
    def __init__(self, vocab_size, d_model=48, num_layers=2, num_neurons=96, use_swish=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        self.layers = nn.ModuleList([
            ModularResonantLayer(d_model, num_neurons, use_swish=use_swish)
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
        
        # Euler hidden state
        h_real = torch.zeros(batch_size, self.d_model, device=input_ids.device)
        h_imag = torch.zeros(batch_size, self.d_model, device=input_ids.device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        for t in range(seq_len):
            t_val = t * PHI
            wavelength = 1.0 + w_emb[:, t, :].abs()
            
            h_combined = h_real + h_imag
            theta = h_combined / wavelength + b_emb[:, t, :] + t_val
            
            # Euler decomposition
            h_imag, h_real = lut.lookup_sin_cos(theta)
            
            # Process through layers
            h = h_real + h_imag
            for layer in self.layers:
                h = h + layer(h, t_val)
        
        return self.output_proj(h)


class ModularResonantLayer(nn.Module):
    """Euler resonant layer for modular arithmetic."""
    
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


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Golden ratio φ = {PHI:.6f}")
    print(f"Task: (a + b) mod {args.p}")
    
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
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("RIN - Modular Arithmetic (Grokking)")
    print("=" * 50)
    
    train(args)


if __name__ == "__main__":
    main()
