#!/usr/bin/env python3
"""
Needle in a Haystack Test for RIN

Tests the network's ability to recall a signal token through varying
amounts of noise/distractor tokens. This reveals the effective memory
horizon of the architecture.

Test structure:
    [TRIGGER] [SIGNAL=k] [noise...] [TRIGGER] -> predict k

The signal is placed at varying distances before the recall trigger.
A model with good memory should maintain high accuracy regardless of distance.

Usage:
    python train_needle.py
    python train_needle.py --max_distance 50 --epochs 100
"""

import argparse
import math
import random
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rin import get_global_lut, PHI


class NeedleDataset(Dataset):
    """
    Dataset for needle-in-haystack recall task.
    
    Sequence: [TRIGGER] [SIGNAL] [noise tokens...] [TRIGGER]
    Target: SIGNAL value
    
    The distance between signal and recall trigger varies.
    """
    
    def __init__(
        self, 
        num_samples: int,
        num_signals: int,      # Number of possible signal values
        min_distance: int,     # Minimum noise tokens between signal and trigger
        max_distance: int,     # Maximum noise tokens
        num_noise_tokens: int, # Number of distinct noise tokens
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.num_signals = num_signals
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.num_noise_tokens = num_noise_tokens
        
        # Token IDs:
        # 0 = TRIGGER token
        # 1 to num_signals = SIGNAL tokens
        # num_signals+1 to num_signals+num_noise_tokens = NOISE tokens
        self.trigger_id = 0
        self.signal_start = 1
        self.noise_start = num_signals + 1
        self.vocab_size = num_signals + num_noise_tokens + 1
        
        if seed is not None:
            random.seed(seed)
        
        self.data = []
        for _ in range(num_samples):
            signal = random.randint(0, num_signals - 1)
            distance = random.randint(min_distance, max_distance)
            
            # Build sequence: [TRIGGER, SIGNAL, noise..., TRIGGER]
            seq = [self.trigger_id, self.signal_start + signal]
            for _ in range(distance):
                noise = random.randint(self.noise_start, self.vocab_size - 1)
                seq.append(noise)
            seq.append(self.trigger_id)
            
            self.data.append((torch.tensor(seq, dtype=torch.long), signal, distance))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        seq, signal, distance = self.data[idx]
        return seq, signal, distance


def collate_fn(batch):
    """Collate with padding for variable length sequences."""
    seqs, signals, distances = zip(*batch)
    max_len = max(len(s) for s in seqs)
    
    # Pad sequences
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    
    return padded, torch.tensor(signals), torch.tensor(distances)


class NeedleRIN(nn.Module):
    """RIN model for needle-in-haystack task."""
    
    def __init__(
        self, 
        vocab_size: int,
        num_signals: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_neurons: int = 128,
        use_swish: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_signals = num_signals
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        self.layers = nn.ModuleList([
            NeedleResonantLayer(d_model, num_neurons, use_swish=use_swish)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(d_model, num_signals, bias=False)
        
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
            wavelength = 1.0 + w_emb[:, t, :].abs()
            
            h_combined = h_real + h_imag
            theta = h_combined / wavelength + b_emb[:, t, :] + t_val
            
            h_imag, h_real = lut.lookup_sin_cos(theta)
            
            h = h_real + h_imag
            for layer in self.layers:
                h = h + layer(h, t_val)
        
        return self.output_proj(h)


class NeedleResonantLayer(nn.Module):
    """Euler resonant layer for needle task."""
    
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


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for seqs, signals, distances in loader:
        seqs = seqs.to(device, non_blocking=True)
        signals = signals.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        logits = model(seqs)
        loss = F.cross_entropy(logits, signals)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        correct += (logits.argmax(-1) == signals).sum().item()
        total += signals.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate_by_distance(model, loader, device, max_distance):
    """Evaluate accuracy broken down by distance."""
    model.eval()
    
    # Track accuracy per distance
    correct_by_dist = {d: 0 for d in range(max_distance + 1)}
    total_by_dist = {d: 0 for d in range(max_distance + 1)}
    
    with torch.no_grad():
        for seqs, signals, distances in loader:
            seqs = seqs.to(device)
            signals = signals.to(device)
            
            logits = model(seqs)
            preds = logits.argmax(-1)
            
            for pred, signal, dist in zip(preds, signals, distances):
                d = dist.item()
                total_by_dist[d] += 1
                if pred == signal:
                    correct_by_dist[d] += 1
    
    # Compute accuracy per distance
    acc_by_dist = {}
    for d in range(max_distance + 1):
        if total_by_dist[d] > 0:
            acc_by_dist[d] = correct_by_dist[d] / total_by_dist[d]
    
    return acc_by_dist


def find_memory_horizon(acc_by_dist: Dict[int, float], threshold: float = 0.9) -> int:
    """Find the distance at which accuracy drops below threshold."""
    for d in sorted(acc_by_dist.keys()):
        if acc_by_dist[d] < threshold:
            return d
    return max(acc_by_dist.keys())


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Golden ratio φ = {PHI:.6f}")
    
    # Data
    train_ds = NeedleDataset(
        num_samples=args.train_samples,
        num_signals=args.num_signals,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        num_noise_tokens=args.num_noise_tokens,
        seed=42,
    )
    
    test_ds = NeedleDataset(
        num_samples=args.test_samples,
        num_signals=args.num_signals,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        num_noise_tokens=args.num_noise_tokens,
        seed=123,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, 
        collate_fn=collate_fn, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"Vocab size: {train_ds.vocab_size}")
    print(f"Signals: {args.num_signals}")
    print(f"Distance range: {args.min_distance} - {args.max_distance}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Test samples: {len(test_ds)}")
    
    # Model
    model = NeedleRIN(
        vocab_size=train_ds.vocab_size,
        num_signals=args.num_signals,
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
    
    # Training
    print(f"\n{'Epoch':>6} | {'Loss':>8} | {'Train':>7} | {'Horizon':>8}")
    print("-" * 45)
    
    best_horizon = 0
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        scheduler.step()
        
        # Evaluate by distance
        acc_by_dist = evaluate_by_distance(model, test_loader, device, args.max_distance)
        horizon = find_memory_horizon(acc_by_dist, threshold=0.9)
        
        if horizon > best_horizon:
            best_horizon = horizon
        
        if (epoch + 1) % args.log_interval == 0 or epoch == args.epochs - 1:
            print(f"{epoch+1:6d} | {train_loss:8.4f} | {train_acc*100:6.1f}% | {horizon:>8}")
    
    # Final evaluation
    print("\n" + "=" * 60)
    print("Final Memory Analysis")
    print("=" * 60)
    
    acc_by_dist = evaluate_by_distance(model, test_loader, device, args.max_distance)
    
    print(f"\n{'Distance':>10} | {'Accuracy':>10} | {'Visualization'}")
    print("-" * 50)
    
    for d in sorted(acc_by_dist.keys()):
        acc = acc_by_dist[d]
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        marker = " ← horizon" if d == find_memory_horizon(acc_by_dist, 0.9) else ""
        print(f"{d:10d} | {acc*100:9.1f}% | {bar}{marker}")
    
    final_horizon = find_memory_horizon(acc_by_dist, 0.9)
    print(f"\nMemory Horizon (90% threshold): {final_horizon} steps")
    print(f"Best horizon during training: {best_horizon} steps")
    
    return {"memory_horizon": final_horizon, "acc_by_dist": acc_by_dist}


def main():
    parser = argparse.ArgumentParser(description="Needle in a Haystack test for RIN")
    parser.add_argument("--num_signals", type=int, default=10, help="Number of signal values")
    parser.add_argument("--min_distance", type=int, default=1, help="Min noise distance")
    parser.add_argument("--max_distance", type=int, default=30, help="Max noise distance")
    parser.add_argument("--num_noise_tokens", type=int, default=50, help="Number of noise tokens")
    parser.add_argument("--train_samples", type=int, default=10000, help="Training samples")
    parser.add_argument("--test_samples", type=int, default=3000, help="Test samples")
    parser.add_argument("--d_model", type=int, default=64, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--num_neurons", type=int, default=128, help="Neurons per layer")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--log_interval", type=int, default=5, help="Log every N epochs")
    parser.add_argument("--use_swish", action="store_true", default=True, help="Use swish")
    parser.add_argument("--no_swish", action="store_false", dest="use_swish")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("RIN - Needle in a Haystack Memory Test")
    print("=" * 60)
    
    train(args)


if __name__ == "__main__":
    main()
