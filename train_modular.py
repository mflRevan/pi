#!/usr/bin/env python3
"""
Train RIN on Modular Arithmetic (Grokking Task)

The classic test for generalization: (a + b) mod p

This task famously exhibits "grokking" - sudden generalization after
prolonged training using the modern RIN architecture with time-dependent
measurement basis.

Usage:
    python train_modular.py
    python train_modular.py --p 113 --epochs 500 --wd 0.1
"""

import argparse
import math
import random
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from rin.model import RINModel


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
            # We'll use the last token position for prediction
            self.data.append((torch.tensor([a, b, p]), (a + b) % p))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


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
    """Train the model using modern RIN architecture with gradient tracking."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Task: (a + b) mod {args.p}")
    print(f"Modern RIN with time-dependent measurement (amplitude/angle)")
    
    # Data
    train_ds = ModularAdditionDataset(args.p, "train", args.train_frac)
    test_ds = ModularAdditionDataset(args.p, "test", args.train_frac)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, pin_memory=True)
    
    print(f"Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    
    # Model - using modern RINModel
    vocab_size = args.p + 1  # 0 to p-1, plus the '=' token at position p
    model = RINModel(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_neurons=args.num_neurons,
        use_swish=args.use_swish,
        wrap_time=args.wrap_time,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    print(f"d_model={args.d_model}, layers={args.num_layers}, neurons={args.num_neurons}")
    print(f"wrap_time={args.wrap_time}, use_swish={args.use_swish}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    
    # Compile model for faster execution (PyTorch 2.0+)
    if args.compile and hasattr(torch, 'compile'):
        print("Compiling model with torch.compile...")
        model = torch.compile(model, mode='default')
    
    # Gradient tracking setup (optional)
    gradient_history = None
    if args.track_gradients:
        gradient_history = {
            'epochs': [],
            'embeddings': [],
            'layer_0_amplitude': [],
            'layer_0_angle': [],
            'layer_0_wavelength': [],
            'layer_0_phase_offset': [],
            'layer_0_attn_cos': [],
            'layer_0_attn_sin': [],
            'layer_0_mixer_real': [],
            'layer_0_mixer_imag': [],
            'output_layer': [],
        }
    
    # Training loop
    best_test_acc = 0
    stability_dips = 0
    prev_test_acc = 0
    last_log = -1
    grokking_epoch = None
    
    if args.track_gradients:
        print(f"\n{'Epoch':>6} | {'Loss':>8} | {'Train':>7} | {'Test':>7} | {'Embed':>10} | {'Ampl':>10} | {'Angle':>10} | {'W':>10} | {'B':>10}")
        print("-" * 110)
    else:
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
            logits, _ = model(x)
            final_logits = logits[:, -1, :]
            loss = F.cross_entropy(final_logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            correct += (final_logits.argmax(-1) == y).sum().item()
            total += y.size(0)
        
        scheduler.step()
        train_acc = correct / total
        
        # Collect gradient statistics (only if tracking enabled)
        grad_stats = None
        if args.track_gradients:
            def get_grad_norm(param):
                if param.grad is not None:
                    return param.grad.norm().item()
                return 0.0
            
            grad_stats = {
                'epoch': epoch + 1,
                'embeddings': get_grad_norm(model.token_embedding.weight),
                'layer_0_amplitude': get_grad_norm(model.layers[0]['up'].measure_amplitude),
                'layer_0_angle': get_grad_norm(model.layers[0]['up'].measure_angle),
                'layer_0_wavelength': get_grad_norm(model.layers[0]['up'].W),
                'layer_0_phase_offset': get_grad_norm(model.layers[0]['up'].B),
                'layer_0_attn_cos': get_grad_norm(model.layers[0]['up'].attn_cos),
                'layer_0_attn_sin': get_grad_norm(model.layers[0]['up'].attn_sin),
                'layer_0_mixer_real': get_grad_norm(model.layers[0]['mixer'].weight_real),
                'layer_0_mixer_imag': get_grad_norm(model.layers[0]['mixer'].weight_imag),
                'output_layer': get_grad_norm(model.output_layer.weight),
            }
            
            # Store gradient history (every epoch for first args.track_epochs epochs)
            if epoch < args.track_epochs:
                for key in gradient_history.keys():
                    if key == 'epochs':
                        gradient_history[key].append(grad_stats['epoch'])
                    elif key in grad_stats:
                        gradient_history[key].append(grad_stats[key])
        
        # Test
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits, _ = model(x)
                final_logits = logits[:, -1, :]
                correct += (final_logits.argmax(-1) == y).sum().item()
                total += y.size(0)
        
        test_acc = correct / total
        
        # Track grokking moment (when test acc jumps above 90%)
        if grokking_epoch is None and test_acc > 0.9:
            grokking_epoch = epoch + 1
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        
        if test_acc < prev_test_acc - 0.05:
            stability_dips += 1
        prev_test_acc = test_acc
        
        # Dynamic logging
        interval = get_log_interval(epoch, args.epochs)
        should_log = (epoch + 1) - last_log >= interval or epoch == args.epochs - 1
        if args.track_gradients:
            should_log = should_log or epoch < args.track_epochs
        
        if should_log:
            if args.track_gradients and grad_stats:
                print(f"{epoch+1:6d} | {total_loss/len(train_loader):8.4f} | {train_acc*100:6.1f}% | {test_acc*100:6.1f}% | "
                      f"{grad_stats['embeddings']:10.6f} | {grad_stats['layer_0_amplitude']:10.6f} | "
                      f"{grad_stats['layer_0_angle']:10.6f} | {grad_stats['layer_0_wavelength']:10.6f} | "
                      f"{grad_stats['layer_0_phase_offset']:10.6f}")
            else:
                print(f"{epoch+1:6d} | {total_loss/len(train_loader):8.4f} | {train_acc*100:6.1f}% | {test_acc*100:6.1f}%")
            last_log = epoch + 1
    
    if args.track_gradients:
        print("-" * 110)
    else:
        print("-" * 40)
    print(f"Best test accuracy: {best_test_acc*100:.2f}%")
    print(f"Stability dips: {stability_dips}")
    if grokking_epoch:
        print(f"Grokking at epoch: {grokking_epoch}")
    
    # Save gradient history
    if args.track_gradients and gradient_history and len(gradient_history['epochs']) > 0:
        output_dir = Path("results/gradients")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"gradient_analysis_p{args.p}_d{args.d_model}_l{args.num_layers}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'p': args.p,
                    'd_model': args.d_model,
                    'num_layers': args.num_layers,
                    'num_neurons': args.num_neurons,
                    'lr': args.lr,
                    'weight_decay': args.weight_decay,
                    'use_swish': args.use_swish,
                    'wrap_time': args.wrap_time,
                },
                'results': {
                    'best_test_acc': best_test_acc,
                    'stability_dips': stability_dips,
                    'grokking_epoch': grokking_epoch,
                },
                'gradient_history': gradient_history,
            }, f, indent=2)
        
        print(f"\nGradient history saved to: {output_file}")
        
        # Print gradient summary
        print("\n" + "=" * 80)
        print("GRADIENT ANALYSIS SUMMARY")
        print("=" * 80)
        
        if len(gradient_history['epochs']) > 0:
            for key in gradient_history.keys():
                if key != 'epochs' and len(gradient_history[key]) > 0:
                    values = gradient_history[key]
                    print(f"\n{key}:")
                    print(f"  Initial (epoch 1): {values[0]:.6f}")
                    if len(values) > 1:
                        print(f"  Final (epoch {len(values)}): {values[-1]:.6f}")
                        print(f"  Mean: {sum(values) / len(values):.6f}")
                        print(f"  Max: {max(values):.6f}")
                        print(f"  Min: {min(values):.6f}")
    
    return {"best_test_acc": best_test_acc, "stability_dips": stability_dips, "grokking_epoch": grokking_epoch}


def main():
    parser = argparse.ArgumentParser(description="Train RIN on modular arithmetic")
    parser.add_argument("--p", type=int, default=97, help="Modulus (prime)")
    parser.add_argument("--train_frac", type=float, default=0.5, help="Training fraction")
    parser.add_argument("--d_model", type=int, default=32, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of layers")
    parser.add_argument("--num_neurons", type=int, default=64, help="Neurons per layer")
    parser.add_argument("--epochs", type=int, default=300, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1.0, help="Weight decay (important for grokking!)")
    parser.add_argument("--use_swish", action="store_true", default=False, help="Use swish activation")
    parser.add_argument("--wrap_time", action="store_true", default=False, help="Use t mod 2π")
    parser.add_argument("--compile", action="store_true", default=False, help="Use torch.compile for speedup")
    parser.add_argument("--track_gradients", action="store_true", default=False, help="Track gradient statistics (slower)")
    parser.add_argument("--track_epochs", type=int, default=20, help="Number of epochs to track gradients")
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("RIN - Modular Arithmetic (Grokking)")
    print("Modern Architecture with Time-Dependent Measurement")
    print("=" * 50)
    
    train(args)


if __name__ == "__main__":
    main()