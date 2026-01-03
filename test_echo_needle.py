#!/usr/bin/env python3
"""
Needle in a Haystack Test for RIN Echo Model

Compares the Echo Chamber architecture against vanilla RIN on the
needle-in-haystack memory retrieval task.

Test structure:
    [TRIGGER] [SIGNAL=k] [noise...] [TRIGGER] -> predict k

The signal is placed at varying distances before the recall trigger.
A model with good memory should maintain high accuracy regardless of distance.

This tests whether the Echo Chamber's EMA value states provide
better long-range memory than vanilla RIN.
"""

import argparse
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
import sys
sys.path.insert(0, '/home/aiman/pi')

from rin import RINModel, RINEchoModel, PHI, get_global_lut


class NeedleDataset(Dataset):
    """Dataset for needle-in-haystack recall task."""
    
    def __init__(
        self, 
        num_samples: int,
        num_signals: int,
        min_distance: int,
        max_distance: int,
        num_noise_tokens: int,
        seed: int = None,
    ):
        self.num_samples = num_samples
        self.num_signals = num_signals
        self.min_distance = min_distance
        self.max_distance = max_distance
        self.num_noise_tokens = num_noise_tokens
        
        # Token IDs
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
            
            seq = [self.trigger_id, self.signal_start + signal]
            for _ in range(distance):
                seq.append(random.randint(self.noise_start, self.vocab_size - 1))
            seq.append(self.trigger_id)
            
            self.data.append((torch.tensor(seq, dtype=torch.long), signal, distance))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Collate with padding."""
    seqs, signals, distances = zip(*batch)
    max_len = max(len(s) for s in seqs)
    
    padded = torch.zeros(len(seqs), max_len, dtype=torch.long)
    for i, s in enumerate(seqs):
        padded[i, :len(s)] = s
    
    return padded, torch.tensor(signals), torch.tensor(distances)


class VanillaNeedleRIN(nn.Module):
    """
    Vanilla RIN with proper complex-valued architecture.
    
    NO RESIDUALS - pure rotary phase flow.
    MAINTAINS COMPLEX STATE throughout (only collapse at final output).
    """
    
    def __init__(self, vocab_size, num_signals, d_model=64, num_layers=2, num_neurons=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_signals = num_signals
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Import complex-valued components
        from rin.model import ResonantLayer, ComplexLinear
        
        self.layers = nn.ModuleList([
            ResonantLayer(d_model, num_neurons, use_swish=True, wrap_time=True)
            for _ in range(num_layers)
        ])
        
        # Complex output projection (only place we collapse)
        self.output_proj = ComplexLinear(d_model, num_signals, bias=False)
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.5)
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        """
        Separated euler transform preserving complex state.
        
        theta_real = h_real / wavelength + b + t*phi
        theta_imag = h_imag / wavelength + b + t*phi
        
        Then complex multiplication: e^(i*theta_real) * e^(i*theta_imag)
        """
        from rin.utils import wrap_time_periodic
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        t_phi = wrap_time_periodic(t_phi)
        
        # SEPARATED theta (preserving complex structure)
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        # Euler on each
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        # Complex multiplication: e^(i*theta_r) * e^(i*theta_i)
        # = (cos_r + i*sin_r)(cos_i + i*sin_i)
        # = (cos_r*cos_i - sin_r*sin_i) + i*(cos_r*sin_i + sin_r*cos_i)
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        
        return h_real_new, h_imag_new
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize complex state
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        for t in range(seq_len):
            t_val = t_indices[t].expand(batch_size)
            
            # Euler transform (maintaining complex state)
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_emb[:, t], b_emb[:, t], t_val)
            
            # Resonant layers (NO RESIDUALS - pure transformation)
            x_real, x_imag = h_real, h_imag
            for layer in self.layers:
                x_real, x_imag = layer(x_real, x_imag, t_val * PHI)
        
        # Final output: collapse only here
        logits_real, logits_imag = self.output_proj(x_real, x_imag)
        return logits_real + logits_imag


class EchoNeedleRIN(nn.Module):
    """
    RIN with Echo Chamber - proper complex-valued architecture.
    
    NO RESIDUALS - pure rotary phase flow.
    MAINTAINS COMPLEX STATE throughout (only collapse at final output).
    """
    
    def __init__(
        self, 
        vocab_size, 
        num_signals, 
        d_model=64, 
        num_layers=2, 
        num_neurons=128,
        n_heads=4,
        alpha=1.0,
        output_mode='complex_linear',
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_signals = num_signals
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        from rin.echo_chamber import ResonantBlock
        from rin.model import ComplexLinear
        
        # ResonantBlocks: ResonantLayer || EchoChamber in parallel
        self.blocks = nn.ModuleList([
            ResonantBlock(
                d_model=d_model,
                num_neurons=num_neurons,
                n_heads=n_heads,
                alpha=alpha,
                output_mode=output_mode,
                wrap_time=True,
            )
            for _ in range(num_layers)
        ])
        
        # Complex output projection (only place we collapse)
        self.output_proj = ComplexLinear(d_model, num_signals, bias=False)
        
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.5)
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        """Separated euler transform preserving complex state."""
        from rin.utils import wrap_time_periodic
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        t_phi = wrap_time_periodic(t_phi)
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        
        return h_real_new, h_imag_new
    
    def init_echo_states(self, batch_size, device):
        return [block.echo_chamber.init_state(batch_size, device) for block in self.blocks]
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Initialize complex state
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        echo_states = self.init_echo_states(batch_size, device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        for t in range(seq_len):
            t_val = t_indices[t].expand(batch_size)
            
            # Euler transform (maintaining complex state)
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_emb[:, t], b_emb[:, t], t_val)
            
            # ResonantBlocks (NO RESIDUALS - pure transformation)
            x_real, x_imag = h_real, h_imag
            new_echo_states = []
            
            for i, block in enumerate(self.blocks):
                (x_real, x_imag), new_state = block.forward_step(
                    x_real, x_imag, echo_states[i], t_val * PHI
                )
                new_echo_states.append(new_state)
            
            echo_states = new_echo_states
        
        # Final output: collapse only here
        logits_real, logits_imag = self.output_proj(x_real, x_imag)
        return logits_real + logits_imag


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
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        correct += (logits.argmax(-1) == signals).sum().item()
        total += signals.size(0)
    
    return total_loss / len(loader), correct / total


def evaluate_by_distance(model, loader, device, max_distance):
    """Evaluate accuracy by distance."""
    model.eval()
    
    correct_by_dist = {d: 0 for d in range(max_distance + 1)}
    total_by_dist = {d: 0 for d in range(max_distance + 1)}
    
    with torch.no_grad():
        for seqs, signals, distances in loader:
            seqs = seqs.to(device)
            signals = signals.to(device)
            
            logits = model(seqs)
            preds = logits.argmax(-1)
            
            for i, d in enumerate(distances.tolist()):
                if d <= max_distance:
                    total_by_dist[d] += 1
                    if preds[i] == signals[i]:
                        correct_by_dist[d] += 1
    
    acc_by_dist = {}
    for d in range(max_distance + 1):
        if total_by_dist[d] > 0:
            acc_by_dist[d] = correct_by_dist[d] / total_by_dist[d]
    
    return acc_by_dist


def find_memory_horizon(acc_by_dist: Dict[int, float], threshold: float = 0.9) -> int:
    """Find distance where accuracy drops below threshold."""
    for d in sorted(acc_by_dist.keys()):
        if acc_by_dist[d] < threshold:
            return d - 1
    return max(acc_by_dist.keys())


def compare_models(args):
    """Compare vanilla RIN vs Echo RIN on needle task."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create datasets
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
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, collate_fn=collate_fn, pin_memory=True)
    
    print(f"\nDataset: vocab={train_ds.vocab_size}, signals={args.num_signals}, distance=[{args.min_distance}, {args.max_distance}]")
    
    # Models to compare - additive interference/superposition
    models_config = {
        'Vanilla RIN': VanillaNeedleRIN(
            vocab_size=train_ds.vocab_size,
            num_signals=args.num_signals,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_neurons=args.num_neurons,
        ),
        'Echo (complex)': EchoNeedleRIN(
            vocab_size=train_ds.vocab_size,
            num_signals=args.num_signals,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_neurons=args.num_neurons,
            n_heads=args.n_heads,
            alpha=1.0,
            output_mode='complex_linear',
        ),
        'Echo (resonant)': EchoNeedleRIN(
            vocab_size=train_ds.vocab_size,
            num_signals=args.num_signals,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_neurons=args.num_neurons,
            n_heads=args.n_heads,
            alpha=1.0,
            output_mode='resonant',  # Retrieved value -> Euler transform
        ),
    }
    
    # Test alpha=0.5 variants if requested
    if args.test_alpha_variants:
        models_config['Echo (α=0.5)'] = EchoNeedleRIN(
            vocab_size=train_ds.vocab_size,
            num_signals=args.num_signals,
            d_model=args.d_model,
            num_layers=args.num_layers,
            num_neurons=args.num_neurons,
            n_heads=args.n_heads,
            alpha=0.5,
            output_mode='complex_linear',
        )
    
    results = {}
    
    for name, model in models_config.items():
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        model = model.to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {params:,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        
        best_horizon = 0
        training_history = []
        
        print(f"\n{'Epoch':>6} | {'Loss':>8} | {'Train':>7} | {'Horizon':>8}")
        print("-" * 45)
        
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
            scheduler.step()
            
            acc_by_dist = evaluate_by_distance(model, test_loader, device, args.max_distance)
            horizon = find_memory_horizon(acc_by_dist, threshold=0.9)
            
            training_history.append({
                'epoch': epoch,
                'loss': train_loss,
                'train_acc': train_acc,
                'horizon': horizon,
            })
            
            if horizon > best_horizon:
                best_horizon = horizon
            
            if (epoch + 1) % args.log_interval == 0 or epoch == args.epochs - 1:
                print(f"{epoch+1:>6} | {train_loss:>8.4f} | {train_acc*100:>6.1f}% | {horizon:>8}")
        
        # Final evaluation
        final_acc = evaluate_by_distance(model, test_loader, device, args.max_distance)
        
        results[name] = {
            'params': params,
            'best_horizon': best_horizon,
            'final_horizon': find_memory_horizon(final_acc, 0.9),
            'acc_by_dist': final_acc,
            'history': training_history,
        }
    
    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)
    
    print(f"\n{'Model':<25} | {'Params':>10} | {'Best Horizon':>12} | {'Final Horizon':>12}")
    print("-" * 70)
    for name, r in results.items():
        print(f"{name:<25} | {r['params']:>10,} | {r['best_horizon']:>12} | {r['final_horizon']:>12}")
    
    # Print accuracy by distance for all models
    print("\n" + "-" * 70)
    print("Accuracy by Distance:")
    print("-" * 70)
    
    header = f"{'Distance':>10}"
    for name in results.keys():
        header += f" | {name[:12]:>12}"
    print(header)
    print("-" * len(header))
    
    for d in range(0, args.max_distance + 1, max(1, args.max_distance // 10)):
        row = f"{d:>10}"
        for name, r in results.items():
            acc = r['acc_by_dist'].get(d, 0) * 100
            row += f" | {acc:>11.1f}%"
        print(row)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Needle test for Echo Chamber")
    parser.add_argument("--num_signals", type=int, default=10)
    parser.add_argument("--min_distance", type=int, default=1)
    parser.add_argument("--max_distance", type=int, default=30)
    parser.add_argument("--num_noise_tokens", type=int, default=50)
    parser.add_argument("--train_samples", type=int, default=10000)
    parser.add_argument("--test_samples", type=int, default=3000)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neurons", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0, help="EMA decay rate (1.0=instant)")
    parser.add_argument("--epochs", type=int, default=10)  # Reduced for faster testing
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=2)
    parser.add_argument("--test_resonant_output", action="store_true", help="Also test resonant output mode")
    parser.add_argument("--test_alpha_variants", action="store_true", help="Also test alpha=0.5")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ECHO CHAMBER - NEEDLE IN HAYSTACK TEST")
    print("=" * 70)
    
    compare_models(args)


if __name__ == "__main__":
    main()
