"""
Test RIN with Attenuation - The Missing Piece

The PROBLEM: RIN neurons output unit vectors (|e^iθ| = 1).
They can rotate phases perfectly but cannot control AMPLITUDE.
This means they can't distinguish signal importance from noise.

The SOLUTION: Add learnable attenuation per neuron, per dimension.
    
    att = 1 / (1 + |A|)  -- learns what to IGNORE
    
    cos_weighted = cos(θ) * att
    sin_weighted = sin(θ) * att
    
    cos_sum = Σ_d cos_weighted  -- attenuated interference
    sin_sum = Σ_d sin_weighted

This allows neurons to selectively suppress certain input dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import argparse
from typing import Optional, Tuple
import random
import numpy as np

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut
from rin.model import ResonantLayer, ComplexLinear, PHI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")


class AttenuatedResonantLayer(nn.Module):
    """
    Resonant Layer with ATTENUATION - amplitude control for interference.
    
    Same as ResonantLayer but with learnable attenuation per neuron, per dimension.
    Attenuation modulates the contribution of each sin/cos score before summing.
    
    att = 1 / (1 + |A|)  -- large |A| = suppress, small |A| = preserve
    
    cos_weighted = cos(θ) * att
    sin_weighted = sin(θ) * att
    
    Then sum: cos_sum = Σ_d cos_weighted, sin_sum = Σ_d sin_weighted
    """
    
    def __init__(
        self,
        d_model: int,
        num_neurons: int,
        lut_resolution: int = 4096,
        use_swish: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.use_swish = use_swish
        
        # Input collapse
        self.input_collapse = nn.Linear(2 * d_model, d_model, bias=True)
        
        # Per-neuron, per-dimension parameters
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        self.A = nn.Parameter(torch.zeros(num_neurons, d_model))  # NEW: attenuation
        
        # Output projections
        self.out_proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.out_proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_collapse.weight, gain=0.5)
        nn.init.zeros_(self.input_collapse.bias)
        nn.init.xavier_uniform_(self.out_proj_real.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj_imag.weight, gain=0.5)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x_real, x_imag, t):
        lut = self._get_lut(x_real.device)
        
        # Collapse complex plane
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        x_collapsed = self.input_collapse(x_combined)
        
        # Expand for broadcasting
        x_expanded = x_collapsed.unsqueeze(1)  # (batch, 1, d_model)
        wavelength = 1.0 + self.W.abs()
        
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)
        
        # Phase computation: (batch, num_neurons, d_model)
        theta = x_expanded / wavelength + self.B + t
        
        # Euler
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # ATTENUATION: Modulate each dimension's contribution before summing
        attenuation = 1.0 / (1.0 + self.A.abs())  # (num_neurons, d_model)
        cos_weighted = cos_theta * attenuation
        sin_weighted = sin_theta * attenuation
        
        # Interference sum (attenuated)
        cos_sum = cos_weighted.sum(dim=-1)  # (batch, num_neurons)
        sin_sum = sin_weighted.sum(dim=-1)
        
        # Project to output
        out_real = self.out_proj_real(cos_sum)
        out_imag = self.out_proj_imag(sin_sum)
        
        if self.use_swish:
            out_real = F.silu(out_real)
            out_imag = F.silu(out_imag)
        
        return out_real, out_imag


class VanillaRIN(nn.Module):
    """Vanilla RIN using FIXED ResonantLayer (no attenuation)."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_neurons: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        self.layers = nn.ModuleList([
            ResonantLayer(d_model, num_neurons)
            for _ in range(num_layers)
        ])
        
        self.output_proj = ComplexLinear(d_model, vocab_size, bias=False)
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.d_model, device=device),
                torch.zeros(batch_size, self.d_model, device=device))
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        lut = self._get_lut(h_real.device)
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        return h_real_new, h_imag_new
    
    def forward(self, input_ids, hidden=None, t_start=0):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if hidden is None:
            h_real, h_imag = self.init_hidden(batch_size, device)
        else:
            h_real, h_imag = hidden
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) + t_start
        all_logits = []
        
        for t in range(seq_len):
            w_t, b_t = w_emb[:, t], b_emb[:, t]
            t_val = t_indices[t].expand(batch_size)
            
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            x_real, x_imag = h_real, h_imag
            t_phi = t_val * PHI
            
            for layer in self.layers:
                d_real, d_imag = layer(x_real, x_imag, t_phi)
                x_real = x_real + d_real
                x_imag = x_imag + d_imag
            
            logits_r, logits_i = self.output_proj(x_real, x_imag)
            all_logits.append(logits_r + logits_i)
        
        return torch.stack(all_logits, dim=1), (h_real, h_imag)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


class AttenuatedRIN(nn.Module):
    """RIN with attenuated resonant layers."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_neurons: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        self.layers = nn.ModuleList([
            AttenuatedResonantLayer(d_model, num_neurons)
            for _ in range(num_layers)
        ])
        
        self.output_proj = ComplexLinear(d_model, vocab_size, bias=False)
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def init_hidden(self, batch_size, device):
        return (torch.zeros(batch_size, self.d_model, device=device),
                torch.zeros(batch_size, self.d_model, device=device))
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        lut = self._get_lut(h_real.device)
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        return h_real_new, h_imag_new
    
    def forward(self, input_ids, hidden=None, t_start=0):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if hidden is None:
            h_real, h_imag = self.init_hidden(batch_size, device)
        else:
            h_real, h_imag = hidden
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) + t_start
        all_logits = []
        
        for t in range(seq_len):
            w_t, b_t = w_emb[:, t], b_emb[:, t]
            t_val = t_indices[t].expand(batch_size)
            
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            x_real, x_imag = h_real, h_imag
            t_phi = t_val * PHI
            
            for layer in self.layers:
                d_real, d_imag = layer(x_real, x_imag, t_phi)
                x_real = x_real + d_real
                x_imag = x_imag + d_imag
            
            logits_r, logits_i = self.output_proj(x_real, x_imag)
            all_logits.append(logits_r + logits_i)
        
        return torch.stack(all_logits, dim=1), (h_real, h_imag)
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Needle-in-Haystack Dataset
# =============================================================================

class NeedleDataset:
    def __init__(self, vocab_size=100, num_signals=10, min_distance=3, max_distance=10):
        self.vocab_size = vocab_size
        self.num_signals = num_signals
        self.min_distance = min_distance
        self.max_distance = max_distance
        
        self.trigger_token = 0
        self.signal_tokens = list(range(1, num_signals + 1))
        self.noise_tokens = list(range(num_signals + 1, vocab_size))
    
    def generate_batch(self, batch_size, distance=None):
        if distance is None:
            distances = [random.randint(self.min_distance, self.max_distance) 
                        for _ in range(batch_size)]
        else:
            distances = [distance] * batch_size
        
        max_d = max(distances)
        seq_len = max_d + 2
        
        sequences, targets = [], []
        for d in distances:
            signal = random.choice(self.signal_tokens)
            prefix_len = seq_len - d - 2
            between_len = d
            
            prefix = [random.choice(self.noise_tokens) for _ in range(prefix_len)]
            between = [random.choice(self.noise_tokens) for _ in range(between_len)]
            
            seq = prefix + [signal] + between + [self.trigger_token]
            sequences.append(seq)
            targets.append(signal)
        
        return (torch.tensor(sequences, dtype=torch.long, device=device),
                torch.tensor(targets, dtype=torch.long, device=device),
                distances)


def evaluate(model, dataset, batch_size=64, num_batches=10, distance=None):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for _ in range(num_batches):
            seqs, targets, _ = dataset.generate_batch(batch_size, distance)
            logits, _ = model(seqs)
            preds = logits[:, -1, :].argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += batch_size
    return correct / total


def train_and_compare(
    d_model=64,
    num_layers=2,
    num_neurons=128,
    min_distance=5,
    max_distance=15,
    epochs=50,
    batch_size=64,
    lr=1e-3,
):
    vocab_size, num_signals = 100, 10
    
    dataset = NeedleDataset(vocab_size, num_signals, min_distance, max_distance)
    
    print("=" * 70)
    print("ATTENUATION TEST: Vanilla RIN vs Attenuated RIN (FIXED)")
    print("=" * 70)
    print(f"Task: Needle-in-haystack retrieval")
    print(f"Distance range: {min_distance} to {max_distance} tokens")
    print(f"Random baseline: {100/num_signals:.1f}%")
    print()
    
    results = {}
    
    for name, model_class in [("Vanilla RIN", VanillaRIN), ("Attenuated RIN", AttenuatedRIN)]:
        print(f"\n{'='*70}")
        print(f"Training: {name}")
        print("="*70)
        
        model = model_class(vocab_size, d_model, num_layers, num_neurons).to(device)
        print(f"Parameters: {model.get_num_params():,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        best_acc = 0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            for _ in range(20):
                seqs, targets, _ = dataset.generate_batch(batch_size)
                optimizer.zero_grad()
                logits, _ = model(seqs)
                loss = F.cross_entropy(logits[:, -1, :], targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            acc = evaluate(model, dataset)
            best_acc = max(best_acc, acc)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}: Loss={total_loss/20:.4f}, Acc={acc*100:.1f}%, Best={best_acc*100:.1f}%")
        
        results[name] = {'model': model, 'best_acc': best_acc, 'final_acc': acc}
    
    # Summary
    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)
    for name, res in results.items():
        print(f"{name:<20} Best: {res['best_acc']*100:.1f}%, Final: {res['final_acc']*100:.1f}%")
    
    # Test by distance
    print("\n" + "=" * 70)
    print("ACCURACY BY DISTANCE")
    print("=" * 70)
    
    for dist in [5, 10, 15, 20]:
        if dist > max_distance + 5:
            continue
        print(f"Distance {dist:2d}:", end=" ")
        for name, res in results.items():
            acc = evaluate(res['model'], dataset, num_batches=20, distance=dist)
            print(f"{name}: {acc*100:5.1f}%", end="  ")
        print()
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_neurons", type=int, default=128)
    parser.add_argument("--min_distance", type=int, default=5)
    parser.add_argument("--max_distance", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    
    train_and_compare(
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_neurons=args.num_neurons,
        min_distance=args.min_distance,
        max_distance=args.max_distance,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
