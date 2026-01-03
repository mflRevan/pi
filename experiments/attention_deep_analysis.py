#!/usr/bin/env python3
"""
Deep Analysis of Echo Chamber Attention

This script performs comprehensive analysis of the resonant attention mechanism:
1. Gradient flow through attention paths
2. Time dependence (t*φ) sensitivity
3. Different projection types (complex linear vs resonant)
4. Attention pattern analysis
5. Long-range recall stress tests
6. Phase dynamics visualization

Run with: python experiments/attention_deep_analysis.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Optional, Tuple, List, Dict
import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut
from rin.model import ResonantLayer, ComplexLinear, PHI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Alternative Projection Types
# =============================================================================

class ResonantProjection(nn.Module):
    """
    Resonant projection using interference analysis instead of linear.
    
    Instead of: out = W @ x
    We do: out = interference_sum(euler_transform(x))
    
    This maintains the wave-based paradigm throughout.
    """
    
    def __init__(self, in_features: int, out_features: int, num_neurons: int = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons or out_features * 2
        
        # Per-neuron, per-input-dimension parameters
        self.W = nn.Parameter(torch.randn(self.num_neurons, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(self.num_neurons, in_features))
        
        # Project interference sums to output
        self.out_proj = nn.Linear(self.num_neurons, out_features, bias=False)
        self._lut = None
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.5)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input (batch, in_features)
            t: Optional timestep for phase modulation
        """
        lut = self._get_lut(x.device)
        
        # x: (batch, in_features) -> (batch, 1, in_features)
        x_expanded = x.unsqueeze(1)
        wavelength = 1.0 + self.W.abs()
        
        # Phase computation
        theta = x_expanded / wavelength + self.B
        if t is not None:
            if t.dim() == 0:
                theta = theta + t
            elif t.dim() == 1:
                theta = theta + t.unsqueeze(-1).unsqueeze(-1)
        
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # Interference sum
        cos_sum = cos_theta.sum(dim=-1)  # (batch, num_neurons)
        sin_sum = sin_theta.sum(dim=-1)
        
        # Combine and project
        interference = cos_sum + sin_sum  # Could also try cos_sum only or magnitude
        return self.out_proj(interference)


class ResonantComplexProjection(nn.Module):
    """
    Resonant projection that outputs complex (real, imag) pairs.
    """
    
    def __init__(self, in_features: int, out_features: int, num_neurons: int = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_neurons = num_neurons or out_features * 2
        
        self.W = nn.Parameter(torch.randn(self.num_neurons, in_features) * 0.02)
        self.B = nn.Parameter(torch.zeros(self.num_neurons, in_features))
        
        self.out_proj_real = nn.Linear(self.num_neurons, out_features, bias=False)
        self.out_proj_imag = nn.Linear(self.num_neurons, out_features, bias=False)
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        lut = self._get_lut(x.device)
        
        x_expanded = x.unsqueeze(1)
        wavelength = 1.0 + self.W.abs()
        
        theta = x_expanded / wavelength + self.B
        if t is not None:
            if t.dim() == 0:
                theta = theta + t
            elif t.dim() == 1:
                theta = theta + t.unsqueeze(-1).unsqueeze(-1)
        
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        cos_sum = cos_theta.sum(dim=-1)
        sin_sum = sin_theta.sum(dim=-1)
        
        return self.out_proj_real(cos_sum), self.out_proj_imag(sin_sum)


# =============================================================================
# Attention Head Variants
# =============================================================================

class ResonantAttentionHeadV2(nn.Module):
    """
    Enhanced attention head with configurable projection types.
    
    Supports:
    - 'euler': Original Euler-transformed queries/keys
    - 'resonant': Full resonant projection for queries/keys
    - 'linear': Standard linear projection (baseline)
    """
    
    def __init__(
        self,
        d_model: int,
        d_head: int,
        head_idx: int,
        query_type: str = 'euler',  # 'euler', 'resonant', 'linear'
        key_type: str = 'euler',
        use_time_in_key: bool = False,  # Whether keys should use time
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.head_idx = head_idx
        self.start_idx = head_idx * d_head
        self.end_idx = (head_idx + 1) * d_head
        self.query_type = query_type
        self.key_type = key_type
        self.use_time_in_key = use_time_in_key
        
        # Query projection
        if query_type == 'euler':
            self.w_query = nn.Parameter(torch.randn(d_head) * 0.02)
            self.b_query = nn.Parameter(torch.zeros(d_head))
        elif query_type == 'resonant':
            self.query_proj = ResonantProjection(d_head, 2 * d_head)
        else:  # linear
            self.query_proj = nn.Linear(d_head, 2 * d_head, bias=False)
        
        # Key projection
        if key_type == 'euler':
            self.w_key = nn.Parameter(torch.randn(d_head) * 0.02)
            self.b_key = nn.Parameter(torch.zeros(d_head))
        elif key_type == 'resonant':
            self.key_proj = ResonantProjection(d_head, 2 * d_head)
        else:  # linear
            self.key_proj = nn.Linear(d_head, 2 * d_head, bias=False)
        
        self.scale = math.sqrt(2 * d_head)
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def compute_query(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x_patch = x[:, self.start_idx:self.end_idx]
        
        if self.query_type == 'euler':
            lut = self._get_lut(x.device)
            wavelength = 1.0 + self.w_query.abs()
            t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
            theta = x_patch / wavelength + self.b_query + t_phi
            sin_q, cos_q = lut.lookup_sin_cos(theta)
            return torch.cat([cos_q, sin_q], dim=-1)
        elif self.query_type == 'resonant':
            return self.query_proj(x_patch, t * PHI if t is not None else None)
        else:
            return self.query_proj(x_patch)
    
    def compute_keys(self, cached_states: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        k_patches = cached_states[:, :, self.start_idx:self.end_idx]
        batch_size, history_len, d_head = k_patches.shape
        
        if self.key_type == 'euler':
            lut = self._get_lut(cached_states.device)
            wavelength = 1.0 + self.w_key.abs()
            theta = k_patches / wavelength + self.b_key
            if self.use_time_in_key and t is not None:
                # Add time-based offset to keys (experimental)
                t_offset = torch.arange(history_len, device=cached_states.device).float() * PHI
                theta = theta + t_offset.unsqueeze(0).unsqueeze(-1)
            sin_k, cos_k = lut.lookup_sin_cos(theta)
            return torch.cat([cos_k, sin_k], dim=-1)
        elif self.key_type == 'resonant':
            # Process each history position
            k_flat = k_patches.reshape(-1, d_head)
            keys_flat = self.key_proj(k_flat)
            return keys_flat.reshape(batch_size, history_len, -1)
        else:
            k_flat = k_patches.reshape(-1, d_head)
            keys_flat = self.key_proj(k_flat)
            return keys_flat.reshape(batch_size, history_len, -1)
    
    def forward(
        self,
        x: torch.Tensor,
        cached_states: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = self.compute_query(x, t)
        keys = self.compute_keys(cached_states, t)
        
        scores = torch.bmm(
            query.unsqueeze(1),
            keys.transpose(1, 2)
        ).squeeze(1)
        
        scores = scores / self.scale
        weights = F.softmax(scores, dim=-1)
        
        output = torch.bmm(
            weights.unsqueeze(1),
            cached_states
        ).squeeze(1)
        
        return output, weights


# =============================================================================
# Configurable Attention Model for Testing
# =============================================================================

class ConfigurableAttentionModel(nn.Module):
    """
    Attention model with configurable components for ablation studies.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_neurons: int = 128,
        n_heads: int = 8,
        query_type: str = 'euler',
        key_type: str = 'euler',
        output_proj_type: str = 'complex_linear',  # 'complex_linear', 'resonant', 'linear'
        use_time_in_key: bool = False,
        time_scale: float = 1.0,  # Multiply time by this factor
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.time_scale = time_scale
        self.output_proj_type = output_proj_type
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Attention heads
        self.heads = nn.ModuleList([
            ResonantAttentionHeadV2(
                d_model, self.d_head, i,
                query_type=query_type,
                key_type=key_type,
                use_time_in_key=use_time_in_key,
            )
            for i in range(n_heads)
        ])
        
        # Context projection
        self.context_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Resonant layer
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        self.res_proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.res_proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        
        # Output projection
        if output_proj_type == 'complex_linear':
            self.output_proj = ComplexLinear(d_model, vocab_size, bias=False)
        elif output_proj_type == 'resonant':
            self.output_proj = ResonantProjection(d_model, vocab_size)
        else:
            self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.context_proj.weight, std=0.02)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        lut = self._get_lut(h_real.device)
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI * self.time_scale if t.dim() == 1 else t * PHI * self.time_scale
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        return h_real_new, h_imag_new
    
    def resonant_forward(self, x, t):
        lut = self._get_lut(x.device)
        x_expanded = x.unsqueeze(1)
        wavelength = 1.0 + self.W.abs()
        
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)
        
        theta = x_expanded / wavelength + self.B + t
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        cos_sum = cos_theta.sum(dim=-1)
        sin_sum = sin_theta.sum(dim=-1)
        
        return F.silu(self.res_proj_real(cos_sum) + self.res_proj_imag(sin_sum))
    
    def forward(self, input_ids, return_attention=False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        # State cache
        cached_states = []
        all_logits = []
        all_attention = [] if return_attention else None
        
        for t_idx in range(seq_len):
            w_t, b_t = w_emb[:, t_idx], b_emb[:, t_idx]
            t_val = t_indices[t_idx].expand(batch_size)
            
            # Euler transform
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            # Collapse for attention
            x = h_real + h_imag
            cached_states.append(x)
            
            # Attention (if we have history)
            if len(cached_states) > 1:
                states_tensor = torch.stack(cached_states[:-1], dim=1)
                
                head_outputs = []
                step_attention = []
                for head in self.heads:
                    out, weights = head(x, states_tensor, t_val)
                    head_outputs.append(out)
                    if return_attention:
                        step_attention.append(weights)
                
                context = torch.stack(head_outputs, dim=0).sum(dim=0)
                x = x + self.context_proj(context)
                
                if return_attention:
                    all_attention.append(step_attention)
            
            # Resonant layer
            t_phi = t_val * PHI * self.time_scale
            x = x + self.resonant_forward(x, t_phi)
            
            # Output
            if self.output_proj_type == 'complex_linear':
                logits_r, logits_i = self.output_proj(x, torch.zeros_like(x))
                logits = logits_r + logits_i
            elif self.output_proj_type == 'resonant':
                logits = self.output_proj(x, t_phi)
            else:
                logits = self.output_proj(x)
            
            all_logits.append(logits)
        
        output = torch.stack(all_logits, dim=1)
        if return_attention:
            return output, all_attention
        return output
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Test Functions
# =============================================================================

def test_gradient_flow(config: Dict, num_samples: int = 100):
    """Test gradient flow through attention mechanism."""
    print("\n" + "="*70)
    print(f"GRADIENT FLOW ANALYSIS")
    print(f"Config: {config}")
    print("="*70)
    
    model = ConfigurableAttentionModel(
        vocab_size=100,
        d_model=64,
        **config
    ).to(device)
    
    # Create sequence
    seq = torch.randint(0, 100, (4, 20), device=device)
    
    # Track gradients for different components
    model.train()
    model.zero_grad()
    
    output = model(seq)
    loss = output[:, -1, :].sum()  # Gradient from last position
    loss.backward()
    
    grad_stats = {}
    
    # Embedding gradients
    grad_stats['embedding'] = {
        'mean': model.token_embedding.weight.grad.abs().mean().item(),
        'max': model.token_embedding.weight.grad.abs().max().item(),
        'nonzero': (model.token_embedding.weight.grad != 0).sum().item(),
    }
    
    # Attention head gradients
    for i, head in enumerate(model.heads):
        if hasattr(head, 'w_query'):
            grad_stats[f'head_{i}_w_query'] = {
                'mean': head.w_query.grad.abs().mean().item() if head.w_query.grad is not None else 0,
            }
        if hasattr(head, 'w_key'):
            grad_stats[f'head_{i}_w_key'] = {
                'mean': head.w_key.grad.abs().mean().item() if head.w_key.grad is not None else 0,
            }
    
    # Resonant layer gradients
    grad_stats['resonant_W'] = {
        'mean': model.W.grad.abs().mean().item(),
        'max': model.W.grad.abs().max().item(),
    }
    grad_stats['resonant_B'] = {
        'mean': model.B.grad.abs().mean().item(),
        'max': model.B.grad.abs().max().item(),
    }
    
    print("\nGradient Statistics:")
    for name, stats in grad_stats.items():
        print(f"  {name}: {stats}")
    
    return grad_stats


def test_attention_recall(config: Dict, distances: List[int] = [3, 5, 10, 15, 20]):
    """Test recall at various distances."""
    print("\n" + "="*70)
    print(f"ATTENTION RECALL TEST")
    print(f"Config: {config}")
    print("="*70)
    
    vocab_size = 50
    num_signals = 10
    
    model = ConfigurableAttentionModel(
        vocab_size=vocab_size,
        d_model=64,
        **config
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    def make_batch(batch_size, distance):
        trigger = 0
        signals = list(range(1, num_signals + 1))
        noise = list(range(num_signals + 1, vocab_size))
        
        seqs, targets = [], []
        for _ in range(batch_size):
            signal = random.choice(signals)
            between = [random.choice(noise) for _ in range(distance)]
            seq = [signal] + between + [trigger]
            seqs.append(seq)
            targets.append(signal)
        
        return (torch.tensor(seqs, device=device),
                torch.tensor(targets, device=device))
    
    # Train
    print("\nTraining...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for _ in range(20):
            # Mix distances during training
            dist = random.choice(distances[:3])  # Train on shorter distances
            seqs, targets = make_batch(64, dist)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: Loss={total_loss/20:.4f}")
    
    # Evaluate at each distance
    print("\nEvaluation by distance:")
    results = {}
    model.eval()
    
    for dist in distances:
        correct, total = 0, 0
        attention_entropy = []
        
        with torch.no_grad():
            for _ in range(20):
                seqs, targets = make_batch(32, dist)
                logits, attention = model(seqs, return_attention=True)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += 32
                
                # Analyze attention at last position
                if attention and len(attention) > 0:
                    last_attn = attention[-1][0]  # Last timestep, first head
                    entropy = -(last_attn * (last_attn + 1e-10).log()).sum(dim=-1).mean()
                    attention_entropy.append(entropy.item())
        
        acc = correct / total
        avg_entropy = np.mean(attention_entropy) if attention_entropy else 0
        results[dist] = {'accuracy': acc, 'entropy': avg_entropy}
        print(f"  Distance {dist:2d}: Acc={acc*100:.1f}%, Entropy={avg_entropy:.3f}")
    
    return results


def test_time_sensitivity(config: Dict):
    """Analyze sensitivity to time scaling."""
    print("\n" + "="*70)
    print("TIME SENSITIVITY ANALYSIS")
    print("="*70)
    
    time_scales = [0.0, 0.5, 1.0, 2.0, 5.0]
    results = {}
    
    for scale in time_scales:
        print(f"\nTime scale: {scale}")
        config_copy = config.copy()
        config_copy['time_scale'] = scale
        
        model = ConfigurableAttentionModel(
            vocab_size=50,
            d_model=64,
            **config_copy
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        # Quick training
        def make_batch(batch_size=32, distance=5):
            seqs, targets = [], []
            for _ in range(batch_size):
                signal = random.randint(1, 10)
                between = [random.randint(11, 49) for _ in range(distance)]
                seq = [signal] + between + [0]
                seqs.append(seq)
                targets.append(signal)
            return torch.tensor(seqs, device=device), torch.tensor(targets, device=device)
        
        for epoch in range(30):
            model.train()
            seqs, targets = make_batch()
            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        model.eval()
        correct = 0
        with torch.no_grad():
            for _ in range(10):
                seqs, targets = make_batch()
                logits = model(seqs)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == targets).sum().item()
        
        acc = correct / 320
        results[scale] = acc
        print(f"  Accuracy: {acc*100:.1f}%")
    
    return results


def test_projection_types():
    """Compare different query/key/output projection types."""
    print("\n" + "="*70)
    print("PROJECTION TYPE COMPARISON")
    print("="*70)
    
    configs = [
        {'query_type': 'euler', 'key_type': 'euler', 'output_proj_type': 'complex_linear', 'name': 'Euler-Euler-ComplexLinear'},
        {'query_type': 'euler', 'key_type': 'euler', 'output_proj_type': 'resonant', 'name': 'Euler-Euler-Resonant'},
        {'query_type': 'euler', 'key_type': 'euler', 'output_proj_type': 'linear', 'name': 'Euler-Euler-Linear'},
        {'query_type': 'resonant', 'key_type': 'resonant', 'output_proj_type': 'complex_linear', 'name': 'Resonant-Resonant-ComplexLinear'},
        {'query_type': 'linear', 'key_type': 'linear', 'output_proj_type': 'linear', 'name': 'Linear-Linear-Linear (baseline)'},
    ]
    
    results = {}
    
    for cfg in configs:
        name = cfg.pop('name')
        print(f"\nTesting: {name}")
        
        model = ConfigurableAttentionModel(
            vocab_size=50,
            d_model=64,
            n_heads=4,
            **cfg
        ).to(device)
        
        print(f"  Parameters: {model.get_num_params():,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        def make_batch(batch_size=32, distance=5):
            seqs, targets = [], []
            for _ in range(batch_size):
                signal = random.randint(1, 10)
                between = [random.randint(11, 49) for _ in range(distance)]
                seq = [signal] + between + [0]
                seqs.append(seq)
                targets.append(signal)
            return torch.tensor(seqs, device=device), torch.tensor(targets, device=device)
        
        # Train
        best_acc = 0
        for epoch in range(50):
            model.train()
            for _ in range(10):
                seqs, targets = make_batch()
                optimizer.zero_grad()
                logits = model(seqs)
                loss = F.cross_entropy(logits[:, -1, :], targets)
                loss.backward()
                optimizer.step()
            
            # Eval
            model.eval()
            correct = 0
            with torch.no_grad():
                for _ in range(5):
                    seqs, targets = make_batch()
                    logits = model(seqs)
                    preds = logits[:, -1, :].argmax(dim=-1)
                    correct += (preds == targets).sum().item()
            acc = correct / 160
            best_acc = max(best_acc, acc)
        
        results[name] = best_acc
        print(f"  Best accuracy: {best_acc*100:.1f}%")
    
    print("\n" + "-"*70)
    print("SUMMARY:")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name}: {acc*100:.1f}%")
    
    return results


def analyze_attention_patterns():
    """Deep analysis of attention patterns."""
    print("\n" + "="*70)
    print("ATTENTION PATTERN ANALYSIS")
    print("="*70)
    
    model = ConfigurableAttentionModel(
        vocab_size=50,
        d_model=64,
        n_heads=4,
    ).to(device)
    
    # Create specific patterns to analyze
    patterns = {
        'uniform': torch.randint(11, 50, (1, 15), device=device),
        'repeated': torch.tensor([[1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]], device=device),
        'signal_trigger': torch.tensor([[5] + [20]*13 + [0]], device=device),
    }
    
    model.eval()
    
    for name, seq in patterns.items():
        print(f"\nPattern: {name}")
        print(f"  Sequence: {seq[0].tolist()}")
        
        with torch.no_grad():
            _, attention = model(seq, return_attention=True)
        
        if attention:
            # Analyze attention at different positions
            for pos_idx, pos_attn in enumerate(attention[-5:], start=max(0, len(attention)-5)):
                if pos_attn:
                    weights = pos_attn[0][0]  # First sample, first head
                    max_pos = weights.argmax().item()
                    max_weight = weights.max().item()
                    entropy = -(weights * (weights + 1e-10).log()).sum().item()
                    print(f"  Position {pos_idx + 5}: max_pos={max_pos}, max_weight={max_weight:.3f}, entropy={entropy:.3f}")


def test_long_range_dependencies():
    """Test ability to maintain long-range dependencies."""
    print("\n" + "="*70)
    print("LONG-RANGE DEPENDENCY TEST")
    print("="*70)
    
    # Test: Signal at position 0, query at position N, predict signal
    distances = [5, 10, 20, 30, 50]
    
    model = ConfigurableAttentionModel(
        vocab_size=100,
        d_model=128,
        num_layers=2,
        num_neurons=256,
        n_heads=8,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100)
    
    def make_batch(batch_size, max_dist):
        seqs, targets, dists = [], [], []
        for _ in range(batch_size):
            dist = random.randint(3, max_dist)
            signal = random.randint(1, 20)
            between = [random.randint(21, 99) for _ in range(dist)]
            seq = [signal] + between + [0]
            seqs.append(seq)
            targets.append(signal)
            dists.append(dist)
        
        max_len = max(len(s) for s in seqs)
        seqs_padded = [s + [99] * (max_len - len(s)) for s in seqs]
        
        return (torch.tensor(seqs_padded, device=device),
                torch.tensor(targets, device=device),
                dists)
    
    print("\nTraining with mixed distances...")
    for epoch in range(100):
        model.train()
        total_loss = 0
        
        for _ in range(20):
            seqs, targets, _ = make_batch(32, 30)
            
            optimizer.zero_grad()
            logits = model(seqs)
            
            # Get last non-padding position for each sequence
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}: Loss={total_loss/20:.4f}")
    
    # Evaluate by distance
    print("\nEvaluation:")
    model.eval()
    
    for dist in distances:
        correct, total = 0, 0
        with torch.no_grad():
            for _ in range(20):
                seqs, targets, _ = make_batch(32, dist)
                logits = model(seqs)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += 32
        
        acc = correct / total
        print(f"  Distance {dist:2d}: {acc*100:.1f}%")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DEEP ATTENTION ANALYSIS SUITE")
    print(f"Device: {device}")
    print(f"Golden ratio φ = {PHI:.6f}")
    print("="*70)
    
    results = {}
    
    # 1. Gradient flow analysis
    print("\n" + "#"*70)
    print("# 1. GRADIENT FLOW ANALYSIS")
    print("#"*70)
    
    default_config = {'query_type': 'euler', 'key_type': 'euler', 'output_proj_type': 'complex_linear'}
    results['gradient_flow'] = test_gradient_flow(default_config)
    
    # 2. Time sensitivity
    print("\n" + "#"*70)
    print("# 2. TIME SENSITIVITY ANALYSIS")
    print("#"*70)
    
    results['time_sensitivity'] = test_time_sensitivity(default_config)
    
    # 3. Projection type comparison
    print("\n" + "#"*70)
    print("# 3. PROJECTION TYPE COMPARISON")
    print("#"*70)
    
    results['projection_types'] = test_projection_types()
    
    # 4. Attention pattern analysis
    print("\n" + "#"*70)
    print("# 4. ATTENTION PATTERN ANALYSIS")
    print("#"*70)
    
    analyze_attention_patterns()
    
    # 5. Attention recall at various distances
    print("\n" + "#"*70)
    print("# 5. ATTENTION RECALL ANALYSIS")
    print("#"*70)
    
    results['recall'] = test_attention_recall(default_config)
    
    # 6. Long-range dependency test
    print("\n" + "#"*70)
    print("# 6. LONG-RANGE DEPENDENCY TEST")
    print("#"*70)
    
    test_long_range_dependencies()
    
    # Save results
    output_dir = Path('/home/aiman/pi/experiments/results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        return obj
    
    with open(output_dir / f'attention_analysis_{timestamp}.json', 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_dir / f'attention_analysis_{timestamp}.json'}")
    print("="*70)
