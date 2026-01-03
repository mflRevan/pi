#!/usr/bin/env python3
"""
Deep Dive: What Makes Echo Attention Work?

This script performs ablation studies and mechanistic analysis
to understand the essential components of Echo Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Dict, List, Optional
import sys
from datetime import datetime

sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut
from rin.model import PHI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def print_section(title, char="="):
    print(f"\n{char*70}")
    print(title)
    print(f"{char*70}")


class FlexibleAttentionModel(nn.Module):
    """Model with configurable attention mechanisms for ablation."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        use_euler_transform: bool = True,
        use_euler_attention: bool = True,
        use_time_encoding: bool = True,
        time_scale: float = PHI,
        use_resonant_layer: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Config
        self.use_euler_transform = use_euler_transform
        self.use_euler_attention = use_euler_attention
        self.use_time_encoding = use_time_encoding
        self.time_scale = time_scale
        self.use_resonant_layer = use_resonant_layer
        
        # Embeddings
        if use_euler_transform:
            self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        else:
            self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Attention
        if use_euler_attention:
            self.w_query = nn.ParameterList([
                nn.Parameter(torch.randn(self.d_head) * 0.02) for _ in range(n_heads)
            ])
            self.b_query = nn.ParameterList([
                nn.Parameter(torch.zeros(self.d_head)) for _ in range(n_heads)
            ])
            self.w_key = nn.ParameterList([
                nn.Parameter(torch.randn(self.d_head) * 0.02) for _ in range(n_heads)
            ])
            self.b_key = nn.ParameterList([
                nn.Parameter(torch.zeros(self.d_head)) for _ in range(n_heads)
            ])
        else:
            # Standard linear attention
            self.q_proj = nn.Linear(d_model, d_model, bias=False)
            self.k_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.context_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Resonant layer
        if use_resonant_layer:
            num_neurons = 128
            self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
            self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
            self.proj_real = nn.Linear(num_neurons, d_model, bias=False)
            self.proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Linear(4 * d_model, d_model),
            )
        
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def euler_state_update(self, h_real, h_imag, w, b, t):
        """Update hidden state using Euler transform."""
        lut = self._get_lut(h_real.device)
        wavelength = 1.0 + w.abs()
        
        if self.use_time_encoding:
            t_scaled = t.unsqueeze(-1) * self.time_scale if t.dim() == 1 else t * self.time_scale
        else:
            t_scaled = 0
        
        theta_real = h_real / wavelength + b + t_scaled
        theta_imag = h_imag / wavelength + b + t_scaled
        
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        return (cos_real * cos_imag - sin_real * sin_imag,
                cos_real * sin_imag + sin_real * cos_imag)
    
    def simple_state_update(self, h, emb):
        """Simple state update without Euler transform."""
        return torch.tanh(h + emb)
    
    def euler_attention_head(self, x, states, t, head_idx):
        """Euler-based attention scoring."""
        lut = self._get_lut(x.device)
        
        start_idx = head_idx * self.d_head
        end_idx = (head_idx + 1) * self.d_head
        
        x_patch = x[:, start_idx:end_idx]
        wl_q = 1.0 + self.w_query[head_idx].abs()
        
        if self.use_time_encoding:
            t_scaled = t.unsqueeze(-1) * self.time_scale if t.dim() == 1 else t * self.time_scale
        else:
            t_scaled = 0
        
        theta_q = x_patch / wl_q + self.b_query[head_idx] + t_scaled
        sin_q, cos_q = lut.lookup_sin_cos(theta_q)
        query = torch.cat([cos_q, sin_q], dim=-1)
        
        k_patches = states[:, :, start_idx:end_idx]
        wl_k = 1.0 + self.w_key[head_idx].abs()
        theta_k = k_patches / wl_k + self.b_key[head_idx]
        sin_k, cos_k = lut.lookup_sin_cos(theta_k)
        keys = torch.cat([cos_k, sin_k], dim=-1)
        
        scale = math.sqrt(2 * self.d_head)
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / scale
        weights = F.softmax(scores, dim=-1)
        
        output = torch.bmm(weights.unsqueeze(1), states).squeeze(1)
        return output, weights
    
    def linear_attention_head(self, x, states, head_idx):
        """Standard linear attention scoring."""
        start_idx = head_idx * self.d_head
        end_idx = (head_idx + 1) * self.d_head
        
        q = self.q_proj(x)[:, start_idx:end_idx]
        k = self.k_proj(states)[:, :, start_idx:end_idx]
        
        scale = math.sqrt(self.d_head)
        scores = torch.bmm(q.unsqueeze(1), k.transpose(1, 2)).squeeze(1) / scale
        weights = F.softmax(scores, dim=-1)
        
        output = torch.bmm(weights.unsqueeze(1), states).squeeze(1)
        return output, weights
    
    def resonant_layer(self, x, t):
        """Resonant FFN layer."""
        lut = self._get_lut(x.device)
        x_exp = x.unsqueeze(1)
        wl = 1.0 + self.W.abs()
        
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)
        
        theta = x_exp / wl + self.B + t
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        cos_sum = cos_theta.sum(dim=-1)
        sin_sum = sin_theta.sum(dim=-1)
        
        return F.silu(self.proj_real(cos_sum) + self.proj_imag(sin_sum))
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        embeddings = self.token_embedding(input_ids)
        
        if self.use_euler_transform:
            h_real = torch.zeros(batch_size, self.d_model, device=device)
            h_imag = torch.zeros(batch_size, self.d_model, device=device)
            w_emb = embeddings[:, :, :self.d_model]
            b_emb = embeddings[:, :, self.d_model:]
        else:
            h = torch.zeros(batch_size, self.d_model, device=device)
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        cached_states = []
        all_logits = []
        
        for t_idx in range(seq_len):
            t_val = t_indices[t_idx].expand(batch_size)
            
            if self.use_euler_transform:
                w_t, b_t = w_emb[:, t_idx], b_emb[:, t_idx]
                h_real, h_imag = self.euler_state_update(h_real, h_imag, w_t, b_t, t_val)
                x = h_real + h_imag
            else:
                x = self.simple_state_update(h, embeddings[:, t_idx])
                h = x
            
            cached_states.append(x)
            
            if len(cached_states) > 1:
                states = torch.stack(cached_states[:-1], dim=1)
                
                head_outputs = []
                for h_idx in range(self.n_heads):
                    if self.use_euler_attention:
                        out, _ = self.euler_attention_head(x, states, t_val, h_idx)
                    else:
                        out, _ = self.linear_attention_head(x, states, h_idx)
                    head_outputs.append(out)
                
                context = torch.stack(head_outputs, dim=0).sum(dim=0)
                x = x + self.context_proj(context)
            
            if self.use_resonant_layer:
                t_scaled = t_val * self.time_scale
                x = x + self.resonant_layer(x, t_scaled)
            else:
                x = x + self.ffn(x)
            
            all_logits.append(self.output_proj(x))
        
        return torch.stack(all_logits, dim=1)


def ablation_study():
    """Systematic ablation of each component."""
    print_section("ABLATION STUDY: What Components Are Essential?")
    
    vocab_size = 50
    
    configurations = [
        # Full model
        {'name': 'Full Echo Attention',
         'euler_transform': True, 'euler_attention': True, 
         'time_encoding': True, 'resonant_layer': True},
        
        # Ablate each component
        {'name': 'No Euler Transform',
         'euler_transform': False, 'euler_attention': True,
         'time_encoding': True, 'resonant_layer': True},
        
        {'name': 'No Euler Attention',
         'euler_transform': True, 'euler_attention': False,
         'time_encoding': True, 'resonant_layer': True},
        
        {'name': 'No Time Encoding',
         'euler_transform': True, 'euler_attention': True,
         'time_encoding': False, 'resonant_layer': True},
        
        {'name': 'No Resonant Layer',
         'euler_transform': True, 'euler_attention': True,
         'time_encoding': True, 'resonant_layer': False},
        
        # Minimal baselines
        {'name': 'Simple RNN + Attention',
         'euler_transform': False, 'euler_attention': False,
         'time_encoding': False, 'resonant_layer': False},
        
        {'name': 'Euler Transform Only',
         'euler_transform': True, 'euler_attention': False,
         'time_encoding': False, 'resonant_layer': False},
        
        {'name': 'Euler Attention Only',
         'euler_transform': False, 'euler_attention': True,
         'time_encoding': False, 'resonant_layer': False},
    ]
    
    results = {}
    
    for config in configurations:
        name = config['name']
        print(f"\nTesting: {name}")
        
        try:
            model = FlexibleAttentionModel(
                vocab_size=vocab_size,
                d_model=64,
                n_heads=4,
                use_euler_transform=config['euler_transform'],
                use_euler_attention=config['euler_attention'],
                use_time_encoding=config['time_encoding'],
                use_resonant_layer=config['resonant_layer'],
            ).to(device)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            
            # Train
            for epoch in range(60):
                model.train()
                for _ in range(20):
                    seqs = []
                    targets = []
                    for _ in range(32):
                        signal = random.randint(1, 10)
                        dist = random.randint(3, 10)
                        seq = [signal] + [random.randint(20, 49) for _ in range(dist)] + [0]
                        seqs.append(seq)
                        targets.append(signal)
                    
                    # Pad to same length
                    max_len = max(len(s) for s in seqs)
                    seqs = [s + [0] * (max_len - len(s)) for s in seqs]
                    
                    seqs = torch.tensor(seqs, device=device)
                    targets = torch.tensor(targets, device=device)
                    
                    optimizer.zero_grad()
                    logits = model(seqs)
                    
                    # Get logits at trigger position
                    trigger_positions = (seqs == 0).float().argmax(dim=1)
                    batch_indices = torch.arange(seqs.size(0), device=device)
                    trigger_logits = logits[batch_indices, trigger_positions]
                    
                    loss = F.cross_entropy(trigger_logits, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
            
            # Evaluate at different distances
            model.eval()
            distance_results = {}
            
            for dist in [5, 10, 15]:
                correct, total = 0, 0
                with torch.no_grad():
                    for _ in range(20):
                        seqs = []
                        targets = []
                        for _ in range(50):
                            signal = random.randint(1, 10)
                            seq = [signal] + [random.randint(20, 49) for _ in range(dist)] + [0]
                            seqs.append(seq)
                            targets.append(signal)
                        
                        seqs = torch.tensor(seqs, device=device)
                        targets = torch.tensor(targets, device=device)
                        
                        logits = model(seqs)
                        preds = logits[:, -1, :].argmax(dim=-1)
                        correct += (preds == targets).sum().item()
                        total += 50
                
                distance_results[dist] = correct / total
            
            results[name] = distance_results
            print(f"  d=5: {distance_results[5]*100:.1f}%, "
                  f"d=10: {distance_results[10]*100:.1f}%, "
                  f"d=15: {distance_results[15]*100:.1f}%")
            
        except Exception as e:
            print(f"  Failed: {e}")
            results[name] = {'error': str(e)}
    
    # Summary
    print("\n" + "-"*70)
    print("ABLATION SUMMARY:")
    print("-"*70)
    print(f"{'Configuration':<30} | d=5   | d=10  | d=15  | Avg")
    print("-"*70)
    
    for name, res in results.items():
        if 'error' not in res:
            avg = np.mean(list(res.values()))
            print(f"{name:<30} | {res[5]*100:4.1f}% | {res[10]*100:4.1f}% | {res[15]*100:4.1f}% | {avg*100:4.1f}%")
    
    return results


def attention_mechanism_analysis():
    """Deep dive into how attention mechanism works."""
    print_section("ATTENTION MECHANISM DEEP DIVE")
    
    vocab_size = 50
    model = FlexibleAttentionModel(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
        use_euler_transform=True,
        use_euler_attention=True,
        use_time_encoding=True,
        use_resonant_layer=True,
    ).to(device)
    
    # Train briefly
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(50):
        seqs = []
        targets = []
        for _ in range(32):
            signal = random.randint(1, 10)
            seq = [signal] + [random.randint(20, 49) for _ in range(6)] + [0]
            seqs.append(seq)
            targets.append(signal)
        
        seqs = torch.tensor(seqs, device=device)
        targets = torch.tensor(targets, device=device)
        
        optimizer.zero_grad()
        logits = model(seqs)
        loss = F.cross_entropy(logits[:, -1, :], targets)
        loss.backward()
        optimizer.step()
    
    # Analyze attention patterns
    print("\n1. Query-Key Angle Analysis")
    print("-"*50)
    
    # Create a specific test case
    model.eval()
    seq = torch.tensor([[5, 25, 26, 27, 28, 29, 30, 0]], device=device)
    
    with torch.no_grad():
        h_real = torch.zeros(1, model.d_model, device=device)
        h_imag = torch.zeros(1, model.d_model, device=device)
        
        embeddings = model.token_embedding(seq)
        w_emb = embeddings[:, :, :model.d_model]
        b_emb = embeddings[:, :, model.d_model:]
        
        t_indices = torch.arange(seq.shape[1], device=device, dtype=torch.float32)
        
        cached_states = []
        lut = model._get_lut(device)
        
        for t_idx in range(seq.shape[1]):
            w_t, b_t = w_emb[:, t_idx], b_emb[:, t_idx]
            t_val = t_indices[t_idx].expand(1)
            
            h_real, h_imag = model.euler_state_update(h_real, h_imag, w_t, b_t, t_val)
            x = h_real + h_imag
            cached_states.append(x)
        
        # At trigger position (last), analyze attention
        query_state = cached_states[-1]  # Query from trigger
        key_states = torch.stack(cached_states[:-1], dim=1)  # Keys from history
        
        print(f"  Sequence: {seq[0].tolist()}")
        print(f"  Query state (trigger): norm={query_state.norm().item():.4f}")
        print(f"  Key states (history): {key_states.shape}")
        
        # For each head, compute Q-K angles
        for h_idx in range(model.n_heads):
            start_idx = h_idx * model.d_head
            end_idx = (h_idx + 1) * model.d_head
            
            # Query angle
            q_patch = query_state[:, start_idx:end_idx]
            wl_q = 1.0 + model.w_query[h_idx].abs()
            t_val = t_indices[-1].expand(1)
            t_scaled = t_val.unsqueeze(-1) * model.time_scale
            theta_q = q_patch / wl_q + model.b_query[h_idx] + t_scaled
            
            # Key angles for each position
            k_patches = key_states[:, :, start_idx:end_idx]
            wl_k = 1.0 + model.w_key[h_idx].abs()
            theta_k = k_patches / wl_k + model.b_key[h_idx]
            
            # Compute angle differences (mean across dimensions)
            angle_diffs = (theta_q.unsqueeze(1) - theta_k).mean(dim=-1)  # (1, seq_len-1)
            cos_diffs = torch.cos(angle_diffs)
            
            # Actual attention weights
            sin_q, cos_q = lut.lookup_sin_cos(theta_q)
            query = torch.cat([cos_q, sin_q], dim=-1)
            sin_k, cos_k = lut.lookup_sin_cos(theta_k)
            keys = torch.cat([cos_k, sin_k], dim=-1)
            
            scale = math.sqrt(2 * model.d_head)
            scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / scale
            weights = F.softmax(scores, dim=-1)
            
            print(f"\n  Head {h_idx}:")
            print(f"    Mean angle diff to signal (pos 0): {angle_diffs[0, 0].item():.4f}")
            print(f"    cos(angle diff): {cos_diffs[0, 0].item():.4f}")
            print(f"    Attention weight on signal: {weights[0, 0].item():.4f}")
            print(f"    Max attention at position: {weights[0].argmax().item()}")


def phase_space_visualization():
    """Visualize state evolution in phase space."""
    print_section("PHASE SPACE VISUALIZATION")
    
    vocab_size = 50
    model = FlexibleAttentionModel(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
    ).to(device)
    
    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(30):
        seqs = []
        targets = []
        for _ in range(32):
            signal = random.randint(1, 10)
            seq = [signal] + [random.randint(20, 49) for _ in range(6)] + [0]
            seqs.append(seq)
            targets.append(signal)
        
        seqs = torch.tensor(seqs, device=device)
        targets = torch.tensor(targets, device=device)
        
        optimizer.zero_grad()
        logits = model(seqs)
        loss = F.cross_entropy(logits[:, -1, :], targets)
        loss.backward()
        optimizer.step()
    
    # Trace state evolution for different signals
    print("\nState evolution for different signal tokens:")
    print("-"*50)
    
    model.eval()
    signals = [1, 5, 10]
    
    for signal in signals:
        seq = torch.tensor([[signal, 25, 26, 27, 28, 29, 30, 0]], device=device)
        
        with torch.no_grad():
            h_real = torch.zeros(1, model.d_model, device=device)
            h_imag = torch.zeros(1, model.d_model, device=device)
            
            embeddings = model.token_embedding(seq)
            w_emb = embeddings[:, :, :model.d_model]
            b_emb = embeddings[:, :, model.d_model:]
            
            t_indices = torch.arange(seq.shape[1], device=device, dtype=torch.float32)
            
            print(f"\nSignal token {signal}:")
            
            for t_idx in range(seq.shape[1]):
                w_t, b_t = w_emb[:, t_idx], b_emb[:, t_idx]
                t_val = t_indices[t_idx].expand(1)
                
                h_real_old = h_real.clone()
                h_imag_old = h_imag.clone()
                
                h_real, h_imag = model.euler_state_update(h_real, h_imag, w_t, b_t, t_val)
                
                # Compute "angle" (mean phase) and "radius" (magnitude)
                combined = h_real + h_imag
                magnitude = combined.norm().item()
                
                # Track first few dimensions
                dim_0_real = h_real[0, 0].item()
                dim_0_imag = h_imag[0, 0].item()
                angle_0 = math.atan2(dim_0_imag, dim_0_real)
                
                print(f"  t={t_idx}: token={seq[0, t_idx].item():2d}, "
                      f"mag={magnitude:.4f}, dim0_angle={angle_0:.4f}")


def information_flow_analysis():
    """Analyze how information flows through the model."""
    print_section("INFORMATION FLOW ANALYSIS")
    
    vocab_size = 50
    model = FlexibleAttentionModel(
        vocab_size=vocab_size,
        d_model=64,
        n_heads=4,
    ).to(device)
    
    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(50):
        seqs = []
        targets = []
        for _ in range(32):
            signal = random.randint(1, 10)
            seq = [signal] + [random.randint(20, 49) for _ in range(6)] + [0]
            seqs.append(seq)
            targets.append(signal)
        
        seqs = torch.tensor(seqs, device=device)
        targets = torch.tensor(targets, device=device)
        
        optimizer.zero_grad()
        logits = model(seqs)
        loss = F.cross_entropy(logits[:, -1, :], targets)
        loss.backward()
        optimizer.step()
    
    # Gradient-based importance
    print("\n1. Gradient-based Input Importance")
    print("-"*50)
    
    model.eval()
    
    seqs = torch.tensor([[5, 25, 26, 27, 28, 29, 30, 0]], device=device)
    seqs.requires_grad = False
    
    # Get embedding and make it require grad
    emb = model.token_embedding(seqs).detach().requires_grad_(True)
    
    # Manual forward pass with embedding
    h_real = torch.zeros(1, model.d_model, device=device)
    h_imag = torch.zeros(1, model.d_model, device=device)
    
    w_emb = emb[:, :, :model.d_model]
    b_emb = emb[:, :, model.d_model:]
    
    t_indices = torch.arange(seqs.shape[1], device=device, dtype=torch.float32)
    
    cached_states = []
    
    for t_idx in range(seqs.shape[1]):
        w_t, b_t = w_emb[:, t_idx], b_emb[:, t_idx]
        t_val = t_indices[t_idx].expand(1)
        
        h_real, h_imag = model.euler_state_update(h_real, h_imag, w_t, b_t, t_val)
        x = h_real + h_imag
        cached_states.append(x)
        
        if len(cached_states) > 1:
            states = torch.stack(cached_states[:-1], dim=1)
            
            head_outputs = []
            for h_idx in range(model.n_heads):
                out, _ = model.euler_attention_head(x, states, t_val, h_idx)
                head_outputs.append(out)
            
            context = torch.stack(head_outputs, dim=0).sum(dim=0)
            x = x + model.context_proj(context)
        
        t_scaled = t_val * model.time_scale
        x = x + model.resonant_layer(x, t_scaled)
    
    logits = model.output_proj(x)
    
    # Gradient of correct class w.r.t. embeddings
    correct_class = 5  # Signal was 5
    loss = logits[0, correct_class]
    loss.backward()
    
    # Importance of each position
    importance = emb.grad.abs().sum(dim=-1).squeeze(0)
    
    print(f"  Input sequence: {seqs[0].tolist()}")
    print(f"  Position importance (gradient magnitude):")
    for i, imp in enumerate(importance):
        bar = "█" * int(imp.item() * 20 / importance.max().item())
        print(f"    Position {i} (token {seqs[0, i].item():2d}): {imp.item():.4f} {bar}")


if __name__ == "__main__":
    print("="*70)
    print("ECHO ATTENTION: WHAT MAKES IT WORK?")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    ablation_study()
    attention_mechanism_analysis()
    phase_space_visualization()
    information_flow_analysis()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
