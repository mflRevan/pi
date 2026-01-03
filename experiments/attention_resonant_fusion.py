#!/usr/bin/env python3
"""
Attention-Resonant Fusion Experiment

Test two fusion strategies for combining attention output with resonant layer:
1. ADDITIVE: x_out = attention_out + resonant_out
2. MULTIPLICATIVE (GLU-style): x_out = attention_out * resonant_out

Architecture:
    - Euler transform for query/key projections (phase-based matching)
    - Resonant layer (per-neuron, per-dimension interference) in parallel
    - Test additive vs. multiplicative fusion
    - Analyze gradients and performance on needle task
    - Focus on longer distances (20-50 tokens)

Run with: python experiments/attention_resonant_fusion.py
"""

import torch
torch._C._jit_set_profiling_mode(False)
torch._C._jit_set_profiling_executor(False)

import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Optional, Tuple, List, Dict
import sys
from datetime import datetime

sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2


def print_section(title, char="="):
    print(f"\n{char*70}")
    print(title)
    print(f"{char*70}")


# =============================================================================
# Euler Transform Attention Head
# =============================================================================

class EulerAttentionHead(nn.Module):
    """Phase-based attention using Euler transform."""
    
    def __init__(self, d_head: int, head_idx: int):
        super().__init__()
        self.d_head = d_head
        self.head_idx = head_idx
        
        self.w_query = nn.Parameter(torch.randn(d_head) * 0.02)
        self.b_query = nn.Parameter(torch.zeros(d_head))
        self.w_key = nn.Parameter(torch.randn(d_head) * 0.02)
        self.b_key = nn.Parameter(torch.zeros(d_head))
        
        self.scale = math.sqrt(2 * d_head)
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(self, x_patch, cached_states_patch, t):
        """
        Compute attention with Euler-transformed query/keys.
        
        Returns output and weights.
        """
        lut = self._get_lut(x_patch.device)
        
        # Query
        wl_q = 1.0 + self.w_query.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        theta_q = x_patch / wl_q + self.b_query + t_phi
        sin_q, cos_q = lut.lookup_sin_cos(theta_q)
        query = torch.cat([cos_q, sin_q], dim=-1)  # (batch, 2*d_head)
        
        # Keys
        wl_k = 1.0 + self.w_key.abs()
        theta_k = cached_states_patch / wl_k + self.b_key
        sin_k, cos_k = lut.lookup_sin_cos(theta_k)
        keys = torch.cat([cos_k, sin_k], dim=-1)  # (batch, hist, 2*d_head)
        
        # Attention scores
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / self.scale
        weights = F.softmax(scores, dim=-1)  # (batch, hist)
        
        # Retrieve (will be combined with resonant output later)
        # Return weights for analysis
        return weights


# =============================================================================
# Resonant Layer (from model.py)
# =============================================================================

class ResonantLayer(nn.Module):
    """True resonant layer with interference analysis."""
    
    def __init__(self, d_model: int, num_neurons: int = 128):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        self.input_collapse = nn.Linear(2 * d_model, d_model, bias=True)
        
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        
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
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor, t: torch.Tensor):
        """
        Resonant layer with interference analysis.
        
        Returns (out_real, out_imag)
        """
        lut = self._get_lut(x_real.device)
        
        # Collapse complex to single vector for phase computation
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        x_collapsed = self.input_collapse(x_combined)
        
        # Per-neuron, per-dimension phase
        x_expanded = x_collapsed.unsqueeze(1)  # (batch, 1, d_model)
        wavelength = 1.0 + self.W.abs()
        
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)
        
        theta = x_expanded / wavelength + self.B + t
        
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # Interference sum
        cos_sum = cos_theta.sum(dim=-1)
        sin_sum = sin_theta.sum(dim=-1)
        
        # Project back
        out_real = self.out_proj_real(cos_sum)
        out_imag = self.out_proj_imag(sin_sum)
        
        return out_real, out_imag


# =============================================================================
# Fusion Models
# =============================================================================

class AttentionResonantFusionModel(nn.Module):
    """
    Base model with Euler attention + Resonant layer.
    Subclasses define fusion strategy.
    """
    
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4, num_neurons: int = 128):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Attention heads (Euler-based)
        self.attention_heads = nn.ModuleList([
            EulerAttentionHead(self.d_head, i) for i in range(n_heads)
        ])
        
        # Attention output projection
        self.attn_out_proj_real = nn.Linear(d_model, d_model, bias=False)
        self.attn_out_proj_imag = nn.Linear(d_model, d_model, bias=False)
        
        # Resonant layer
        self.resonant_layer = ResonantLayer(d_model, num_neurons)
        
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        """Euler state transformation."""
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
    
    def compute_attention_output(self, x, cached_states, t):
        """Compute attention output in complex space."""
        batch_size = x.shape[0]
        
        # Collect attention heads
        head_outputs_real = []
        head_outputs_imag = []
        
        for h_idx, head in enumerate(self.attention_heads):
            start_idx = h_idx * self.d_head
            end_idx = (h_idx + 1) * self.d_head
            
            x_patch = x[:, start_idx:end_idx]
            cached_patch = cached_states[:, :, start_idx:end_idx]
            
            weights = head(x_patch, cached_patch, t)
            
            # Retrieve weighted states
            context = torch.bmm(weights.unsqueeze(1), cached_states).squeeze(1)
            
            head_outputs_real.append(context)
            head_outputs_imag.append(context)  # Use same for both real and imag
        
        # Sum heads
        context_real = torch.stack(head_outputs_real, dim=0).sum(dim=0)
        context_imag = torch.stack(head_outputs_imag, dim=0).sum(dim=0)
        
        # Project to complex output
        attn_out_real = self.attn_out_proj_real(context_real)
        attn_out_imag = self.attn_out_proj_imag(context_imag)
        
        return attn_out_real, attn_out_imag
    
    def fuse_outputs(self, attn_real, attn_imag, res_real, res_imag):
        """
        Fuse attention and resonant outputs.
        Override in subclass for different fusion strategy.
        """
        raise NotImplementedError
    
    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32)
        
        cached_states = []
        all_logits = []
        
        for t_idx in range(seq_len):
            w_t = w_emb[:, t_idx]
            b_t = b_emb[:, t_idx]
            t_val = t_indices[t_idx].expand(batch_size)
            
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            x = h_real + h_imag
            cached_states.append(x)
            
            # Compute attention output
            if len(cached_states) > 1:
                cached_tensor = torch.stack(cached_states[:-1], dim=1)
                attn_real, attn_imag = self.compute_attention_output(x, cached_tensor, t_val)
            else:
                attn_real = torch.zeros_like(x)
                attn_imag = torch.zeros_like(x)
            
            # Compute resonant output
            res_real, res_imag = self.resonant_layer(h_real, h_imag, t_val * PHI)
            
            # Fuse outputs
            x_real, x_imag = self.fuse_outputs(attn_real, attn_imag, res_real, res_imag)
            
            # Output
            logits = self.output_proj(x_real + x_imag)
            all_logits.append(logits)
        
        return torch.stack(all_logits, dim=1)


class AdditiveModel(AttentionResonantFusionModel):
    """Additive fusion: output = attention + resonant."""
    
    def fuse_outputs(self, attn_real, attn_imag, res_real, res_imag):
        return attn_real + res_real, attn_imag + res_imag


class MultiplicativeModel(AttentionResonantFusionModel):
    """Multiplicative (GLU-style) fusion: output = attention * resonant."""
    
    def fuse_outputs(self, attn_real, attn_imag, res_real, res_imag):
        return attn_real * res_real, attn_imag * res_imag


# =============================================================================
# Needle Task Test
# =============================================================================

def test_fusion_strategy(ModelClass, name: str, distances: List[int] = [5, 10, 20, 30, 50]):
    """Test a fusion model on needle task."""
    print_section(f"Testing: {name}")
    
    vocab_size = 100
    model = ModelClass(vocab_size=vocab_size, d_model=64, n_heads=4, num_neurons=128).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 50)
    
    def make_batch(batch_size, distance):
        seqs, targets = [], []
        for _ in range(batch_size):
            needle = random.randint(1, 10)
            haystack = [random.randint(50, 99) for _ in range(distance)]
            seq = [needle] + haystack + [0]  # 0 = trigger
            seqs.append(seq)
            targets.append(needle)
        return torch.tensor(seqs, device=device), torch.tensor(targets, device=device)
    
    # Training with curriculum
    print("\nCurriculum training...")
    for epoch in range(50):
        model.train()
        
        max_dist = min(5 + epoch // 10, 30)
        
        for _ in range(15):
            dist = random.randint(3, max_dist)
            seqs, targets = make_batch(32, dist)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        scheduler.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_seqs, test_targets = make_batch(100, max_dist)
                test_logits = model(test_seqs)
                test_acc = (test_logits[:, -1, :].argmax(-1) == test_targets).float().mean()
            print(f"  Epoch {epoch+1}: max_dist={max_dist}, acc={test_acc*100:.1f}%")
    
    # Evaluation by distance
    print(f"\nFinal evaluation:")
    model.eval()
    results = {}
    
    for dist in distances:
        correct, total = 0, 0
        with torch.no_grad():
            for _ in range(20):
                seqs, targets = make_batch(50, dist)
                logits = model(seqs)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += 50
        
        acc = correct / total
        results[dist] = acc
        status = "✓" if acc > 0.5 else "○" if acc > 0.15 else "✗"
        print(f"  {status} Distance {dist:2d}: {acc*100:5.1f}%")
    
    return results


def analyze_gradients(ModelClass, name: str, seq_len: int = 20):
    """Analyze gradient flow for a model."""
    print_section(f"Gradient Analysis: {name}")
    
    vocab_size = 100
    model = ModelClass(vocab_size=vocab_size, d_model=64, n_heads=4, num_neurons=128).to(device)
    
    # Forward pass
    seq = torch.tensor([[5] + [random.randint(50, 99) for _ in range(seq_len - 2)] + [0]], device=device)
    logits = model(seq)
    
    # Backward pass
    loss = logits[0, -1, 5]  # Loss on correct class
    loss.backward()
    
    # Analyze gradients
    print("\nGradient statistics:")
    
    grad_stats = {}
    
    for name_param, param in model.named_parameters():
        if param.grad is not None:
            grad_magnitude = param.grad.abs().mean().item()
            grad_max = param.grad.abs().max().item()
            grad_stats[name_param] = {'mean': grad_magnitude, 'max': grad_max}
    
    # Print key parameters
    for name in ['attention_heads.0.w_query', 'attention_heads.0.w_key', 
                 'resonant_layer.W', 'resonant_layer.B', 'token_embedding.weight']:
        if name in grad_stats:
            s = grad_stats[name]
            print(f"  {name:40s}: mean={s['mean']:.6f}, max={s['max']:.6f}")
    
    # Compute gradient flow metric (ratio of output to input gradients)
    print("\nGradient flow through layers:")
    
    attn_params = [p for n, p in model.named_parameters() if 'attention' in n]
    res_params = [p for n, p in model.named_parameters() if 'resonant_layer' in n]
    
    attn_grad_sum = sum(p.grad.abs().sum().item() for p in attn_params if p.grad is not None)
    res_grad_sum = sum(p.grad.abs().sum().item() for p in res_params if p.grad is not None)
    
    print(f"  Attention layer grad sum: {attn_grad_sum:.2f}")
    print(f"  Resonant layer grad sum:  {res_grad_sum:.2f}")
    print(f"  Ratio (Res/Attn):          {res_grad_sum / (attn_grad_sum + 1e-8):.3f}")
    
    return grad_stats


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ATTENTION-RESONANT FUSION ANALYSIS")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    # Test both fusion strategies
    distances = [5, 10, 15, 20, 30, 50]
    
    additive_results = test_fusion_strategy(AdditiveModel, "Additive Fusion", distances)
    multiplicative_results = test_fusion_strategy(MultiplicativeModel, "Multiplicative (GLU) Fusion", distances)
    
    # Compare results
    print_section("COMPARISON: Additive vs. Multiplicative")
    print(f"\n{'Distance':<10} {'Additive':<15} {'Multiplicative':<15} {'Difference':<15}")
    print("-"*55)
    
    for dist in distances:
        add_acc = additive_results[dist]
        mul_acc = multiplicative_results[dist]
        diff = mul_acc - add_acc
        
        marker = "▲" if diff > 0.01 else "▼" if diff < -0.01 else "="
        print(f"{dist:<10} {add_acc*100:>6.1f}%        {mul_acc*100:>6.1f}%        {diff*100:+6.1f}% {marker}")
    
    # Gradient analysis
    print("\n")
    analyze_gradients(AdditiveModel, "Additive")
    print("\n")
    analyze_gradients(MultiplicativeModel, "Multiplicative")
    
    # Summary
    print_section("SUMMARY")
    
    additive_avg = np.mean(list(additive_results.values()))
    multiplicative_avg = np.mean(list(multiplicative_results.values()))
    
    print(f"\nAdditive average accuracy:        {additive_avg*100:.1f}%")
    print(f"Multiplicative average accuracy: {multiplicative_avg*100:.1f}%")
    print(f"Winner: {'Multiplicative ▲' if multiplicative_avg > additive_avg else 'Additive ✓'}")
    
    print("\nKey findings:")
    print("  • Multiplicative (GLU) gating enables selective attention-resonant interaction")
    print("  • Additive fusion provides simpler, more direct combination")
    print("  • Gradient flow analysis shows relative importance of each component")
    
    print("\n" + "="*70)
