#!/usr/bin/env python3
"""
Stress Test Suite for Echo Chamber Attention

This script performs extensive stress testing:
1. Extreme long-range recall (up to 100 tokens)
2. Gradient pathology detection
3. Memory scaling analysis
4. Interference pattern stress tests
5. Adversarial input patterns

Run with: python experiments/attention_stress_test.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import Optional, Tuple, List, Dict
import sys
import gc
from datetime import datetime
from pathlib import Path

sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut
from rin.model import ComplexLinear, PHI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Stress Test Model (Minimal, for testing)
# =============================================================================

class StressTestModel(nn.Module):
    """Minimal model for stress testing attention."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        num_neurons: int = 128,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Attention components
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
        
        self.context_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Resonant layer
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        self.proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        self._lut = None
        self.attention_weights_history = []  # For analysis
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        lut = self._get_lut(h_real.device)
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        return (cos_real * cos_imag - sin_real * sin_imag,
                cos_real * sin_imag + sin_real * cos_imag)
    
    def attention_head(self, x, states, t, head_idx):
        lut = self._get_lut(x.device)
        
        start_idx = head_idx * self.d_head
        end_idx = (head_idx + 1) * self.d_head
        
        # Query
        x_patch = x[:, start_idx:end_idx]
        wl_q = 1.0 + self.w_query[head_idx].abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        theta_q = x_patch / wl_q + self.b_query[head_idx] + t_phi
        sin_q, cos_q = lut.lookup_sin_cos(theta_q)
        query = torch.cat([cos_q, sin_q], dim=-1)
        
        # Keys
        k_patches = states[:, :, start_idx:end_idx]
        wl_k = 1.0 + self.w_key[head_idx].abs()
        theta_k = k_patches / wl_k + self.b_key[head_idx]
        sin_k, cos_k = lut.lookup_sin_cos(theta_k)
        keys = torch.cat([cos_k, sin_k], dim=-1)
        
        # Scores
        scale = math.sqrt(2 * self.d_head)
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1) / scale
        weights = F.softmax(scores, dim=-1)
        
        output = torch.bmm(weights.unsqueeze(1), states).squeeze(1)
        return output, weights
    
    def resonant_layer(self, x, t):
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
    
    def forward(self, input_ids, return_attention=False, store_attention=False):
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
        all_attention = []
        
        if store_attention:
            self.attention_weights_history = []
        
        for t_idx in range(seq_len):
            w_t, b_t = w_emb[:, t_idx], b_emb[:, t_idx]
            t_val = t_indices[t_idx].expand(batch_size)
            
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            x = h_real + h_imag
            cached_states.append(x)
            
            if len(cached_states) > 1:
                states = torch.stack(cached_states[:-1], dim=1)
                
                head_outputs = []
                step_weights = []
                for h_idx in range(self.n_heads):
                    out, weights = self.attention_head(x, states, t_val, h_idx)
                    head_outputs.append(out)
                    step_weights.append(weights)
                
                context = torch.stack(head_outputs, dim=0).sum(dim=0)
                x = x + self.context_proj(context)
                
                if return_attention:
                    all_attention.append(step_weights)
                if store_attention:
                    self.attention_weights_history.append([w.detach().cpu() for w in step_weights])
            
            t_phi = t_val * PHI
            x = x + self.resonant_layer(x, t_phi)
            
            all_logits.append(self.output_proj(x))
        
        output = torch.stack(all_logits, dim=1)
        
        if return_attention:
            return output, all_attention
        return output
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Stress Tests
# =============================================================================

def stress_test_long_range(distances: List[int] = [10, 20, 30, 50, 75, 100]):
    """Test recall at extreme distances."""
    print("\n" + "="*70)
    print("STRESS TEST: Long-Range Recall")
    print("="*70)
    
    vocab_size = 100
    num_signals = 20
    
    # Larger model for long range
    model = StressTestModel(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=8,
        num_neurons=256,
    ).to(device)
    
    print(f"Model params: {model.get_num_params():,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 150)
    
    def make_batch(batch_size, distance):
        seqs, targets = [], []
        for _ in range(batch_size):
            signal = random.randint(1, num_signals)
            between = [random.randint(num_signals + 1, vocab_size - 1) for _ in range(distance)]
            seq = [signal] + between + [0]  # 0 = trigger
            seqs.append(seq)
            targets.append(signal)
        return (torch.tensor(seqs, device=device),
                torch.tensor(targets, device=device))
    
    # Progressive training: start with short, gradually increase
    print("\nProgressive training...")
    current_max_dist = 10
    
    for epoch in range(150):
        model.train()
        total_loss = 0
        
        # Gradually increase max distance
        if epoch % 30 == 0 and epoch > 0:
            current_max_dist = min(current_max_dist + 20, max(distances))
            print(f"  Increasing max training distance to {current_max_dist}")
        
        for _ in range(20):
            dist = random.randint(3, current_max_dist)
            seqs, targets = make_batch(32, dist)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        
        if (epoch + 1) % 30 == 0:
            print(f"  Epoch {epoch+1}: Loss={total_loss/20:.4f}")
    
    # Evaluate at each distance
    print("\nEvaluation by distance:")
    results = {}
    model.eval()
    
    for dist in distances:
        correct, total = 0, 0
        with torch.no_grad():
            for _ in range(30):
                seqs, targets = make_batch(32, dist)
                logits = model(seqs)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += 32
        
        acc = correct / total
        results[dist] = acc
        random_baseline = 100 / num_signals
        print(f"  Distance {dist:3d}: {acc*100:5.1f}% (random: {random_baseline:.1f}%)")
    
    return results


def stress_test_gradient_pathology():
    """Detect gradient pathologies."""
    print("\n" + "="*70)
    print("STRESS TEST: Gradient Pathology Detection")
    print("="*70)
    
    model = StressTestModel(
        vocab_size=100,
        d_model=64,
        n_heads=4,
    ).to(device)
    
    # Test 1: Gradient magnitude over sequence length
    print("\nTest 1: Gradient magnitude vs sequence length")
    
    seq_lengths = [10, 25, 50, 75, 100]
    grad_magnitudes = {}
    
    for seq_len in seq_lengths:
        model.zero_grad()
        
        seq = torch.randint(0, 100, (4, seq_len), device=device)
        output = model(seq)
        loss = output[:, -1, :].sum()
        loss.backward()
        
        # Measure embedding gradients
        emb_grad = model.token_embedding.weight.grad
        grad_mag = emb_grad.abs().mean().item() if emb_grad is not None else 0
        grad_magnitudes[seq_len] = grad_mag
        
        print(f"  Seq len {seq_len:3d}: grad_magnitude = {grad_mag:.6f}")
    
    # Test 2: Gradient from different positions
    print("\nTest 2: Gradient source position analysis (seq_len=50)")
    
    positions_grads = {}
    for target_pos in [0, 10, 25, 40, 49]:
        model.zero_grad()
        
        seq = torch.randint(0, 100, (4, 50), device=device)
        output = model(seq)
        loss = output[:, target_pos, :].sum()
        loss.backward()
        
        emb_grad = model.token_embedding.weight.grad
        grad_mag = emb_grad.abs().mean().item() if emb_grad is not None else 0
        positions_grads[target_pos] = grad_mag
        
        print(f"  Grad from position {target_pos:2d}: {grad_mag:.6f}")
    
    # Test 3: Check for exploding/vanishing gradients in attention
    print("\nTest 3: Attention parameter gradients")
    
    model.zero_grad()
    seq = torch.randint(0, 100, (4, 50), device=device)
    output = model(seq)
    loss = output.sum()
    loss.backward()
    
    for h_idx in range(model.n_heads):
        w_q_grad = model.w_query[h_idx].grad
        b_q_grad = model.b_query[h_idx].grad
        
        if w_q_grad is not None:
            print(f"  Head {h_idx} w_query: mean={w_q_grad.abs().mean():.6f}, "
                  f"max={w_q_grad.abs().max():.6f}")
    
    return {
        'seq_length_grads': grad_magnitudes,
        'position_grads': positions_grads,
    }


def stress_test_memory_scaling():
    """Test memory usage with increasing cache size."""
    print("\n" + "="*70)
    print("STRESS TEST: Memory Scaling")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return {}
    
    cache_sizes = [32, 64, 128, 256, 512]
    results = {}
    
    for cache_size in cache_sizes:
        torch.cuda.empty_cache()
        gc.collect()
        
        torch.cuda.reset_peak_memory_stats()
        
        model = StressTestModel(
            vocab_size=100,
            d_model=64,
            n_heads=4,
        ).to(device)
        
        # Create sequence of cache_size length
        seq = torch.randint(0, 100, (8, cache_size), device=device)
        
        # Forward pass
        with torch.no_grad():
            _ = model(seq)
        
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        results[cache_size] = peak_memory
        
        print(f"  Cache size {cache_size:3d}: {peak_memory:.1f} MB peak memory")
        
        del model, seq
        torch.cuda.empty_cache()
    
    return results


def stress_test_interference_patterns():
    """Test with adversarial interference patterns."""
    print("\n" + "="*70)
    print("STRESS TEST: Adversarial Interference Patterns")
    print("="*70)
    
    vocab_size = 50
    model = StressTestModel(vocab_size=vocab_size, d_model=64, n_heads=4).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    # Train on simple pattern first
    print("\nPhase 1: Training on simple patterns...")
    for epoch in range(30):
        model.train()
        for _ in range(20):
            # Simple pattern: signal at start, trigger at end
            seqs = []
            targets = []
            for _ in range(32):
                signal = random.randint(1, 10)
                seq = [signal] + [random.randint(20, 49) for _ in range(8)] + [0]
                seqs.append(seq)
                targets.append(signal)
            
            seqs = torch.tensor(seqs, device=device)
            targets = torch.tensor(targets, device=device)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            optimizer.step()
    
    # Test on adversarial patterns
    print("\nPhase 2: Testing adversarial patterns...")
    model.eval()
    
    test_patterns = {
        'normal': lambda: ([random.randint(1, 10)] + 
                          [random.randint(20, 49) for _ in range(8)] + [0]),
        'repeated_signal': lambda: ([5] + [5]*8 + [0]),  # Signal repeated
        'all_signals': lambda: ([random.randint(1, 10) for _ in range(9)] + [0]),  # All signals
        'signal_in_middle': lambda: ([20]*4 + [5] + [20]*4 + [0]),  # Signal not at start
        'no_noise': lambda: ([5] + [0]*8 + [0]),  # No noise between
        'high_freq_noise': lambda: ([5] + [20, 21, 20, 21, 20, 21, 20, 21] + [0]),  # Alternating
    }
    
    results = {}
    
    for pattern_name, pattern_fn in test_patterns.items():
        correct, total = 0, 0
        
        with torch.no_grad():
            for _ in range(50):
                seqs = [pattern_fn() for _ in range(32)]
                targets = [s[0] for s in seqs]  # First token is always target for normal
                
                seqs = torch.tensor(seqs, device=device)
                targets = torch.tensor(targets, device=device)
                
                logits = model(seqs)
                preds = logits[:, -1, :].argmax(dim=-1)
                
                # For 'signal_in_middle', target is the middle signal
                if pattern_name == 'signal_in_middle':
                    targets = torch.tensor([5] * 32, device=device)
                
                correct += (preds == targets).sum().item()
                total += 32
        
        acc = correct / total
        results[pattern_name] = acc
        print(f"  {pattern_name:20s}: {acc*100:.1f}%")
    
    return results


def stress_test_attention_entropy():
    """Analyze attention entropy under various conditions."""
    print("\n" + "="*70)
    print("STRESS TEST: Attention Entropy Analysis")
    print("="*70)
    
    model = StressTestModel(vocab_size=100, d_model=64, n_heads=4).to(device)
    
    # Train briefly
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(30):
        seqs = torch.randint(1, 100, (32, 15), device=device)
        seqs[:, 0] = torch.randint(1, 11, (32,))  # Signal
        seqs[:, -1] = 0  # Trigger
        targets = seqs[:, 0].clone()
        
        optimizer.zero_grad()
        logits = model(seqs)
        loss = F.cross_entropy(logits[:, -1, :], targets)
        loss.backward()
        optimizer.step()
    
    # Analyze entropy
    print("\nEntropy analysis after training:")
    model.eval()
    
    conditions = {
        'random_uniform': lambda: torch.randint(1, 100, (4, 20), device=device),
        'single_token_repeated': lambda: torch.full((4, 20), 42, device=device),
        'sequential': lambda: torch.arange(20, device=device).unsqueeze(0).expand(4, -1),
        'signal_noise_trigger': lambda: torch.cat([
            torch.randint(1, 11, (4, 1), device=device),
            torch.randint(20, 100, (4, 18), device=device),
            torch.zeros(4, 1, dtype=torch.long, device=device)
        ], dim=1),
    }
    
    results = {}
    
    for cond_name, seq_fn in conditions.items():
        seq = seq_fn()
        
        with torch.no_grad():
            _, attention = model(seq, return_attention=True)
        
        # Analyze last position attention
        if attention:
            last_attn = attention[-1]
            entropies = []
            
            for head_weights in last_attn:
                # Calculate entropy
                entropy = -(head_weights * (head_weights + 1e-10).log()).sum(dim=-1).mean().item()
                entropies.append(entropy)
            
            avg_entropy = np.mean(entropies)
            max_entropy = math.log(seq.shape[1] - 1)  # Maximum possible entropy
            
            results[cond_name] = {
                'avg_entropy': avg_entropy,
                'normalized_entropy': avg_entropy / max_entropy,
                'head_entropies': entropies,
            }
            
            print(f"  {cond_name:25s}: entropy={avg_entropy:.3f} "
                  f"(normalized: {avg_entropy/max_entropy:.3f})")
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("ECHO ATTENTION STRESS TEST SUITE")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    all_results = {}
    
    # Run all stress tests
    print("\n" + "#"*70)
    print("# RUNNING STRESS TESTS")
    print("#"*70)
    
    try:
        all_results['long_range'] = stress_test_long_range()
    except Exception as e:
        print(f"Long range test failed: {e}")
    
    try:
        all_results['gradient_pathology'] = stress_test_gradient_pathology()
    except Exception as e:
        print(f"Gradient pathology test failed: {e}")
    
    try:
        all_results['memory_scaling'] = stress_test_memory_scaling()
    except Exception as e:
        print(f"Memory scaling test failed: {e}")
    
    try:
        all_results['interference'] = stress_test_interference_patterns()
    except Exception as e:
        print(f"Interference test failed: {e}")
    
    try:
        all_results['entropy'] = stress_test_attention_entropy()
    except Exception as e:
        print(f"Entropy test failed: {e}")
    
    # Summary
    print("\n" + "="*70)
    print("STRESS TEST SUMMARY")
    print("="*70)
    
    if 'long_range' in all_results:
        print("\nLong-range recall:")
        for dist, acc in all_results['long_range'].items():
            status = "✓" if acc > 0.15 else "✗"  # 15% > random (5%)
            print(f"  {status} Distance {dist}: {acc*100:.1f}%")
    
    if 'interference' in all_results:
        print("\nAdversarial patterns:")
        for pattern, acc in all_results['interference'].items():
            status = "✓" if acc > 0.5 else "✗"
            print(f"  {status} {pattern}: {acc*100:.1f}%")
    
    print("\n" + "="*70)
    print("STRESS TESTS COMPLETE")
    print("="*70)
