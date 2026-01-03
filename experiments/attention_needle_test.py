#!/usr/bin/env python3
"""
Needle-in-Haystack Test for Echo Attention

Tests the attention mechanism's ability to find specific "needle" tokens
within varying amounts of noise ("haystack").

Tests include:
1. Single needle retrieval at varying distances
2. Multiple needle retrieval
3. Needle retrieval with distractors (similar tokens)
4. Time-varying needle position
5. Needle pattern matching

Run with: python experiments/attention_needle_test.py
"""

import torch
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
from rin.model import ComplexLinear, PHI

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# Needle Model
# =============================================================================

class NeedleModel(nn.Module):
    """Model for needle-in-haystack retrieval."""
    
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
        
        # Multi-head attention
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
        
        # Resonant processing
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        self.proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        self._lut = None
    
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
        
        # Attention
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
    
    def forward(self, input_ids, return_attention=False):
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
            
            t_phi = t_val * PHI
            x = x + self.resonant_layer(x, t_phi)
            
            all_logits.append(self.output_proj(x))
        
        output = torch.stack(all_logits, dim=1)
        
        if return_attention:
            return output, all_attention
        return output


# =============================================================================
# Needle Tests
# =============================================================================

def test_single_needle(distances: List[int] = [3, 5, 10, 20, 30, 50]):
    """Test single needle retrieval at varying distances."""
    print("\n" + "="*70)
    print("NEEDLE TEST: Single Needle at Varying Distances")
    print("="*70)
    
    vocab_size = 100
    needle_range = (1, 10)  # Tokens 1-10 are needles
    haystack_range = (50, 99)  # Tokens 50-99 are haystack
    trigger = 0  # Token 0 triggers retrieval
    
    model = NeedleModel(
        vocab_size=vocab_size,
        d_model=128,
        n_heads=8,
        num_neurons=256,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=30, epochs=100
    )
    
    def make_batch(batch_size, distance):
        seqs, targets = [], []
        for _ in range(batch_size):
            needle = random.randint(*needle_range)
            haystack = [random.randint(*haystack_range) for _ in range(distance)]
            seq = [needle] + haystack + [trigger]
            seqs.append(seq)
            targets.append(needle)
        return torch.tensor(seqs, device=device), torch.tensor(targets, device=device)
    
    # Curriculum: start short, gradually increase
    print("\nCurriculum training...")
    for epoch in range(100):
        model.train()
        
        max_dist = min(3 + epoch // 10, max(distances))
        
        for _ in range(30):
            dist = random.randint(3, max_dist)
            seqs, targets = make_batch(32, dist)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_seqs, test_targets = make_batch(100, max_dist)
                test_logits = model(test_seqs)
                test_acc = (test_logits[:, -1, :].argmax(-1) == test_targets).float().mean()
            print(f"  Epoch {epoch+1}: max_dist={max_dist}, acc={test_acc*100:.1f}%")
    
    # Evaluate
    print("\nFinal evaluation by distance:")
    results = {}
    model.eval()
    
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


def test_multiple_needles():
    """Test retrieval of multiple needles (retrieve the nth needle)."""
    print("\n" + "="*70)
    print("NEEDLE TEST: Multiple Needles (Indexed Retrieval)")
    print("="*70)
    
    vocab_size = 100
    needle_range = (1, 10)
    haystack_range = (50, 99)
    query_tokens = {1: 11, 2: 12, 3: 13}  # Query token for nth needle
    
    model = NeedleModel(
        vocab_size=vocab_size,
        d_model=96,
        n_heads=6,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    def make_batch(batch_size, num_needles):
        seqs, targets = [], []
        for _ in range(batch_size):
            needles = [random.randint(*needle_range) for _ in range(num_needles)]
            segments = []
            for n in needles:
                segments.append(n)
                segments.extend([random.randint(*haystack_range) for _ in range(3)])
            
            query_idx = random.randint(1, num_needles)
            segments.append(query_tokens[query_idx])
            
            seqs.append(segments)
            targets.append(needles[query_idx - 1])  # Target is the queried needle
        
        return torch.tensor(seqs, device=device), torch.tensor(targets, device=device)
    
    print("\nTraining on 2-3 needle sequences...")
    for epoch in range(80):
        model.train()
        
        for _ in range(30):
            num_needles = random.choice([2, 3])
            seqs, targets = make_batch(32, num_needles)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_seqs, test_targets = make_batch(100, 3)
                test_logits = model(test_seqs)
                test_acc = (test_logits[:, -1, :].argmax(-1) == test_targets).float().mean()
            print(f"  Epoch {epoch+1}: acc={test_acc*100:.1f}%")
    
    # Evaluate by query position
    print("\nEvaluation by query position:")
    results = {}
    model.eval()
    
    for query_idx in [1, 2, 3]:
        correct, total = 0, 0
        with torch.no_grad():
            for _ in range(30):
                seqs, targets = [], []
                for _ in range(32):
                    needles = [random.randint(*needle_range) for _ in range(3)]
                    segments = []
                    for n in needles:
                        segments.append(n)
                        segments.extend([random.randint(*haystack_range) for _ in range(3)])
                    segments.append(query_tokens[query_idx])
                    seqs.append(segments)
                    targets.append(needles[query_idx - 1])
                
                seqs = torch.tensor(seqs, device=device)
                targets = torch.tensor(targets, device=device)
                
                logits = model(seqs)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += 32
        
        acc = correct / total
        results[query_idx] = acc
        print(f"  Query 'retrieve needle {query_idx}': {acc*100:.1f}%")
    
    return results


def test_needle_with_distractors():
    """Test needle retrieval with similar distractor tokens."""
    print("\n" + "="*70)
    print("NEEDLE TEST: Needle with Distractors")
    print("="*70)
    
    # Setup: Needle is always from group A (1-5)
    # Distractors are from group B (6-10, similar to A)
    # Haystack is group C (50-99)
    vocab_size = 100
    
    model = NeedleModel(
        vocab_size=vocab_size,
        d_model=96,
        n_heads=6,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    def make_batch(batch_size, num_distractors):
        seqs, targets = [], []
        for _ in range(batch_size):
            needle = random.randint(1, 5)  # True needle
            distractors = [random.randint(6, 10) for _ in range(num_distractors)]
            haystack = [random.randint(50, 99) for _ in range(5)]
            
            # Interleave
            elements = [needle] + distractors + haystack
            random.shuffle(elements)
            elements.append(0)  # Trigger
            
            seqs.append(elements)
            targets.append(needle)
        
        return torch.tensor(seqs, device=device), torch.tensor(targets, device=device)
    
    print("\nTraining with increasing distractors...")
    for epoch in range(80):
        model.train()
        
        max_distractors = min(1 + epoch // 20, 4)
        
        for _ in range(30):
            num_dist = random.randint(0, max_distractors)
            seqs, targets = make_batch(32, num_dist)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_seqs, test_targets = make_batch(100, max_distractors)
                test_logits = model(test_seqs)
                test_acc = (test_logits[:, -1, :].argmax(-1) == test_targets).float().mean()
            print(f"  Epoch {epoch+1}: max_distractors={max_distractors}, acc={test_acc*100:.1f}%")
    
    # Evaluate
    print("\nEvaluation by number of distractors:")
    results = {}
    model.eval()
    
    for num_dist in [0, 1, 2, 3, 4]:
        correct, total = 0, 0
        with torch.no_grad():
            for _ in range(30):
                seqs, targets = make_batch(50, num_dist)
                logits = model(seqs)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += 50
        
        acc = correct / total
        results[num_dist] = acc
        status = "✓" if acc > 0.5 else "○" if acc > 0.25 else "✗"
        print(f"  {status} {num_dist} distractors: {acc*100:.1f}%")
    
    return results


def test_time_varying_needle():
    """Test retrieval where needle position varies systematically."""
    print("\n" + "="*70)
    print("NEEDLE TEST: Time-Varying Needle Position")
    print("="*70)
    
    vocab_size = 100
    seq_len = 15
    
    model = NeedleModel(vocab_size=vocab_size, d_model=64, n_heads=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    def make_batch(batch_size, needle_positions):
        """Needle at specified position, rest is haystack."""
        seqs, targets, positions = [], [], []
        for _ in range(batch_size):
            pos = random.choice(needle_positions)
            needle = random.randint(1, 10)
            seq = [random.randint(50, 99) for _ in range(seq_len)]
            seq[pos] = needle
            seq[-1] = 0  # Trigger
            
            seqs.append(seq)
            targets.append(needle)
            positions.append(pos)
        
        return (torch.tensor(seqs, device=device),
                torch.tensor(targets, device=device),
                positions)
    
    print("\nTraining with needles at various positions...")
    all_positions = list(range(seq_len - 1))  # Anywhere except trigger position
    
    for epoch in range(60):
        model.train()
        
        for _ in range(30):
            seqs, targets, _ = make_batch(32, all_positions)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_seqs, test_targets, _ = make_batch(100, all_positions)
                test_logits = model(test_seqs)
                test_acc = (test_logits[:, -1, :].argmax(-1) == test_targets).float().mean()
            print(f"  Epoch {epoch+1}: overall_acc={test_acc*100:.1f}%")
    
    # Evaluate by position
    print("\nEvaluation by needle position:")
    results = {}
    model.eval()
    
    for pos in [0, 3, 6, 9, 12]:
        correct, total = 0, 0
        with torch.no_grad():
            for _ in range(20):
                seqs, targets, _ = make_batch(50, [pos])
                logits = model(seqs)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += 50
        
        acc = correct / total
        results[pos] = acc
        distance = (seq_len - 1) - pos
        print(f"  Position {pos:2d} (distance {distance:2d}): {acc*100:.1f}%")
    
    return results


def test_pattern_needle():
    """Test needle that is a pattern, not single token."""
    print("\n" + "="*70)
    print("NEEDLE TEST: Pattern Needle (2-token patterns)")
    print("="*70)
    
    vocab_size = 100
    # Patterns are pairs: (1,2), (3,4), (5,6), (7,8), (9,10)
    patterns = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
    pattern_to_answer = {p: i for i, p in enumerate(patterns)}
    
    model = NeedleModel(
        vocab_size=vocab_size,
        d_model=96,
        n_heads=6,
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    
    def make_batch(batch_size, haystack_len):
        seqs, targets = [], []
        for _ in range(batch_size):
            pattern = random.choice(patterns)
            haystack = [random.randint(50, 99) for _ in range(haystack_len)]
            
            # Insert pattern consecutively at start
            seq = list(pattern) + haystack + [0]
            
            seqs.append(seq)
            targets.append(pattern_to_answer[pattern])
        
        return torch.tensor(seqs, device=device), torch.tensor(targets, device=device)
    
    print("\nTraining on 2-token patterns...")
    for epoch in range(80):
        model.train()
        
        for _ in range(30):
            haystack_len = random.randint(3, 10)
            seqs, targets = make_batch(32, haystack_len)
            
            optimizer.zero_grad()
            logits = model(seqs)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_seqs, test_targets = make_batch(100, 8)
                test_logits = model(test_seqs)
                test_acc = (test_logits[:, -1, :].argmax(-1) == test_targets).float().mean()
            print(f"  Epoch {epoch+1}: acc={test_acc*100:.1f}%")
    
    # Evaluate by haystack length
    print("\nEvaluation by haystack length:")
    results = {}
    model.eval()
    
    for hay_len in [3, 5, 8, 12, 15]:
        correct, total = 0, 0
        with torch.no_grad():
            for _ in range(20):
                seqs, targets = make_batch(50, hay_len)
                logits = model(seqs)
                preds = logits[:, -1, :].argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += 50
        
        acc = correct / total
        results[hay_len] = acc
        print(f"  Haystack length {hay_len:2d}: {acc*100:.1f}%")
    
    return results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("NEEDLE-IN-HAYSTACK TEST SUITE")
    print(f"Device: {device}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("="*70)
    
    all_results = {}
    
    print("\n" + "#"*70)
    print("# RUNNING NEEDLE TESTS")
    print("#"*70)
    
    try:
        all_results['single_needle'] = test_single_needle()
    except Exception as e:
        print(f"Single needle test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        all_results['multiple_needles'] = test_multiple_needles()
    except Exception as e:
        print(f"Multiple needles test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        all_results['distractors'] = test_needle_with_distractors()
    except Exception as e:
        print(f"Distractors test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        all_results['time_varying'] = test_time_varying_needle()
    except Exception as e:
        print(f"Time-varying test failed: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        all_results['pattern'] = test_pattern_needle()
    except Exception as e:
        print(f"Pattern test failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("NEEDLE TEST SUMMARY")
    print("="*70)
    
    if 'single_needle' in all_results:
        print("\nSingle needle by distance:")
        for dist, acc in all_results['single_needle'].items():
            status = "✓" if acc > 0.5 else "○" if acc > 0.15 else "✗"
            print(f"  {status} d={dist}: {acc*100:.1f}%")
    
    if 'multiple_needles' in all_results:
        print("\nMultiple needles by query:")
        for q, acc in all_results['multiple_needles'].items():
            status = "✓" if acc > 0.5 else "✗"
            print(f"  {status} query={q}: {acc*100:.1f}%")
    
    if 'distractors' in all_results:
        print("\nDistractors:")
        for n, acc in all_results['distractors'].items():
            status = "✓" if acc > 0.5 else "✗"
            print(f"  {status} n={n}: {acc*100:.1f}%")
    
    print("\n" + "="*70)
    print("NEEDLE TESTS COMPLETE")
    print("="*70)
