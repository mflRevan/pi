"""
Test k sensitivity and interference calculation methods.

Tests:
1. Different k values (2, 5, 10) - verify exponential decay behavior
2. Complex interference calculation (conjugate rule)
3. Query-memory vs query-input interference
4. Alpha statistics and write sparsity per k
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
import time

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut

PHI = (1 + math.sqrt(5)) / 2


class LiquidEchoHead(nn.Module):
    """
    Liquid Echo head with configurable interference calculation.
    """
    
    def __init__(
        self, 
        d_model: int, 
        head_idx: int = 0,
        interference_mode: str = "complex_full"  # "real_only", "complex_real", "complex_full", "memory_match"
    ):
        super().__init__()
        self.d_model = d_model
        self.head_idx = head_idx
        self.interference_mode = interference_mode
        
        # Query projection - detects what should trigger update
        self.w_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_query = nn.Parameter(torch.zeros(d_model))
        
        # Oscillation parameters
        self.w_osc = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_osc = nn.Parameter(torch.zeros(d_model))
        
        # k is FIXED hyperparameter
        self.k = 1.0
        
        self.scale = math.sqrt(d_model)
        
        self._lut = None
        self._memory_real = None
        self._memory_imag = None
    
    def set_k(self, k: float):
        """Set the k (sensitivity) parameter."""
        self.k = k
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def reset_memory(self, batch_size: int, device: torch.device):
        """Initialize memory to zeros."""
        self._memory_real = torch.zeros(batch_size, self.d_model, device=device)
        self._memory_imag = torch.zeros(batch_size, self.d_model, device=device)
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Process one timestep.
        """
        lut = self._get_lut(x_real.device)
        batch_size = x_real.shape[0]
        
        if self._memory_real is None or self._memory_real.shape[0] != batch_size:
            self.reset_memory(batch_size, x_real.device)
        
        t_val = t.view(-1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0)
        t_phi = t_val * PHI
        
        # === 1. QUERY: Detect what should trigger update ===
        wl_query = 1.0 + self.w_query.abs()
        theta_query = x_real / wl_query + self.b_query
        
        sin_q, cos_q = lut.lookup_sin_cos(theta_query)
        query_real = cos_q
        query_imag = sin_q
        
        # === 2. INTERFERENCE: Multiple calculation modes ===
        memory_real_det = self._memory_real.detach()
        memory_imag_det = self._memory_imag.detach()
        
        if self.interference_mode == "real_only":
            # Original: Real part only of query · input
            interference = (query_real * x_real + query_imag * x_imag).sum(dim=-1)
            
        elif self.interference_mode == "complex_real":
            # Real part of complex conjugate: query* · input
            # For z1 = a + bi, z2 = c + di: Re(z1* · z2) = ac + bd
            interference_real = (query_real * x_real + query_imag * x_imag).sum(dim=-1)
            interference = interference_real
            
        elif self.interference_mode == "complex_full":
            # Full complex magnitude: |query* · input|
            # Re(z1* · z2) = ac + bd
            # Im(z1* · z2) = ad - bc
            interference_real = (query_real * x_real + query_imag * x_imag).sum(dim=-1)
            interference_imag = (query_real * x_imag - query_imag * x_real).sum(dim=-1)
            interference = torch.sqrt(interference_real**2 + interference_imag**2)
            
        elif self.interference_mode == "memory_match":
            # Compare query with MEMORY instead of input
            # This checks: "does current memory match what we're looking for?"
            interference_real = (query_real * memory_real_det + query_imag * memory_imag_det).sum(dim=-1)
            interference_imag = (query_real * memory_imag_det - query_imag * memory_real_det).sum(dim=-1)
            interference = torch.sqrt(interference_real**2 + interference_imag**2)
        
        # Normalize to roughly [-1, 1] or [0, 1] depending on mode
        if self.interference_mode == "complex_full" or self.interference_mode == "memory_match":
            # Magnitude is always positive, normalize to [0, 1]
            interference_norm = torch.sigmoid(interference / self.scale - 2.0)  # Center around 0
        else:
            # Can be negative, normalize to [-1, 1]
            interference_norm = torch.tanh(interference / self.scale)
        
        # === 3. EXPONENTIAL DECAY: Convert interference to update weight ===
        # x_inv: 0 when match (interference=1), 1 when no match (interference=-1 or 0)
        if self.interference_mode == "complex_full" or self.interference_mode == "memory_match":
            # For magnitude-based: 1 when match, 0 when no match
            x_inv = 1.0 - interference_norm
        else:
            # For signed: map [-1, 1] to [1, 0]
            x_inv = (1.0 - interference_norm) / 2.0
        
        # alpha: 1 when match, exp(-k) when no match
        alpha = torch.exp(-self.k * x_inv)
        
        # === 4. EMA UPDATE: Blend input when triggered ===
        alpha_exp = alpha.unsqueeze(-1)
        
        # Gradient flows through input blend ratio
        blended_real = alpha_exp * x_real + (1 - alpha_exp) * memory_real_det
        blended_imag = alpha_exp * x_imag + (1 - alpha_exp) * memory_imag_det
        
        # === 5. EVOLVE: Oscillate with learned frequency parameters ===
        wl_osc = 1.0 + self.w_osc.abs()
        
        theta_osc_real = blended_real / wl_osc + self.b_osc + t_phi
        theta_osc_imag = blended_imag / wl_osc + self.b_osc + t_phi
        
        sin_or, cos_or = lut.lookup_sin_cos(theta_osc_real)
        sin_oi, cos_oi = lut.lookup_sin_cos(theta_osc_imag)
        
        # Complex multiplication for oscillation
        evolved_real = cos_or * cos_oi - sin_or * sin_oi
        evolved_imag = cos_or * sin_oi + sin_or * cos_oi
        
        # Store for next step (detached)
        self._memory_real = evolved_real.detach()
        self._memory_imag = evolved_imag.detach()
        
        diagnostics = {
            'interference_raw': interference.detach(),
            'interference_norm': interference_norm.detach(),
            'x_inv': x_inv.detach(),
            'alpha': alpha.detach(),
            'query_mag': (query_real**2 + query_imag**2).sum(-1).sqrt().detach(),
            'memory_mag': (memory_real_det**2 + memory_imag_det**2).sum(-1).sqrt().detach(),
            'evolved_mag': (evolved_real**2 + evolved_imag**2).sum(-1).sqrt().detach(),
        }
        
        return evolved_real, evolved_imag, diagnostics


def test_k_values_detailed():
    """Test different k values and show alpha/sparsity behavior."""
    print("="*80)
    print("K SENSITIVITY AND WRITE SPARSITY TEST")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Test with simple sequence
    batch_size = 4
    seq_len = 20
    d_model = 32
    
    results = {}
    
    for k_val in [1.0, 2.0, 5.0, 10.0]:
        print(f"\n{'='*80}")
        print(f"K = {k_val}")
        print('='*80)
        
        head = LiquidEchoHead(
            d_model=d_model, 
            head_idx=0,
            interference_mode="complex_full"
        ).to(device)
        head.set_k(k_val)
        head.reset_memory(batch_size, device)
        
        # Random input sequence
        x_real = torch.randn(batch_size, d_model, device=device) * 0.1
        x_imag = torch.randn(batch_size, d_model, device=device) * 0.1
        
        alphas = []
        x_invs = []
        interference_norms = []
        
        for t in range(seq_len):
            t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
            _, _, diag = head(x_real, x_imag, t_tensor)
            
            alphas.append(diag['alpha'].mean().item())
            x_invs.append(diag['x_inv'].mean().item())
            interference_norms.append(diag['interference_norm'].mean().item())
        
        # Calculate statistics
        alpha_mean = sum(alphas) / len(alphas)
        alpha_min = min(alphas)
        alpha_max = max(alphas)
        alpha_std = (sum((a - alpha_mean)**2 for a in alphas) / len(alphas)) ** 0.5
        
        # Write sparsity: how often does alpha exceed threshold?
        threshold_50 = sum(1 for a in alphas if a > 0.5) / len(alphas)
        threshold_80 = sum(1 for a in alphas if a > 0.8) / len(alphas)
        threshold_95 = sum(1 for a in alphas if a > 0.95) / len(alphas)
        
        print(f"\nAlpha statistics:")
        print(f"  Mean:  {alpha_mean:.4f}")
        print(f"  Std:   {alpha_std:.4f}")
        print(f"  Min:   {alpha_min:.4f}")
        print(f"  Max:   {alpha_max:.4f}")
        
        print(f"\nWrite sparsity (% of steps with alpha > threshold):")
        print(f"  α > 0.50: {threshold_50*100:5.1f}%")
        print(f"  α > 0.80: {threshold_80*100:5.1f}%")
        print(f"  α > 0.95: {threshold_95*100:5.1f}%")
        
        print(f"\nSequence alpha pattern:")
        for t in range(0, seq_len, 2):  # Show every 2nd step
            bar = '█' * int(alphas[t] * 40)
            print(f"  t={t:2d}: α={alphas[t]:.3f} {bar}")
        
        # Verify exponential decay formula
        print(f"\nExponential decay verification:")
        print(f"  x_inv range: [{min(x_invs):.3f}, {max(x_invs):.3f}]")
        print(f"  interference_norm range: [{min(interference_norms):.3f}, {max(interference_norms):.3f}]")
        
        # Show theoretical alpha values for k
        print(f"\nTheoretical alpha values for k={k_val}:")
        print(f"  Perfect match (x_inv=0.0):  α = exp(-{k_val}*0.0) = {math.exp(-k_val*0.0):.4f}")
        print(f"  Half match (x_inv=0.5):     α = exp(-{k_val}*0.5) = {math.exp(-k_val*0.5):.4f}")
        print(f"  No match (x_inv=1.0):       α = exp(-{k_val}*1.0) = {math.exp(-k_val*1.0):.4f}")
        
        results[k_val] = {
            'alpha_mean': alpha_mean,
            'alpha_std': alpha_std,
            'threshold_50': threshold_50,
            'threshold_80': threshold_80,
            'threshold_95': threshold_95,
            'alphas': alphas,
        }
    
    # Comparison table
    print("\n" + "="*80)
    print("K COMPARISON SUMMARY")
    print("="*80)
    print(f"{'k':<8} {'α mean':<10} {'α std':<10} {'>50%':<8} {'>80%':<8} {'>95%':<8}")
    print("-"*80)
    for k, r in results.items():
        print(f"{k:<8} {r['alpha_mean']:<10.4f} {r['alpha_std']:<10.4f} "
              f"{r['threshold_50']*100:<8.1f} {r['threshold_80']*100:<8.1f} {r['threshold_95']*100:<8.1f}")


def test_interference_modes():
    """Compare different interference calculation methods."""
    print("\n" + "="*80)
    print("INTERFERENCE CALCULATION MODES COMPARISON")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size = 1
    seq_len = 15
    d_model = 32
    k = 2.0
    
    modes = ["real_only", "complex_real", "complex_full", "memory_match"]
    
    mode_results = {}
    
    for mode in modes:
        print(f"\n{'-'*80}")
        print(f"Mode: {mode}")
        print('-'*80)
        
        head = LiquidEchoHead(
            d_model=d_model,
            head_idx=0,
            interference_mode=mode
        ).to(device)
        head.set_k(k)
        head.reset_memory(batch_size, device)
        
        # Create predictable input pattern
        x_real = torch.randn(batch_size, d_model, device=device) * 0.1
        x_imag = torch.randn(batch_size, d_model, device=device) * 0.1
        
        alphas = []
        interference_vals = []
        
        for t in range(seq_len):
            t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
            _, _, diag = head(x_real, x_imag, t_tensor)
            
            alphas.append(diag['alpha'].mean().item())
            interference_vals.append(diag['interference_raw'].mean().item())
        
        alpha_mean = sum(alphas) / len(alphas)
        
        print(f"Alpha mean: {alpha_mean:.4f}")
        print(f"Interference range: [{min(interference_vals):.3f}, {max(interference_vals):.3f}]")
        
        print(f"\nAlpha sequence:")
        for t in range(0, seq_len, 3):
            bar = '█' * int(alphas[t] * 30)
            print(f"  t={t:2d}: α={alphas[t]:.3f} int={interference_vals[t]:6.2f} {bar}")
        
        mode_results[mode] = {
            'alpha_mean': alpha_mean,
            'alphas': alphas,
            'interference': interference_vals,
        }
    
    # Comparison
    print("\n" + "="*80)
    print("MODE COMPARISON")
    print("="*80)
    print(f"{'Mode':<20} {'α mean':<12} {'α range':<20}")
    print("-"*80)
    for mode, r in mode_results.items():
        alpha_min = min(r['alphas'])
        alpha_max = max(r['alphas'])
        print(f"{mode:<20} {r['alpha_mean']:<12.4f} [{alpha_min:.3f}, {alpha_max:.3f}]")


def test_retrieval_with_k():
    """Test retrieval task with different k values."""
    print("\n" + "="*80)
    print("RETRIEVAL TASK WITH DIFFERENT K VALUES")
    print("="*80)
    
    # We need to import the full model
    from rin.liquid_echo_correct import LiquidEchoModel
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 300
    
    results = {}
    
    for k_val in [1.0, 2.0, 5.0, 10.0]:
        print(f"\n{'='*80}")
        print(f"K = {k_val}")
        print('='*80)
        
        model = LiquidEchoModel(
            vocab_size=vocab_size,
            d_model=64,
            num_layers=2,
            num_neurons=128,
            n_echo_heads=2,
            k=k_val,
            fusion_mode="multiplicative",
        ).to(device)
        
        print(f"Parameters: {model.get_num_params():,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        
        best_acc = 0.0
        best_epoch = 0
        start = time.time()
        
        for epoch in range(num_epochs):
            model.train()
            
            seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
            targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
            
            for i in range(batch_size):
                pos = torch.randint(2, seq_len//2, (1,)).item()
                seq[i, pos] = marker
                seq[i, pos+1] = targets[i]
            
            seq[:, -2] = marker
            
            logits = model(seq)
            loss = F.cross_entropy(logits[:, -1, :], targets)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if epoch % 100 == 99:
                model.eval()
                with torch.no_grad():
                    pred = logits[:, -1, :].argmax(dim=-1)
                    acc = (pred == targets).float().mean().item()
                    if acc > best_acc:
                        best_acc = acc
                        best_epoch = epoch + 1
                
                print(f"  Epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc:5.1%}, best={best_acc:5.1%} @{best_epoch}")
        
        elapsed = time.time() - start
        
        # Get alpha statistics from final batch
        model.eval()
        with torch.no_grad():
            _, diag = model(seq, return_diagnostics=True)
        
        alphas = [d['layer0_echo'][0]['alpha'].mean().item() for d in diag]
        alpha_mean = sum(alphas) / len(alphas)
        alpha_std = (sum((a - alpha_mean)**2 for a in alphas) / len(alphas)) ** 0.5
        
        threshold_80 = sum(1 for a in alphas if a > 0.8) / len(alphas)
        
        print(f"\nFinal alpha stats: mean={alpha_mean:.3f}, std={alpha_std:.3f}, >80%={threshold_80*100:.1f}%")
        print(f"Time: {elapsed:.1f}s")
        
        results[k_val] = {
            'acc': best_acc,
            'best_epoch': best_epoch,
            'time': elapsed,
            'alpha_mean': alpha_mean,
            'alpha_std': alpha_std,
            'write_sparsity': threshold_80,
        }
    
    # Final comparison
    print("\n" + "="*80)
    print("K VALUE PERFORMANCE COMPARISON")
    print("="*80)
    print(f"{'k':<8} {'Best Acc':<12} {'@Epoch':<10} {'Time':<10} {'α mean':<10} {'Write%':<10}")
    print("-"*80)
    for k, r in results.items():
        print(f"{k:<8} {r['acc']:5.1%}        {r['best_epoch']:<10} {r['time']:<10.1f} "
              f"{r['alpha_mean']:<10.3f} {r['write_sparsity']*100:<10.1f}")


if __name__ == "__main__":
    # Test 1: K sensitivity and write sparsity
    test_k_values_detailed()
    
    # Test 2: Different interference calculation methods
    test_interference_modes()
    
    # Test 3: Retrieval task with different k
    test_retrieval_with_k()
