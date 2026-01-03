"""
Test memory_match mode with different configurations.

Tests:
1. memory_match with k=10, 4 heads
2. memory_match with sigmoid (no exponential decay)

Both with detailed diagnostics: gradients, write sparsity, alpha stats per timestep.
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
    Liquid Echo head with memory_match interference.
    
    Supports two alpha calculation modes:
    - "exponential": alpha = exp(-k * x_inv)
    - "sigmoid": alpha = sigmoid(interference / sqrt(d_model))
    """
    
    def __init__(
        self, 
        d_model: int, 
        head_idx: int = 0,
        alpha_mode: str = "exponential",  # or "sigmoid"
        k: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.head_idx = head_idx
        self.alpha_mode = alpha_mode
        self.k = k
        
        # Query projection - detects what pattern to look for in memory
        self.w_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_query = nn.Parameter(torch.zeros(d_model))
        
        # Oscillation parameters
        self.w_osc = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_osc = nn.Parameter(torch.zeros(d_model))
        
        self.scale = math.sqrt(d_model)
        
        self._lut = None
        self._memory_real = None
        self._memory_imag = None
    
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
        Process one timestep with memory_match interference.
        """
        lut = self._get_lut(x_real.device)
        batch_size = x_real.shape[0]
        
        if self._memory_real is None or self._memory_real.shape[0] != batch_size:
            self.reset_memory(batch_size, x_real.device)
        
        t_val = t.view(-1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0)
        t_phi = t_val * PHI
        
        # === 1. QUERY: Learn what pattern to look for ===
        wl_query = 1.0 + self.w_query.abs()
        theta_query = x_real / wl_query + self.b_query
        
        sin_q, cos_q = lut.lookup_sin_cos(theta_query)
        query_real = cos_q
        query_imag = sin_q
        
        # === 2. MEMORY MATCH: Compare query with current memory ===
        memory_real_det = self._memory_real.detach()
        memory_imag_det = self._memory_imag.detach()
        
        # Complex conjugate: query* · memory
        # Re(z1* · z2) = Re(z1) * Re(z2) + Im(z1) * Im(z2)
        # Im(z1* · z2) = Re(z1) * Im(z2) - Im(z1) * Re(z2)
        interference_real = (query_real * memory_real_det + query_imag * memory_imag_det).sum(dim=-1)
        interference_imag = (query_real * memory_imag_det - query_imag * memory_real_det).sum(dim=-1)
        
        # Magnitude: |query* · memory|
        interference_mag = torch.sqrt(interference_real**2 + interference_imag**2)
        
        # === 3. ALPHA CALCULATION: Two modes ===
        if self.alpha_mode == "exponential":
            # Normalize magnitude to [0, 1] range
            interference_norm = torch.sigmoid(interference_mag / self.scale - 2.0)
            
            # x_inv: 0 when match, 1 when no match
            x_inv = 1.0 - interference_norm
            
            # Exponential decay
            alpha = torch.exp(-self.k * x_inv)
            
        else:  # sigmoid
            # Direct sigmoid on normalized interference
            # No exponential decay, just scaled sigmoid
            alpha = torch.sigmoid(interference_mag / self.scale - 2.0)
        
        # === 4. EMA UPDATE: Blend input when triggered ===
        alpha_exp = alpha.unsqueeze(-1)
        
        # High alpha (memory matches query) → overwrite with input
        # Low alpha (memory doesn't match) → preserve memory
        blended_real = alpha_exp * x_real + (1 - alpha_exp) * memory_real_det
        blended_imag = alpha_exp * x_imag + (1 - alpha_exp) * memory_imag_det
        
        # === 5. EVOLVE: Oscillate with learned frequency parameters ===
        wl_osc = 1.0 + self.w_osc.abs()
        
        theta_osc_real = blended_real / wl_osc + self.b_osc + t_phi
        theta_osc_imag = blended_imag / wl_osc + self.b_osc + t_phi
        
        sin_or, cos_or = lut.lookup_sin_cos(theta_osc_real)
        sin_oi, cos_oi = lut.lookup_sin_cos(theta_osc_imag)
        
        # Complex multiplication
        evolved_real = cos_or * cos_oi - sin_or * sin_oi
        evolved_imag = cos_or * sin_oi + sin_or * cos_oi
        
        # Store for next step (detached)
        self._memory_real = evolved_real.detach()
        self._memory_imag = evolved_imag.detach()
        
        diagnostics = {
            'interference_real': interference_real.detach(),
            'interference_imag': interference_imag.detach(),
            'interference_mag': interference_mag.detach(),
            'alpha': alpha.detach(),
            'query_mag': (query_real**2 + query_imag**2).sum(-1).sqrt().detach(),
            'memory_mag': (memory_real_det**2 + memory_imag_det**2).sum(-1).sqrt().detach(),
            'evolved_mag': (evolved_real**2 + evolved_imag**2).sum(-1).sqrt().detach(),
        }
        
        return evolved_real, evolved_imag, diagnostics


class LiquidEchoModule(nn.Module):
    """Multi-head liquid echo with memory_match."""
    
    def __init__(
        self, 
        d_model: int, 
        n_heads: int = 1,
        alpha_mode: str = "exponential",
        k: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.heads = nn.ModuleList([
            LiquidEchoHead(d_model, head_idx=i, alpha_mode=alpha_mode, k=k)
            for i in range(n_heads)
        ])
    
    def reset_memory(self, batch_size: int, device: torch.device):
        for head in self.heads:
            head.reset_memory(batch_size, device)
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        gate_real = torch.zeros_like(x_real)
        gate_imag = torch.zeros_like(x_imag)
        all_diag = []
        
        for head in self.heads:
            h_real, h_imag, diag = head(x_real, x_imag, t)
            gate_real = gate_real + h_real
            gate_imag = gate_imag + h_imag
            all_diag.append(diag)
        
        # Average over heads
        gate_real = gate_real / self.n_heads
        gate_imag = gate_imag / self.n_heads
        
        return gate_real, gate_imag, all_diag


class ResonantLayer(nn.Module):
    """Resonant layer with attenuation."""
    
    def __init__(self, d_model: int, num_neurons: int):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        self.input_collapse = nn.Linear(2 * d_model, d_model, bias=True)
        
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))
        
        self.attn_cos = nn.Parameter(torch.ones(num_neurons, d_model))
        self.attn_sin = nn.Parameter(torch.ones(num_neurons, d_model))
        
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
        lut = self._get_lut(x_real.device)
        
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        x_collapsed = self.input_collapse(x_combined)
        
        x_exp = x_collapsed.unsqueeze(1)
        wavelength = 1.0 + self.W.abs()
        t_val = t.view(-1, 1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        theta = x_exp / wavelength + self.B + t_val
        
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        cos_weighted = cos_theta * self.attn_cos
        sin_weighted = sin_theta * self.attn_sin
        
        cos_sum = cos_weighted.sum(dim=-1)
        sin_sum = sin_weighted.sum(dim=-1)
        
        out_real = self.out_proj_real(cos_sum)
        out_imag = self.out_proj_imag(sin_sum)
        
        return F.silu(out_real), F.silu(out_imag)


class LiquidEchoModel(nn.Module):
    """Liquid Echo with memory_match and configurable alpha mode."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        n_echo_heads: int = 1,
        alpha_mode: str = "exponential",  # or "sigmoid"
        k: float = 1.0,
        fusion_mode: str = "multiplicative",
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_echo_heads = n_echo_heads
        self.alpha_mode = alpha_mode
        self.k = k
        self.fusion_mode = fusion_mode
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        self.resonant_layers = nn.ModuleList([
            ResonantLayer(d_model, num_neurons)
            for _ in range(num_layers)
        ])
        
        self.echo_modules = nn.ModuleList([
            LiquidEchoModule(d_model, n_echo_heads, alpha_mode, k)
            for _ in range(num_layers)
        ])
        
        self.gate_scales = nn.ParameterList([
            nn.Parameter(torch.tensor(0.1))
            for _ in range(num_layers)
        ])
        
        self.output_proj_real = nn.Linear(d_model, vocab_size, bias=False)
        self.output_proj_imag = nn.Linear(d_model, vocab_size, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj_real.weight, std=0.02)
        nn.init.normal_(self.output_proj_imag.weight, std=0.02)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def reset_memory(self, batch_size: int, device: torch.device):
        for echo in self.echo_modules:
            echo.reset_memory(batch_size, device)
    
    def euler_transform(self, h_real, h_imag, w, b, t):
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t * PHI
        t_phi = t_phi.view(-1, 1) if t_phi.dim() >= 1 else t_phi.unsqueeze(0).unsqueeze(0)
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_r, cos_r = lut.lookup_sin_cos(theta_real)
        sin_i, cos_i = lut.lookup_sin_cos(theta_imag)
        
        return cos_r * cos_i - sin_r * sin_i, cos_r * sin_i + sin_r * cos_i
    
    def forward(self, input_ids: torch.Tensor, return_diagnostics: bool = False):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        self.reset_memory(batch_size, device)
        
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        all_logits = []
        all_diagnostics = [] if return_diagnostics else None
        
        for t in range(seq_len):
            w_t = w_emb[:, t, :]
            b_t = b_emb[:, t, :]
            t_val = torch.tensor(t, dtype=torch.float32, device=device)
            
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            x_real, x_imag = h_real, h_imag
            t_phi = t_val * PHI
            
            step_diag = {'t': t} if return_diagnostics else None
            
            for layer_idx in range(self.num_layers):
                res_real, res_imag = self.resonant_layers[layer_idx](x_real, x_imag, t_phi)
                
                gate_real, gate_imag, echo_diag = self.echo_modules[layer_idx](
                    x_real, x_imag, t_val
                )
                
                if return_diagnostics:
                    step_diag[f'layer{layer_idx}_echo'] = echo_diag
                
                scale = self.gate_scales[layer_idx]
                
                if self.fusion_mode == "additive":
                    gated_real = res_real + scale * gate_real
                    gated_imag = res_imag + scale * gate_imag
                else:
                    gated_real = res_real * (1.0 + scale * gate_real)
                    gated_imag = res_imag * (1.0 + scale * gate_imag)
                
                if return_diagnostics:
                    step_diag[f'layer{layer_idx}_res_mag'] = (res_real**2 + res_imag**2).sum(-1).sqrt().mean().item()
                    step_diag[f'layer{layer_idx}_gate_mag'] = (gate_real**2 + gate_imag**2).sum(-1).sqrt().mean().item()
                
                x_real = x_real + gated_real
                x_imag = x_imag + gated_imag
            
            if return_diagnostics:
                all_diagnostics.append(step_diag)
            
            logits = self.output_proj_real(x_real) + self.output_proj_imag(x_imag)
            all_logits.append(logits)
        
        result = torch.stack(all_logits, dim=1)
        
        if return_diagnostics:
            return result, all_diagnostics
        return result
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def analyze_alpha_dynamics(diagnostics, n_heads):
    """Extract detailed alpha statistics from diagnostics."""
    seq_len = len(diagnostics)
    
    # Per-head alpha over time
    head_alphas = [[] for _ in range(n_heads)]
    
    for step_diag in diagnostics:
        echo_diag = step_diag['layer0_echo']
        for head_idx in range(n_heads):
            alpha = echo_diag[head_idx]['alpha'].mean().item()
            head_alphas[head_idx].append(alpha)
    
    # Statistics
    stats = {}
    for head_idx in range(n_heads):
        alphas = head_alphas[head_idx]
        mean = sum(alphas) / len(alphas)
        std = (sum((a - mean)**2 for a in alphas) / len(alphas)) ** 0.5
        min_a = min(alphas)
        max_a = max(alphas)
        
        # Write sparsity
        write_50 = sum(1 for a in alphas if a > 0.5) / len(alphas)
        write_80 = sum(1 for a in alphas if a > 0.8) / len(alphas)
        write_95 = sum(1 for a in alphas if a > 0.95) / len(alphas)
        
        stats[head_idx] = {
            'mean': mean,
            'std': std,
            'min': min_a,
            'max': max_a,
            'write_50': write_50,
            'write_80': write_80,
            'write_95': write_95,
            'alphas': alphas,
        }
    
    return stats


def test_memory_match_k10_4heads():
    """Test memory_match with k=10, 4 heads."""
    print("="*80)
    print("MEMORY MATCH: k=10, 4 heads, exponential decay")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 400
    
    model = LiquidEchoModel(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_neurons=128,
        n_echo_heads=4,
        alpha_mode="exponential",
        k=10.0,
        fusion_mode="multiplicative",
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Config: k=10.0, 4 heads, exponential decay")
    
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
        
        if epoch % 50 == 49:
            model.eval()
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == targets).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1
            
            print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc:5.1%}, best={best_acc:5.1%} @{best_epoch}")
    
    elapsed = time.time() - start
    
    # Final diagnostics
    model.eval()
    with torch.no_grad():
        # Check gradients on one batch
        seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
        for i in range(batch_size):
            pos = torch.randint(2, seq_len//2, (1,)).item()
            seq[i, pos] = marker
            seq[i, pos+1] = targets[i]
        seq[:, -2] = marker
        
        logits, diag = model(seq, return_diagnostics=True)
    
    # Gradient check
    loss = F.cross_entropy(logits[:, -1, :], targets)
    optimizer.zero_grad()
    loss.backward()
    
    print("\n" + "="*80)
    print("GRADIENT ANALYSIS")
    print("="*80)
    
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.norm().item() > 0:
            grad_norm = param.grad.norm().item()
            if 'echo' in name:
                if 'query' in name:
                    grad_stats.setdefault('query', []).append(grad_norm)
                elif 'osc' in name:
                    grad_stats.setdefault('osc', []).append(grad_norm)
    
    for cat, norms in grad_stats.items():
        avg = sum(norms) / len(norms)
        print(f"{cat}: avg={avg:.6f}, count={len(norms)}")
    
    # Alpha dynamics
    print("\n" + "="*80)
    print("ALPHA DYNAMICS (per head)")
    print("="*80)
    
    stats = analyze_alpha_dynamics(diag, n_heads=4)
    
    for head_idx, head_stats in stats.items():
        print(f"\nHead {head_idx}:")
        print(f"  Mean:  {head_stats['mean']:.4f}")
        print(f"  Std:   {head_stats['std']:.4f}")
        print(f"  Range: [{head_stats['min']:.4f}, {head_stats['max']:.4f}]")
        print(f"  Write sparsity: >50%={head_stats['write_50']*100:.1f}%, "
              f">80%={head_stats['write_80']*100:.1f}%, >95%={head_stats['write_95']*100:.1f}%")
        
        print(f"  Alpha sequence:")
        for t in range(0, len(head_stats['alphas']), 2):
            a = head_stats['alphas'][t]
            bar = '█' * int(a * 30)
            print(f"    t={t:2d}: α={a:.3f} {bar}")
    
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Best accuracy: {best_acc:.1%} @epoch {best_epoch}")
    
    return best_acc


def test_memory_match_sigmoid():
    """Test memory_match with sigmoid (no exponential decay)."""
    print("\n" + "="*80)
    print("MEMORY MATCH: sigmoid alpha (no exponential decay)")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 400
    
    model = LiquidEchoModel(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_neurons=128,
        n_echo_heads=4,
        alpha_mode="sigmoid",
        k=1.0,  # Not used in sigmoid mode
        fusion_mode="multiplicative",
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Config: 4 heads, sigmoid(interference/√d - 2)")
    
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
        
        if epoch % 50 == 49:
            model.eval()
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == targets).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_epoch = epoch + 1
            
            print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc:5.1%}, best={best_acc:5.1%} @{best_epoch}")
    
    elapsed = time.time() - start
    
    # Final diagnostics
    model.eval()
    with torch.no_grad():
        seq = torch.randint(0, vocab_size-2, (batch_size, seq_len), device=device)
        targets = torch.randint(0, vocab_size-2, (batch_size,), device=device)
        for i in range(batch_size):
            pos = torch.randint(2, seq_len//2, (1,)).item()
            seq[i, pos] = marker
            seq[i, pos+1] = targets[i]
        seq[:, -2] = marker
        
        logits, diag = model(seq, return_diagnostics=True)
    
    # Gradient check
    loss = F.cross_entropy(logits[:, -1, :], targets)
    optimizer.zero_grad()
    loss.backward()
    
    print("\n" + "="*80)
    print("GRADIENT ANALYSIS")
    print("="*80)
    
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.norm().item() > 0:
            grad_norm = param.grad.norm().item()
            if 'echo' in name:
                if 'query' in name:
                    grad_stats.setdefault('query', []).append(grad_norm)
                elif 'osc' in name:
                    grad_stats.setdefault('osc', []).append(grad_norm)
    
    for cat, norms in grad_stats.items():
        avg = sum(norms) / len(norms)
        print(f"{cat}: avg={avg:.6f}, count={len(norms)}")
    
    # Alpha dynamics
    print("\n" + "="*80)
    print("ALPHA DYNAMICS (per head)")
    print("="*80)
    
    stats = analyze_alpha_dynamics(diag, n_heads=4)
    
    for head_idx, head_stats in stats.items():
        print(f"\nHead {head_idx}:")
        print(f"  Mean:  {head_stats['mean']:.4f}")
        print(f"  Std:   {head_stats['std']:.4f}")
        print(f"  Range: [{head_stats['min']:.4f}, {head_stats['max']:.4f}]")
        print(f"  Write sparsity: >50%={head_stats['write_50']*100:.1f}%, "
              f">80%={head_stats['write_80']*100:.1f}%, >95%={head_stats['write_95']*100:.1f}%")
        
        print(f"  Alpha sequence:")
        for t in range(0, len(head_stats['alphas']), 2):
            a = head_stats['alphas'][t]
            bar = '█' * int(a * 30)
            print(f"    t={t:2d}: α={a:.3f} {bar}")
    
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Best accuracy: {best_acc:.1%} @epoch {best_epoch}")
    
    return best_acc


if __name__ == "__main__":
    # Test 1: k=10, 4 heads, exponential
    acc1 = test_memory_match_k10_4heads()
    
    # Test 2: sigmoid, 4 heads
    acc2 = test_memory_match_sigmoid()
    
    print("\n" + "="*80)
    print("FINAL COMPARISON")
    print("="*80)
    print(f"Exponential (k=10, 4 heads): {acc1:.1%}")
    print(f"Sigmoid (4 heads):           {acc2:.1%}")
