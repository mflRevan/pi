"""
Liquid Echo - Corrected Implementation

The echo state is a FREQUENCY PATTERN that:
1. Oscillates with learned (w, b) parameters
2. Only updates (via EMA) when input interferes with learned query
3. Is directly used as multiplicative gate (no projection)

Key fixes:
- k is FIXED hyperparameter (not learnable), default 1
- NO output projection - evolved state IS the contribution
- NO tanh gating - direct exponential decay for alpha
- Memory oscillates with same w/b parameters continuously
- Supports both additive and multiplicative fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut

PHI = (1 + math.sqrt(5)) / 2


class LiquidEchoHead(nn.Module):
    """
    Liquid Echo head - frequency latch.
    
    The head maintains a frequency pattern (complex oscillation) that:
    - Evolves with learned w/b over time
    - Updates (blends with input) only when triggered by query match
    - Is modulated by exponential decay of interference signal
    
    No projection, no attention - just a resonant oscillator with
    selective update gating.
    """
    
    def __init__(self, d_model: int, head_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.head_idx = head_idx
        
        # Query projection - detects what should trigger update
        self.w_query = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_query = nn.Parameter(torch.zeros(d_model))
        
        # Oscillation parameters - the learned frequency pattern
        # Same for both trigger detection and state evolution
        self.w_osc = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_osc = nn.Parameter(torch.zeros(d_model))
        
        # k is FIXED hyperparameter for exponential decay
        self.k = 1.0  # Can be set via config, default 1
        
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
        
        Returns:
            echo_real, echo_imag: The frequency pattern output
            diagnostics: Debug info
        """
        lut = self._get_lut(x_real.device)
        batch_size = x_real.shape[0]
        
        if self._memory_real is None or self._memory_real.shape[0] != batch_size:
            self.reset_memory(batch_size, x_real.device)
        
        t_val = t.view(-1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0)
        t_phi = t_val * PHI
        
        # === 1. QUERY: Detect what should trigger update ===
        # Euler transform of input through query lens
        wl_query = 1.0 + self.w_query.abs()
        theta_query = x_real / wl_query + self.b_query
        
        sin_q, cos_q = lut.lookup_sin_cos(theta_query)
        query_real = cos_q
        query_imag = sin_q
        
        # === 2. INTERFERENCE: How well does input match query pattern? ===
        # Dot product in complex space
        interference = (query_real * x_real + query_imag * x_imag).sum(dim=-1)
        
        # Normalize to roughly [-1, 1]
        interference_norm = torch.tanh(interference / self.scale)
        
        # === 3. EXPONENTIAL DECAY: Convert interference to update weight ===
        # x_inv: 0 when match (interference=1), 1 when no match (interference=-1)
        x_inv = (1.0 - interference_norm) / 2.0
        
        # alpha: 1 when match, exp(-k) when no match
        # k is FIXED, not learnable
        alpha = torch.exp(-self.k * x_inv)
        
        # === 4. EMA UPDATE: Blend input when triggered ===
        # Memory is detached to prevent BPTT
        memory_real_det = self._memory_real.detach()
        memory_imag_det = self._memory_imag.detach()
        
        alpha_exp = alpha.unsqueeze(-1)
        
        # Gradient flows through input blend ratio
        blended_real = alpha_exp * x_real + (1 - alpha_exp) * memory_real_det
        blended_imag = alpha_exp * x_imag + (1 - alpha_exp) * memory_imag_det
        
        # === 5. EVOLVE: Oscillate with learned frequency parameters ===
        # The "frequency" of oscillation is controlled by w_osc, b_osc
        # These are the ONLY parameters determining the pattern
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
        
        # === OUTPUT: The evolved frequency pattern IS the echo ===
        # No projection - direct use as gate modulation
        
        diagnostics = {
            'interference_raw': interference.detach(),
            'interference_norm': interference_norm.detach(),
            'alpha': alpha.detach(),
            'query_mag': (query_real**2 + query_imag**2).sum(-1).sqrt().detach(),
            'evolved_mag': (evolved_real**2 + evolved_imag**2).sum(-1).sqrt().detach(),
        }
        
        return evolved_real, evolved_imag, diagnostics


class LiquidEchoModule(nn.Module):
    """
    Multi-head liquid echo.
    
    Each head maintains independent frequency patterns.
    Outputs are summed (superposition) to create composite gate.
    """
    
    def __init__(self, d_model: int, n_heads: int = 1, k: float = 1.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.k = k
        
        self.heads = nn.ModuleList([
            LiquidEchoHead(d_model, head_idx=i)
            for i in range(n_heads)
        ])
        
        # Set k on all heads
        for head in self.heads:
            head.set_k(k)
    
    def reset_memory(self, batch_size: int, device: torch.device):
        for head in self.heads:
            head.reset_memory(batch_size, device)
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        """
        Returns:
            gate_real, gate_imag: Summed echo patterns (direct gate, no projection)
            diagnostics: Per-head debug info
        """
        gate_real = torch.zeros_like(x_real)
        gate_imag = torch.zeros_like(x_imag)
        all_diag = []
        
        for head in self.heads:
            h_real, h_imag, diag = head(x_real, x_imag, t)
            gate_real = gate_real + h_real
            gate_imag = gate_imag + h_imag
            all_diag.append(diag)
        
        # Average over heads (to keep magnitude reasonable)
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
        
        # Attenuation weights
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
    """
    Liquid Echo Model - Frequency Latch with Configurable Fusion.
    
    Architecture:
        Additive: output = resonant + scale * echo_gate + residual
        Multiplicative: output = resonant ⊙ (1 + scale * echo_gate) + residual
        
    Where echo_gate is the evolved frequency pattern (no projection).
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        n_echo_heads: int = 1,
        k: float = 1.0,
        fusion_mode: str = "multiplicative",  # or "additive"
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_echo_heads = n_echo_heads
        self.k = k
        self.fusion_mode = fusion_mode
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        self.resonant_layers = nn.ModuleList([
            ResonantLayer(d_model, num_neurons)
            for _ in range(num_layers)
        ])
        
        self.echo_modules = nn.ModuleList([
            LiquidEchoModule(d_model, n_echo_heads, k)
            for _ in range(num_layers)
        ])
        
        # Scaling factor for gate strength
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
            
            # Euler state evolution
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            x_real, x_imag = h_real, h_imag
            t_phi = t_val * PHI
            
            step_diag = {'t': t} if return_diagnostics else None
            
            for layer_idx in range(self.num_layers):
                # Resonant processing (parallel)
                res_real, res_imag = self.resonant_layers[layer_idx](x_real, x_imag, t_phi)
                
                # Echo gate (frequency patterns, no projection)
                gate_real, gate_imag, echo_diag = self.echo_modules[layer_idx](
                    x_real, x_imag, t_val
                )
                
                if return_diagnostics:
                    step_diag[f'layer{layer_idx}_echo'] = echo_diag
                
                # Fusion: Additive or Multiplicative
                scale = self.gate_scales[layer_idx]
                
                if self.fusion_mode == "additive":
                    # Additive: echo pattern summed with resonant
                    gated_real = res_real + scale * gate_real
                    gated_imag = res_imag + scale * gate_imag
                else:  # multiplicative
                    # Multiplicative: echo pattern modulates resonant (GLU-like)
                    gated_real = res_real * (1.0 + scale * gate_real)
                    gated_imag = res_imag * (1.0 + scale * gate_imag)
                
                if return_diagnostics:
                    step_diag[f'layer{layer_idx}_res_mag'] = (res_real**2 + res_imag**2).sum(-1).sqrt().mean().item()
                    step_diag[f'layer{layer_idx}_gate_mag'] = (gate_real**2 + gate_imag**2).sum(-1).sqrt().mean().item()
                
                # Residual connection
                x_real = x_real + gated_real
                x_imag = x_imag + gated_imag
                
                if return_diagnostics:
                    step_diag[f'layer{layer_idx}_x_mag'] = (x_real**2 + x_imag**2).sum(-1).sqrt().mean().item()
            
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


# ============================================================================
# TESTING
# ============================================================================

def test_gradient_flow():
    """Verify gradients flow properly."""
    print("="*70)
    print("GRADIENT FLOW TEST")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LiquidEchoModel(
        vocab_size=64,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=2,
        k=1.0,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    
    x = torch.randint(0, 64, (4, 12), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits[:, -1, :], torch.zeros(4, dtype=torch.long, device=device))
    loss.backward()
    
    print("\nGradient norms:")
    print("-"*70)
    
    cats = {
        'query': [], 'osc': [], 'resonant': [], 'scale': [], 'output': []
    }
    
    for name, param in model.named_parameters():
        if param.grad is None or param.grad.norm().item() == 0:
            continue
        
        grad = param.grad.norm().item()
        
        if 'query' in name:
            cats['query'].append((name, grad))
        elif 'osc' in name:
            cats['osc'].append((name, grad))
        elif 'resonant' in name or 'W' in name or 'B' in name or 'attn' in name:
            cats['resonant'].append((name, grad))
        elif 'gate_scale' in name:
            cats['scale'].append((name, grad))
        elif 'output' in name:
            cats['output'].append((name, grad))
    
    for cat, items in cats.items():
        if items:
            avg = sum(g for _, g in items) / len(items)
            print(f"\n{cat.upper()} (avg={avg:.6f}):")
            for name, g in items[:3]:
                short = '.'.join(name.split('.')[-2:])
                print(f"  {short}: {g:.6f}")
    
    print("\n✓ Gradient flow test complete")


def test_alpha_pattern():
    """Visualize alpha (trigger) pattern."""
    print("\n" + "="*70)
    print("ALPHA TRIGGER PATTERN")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LiquidEchoModel(
        vocab_size=64,
        d_model=32,
        num_layers=1,
        num_neurons=32,
        n_echo_heads=1,
        k=1.0,
    ).to(device)
    
    x = torch.randint(0, 64, (1, 20), device=device)
    _, diag = model(x, return_diagnostics=True)
    
    print("\nAlpha values (k=1.0) over sequence:")
    print("-"*70)
    
    # Check structure of diagnostics first
    if diag and 'layer0_echo' in diag[0]:
        alphas = [d['layer0_echo'][0]['alpha'].mean().item() for d in diag]
        
        for t, a in enumerate(alphas):
            bar = '█' * int(a * 50)
            print(f"t={t:2d}: {a:.3f} {bar}")
        
        print(f"\nMean alpha: {sum(alphas)/len(alphas):.3f}")
    else:
        print("No echo diagnostics found - model may not be recording echo data")


def test_fusion_comparison():
    """Compare additive vs multiplicative fusion."""
    print("\n" + "="*70)
    print("FUSION MODE COMPARISON (Additive vs Multiplicative)")
    print("="*70)
    
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 500
    
    results = {}
    
    for fusion_mode in ["additive", "multiplicative"]:
        print(f"\n{'='*70}")
        print(f"MODE: {fusion_mode.upper()}")
        print('='*70)
        
        model = LiquidEchoModel(
            vocab_size=vocab_size,
            d_model=64,
            num_layers=2,
            num_neurons=128,
            n_echo_heads=2,
            k=1.0,
            fusion_mode=fusion_mode,
        ).to(device)
        
        print(f"Parameters: {model.get_num_params():,}")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        
        best_acc = 0.0
        best_epoch = 0
        start = time.time()
        
        epoch_accs = []
        
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
                    epoch_accs.append(acc)
                    if acc > best_acc:
                        best_acc = acc
                        best_epoch = epoch + 1
                
                print(f"Epoch {epoch+1:3d}: loss={loss.item():.4f}, acc={acc:5.1%}, best={best_acc:5.1%} @{best_epoch}")
        
        elapsed = time.time() - start
        
        # Get final alpha distribution
        model.eval()
        with torch.no_grad():
            _, diag = model(seq, return_diagnostics=True)
        
        alphas = [d['layer0_echo'][0]['alpha'].mean().item() for d in diag]
        alpha_mean = sum(alphas) / len(alphas)
        alpha_min = min(alphas)
        alpha_max = max(alphas)
        
        print(f"\nAlpha stats: mean={alpha_mean:.3f}, min={alpha_min:.3f}, max={alpha_max:.3f}")
        print(f"Time: {elapsed:.1f}s")
        
        results[fusion_mode] = {
            'acc': best_acc,
            'best_epoch': best_epoch,
            'time': elapsed,
            'alpha_mean': alpha_mean,
            'epoch_accs': epoch_accs,
        }
    
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    print(f"{'Mode':<18} {'Best Acc':<12} {'Best @':<10} {'Time':<10} {'Alpha Mean':<12}")
    print("-"*70)
    for mode, r in results.items():
        print(f"{mode:<18} {r['acc']:5.1%}        {r['best_epoch']:<10} {r['time']:<10.1f} {r['alpha_mean']:.3f}")
    
    # Winner analysis
    print("\n" + "="*70)
    additive_acc = results['additive']['acc']
    mult_acc = results['multiplicative']['acc']
    
    if additive_acc > mult_acc:
        diff = (additive_acc - mult_acc) / mult_acc * 100
        print(f"✓ ADDITIVE wins by {diff:.1f}% relative improvement")
    elif mult_acc > additive_acc:
        diff = (mult_acc - additive_acc) / additive_acc * 100
        print(f"✓ MULTIPLICATIVE wins by {diff:.1f}% relative improvement")
    else:
        print("TIE")
    
    # Learning curves
    print("\n" + "="*70)
    print("LEARNING CURVES (every 50 epochs)")
    print("="*70)
    print(f"{'Epoch':<10} {'Additive':<12} {'Multiplicative':<15}")
    print("-"*40)
    for i, epoch in enumerate(range(50, num_epochs + 1, 50)):
        add_acc = results['additive']['epoch_accs'][i] if i < len(results['additive']['epoch_accs']) else 0
        mul_acc = results['multiplicative']['epoch_accs'][i] if i < len(results['multiplicative']['epoch_accs']) else 0
        print(f"{epoch:<10} {add_acc:5.1%}        {mul_acc:5.1%}")


if __name__ == "__main__":
    test_gradient_flow()
    test_alpha_pattern()
    test_fusion_comparison()
