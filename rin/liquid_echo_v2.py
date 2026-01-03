"""
Liquid Echo V2 - Multiplicative Gating with Exponential Decay

Changes from V1:
1. Multiplicative gate instead of additive interference
2. Exponential decay for alpha: exp(-k * |x_inverted|)
   - x_inverted = 1 - (interference + 1) * 0.5
   - High interference (match) → x_inverted ≈ 0 → alpha ≈ 1
   - Low interference (no match) → x_inverted ≈ 1 → alpha ≈ exp(-k)
3. Detailed activation and gradient tracking

The multiplicative gate:
    output = resonant * echo_gate
    
Where echo_gate modulates the resonant stream based on memory state.
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


class LiquidEchoHeadV2(nn.Module):
    """
    Liquid Echo head with exponential decay gating.
    
    Alpha computation:
        1. Compute interference score (complex dot product)
        2. Normalize to [-1, 1] range
        3. Invert: x_inv = 1 - (score + 1) * 0.5 = (1 - score) / 2
        4. Apply exponential decay: alpha = exp(-k * |x_inv|)
        
    k (sensitivity) controls selectivity:
        - High k → sharp transitions, very selective
        - Low k → smooth transitions, more lenient
    """
    
    def __init__(self, d_model: int, head_idx: int = 0, init_k: float = 3.0):
        super().__init__()
        self.d_model = d_model
        self.head_idx = head_idx
        
        # Trigger projection (receives gradients through time)
        self.w_trigger = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_trigger = nn.Parameter(torch.zeros(d_model))
        
        # State evolution (clock ticking)
        self.w_state = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_state = nn.Parameter(torch.zeros(d_model))
        
        # Sensitivity factor k (learnable)
        # Higher k = more selective, lower k = more lenient
        self.k = nn.Parameter(torch.tensor(init_k))
        
        # Normalization scale
        self.scale = math.sqrt(d_model)
        
        self._lut = None
        self._memory_real = None
        self._memory_imag = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def reset_memory(self, batch_size: int, device: torch.device):
        self._memory_real = torch.zeros(batch_size, self.d_model, device=device)
        self._memory_imag = torch.zeros(batch_size, self.d_model, device=device)
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            echo_real, echo_imag: Echo output
            diagnostics: Dict with intermediate values for debugging
        """
        lut = self._get_lut(x_real.device)
        batch_size = x_real.shape[0]
        
        if self._memory_real is None or self._memory_real.shape[0] != batch_size:
            self.reset_memory(batch_size, x_real.device)
        
        t_val = t.view(-1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0)
        t_phi = t_val * PHI
        
        # === 1. TRIGGER RESPONSE ===
        wl_trigger = 1.0 + self.w_trigger.abs()
        
        theta_tr = x_real / wl_trigger + self.b_trigger + t_phi
        theta_ti = x_imag / wl_trigger + self.b_trigger + t_phi
        
        sin_tr, cos_tr = lut.lookup_sin_cos(theta_tr)
        sin_ti, cos_ti = lut.lookup_sin_cos(theta_ti)
        
        trigger_real = cos_tr * cos_ti - sin_tr * sin_ti
        trigger_imag = cos_tr * sin_ti + sin_tr * cos_ti
        
        # === 2. INTERFERENCE SCORE ===
        # Complex dot product: Re(trigger^* · input)
        interference_raw = (
            trigger_real * x_real + trigger_imag * x_imag
        ).sum(dim=-1)  # (batch,)
        
        # Normalize to roughly [-1, 1]
        interference_norm = interference_raw / self.scale
        # Clamp to ensure valid range
        interference_clamped = torch.clamp(interference_norm, -1.0, 1.0)
        
        # === 3. EXPONENTIAL DECAY ALPHA ===
        # x_inv: 1 when no match, 0 when perfect match
        x_inv = (1.0 - interference_clamped) / 2.0  # Maps [1, -1] → [0, 1]
        
        # Ensure k is positive
        k = self.k.abs() + 0.1  # Min k of 0.1 to avoid flat response
        
        # alpha: 1 when match (x_inv=0), exp(-k) when no match (x_inv=1)
        alpha = torch.exp(-k * x_inv)  # (batch,)
        
        # === 4. EMA UPDATE ===
        memory_real_detached = self._memory_real.detach()
        memory_imag_detached = self._memory_imag.detach()
        
        alpha_exp = alpha.unsqueeze(-1)
        
        blended_real = alpha_exp * x_real + (1 - alpha_exp) * memory_real_detached
        blended_imag = alpha_exp * x_imag + (1 - alpha_exp) * memory_imag_detached
        
        # === 5. EVOLVE STATE ===
        wl_state = 1.0 + self.w_state.abs()
        
        theta_sr = blended_real / wl_state + self.b_state + t_phi
        theta_si = blended_imag / wl_state + self.b_state + t_phi
        
        sin_sr, cos_sr = lut.lookup_sin_cos(theta_sr)
        sin_si, cos_si = lut.lookup_sin_cos(theta_si)
        
        evolved_real = cos_sr * cos_si - sin_sr * sin_si
        evolved_imag = cos_sr * sin_si + sin_sr * cos_si
        
        # Store for next step
        self._memory_real = evolved_real.detach()
        self._memory_imag = evolved_imag.detach()
        
        # Diagnostics
        diagnostics = {
            'interference_raw': interference_raw.detach(),
            'interference_norm': interference_norm.detach(),
            'x_inv': x_inv.detach(),
            'alpha': alpha.detach(),
            'k': k.detach(),
            'trigger_mag': (trigger_real**2 + trigger_imag**2).sum(-1).sqrt().detach(),
            'memory_mag': (memory_real_detached**2 + memory_imag_detached**2).sum(-1).sqrt().detach(),
            'evolved_mag': (evolved_real**2 + evolved_imag**2).sum(-1).sqrt().detach(),
        }
        
        return evolved_real, evolved_imag, diagnostics


class LiquidEchoModuleV2(nn.Module):
    """Multi-head liquid echo with multiplicative output."""
    
    def __init__(self, d_model: int, n_heads: int = 1, init_k: float = 3.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.heads = nn.ModuleList([
            LiquidEchoHeadV2(d_model, head_idx=i, init_k=init_k)
            for i in range(n_heads)
        ])
        
        # Gate projection: combines head outputs into a gate signal
        # Output in [0, 2] range centered at 1 for multiplicative identity
        self.gate_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)  # Start at identity
    
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
            gate_real, gate_imag: Multiplicative gate values
            all_diagnostics: List of diagnostics per head
        """
        echo_sum_real = torch.zeros_like(x_real)
        echo_sum_imag = torch.zeros_like(x_imag)
        all_diagnostics = []
        
        for head in self.heads:
            h_real, h_imag, diag = head(x_real, x_imag, t)
            echo_sum_real = echo_sum_real + h_real
            echo_sum_imag = echo_sum_imag + h_imag
            all_diagnostics.append(diag)
        
        # Average over heads
        echo_avg_real = echo_sum_real / self.n_heads
        echo_avg_imag = echo_sum_imag / self.n_heads
        
        # Project to gate - use sigmoid to bound in [0, 2] centered at 1
        gate_real = 1.0 + torch.tanh(self.gate_proj(echo_avg_real))  # [0, 2]
        gate_imag = 1.0 + torch.tanh(self.gate_proj(echo_avg_imag))  # [0, 2]
        
        return gate_real, gate_imag, all_diagnostics


class ResonantLayerV2(nn.Module):
    """Resonant layer with activation tracking."""
    
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
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        t: torch.Tensor,
        return_activations: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        out_real = F.silu(out_real)
        out_imag = F.silu(out_imag)
        
        if return_activations:
            activations = {
                'collapsed_mag': x_collapsed.norm(dim=-1).detach(),
                'cos_sum_mag': cos_sum.norm(dim=-1).detach(),
                'sin_sum_mag': sin_sum.norm(dim=-1).detach(),
                'out_mag': (out_real**2 + out_imag**2).sum(-1).sqrt().detach(),
            }
            return out_real, out_imag, activations
        
        return out_real, out_imag


class LiquidEchoModelV2(nn.Module):
    """
    Liquid Echo V2 with Multiplicative Gating.
    
    Architecture:
        output = resonant * echo_gate
        
    The echo_gate modulates (gates) the resonant stream, allowing
    the memory circuit to control information flow.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        n_echo_heads: int = 1,
        init_k: float = 3.0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_echo_heads = n_echo_heads
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        self.resonant_layers = nn.ModuleList([
            ResonantLayerV2(d_model, num_neurons)
            for _ in range(num_layers)
        ])
        
        self.echo_modules = nn.ModuleList([
            LiquidEchoModuleV2(d_model, n_echo_heads, init_k)
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
    
    def euler_transform(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t * PHI
        t_phi = t_phi.view(-1, 1) if t_phi.dim() >= 1 else t_phi.unsqueeze(0).unsqueeze(0)
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_r, cos_r = lut.lookup_sin_cos(theta_real)
        sin_i, cos_i = lut.lookup_sin_cos(theta_imag)
        
        h_real_new = cos_r * cos_i - sin_r * sin_i
        h_imag_new = cos_r * sin_i + sin_r * cos_i
        
        return h_real_new, h_imag_new
    
    def forward(
        self,
        input_ids: torch.Tensor,
        return_diagnostics: bool = False,
    ) -> torch.Tensor:
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
                # Resonant processing
                if return_diagnostics:
                    res_real, res_imag, res_act = self.resonant_layers[layer_idx](
                        x_real, x_imag, t_phi, return_activations=True
                    )
                    step_diag[f'layer{layer_idx}_res'] = res_act
                else:
                    res_real, res_imag = self.resonant_layers[layer_idx](
                        x_real, x_imag, t_phi
                    )
                
                # Echo gate (multiplicative)
                gate_real, gate_imag, echo_diag = self.echo_modules[layer_idx](
                    x_real, x_imag, t_val
                )
                
                if return_diagnostics:
                    step_diag[f'layer{layer_idx}_echo'] = echo_diag
                    step_diag[f'layer{layer_idx}_gate_real'] = gate_real.mean().item()
                    step_diag[f'layer{layer_idx}_gate_imag'] = gate_imag.mean().item()
                
                # MULTIPLICATIVE GATING: resonant * gate
                gated_real = res_real * gate_real
                gated_imag = res_imag * gate_imag
                
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
# TESTING WITH DETAILED DIAGNOSTICS
# ============================================================================

def test_activation_magnitudes():
    """Test activation magnitudes at each step."""
    print("="*70)
    print("ACTIVATION MAGNITUDE TEST")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LiquidEchoModelV2(
        vocab_size=64,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=2,
        init_k=3.0,
    ).to(device)
    
    x = torch.randint(0, 64, (4, 12), device=device)
    logits, diagnostics = model(x, return_diagnostics=True)
    
    print("\nActivation magnitudes over sequence:")
    print("-"*70)
    
    # Show key metrics for each timestep
    for diag in diagnostics:
        t = diag['t']
        
        # Echo diagnostics (first head of each layer)
        l0_alpha = diag['layer0_echo'][0]['alpha'].mean().item()
        l1_alpha = diag['layer1_echo'][0]['alpha'].mean().item()
        l0_k = diag['layer0_echo'][0]['k'].item()
        
        # Gate values
        g0_r = diag['layer0_gate_real']
        g1_r = diag['layer1_gate_real']
        
        # State magnitude
        x_mag = diag['layer1_x_mag']
        
        print(f"t={t:2d} | α0={l0_alpha:.3f} α1={l1_alpha:.3f} | "
              f"g0={g0_r:.3f} g1={g1_r:.3f} | x_mag={x_mag:.3f} | k={l0_k:.2f}")
    
    print(f"\nLogits range: [{logits.min().item():.2f}, {logits.max().item():.2f}]")


def test_gradient_magnitudes():
    """Test gradient magnitudes through the model."""
    print("\n" + "="*70)
    print("GRADIENT MAGNITUDE TEST")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LiquidEchoModelV2(
        vocab_size=64,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=2,
        init_k=3.0,
    ).to(device)
    
    # Hook to capture gradients
    grad_magnitudes = {}
    
    def hook_fn(name):
        def hook(grad):
            grad_magnitudes[name] = grad.norm().item()
        return hook
    
    x = torch.randint(0, 64, (4, 12), device=device)
    
    # Register hooks on key parameters
    hooks = []
    for name, param in model.named_parameters():
        if 'trigger' in name or 'state' in name or 'k' in name or 'gate_proj' in name:
            h = param.register_hook(hook_fn(name))
            hooks.append(h)
    
    logits = model(x)
    loss = F.cross_entropy(logits[:, -1, :], torch.zeros(4, dtype=torch.long, device=device))
    loss.backward()
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    print("\nGradient magnitudes for key parameters:")
    print("-"*70)
    
    categories = {'trigger': [], 'state': [], 'k': [], 'gate': []}
    
    for name, mag in sorted(grad_magnitudes.items()):
        if 'trigger' in name:
            categories['trigger'].append((name, mag))
        elif 'state' in name and 'k' not in name:
            categories['state'].append((name, mag))
        elif '.k' in name:
            categories['k'].append((name, mag))
        elif 'gate' in name:
            categories['gate'].append((name, mag))
    
    for cat_name, items in categories.items():
        if items:
            avg = sum(m for _, m in items) / len(items)
            print(f"\n{cat_name.upper()} params (avg={avg:.4f}):")
            for name, mag in items:
                short_name = '.'.join(name.split('.')[-3:])
                print(f"  {short_name}: {mag:.6f}")
    
    # Also check parameter gradient norms
    print("\n\nAll parameter gradient norms:")
    print("-"*70)
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm > 0.01:  # Only show non-trivial
                short_name = '.'.join(name.split('.')[-2:])
                print(f"  {short_name}: {grad_norm:.4f}")


def test_exponential_decay_behavior():
    """Visualize the exponential decay alpha function."""
    print("\n" + "="*70)
    print("EXPONENTIAL DECAY BEHAVIOR")
    print("="*70)
    
    # Test different k values
    interference = torch.linspace(-1, 1, 21)
    
    print("\nAlpha = exp(-k * |x_inv|) where x_inv = (1-interference)/2")
    print("-"*70)
    print(f"{'interf':<8}", end="")
    for k in [1.0, 2.0, 3.0, 5.0, 10.0]:
        print(f"k={k:<5}", end=" ")
    print()
    print("-"*70)
    
    for i_val in interference:
        x_inv = (1.0 - i_val) / 2.0
        print(f"{i_val.item():>6.2f}  ", end="")
        for k in [1.0, 2.0, 3.0, 5.0, 10.0]:
            alpha = torch.exp(-k * x_inv.abs())
            print(f"{alpha.item():.3f}  ", end="")
        print()
    
    print("\nInterpretation:")
    print("  interference=1.0 (perfect match)  → x_inv=0 → alpha≈1 (full update)")
    print("  interference=0.0 (neutral)        → x_inv=0.5 → alpha varies with k")
    print("  interference=-1.0 (anti-match)    → x_inv=1 → alpha=exp(-k) (minimal update)")


def test_retrieval_task():
    """Test on marker retrieval."""
    print("\n" + "="*70)
    print("RETRIEVAL TASK TEST - Liquid Echo V2 (Multiplicative)")
    print("="*70)
    
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 500
    
    model = LiquidEchoModelV2(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=2,
        num_neurons=128,
        n_echo_heads=2,
        init_k=3.0,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    start = time.time()
    best_acc = 0.0
    
    # Track k evolution
    k_history = []
    
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
        optimizer.step()
        
        # Track k values
        k_vals = [h.k.item() for m in model.echo_modules for h in m.heads]
        k_history.append(sum(k_vals) / len(k_vals))
        
        if epoch % 100 == 99:
            model.eval()
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == targets).float().mean().item()
                best_acc = max(best_acc, acc)
                k_avg = sum(k_vals) / len(k_vals)
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1%}, k_avg={k_avg:.2f}")
    
    elapsed = time.time() - start
    
    print(f"\nTime: {elapsed:.1f}s")
    print(f"Best accuracy: {best_acc:.1%}")
    print(f"Final k values: {[f'{h.k.item():.2f}' for m in model.echo_modules for h in m.heads]}")
    
    # Show alpha distribution at end
    model.eval()
    with torch.no_grad():
        _, diag = model(seq, return_diagnostics=True)
        
        print("\nFinal alpha distribution over sequence:")
        alphas_l0 = [d['layer0_echo'][0]['alpha'].mean().item() for d in diag]
        for t, a in enumerate(alphas_l0):
            bar = '█' * int(a * 40)
            print(f"  t={t:2d}: {a:.3f} {bar}")


def test_different_k_init():
    """Compare different initial k values."""
    print("\n" + "="*70)
    print("K SENSITIVITY COMPARISON")
    print("="*70)
    
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 300
    
    results = {}
    
    for init_k in [1.0, 3.0, 5.0, 10.0]:
        print(f"\n--- init_k = {init_k} ---")
        
        model = LiquidEchoModelV2(
            vocab_size=vocab_size,
            d_model=64,
            num_layers=2,
            num_neurons=128,
            n_echo_heads=2,
            init_k=init_k,
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
        
        best_acc = 0.0
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
            optimizer.step()
            
            if epoch % 100 == 99:
                model.eval()
                with torch.no_grad():
                    pred = logits[:, -1, :].argmax(dim=-1)
                    acc = (pred == targets).float().mean().item()
                    best_acc = max(best_acc, acc)
                
                k_avg = sum(h.k.item() for m in model.echo_modules for h in m.heads) / (
                    model.num_layers * model.n_echo_heads
                )
                print(f"  Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1%}, k_avg={k_avg:.2f}")
        
        elapsed = time.time() - start
        final_k = sum(h.k.item() for m in model.echo_modules for h in m.heads) / (
            model.num_layers * model.n_echo_heads
        )
        results[init_k] = {'acc': best_acc, 'time': elapsed, 'final_k': final_k}
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'init_k':<10} {'Best Acc':<12} {'Final k':<12} {'Time':<10}")
    print("-"*44)
    for k, r in results.items():
        print(f"{k:<10} {r['acc']:.1%}        {r['final_k']:.2f}         {r['time']:.1f}s")


if __name__ == "__main__":
    test_exponential_decay_behavior()
    test_activation_magnitudes()
    test_gradient_magnitudes()
    test_retrieval_task()
    test_different_k_init()
