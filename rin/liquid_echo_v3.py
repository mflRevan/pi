"""
Liquid Echo V3 - Fixed Gradient Flow

KEY FIX: The trigger/state params MUST receive gradients through the CURRENT step.

The issue in V2: Even though we detach memory history, the current step's
computation using trigger/state params should still get gradients. But the
gate projection was taking over all gradient flow.

Solution: Remove gate projection, use the evolved state directly as a
multiplicative modulation with proper gradient paths.

New architecture:
    1. Trigger computes interference with input (gets gradients)
    2. EMA blends input and memory based on alpha (input path has gradients)
    3. State evolution transforms blended state (gets gradients via input blend)
    4. Evolved state DIRECTLY modulates resonant output (no extra projection)
    
The key insight: gradients flow through the INPUT portion of the EMA blend:
    blended = α * input + (1-α) * memory.detach()
    
    ∂loss/∂trigger flows through:
    - ∂blended/∂α (how trigger affects blend ratio)
    - ∂α/∂trigger (how trigger params affect α)
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


class LiquidEchoHeadV3(nn.Module):
    """
    Liquid Echo head with proper gradient flow.
    
    Gradient paths:
        loss → evolved_state → state_transform(blended) → blended
        blended = α * input + (1-α) * memory.detach()
        
        Path 1: ∂loss/∂blended → α * ∂loss/∂input (backprop to earlier layers)
        Path 2: ∂loss/∂blended → (input - memory) * ∂loss/∂α → ∂α/∂trigger
        
    This ensures trigger params get gradients proportional to how much
    the blend ratio MATTERS for the loss (input vs memory difference).
    """
    
    def __init__(self, d_model: int, head_idx: int = 0, init_k: float = 3.0):
        super().__init__()
        self.d_model = d_model
        self.head_idx = head_idx
        
        # Trigger projection
        self.w_trigger = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_trigger = nn.Parameter(torch.zeros(d_model))
        
        # State evolution
        self.w_state = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_state = nn.Parameter(torch.zeros(d_model))
        
        # Sensitivity (k) - controls sharpness of gating
        self.k = nn.Parameter(torch.tensor(init_k))
        
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
        lut = self._get_lut(x_real.device)
        batch_size = x_real.shape[0]
        
        if self._memory_real is None or self._memory_real.shape[0] != batch_size:
            self.reset_memory(batch_size, x_real.device)
        
        t_val = t.view(-1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0)
        t_phi = t_val * PHI
        
        # === 1. TRIGGER: Euler transform of input ===
        wl_trigger = 1.0 + self.w_trigger.abs()
        
        theta_tr = x_real / wl_trigger + self.b_trigger + t_phi
        theta_ti = x_imag / wl_trigger + self.b_trigger + t_phi
        
        sin_tr, cos_tr = lut.lookup_sin_cos(theta_tr)
        sin_ti, cos_ti = lut.lookup_sin_cos(theta_ti)
        
        trigger_real = cos_tr * cos_ti - sin_tr * sin_ti
        trigger_imag = cos_tr * sin_ti + sin_tr * cos_ti
        
        # === 2. INTERFERENCE SCORE ===
        # Compute match between trigger pattern and input
        interference_raw = (
            trigger_real * x_real + trigger_imag * x_imag
        ).sum(dim=-1)
        
        # Normalize to [-1, 1]
        interference_norm = torch.tanh(interference_raw / self.scale)
        
        # === 3. EXPONENTIAL DECAY ALPHA ===
        x_inv = (1.0 - interference_norm) / 2.0  # [1,-1] → [0,1]
        k = self.k.abs() + 0.1
        alpha = torch.exp(-k * x_inv)
        
        # === 4. EMA BLEND ===
        # CRITICAL: memory is detached, but input path retains gradients
        memory_real_det = self._memory_real.detach()
        memory_imag_det = self._memory_imag.detach()
        
        alpha_exp = alpha.unsqueeze(-1)
        
        # This is where gradients flow through:
        # - To trigger via ∂α/∂trigger
        # - To earlier layers via α * ∂loss/∂x
        blended_real = alpha_exp * x_real + (1 - alpha_exp) * memory_real_det
        blended_imag = alpha_exp * x_imag + (1 - alpha_exp) * memory_imag_det
        
        # === 5. STATE EVOLUTION ===
        wl_state = 1.0 + self.w_state.abs()
        
        theta_sr = blended_real / wl_state + self.b_state + t_phi
        theta_si = blended_imag / wl_state + self.b_state + t_phi
        
        sin_sr, cos_sr = lut.lookup_sin_cos(theta_sr)
        sin_si, cos_si = lut.lookup_sin_cos(theta_si)
        
        evolved_real = cos_sr * cos_si - sin_sr * sin_si
        evolved_imag = cos_sr * sin_si + sin_sr * cos_si
        
        # Store for next step (detached)
        self._memory_real = evolved_real.detach()
        self._memory_imag = evolved_imag.detach()
        
        diagnostics = {
            'interference_raw': interference_raw.detach(),
            'interference_norm': interference_norm.detach(),
            'alpha': alpha.detach(),
            'k': k.detach(),
            'trigger_mag': (trigger_real**2 + trigger_imag**2).sum(-1).sqrt().detach(),
            'blended_mag': (blended_real**2 + blended_imag**2).sum(-1).sqrt().detach(),
            'evolved_mag': (evolved_real**2 + evolved_imag**2).sum(-1).sqrt().detach(),
        }
        
        return evolved_real, evolved_imag, diagnostics


class LiquidEchoModuleV3(nn.Module):
    """Multi-head module with direct multiplicative gating."""
    
    def __init__(self, d_model: int, n_heads: int = 1, init_k: float = 3.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        self.heads = nn.ModuleList([
            LiquidEchoHeadV3(d_model, head_idx=i, init_k=init_k)
            for i in range(n_heads)
        ])
        
        # Scale for the gate magnitude
        self.gate_scale = nn.Parameter(torch.tensor(0.1))
    
    def reset_memory(self, batch_size: int, device: torch.device):
        for head in self.heads:
            head.reset_memory(batch_size, device)
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
        echo_sum_real = torch.zeros_like(x_real)
        echo_sum_imag = torch.zeros_like(x_imag)
        all_diagnostics = []
        
        for head in self.heads:
            h_real, h_imag, diag = head(x_real, x_imag, t)
            echo_sum_real = echo_sum_real + h_real
            echo_sum_imag = echo_sum_imag + h_imag
            all_diagnostics.append(diag)
        
        # Scale the gate contribution
        # Gate is centered at 1 (identity) with echo modulating around it
        gate_real = 1.0 + self.gate_scale * echo_sum_real
        gate_imag = 1.0 + self.gate_scale * echo_sum_imag
        
        return gate_real, gate_imag, all_diagnostics


class ResonantLayerV3(nn.Module):
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


class LiquidEchoModelV3(nn.Module):
    """
    Liquid Echo V3 with proper gradient flow.
    
    Architecture: output = resonant * echo_gate + residual
    
    The echo_gate is (1 + scale * echo_output), centered at 1.
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
            ResonantLayerV3(d_model, num_neurons)
            for _ in range(num_layers)
        ])
        
        self.echo_modules = nn.ModuleList([
            LiquidEchoModuleV3(d_model, n_echo_heads, init_k)
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
                # Resonant
                res_real, res_imag = self.resonant_layers[layer_idx](x_real, x_imag, t_phi)
                
                # Echo gate
                gate_real, gate_imag, echo_diag = self.echo_modules[layer_idx](
                    x_real, x_imag, t_val
                )
                
                if return_diagnostics:
                    step_diag[f'layer{layer_idx}_echo'] = echo_diag
                    step_diag[f'layer{layer_idx}_gate_real_mean'] = gate_real.mean().item()
                    step_diag[f'layer{layer_idx}_gate_imag_mean'] = gate_imag.mean().item()
                
                # Multiplicative gating
                gated_real = res_real * gate_real
                gated_imag = res_imag * gate_imag
                
                # Residual
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
    """Verify gradients flow to trigger/state params."""
    print("="*70)
    print("GRADIENT FLOW TEST - V3")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LiquidEchoModelV3(
        vocab_size=64,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=2,
        init_k=3.0,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    
    x = torch.randint(0, 64, (4, 12), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits[:, -1, :], torch.zeros(4, dtype=torch.long, device=device))
    loss.backward()
    
    print("\nGradient norms by parameter type:")
    print("-"*70)
    
    cats = {'trigger': [], 'state': [], 'k': [], 'gate_scale': [], 'resonant': [], 'other': []}
    
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f"  ⚠️  NO GRAD: {name}")
            continue
        
        grad = param.grad.norm().item()
        
        if 'trigger' in name:
            cats['trigger'].append((name, grad))
        elif 'state' in name and '.k' not in name:
            cats['state'].append((name, grad))
        elif '.k' in name:
            cats['k'].append((name, grad))
        elif 'gate_scale' in name:
            cats['gate_scale'].append((name, grad))
        elif 'resonant' in name or 'W' in name or 'B' in name or 'attn' in name or 'proj' in name:
            cats['resonant'].append((name, grad))
        else:
            cats['other'].append((name, grad))
    
    for cat, items in cats.items():
        if items:
            avg = sum(g for _, g in items) / len(items)
            print(f"\n{cat.upper()} (avg={avg:.6f}):")
            for name, g in items[:4]:
                short = '.'.join(name.split('.')[-2:])
                status = "✓" if g > 1e-6 else "⚠️"
                print(f"  {status} {short}: {g:.6f}")
    
    # Check specifically the echo params
    echo_has_grad = all(
        p.grad is not None and p.grad.norm().item() > 1e-6
        for n, p in model.named_parameters()
        if 'trigger' in n or ('state' in n and '.k' not in n) or '.k' in n
    )
    
    if echo_has_grad:
        print("\n✓ Echo parameters receiving gradients!")
    else:
        print("\n⚠️  Some echo parameters have zero/missing gradients!")


def test_alpha_dynamics():
    """Test how alpha changes with training."""
    print("\n" + "="*70)
    print("ALPHA DYNAMICS TEST")
    print("="*70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = LiquidEchoModelV3(
        vocab_size=64,
        d_model=32,
        num_layers=1,
        num_neurons=32,
        n_echo_heads=1,
        init_k=3.0,
    ).to(device)
    
    # Initial alpha distribution
    x = torch.randint(0, 64, (1, 16), device=device)
    _, diag = model(x, return_diagnostics=True)
    
    print("\nInitial alpha distribution:")
    alphas = [d['layer0_echo'][0]['alpha'].mean().item() for d in diag]
    for t, a in enumerate(alphas):
        bar = '█' * int(a * 40)
        print(f"  t={t:2d}: {a:.3f} {bar}")
    
    print(f"\nInitial k: {model.echo_modules[0].heads[0].k.item():.3f}")


def test_retrieval():
    """Test retrieval task."""
    print("\n" + "="*70)
    print("RETRIEVAL TASK - V3")
    print("="*70)
    
    import time
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    num_epochs = 500
    
    model = LiquidEchoModelV3(
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
            
            k_vals = [h.k.item() for m in model.echo_modules for h in m.heads]
            k_avg = sum(k_vals) / len(k_vals)
            scale_vals = [m.gate_scale.item() for m in model.echo_modules]
            
            print(f"Epoch {epoch+1}: loss={loss.item():.4f}, acc={acc:.1%}, k={k_avg:.2f}, scales={scale_vals}")
    
    print(f"\nTime: {time.time() - start:.1f}s")
    print(f"Best: {best_acc:.1%}")
    
    # Final alpha distribution
    model.eval()
    with torch.no_grad():
        _, diag = model(seq, return_diagnostics=True)
    
    print("\nFinal alpha (head 0, layer 0):")
    alphas = [d['layer0_echo'][0]['alpha'].mean().item() for d in diag]
    for t, a in enumerate(alphas):
        bar = '█' * int(a * 40)
        print(f"  t={t:2d}: {a:.3f} {bar}")


if __name__ == "__main__":
    test_gradient_flow()
    test_alpha_dynamics()
    test_retrieval()
