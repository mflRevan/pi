"""
Echo Chamber V2 - Constant-Q Decay with Raw Interference

Key changes from V1:
1. NO EMA interpolation - use: memory * decay + input * raw_interference
2. RAW interference score - no normalization, full dynamic range from conjugate
3. FREQUENCY-DEPENDENT DECAY: γ(w) = exp(-β * |w|)
   - Higher frequency → faster decay
   - Lower frequency → slower decay  
   - "Constant Q" - each freq rings for same number of cycles

The Mathematical Law:
    decay(w) = clamp(exp(-β * |w|), max=0.9999)
    
    where w is the output projection's learned frequency (d_model sized)
    and β is a global damping constant

Memory Update:
    memory = memory * decay + input * raw_interference
    
    NOT: memory = α * input + (1-α) * memory  (old EMA approach)

Expected behavior:
    - Early training: high random interference → chaotic → pressure to specialize
    - Late training: specialized queries → sparse interference → stable selective memory
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict
import time

import sys
sys.path.insert(0, '/home/aiman/pi')

from rin.lut import get_global_lut

PHI = (1 + math.sqrt(5)) / 2


class EchoHeadV2(nn.Module):
    """
    Echo head with RAW interference output.
    
    Returns raw conjugate interference (not normalized) - has dynamic range
    proportional to d_head. This creates selection pressure for specialization.
    """
    
    def __init__(self, d_head: int, head_idx: int):
        super().__init__()
        self.d_head = d_head
        self.head_idx = head_idx
        
        # Learned trigger - the pattern this head detects
        # Initialize with larger magnitude for meaningful interference
        self.trigger_real = nn.Parameter(torch.randn(d_head) * 0.5)
        self.trigger_imag = nn.Parameter(torch.randn(d_head) * 0.5)
        
        # Query Euler projection
        self.w_query = nn.Parameter(torch.randn(d_head) * 0.02)
        self.b_query = nn.Parameter(torch.zeros(d_head))
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def forward(
        self,
        x_real: torch.Tensor,  # (batch, d_head)
        x_imag: torch.Tensor,  # (batch, d_head)
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute RAW interference score (not normalized).
        
        Returns:
            interference_real: (batch,) - real part of conjugate interference
            interference_imag: (batch,) - imag part of conjugate interference
            diagnostics: debug info
        """
        lut = self._get_lut(x_real.device)
        
        t_val = t.view(-1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0)
        t_phi = t_val * PHI
        
        # Query Euler transform
        wavelength = 1.0 + self.w_query.abs()
        theta_real = x_real / wavelength + self.b_query + t_phi
        theta_imag = x_imag / wavelength + self.b_query + t_phi
        
        sin_r, cos_r = lut.lookup_sin_cos(theta_real)
        sin_i, cos_i = lut.lookup_sin_cos(theta_imag)
        
        query_real = cos_r * cos_i - sin_r * sin_i
        query_imag = cos_r * sin_i + sin_r * cos_i
        
        # RAW conjugate interference: query* · trigger
        # Re(z1* · z2) = Re(z1)·Re(z2) + Im(z1)·Im(z2)
        # Im(z1* · z2) = Re(z1)·Im(z2) - Im(z1)·Re(z2)
        # NOT normalized - has range proportional to d_head
        interference_real = (query_real * self.trigger_real + query_imag * self.trigger_imag).sum(dim=-1)
        interference_imag = (query_real * self.trigger_imag - query_imag * self.trigger_real).sum(dim=-1)
        
        # Compute magnitudes for diagnostics only
        query_mag = (query_real**2 + query_imag**2).sum(dim=-1).sqrt()
        trigger_mag = (self.trigger_real**2 + self.trigger_imag**2).sum().sqrt()
        interference_mag = (interference_real**2 + interference_imag**2).sqrt()
        
        diagnostics = {
            'interference_real': interference_real.detach(),
            'interference_imag': interference_imag.detach(),
            'interference_mag': interference_mag.detach(),
            'query_mag': query_mag.detach(),
            'trigger_mag': trigger_mag.detach(),
            # Normalized for comparison
            'normalized_real': (interference_real / (query_mag * trigger_mag + 1e-8)).detach(),
        }
        
        return interference_real, interference_imag, diagnostics


class EchoChamberV2(nn.Module):
    """
    Echo Chamber with Constant-Q decay and raw interference.
    
    Memory update (Q-EMA):
        w_eff = 1 / (1 + |w|)  # Effective wavelength (same as Euler projection)
        β_eff = 1 / (1 + |β|)  # Effective decay rate (SAME parameterization as w!)
        decay = clamp(exp(-β_eff * w_eff), max=0.9999)
        memory = memory * decay + input * |interference| * w_eff * (1 - decay)
    
    Key insight: Both decay AND write strength are wavelength-respecting.
    Higher frequency (smaller w_eff) = faster decay AND smaller write contribution.
    This creates proper Q-factor behavior where each frequency has proportional
    decay rate and update sensitivity.
    
    Beta uses SAME parameterization as wavelength:
        β_eff = 1 / (1 + |β|) ∈ (0, 1]
    - Can't be zero (division by 1+x where x≥0)
    - Can't exceed 1 (bounded by formula)
    - Naturally stable, no eps needed
    """
    
    def __init__(self, d_model: int, n_heads: int = 4, detach_memory: bool = True):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Whether to detach memory between timesteps
        # True = faster training, no gradient flow through memory history
        # False = full BPTT through memory, enables learning long-term dependencies
        self.detach_memory = detach_memory
        
        # Echo heads
        self.heads = nn.ModuleList([
            EchoHeadV2(self.d_head, head_idx=i)
            for i in range(n_heads)
        ])
        
        # Output Euler projection frequency
        self.w_out = nn.Parameter(torch.randn(d_model) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(d_model))
        
        # Per-dimension damping constant β (same style as wavelength!)
        # β_eff = 1 / (1 + |β|)
        # 
        # Initialize with mean ~10 to give moderate initial decay:
        #   β=10 → β_eff≈0.09 → decay≈0.91 → 10-step persistence≈0.39
        # This gives the model a fair chance to learn memory, but weight decay
        # will still push toward faster decay if memory isn't useful.
        self.beta = nn.Parameter(torch.abs(torch.randn(d_model)) * 5.0 + 5.0)
        
        # Memory state
        self._memory_real = None
        self._memory_imag = None
        
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(4096, device)
        return self._lut
    
    def reset_memory(self, batch_size: int, device: torch.device):
        self._memory_real = torch.zeros(batch_size, self.d_model, device=device)
        self._memory_imag = torch.zeros(batch_size, self.d_model, device=device)
    
    def get_beta(self) -> torch.Tensor:
        """
        Get effective beta using wavelength-style parameterization.
        β_eff = 1 / (1 + |β|)
        
        For slow decay (long memory), we need SMALL β_eff:
        - Large |β| (e.g. 100) → small β_eff (≈0.01) → decay ≈ exp(-0.01) ≈ 0.99 ✓
        - Small |β| (e.g. 0.01) → large β_eff (≈0.99) → decay ≈ exp(-0.99) ≈ 0.37 ✗
        
        So unlike wavelength (where small w = slow rotation), here LARGE beta = slow decay!
        The formula inverts the effect compared to wavelength.
        """
        return 1.0 / (1.0 + self.beta.abs())
    
    def get_effective_wavelength(self) -> torch.Tensor:
        """
        Compute effective wavelength: w_eff = 1 / (1 + |w|)
        
        This matches the Euler projection formula and ensures:
        - Higher frequency (larger |w|) → smaller w_eff → faster decay
        - Lower frequency (smaller |w|) → larger w_eff → slower decay
        """
        return 1.0 / (1.0 + self.w_out.abs())
    
    def compute_decay(self) -> torch.Tensor:
        """
        Compute frequency-dependent decay: γ(w) = exp(-β * w_eff)
        
        where w_eff = 1 / (1 + |w|) is the effective wavelength.
        
        Returns: (d_model,) decay values clamped to max 0.9999
        """
        beta = self.get_beta()  # (d_model,)
        w_eff = self.get_effective_wavelength()  # (d_model,)
        decay = torch.exp(-beta * w_eff)
        return decay.clamp(max=0.9999)
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Process input and update memory with constant-Q decay.
        """
        lut = self._get_lut(x_real.device)
        batch_size = x_real.shape[0]
        
        if self._memory_real is None or self._memory_real.shape[0] != batch_size:
            self.reset_memory(batch_size, x_real.device)
        
        t_val = t.view(-1, 1) if t.dim() >= 1 else t.unsqueeze(0).unsqueeze(0)
        t_phi = t_val * PHI
        
        # Compute head interference scores
        head_int_real = []
        head_int_imag = []
        head_diagnostics = []
        
        for head_idx, head in enumerate(self.heads):
            start = head_idx * self.d_head
            end = start + self.d_head
            
            patch_real = x_real[:, start:end]
            patch_imag = x_imag[:, start:end]
            
            int_r, int_i, diag = head(patch_real, patch_imag, t)
            head_int_real.append(int_r)
            head_int_imag.append(int_i)
            head_diagnostics.append(diag)
        
        # Combine interference across heads (mean)
        # Shape: (batch,)
        total_int_real = torch.stack(head_int_real, dim=-1).mean(dim=-1)
        total_int_imag = torch.stack(head_int_imag, dim=-1).mean(dim=-1)
        
        # === CONSTANT-Q DECAY ===
        decay = self.compute_decay()  # (d_model,)
        
        # === Q-EMA MEMORY UPDATE ===
        # memory = memory * decay + input * |interference| * w_eff * (1 - decay)
        # 
        # Key components:
        # - |interference|: Write gate (both constructive AND destructive signals)
        # - w_eff: Wavelength-respecting scale (higher freq = smaller contribution)
        # - (1 - decay): Bounds growth to prevent explosion
        #
        # This is a proper Q-EMA where both decay and write respect frequency.
        #
        # Gradient flow control:
        # - detach_memory=True: Faster training, but can't learn long-term dependencies
        # - detach_memory=False: Full BPTT through memory history, enables learning
        if self.detach_memory:
            memory_real = self._memory_real.detach()
            memory_imag = self._memory_imag.detach()
        else:
            memory_real = self._memory_real
            memory_imag = self._memory_imag
        
        # Effective wavelength for Q-EMA (same formula as Euler projection)
        w_eff = self.get_effective_wavelength()  # (d_model,)
        
        # Compute interference magnitude (always positive)
        # Both constructive AND destructive interference = strong pattern detected
        int_mag = torch.sqrt(total_int_real**2 + total_int_imag**2 + 1e-8)
        
        # Expand to (batch, 1) for broadcasting
        int_mag_exp = int_mag.unsqueeze(-1)
        
        # Q-EMA write scale: w_eff * (1 - decay)
        # Higher frequency = smaller w_eff = smaller write contribution
        # This couples write strength to the natural time constant of each dimension
        write_scale = (w_eff * (1.0 - decay)).unsqueeze(0)  # (1, d_model)
        
        # Additive write: input * |interference| * w_eff * (1 - decay)
        new_memory_real = memory_real * decay + x_real * int_mag_exp * write_scale
        new_memory_imag = memory_imag * decay + x_imag * int_mag_exp * write_scale
        
        # Store updated memory (always detach for autograd graph management)
        if self.detach_memory:
            self._memory_real = new_memory_real.detach()
            self._memory_imag = new_memory_imag.detach()
        else:
            # Keep connected for BPTT - memory is part of compute graph
            self._memory_real = new_memory_real
            self._memory_imag = new_memory_imag
        
        # === OUTPUT: Euler projection of memory ===
        wavelength = 1.0 + self.w_out.abs()
        theta_out_real = new_memory_real / wavelength + self.b_out + t_phi
        theta_out_imag = new_memory_imag / wavelength + self.b_out + t_phi
        
        sin_or, cos_or = lut.lookup_sin_cos(theta_out_real)
        sin_oi, cos_oi = lut.lookup_sin_cos(theta_out_imag)
        out_real = cos_or * cos_oi - sin_or * sin_oi
        out_imag = cos_or * sin_oi + sin_or * cos_oi
        
        # Get effective beta for diagnostics
        beta_eff = self.get_beta()
        
        diagnostics = {
            'total_int_real': total_int_real.detach(),
            'total_int_imag': total_int_imag.detach(),
            'total_int_mag': (total_int_real**2 + total_int_imag**2).sqrt().detach(),
            'decay_mean': decay.mean().detach(),
            'decay_min': decay.min().detach(),
            'decay_max': decay.max().detach(),
            'beta_eff_mean': beta_eff.mean().detach(),
            'beta_eff_min': beta_eff.min().detach(),
            'beta_eff_max': beta_eff.max().detach(),
            'beta_raw_mean': self.beta.mean().detach(),
            'beta_raw_std': self.beta.std().detach(),
            'w_eff_mean': w_eff.mean().detach(),
            'w_eff_min': w_eff.min().detach(),
            'w_eff_max': w_eff.max().detach(),
            'write_scale_mean': write_scale.mean().detach(),
            'memory_mag': (new_memory_real**2 + new_memory_imag**2).sum(-1).sqrt().detach(),
            'output_mag': (out_real**2 + out_imag**2).sum(-1).sqrt().detach(),
            'head_details': head_diagnostics,
        }
        
        return out_real, out_imag, diagnostics


class ResonantLayer(nn.Module):
    """Resonant layer (from model.py)."""
    
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


class EchoChamberModelV2(nn.Module):
    """Model with Echo Chamber V2 (constant-Q decay, raw interference)."""
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        n_echo_heads: int = 4,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.n_echo_heads = n_echo_heads
        
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        self.resonant_layers = nn.ModuleList([
            ResonantLayer(d_model, num_neurons)
            for _ in range(num_layers)
        ])
        
        self.echo_chambers = nn.ModuleList([
            EchoChamberV2(d_model, n_echo_heads)
            for _ in range(num_layers)
        ])
        
        self.echo_scales = nn.ParameterList([
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
        for chamber in self.echo_chambers:
            chamber.reset_memory(batch_size, device)
    
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
                
                echo_real, echo_imag, echo_diag = self.echo_chambers[layer_idx](
                    x_real, x_imag, t_val
                )
                
                if return_diagnostics:
                    step_diag[f'layer{layer_idx}_echo'] = echo_diag
                
                scale = self.echo_scales[layer_idx]
                combined_real = res_real + scale * echo_real
                combined_imag = res_imag + scale * echo_imag
                
                x_real = x_real + combined_real
                x_imag = x_imag + combined_imag
            
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
    """Verify gradients flow to all components."""
    print("="*80)
    print("GRADIENT FLOW TEST - Echo Chamber V2")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = EchoChamberModelV2(
        vocab_size=64,
        d_model=64,
        num_layers=2,
        num_neurons=64,
        n_echo_heads=4,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    print(f"d_head = {64 // 4} = 16")
    
    x = torch.randint(0, 64, (4, 12), device=device)
    logits = model(x)
    loss = F.cross_entropy(logits[:, -1, :], torch.zeros(4, dtype=torch.long, device=device))
    loss.backward()
    
    print("\nGradient norms by component:")
    print("-"*80)
    
    cats = {
        'trigger': [],
        'query': [],
        'output_euler': [],
        'beta': [],
        'echo_scale': [],
        'resonant': [],
        'output': []
    }
    
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        
        grad = param.grad.norm().item()
        if grad == 0:
            continue
        
        if 'trigger' in name:
            cats['trigger'].append((name, grad))
        elif 'w_query' in name or 'b_query' in name:
            cats['query'].append((name, grad))
        elif 'w_out' in name or 'b_out' in name:
            cats['output_euler'].append((name, grad))
        elif 'beta' in name:
            cats['beta'].append((name, grad))
        elif 'echo_scale' in name:
            cats['echo_scale'].append((name, grad))
        elif 'output_proj' in name:
            cats['output'].append((name, grad))
        else:
            cats['resonant'].append((name, grad))
    
    for cat, items in cats.items():
        if items:
            avg = sum(g for _, g in items) / len(items)
            print(f"\n{cat.upper()} (avg={avg:.6f}, count={len(items)}):")
            for name, g in items[:3]:
                short = '.'.join(name.split('.')[-2:])
                print(f"  {short}: {g:.6f}")
    
    print("\n✓ Gradient flow test complete")


def test_interference_distribution():
    """Test raw interference score distribution."""
    print("\n" + "="*80)
    print("INTERFERENCE DISTRIBUTION TEST (before training)")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = EchoChamberModelV2(
        vocab_size=64,
        d_model=64,
        num_layers=1,
        num_neurons=64,
        n_echo_heads=4,
    ).to(device)
    
    print(f"d_head = {64 // 4} = 16")
    print(f"Expected interference range: roughly [-{16}, +{16}] for aligned, 0 for orthogonal")
    
    x = torch.randint(0, 64, (1, 20), device=device)
    with torch.no_grad():
        _, diag = model(x, return_diagnostics=True)
    
    print("\n" + "-"*100)
    print(f"{'t':>3} | {'int_real':>10} | {'int_imag':>10} | {'int_mag':>10} | {'decay':>8} | {'mem_mag':>10}")
    print("-"*100)
    
    all_int_real = []
    all_int_mag = []
    
    for step in diag:
        t = step['t']
        echo = step['layer0_echo']
        
        int_real = echo['total_int_real'].mean().item()
        int_imag = echo['total_int_imag'].mean().item()
        int_mag = echo['total_int_mag'].mean().item()
        decay = echo['decay_mean'].item()
        mem_mag = echo['memory_mag'].mean().item()
        
        all_int_real.append(int_real)
        all_int_mag.append(int_mag)
        
        print(f"{t:3d} | {int_real:10.4f} | {int_imag:10.4f} | {int_mag:10.4f} | {decay:8.4f} | {mem_mag:10.4f}")
    
    print("-"*100)
    
    print(f"\nInterference stats:")
    print(f"  int_real: min={min(all_int_real):.4f}, max={max(all_int_real):.4f}, range={max(all_int_real)-min(all_int_real):.4f}")
    print(f"  int_mag:  min={min(all_int_mag):.4f}, max={max(all_int_mag):.4f}")
    
    # Decay stats
    echo = diag[0]['layer0_echo']
    print(f"\nDecay (constant-Q):")
    print(f"  mean={echo['decay_mean'].item():.4f}, min={echo['decay_min'].item():.4f}, max={echo['decay_max'].item():.4f}")
    print(f"  β = {model.echo_chambers[0].beta.item():.4f}")


def test_learning_dynamics():
    """Train and track interference evolution."""
    print("\n" + "="*80)
    print("LEARNING DYNAMICS - Before/After Training")
    print("="*80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    vocab_size = 64
    marker = vocab_size - 1
    seq_len = 16
    batch_size = 32
    
    model = EchoChamberModelV2(
        vocab_size=vocab_size,
        d_model=64,
        num_layers=1,
        num_neurons=64,
        n_echo_heads=4,
    ).to(device)
    
    print(f"Parameters: {model.get_num_params():,}")
    print(f"Task: Marker retrieval")
    
    def get_stats(model, test_seq):
        model.eval()
        with torch.no_grad():
            _, diag = model(test_seq, return_diagnostics=True)
        
        int_reals = [d['layer0_echo']['total_int_real'].mean().item() for d in diag]
        int_mags = [d['layer0_echo']['total_int_mag'].mean().item() for d in diag]
        mem_mags = [d['layer0_echo']['memory_mag'].mean().item() for d in diag]
        
        stats = {
            'int_real_mean': sum(int_reals) / len(int_reals),
            'int_real_std': (sum((r - sum(int_reals)/len(int_reals))**2 for r in int_reals) / len(int_reals)) ** 0.5,
            'int_real_range': max(int_reals) - min(int_reals),
            'int_mag_mean': sum(int_mags) / len(int_mags),
            'mem_mag_final': mem_mags[-1],
            'decay_mean': diag[0]['layer0_echo']['decay_mean'].item(),
            'beta': model.echo_chambers[0].beta.item(),
        }
        
        # Get per-head trigger mags
        trigger_mags = []
        for head in model.echo_chambers[0].heads:
            t_mag = (head.trigger_real**2 + head.trigger_imag**2).sum().sqrt().item()
            trigger_mags.append(t_mag)
        stats['trigger_mags'] = trigger_mags
        
        return stats, int_reals
    
    # Create test sequence
    test_seq = torch.randint(0, vocab_size-2, (1, seq_len), device=device)
    test_seq[0, 4] = marker
    test_seq[0, 5] = 42  # target value
    test_seq[0, -2] = marker
    
    # BEFORE training
    print("\n--- BEFORE TRAINING ---")
    stats_before, int_reals_before = get_stats(model, test_seq)
    print(f"  int_real: mean={stats_before['int_real_mean']:.4f}, std={stats_before['int_real_std']:.4f}, range={stats_before['int_real_range']:.4f}")
    print(f"  int_mag:  mean={stats_before['int_mag_mean']:.4f}")
    print(f"  memory:   final_mag={stats_before['mem_mag_final']:.4f}")
    print(f"  decay:    mean={stats_before['decay_mean']:.4f}, β={stats_before['beta']:.4f}")
    print(f"  triggers: {[f'{m:.3f}' for m in stats_before['trigger_mags']]}")
    
    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    
    track_epochs = [25, 50, 100]
    
    for epoch in range(max(track_epochs) + 1):
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
        
        if epoch in track_epochs:
            model.eval()
            with torch.no_grad():
                pred = logits[:, -1, :].argmax(dim=-1)
                acc = (pred == targets).float().mean().item()
            
            stats, int_reals = get_stats(model, test_seq)
            
            print(f"\n--- EPOCH {epoch} (loss={loss.item():.4f}, acc={acc:.1%}) ---")
            print(f"  int_real: mean={stats['int_real_mean']:.4f}, std={stats['int_real_std']:.4f}, range={stats['int_real_range']:.4f}")
            print(f"  int_mag:  mean={stats['int_mag_mean']:.4f}")
            print(f"  memory:   final_mag={stats['mem_mag_final']:.4f}")
            print(f"  decay:    mean={stats['decay_mean']:.4f}, β={stats['beta']:.4f}")
            print(f"  triggers: {[f'{m:.3f}' for m in stats['trigger_mags']]}")
    
    # AFTER training
    print("\n--- AFTER TRAINING (200 epochs) ---")
    stats_after, int_reals_after = get_stats(model, test_seq)
    
    print(f"\nComparison:")
    print(f"  int_real range:  {stats_before['int_real_range']:.4f} → {stats_after['int_real_range']:.4f}")
    print(f"  int_real std:    {stats_before['int_real_std']:.4f} → {stats_after['int_real_std']:.4f}")
    print(f"  β (damping):     {stats_before['beta']:.4f} → {stats_after['beta']:.4f}")
    print(f"  decay mean:      {stats_before['decay_mean']:.4f} → {stats_after['decay_mean']:.4f}")
    
    print(f"\nPer-timestep interference (before → after):")
    for t in [0, 4, 5, 14, 15]:  # marker at 4, value at 5, query at 14, final at 15
        print(f"  t={t:2d}: {int_reals_before[t]:7.4f} → {int_reals_after[t]:7.4f}")


if __name__ == "__main__":
    test_gradient_flow()
    test_interference_distribution()
    test_learning_dynamics()
