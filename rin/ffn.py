"""
Resonant FFN - Unified Entry Point for All FFN Variants

This module provides the single source of truth for FFN implementations
in the Holographic Transformer architecture.

Usage:
    from rin.ffn import ResonantFFN
    
    # Content-only gating (original)
    ffn = ResonantFFN(d_model=512, n_phase=64, gate_mode='content')
    
    # Position-aware gating
    ffn = ResonantFFN(d_model=512, n_phase=64, gate_mode='time')
    
    # Parallel time*content multiplicative gating
    ffn = ResonantFFN(d_model=512, n_phase=64, gate_mode='parallel')
    
    # Omniware: unified time×content (recommended for most tasks)
    ffn = ResonantFFN(d_model=512, n_phase=64, gate_mode='omniware', use_triton=True)
    
    # Forward pass
    out_real, out_imag = ffn(x_real, x_imag)

Gate Modes:
    - 'content': Original resonant gating (content-only)
    - 'time': Position-aware gating (RoPE-style)
    - 'parallel': Multiplicative time × content gating
    - 'omniware': Unified time*content in single theta (recommended)

Triton Acceleration:
    When use_triton=True (default), uses optimized Triton kernels for
    'omniware' mode with <2x overhead vs SwiGLU baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional, Literal


# =============================================================================
# LOGARITHMIC GRADIENT SCALING
# =============================================================================
# These autograd functions apply ln(1 + |grad|) * sign(grad) scaling
# to gradients during backward pass. Critical for stable training
# with multiscale time frequencies.

class _LogGradScale(torch.autograd.Function):
    """
    Identity forward, log-compressed gradient backward.
    
    Applies: grad_scaled = ln(1 + |grad|) * sign(grad)
    
    This provides natural geometric compression for gradients from
    multiscale time frequencies, reducing gradient ratio from
    ~7000x to ~60-120x.
    """
    @staticmethod
    def forward(ctx, x):
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad):
        # Log compression: large gradients compressed, small preserved
        return torch.log1p(grad.abs()) * grad.sign()


def _log_grad_scale(x: torch.Tensor) -> torch.Tensor:
    """Apply log gradient scaling during backward pass."""
    return _LogGradScale.apply(x)


class _OmniwareGateWithLogGrad(torch.autograd.Function):
    """
    Custom autograd for Omniware gate with log gradient scaling.
    
    This applies log scaling AFTER the full chain rule computation,
    ensuring mathematically correct gradient flow with compression
    only at the final step.
    
    The key insight is that we want to compress the FINAL gradients
    w.r.t. x_imag and w, not intermediate gradients.
    """
    @staticmethod
    def forward(ctx, x_imag, pos_freq, w, b, energy_scale):
        """
        Forward: gate = sum_p(cos(w * x_imag * pos_freq + b)) * energy_scale
        """
        # Compute time × content
        time_content = x_imag * pos_freq.unsqueeze(0)  # (B, L, P)
        
        # Compute theta
        time_content_exp = time_content.unsqueeze(-1)  # (B, L, P, 1)
        w_exp = w.unsqueeze(0).unsqueeze(0)  # (1, 1, P, H)
        b_exp = b.unsqueeze(0).unsqueeze(0)  # (1, 1, P, H)
        theta = w_exp * time_content_exp + b_exp  # (B, L, P, H)
        
        # Compute gate
        cos_theta = torch.cos(theta)
        cos_sum = cos_theta.sum(dim=-2)  # (B, L, H)
        gate = cos_sum * energy_scale
        
        # Save for backward
        ctx.save_for_backward(x_imag, pos_freq, w, b, theta)
        ctx.energy_scale = energy_scale
        
        return gate
    
    @staticmethod
    def backward(ctx, grad_gate):
        """
        Backward with log gradient scaling on x_imag and w.
        
        Chain rule:
            d_gate/d_theta = -sin(theta) * energy_scale  (summed over P)
            d_theta/d_time_content = w
            d_time_content/d_x_imag = pos_freq
            d_theta/d_w = time_content
            d_theta/d_b = 1
        
        Final gradients have log scaling applied.
        """
        x_imag, pos_freq, w, b, theta = ctx.saved_tensors
        energy_scale = ctx.energy_scale
        
        # d_gate/d_theta: shape matches theta (B, L, P, H)
        # But grad_gate is (B, L, H), need to broadcast
        d_gate_d_theta = -torch.sin(theta) * energy_scale  # (B, L, P, H)
        
        # grad_gate (B, L, H) -> expand for P dimension
        grad_gate_exp = grad_gate.unsqueeze(-2)  # (B, L, 1, H)
        grad_theta = grad_gate_exp * d_gate_d_theta  # (B, L, P, H)
        
        # Compute time_content for gradient calculation
        time_content = x_imag * pos_freq.unsqueeze(0)  # (B, L, P)
        
        # Gradient w.r.t. w: d_theta/d_w = time_content
        # grad_w = sum over (B, L) of grad_theta * time_content
        time_content_exp = time_content.unsqueeze(-1)  # (B, L, P, 1)
        grad_w = (grad_theta * time_content_exp).sum(dim=(0, 1))  # (P, H)
        
        # Gradient w.r.t. b: d_theta/d_b = 1
        grad_b = grad_theta.sum(dim=(0, 1))  # (P, H)
        
        # Gradient w.r.t. x_imag: chain through time_content
        # d_theta/d_time_content = w
        # d_time_content/d_x_imag = pos_freq
        w_exp = w.unsqueeze(0).unsqueeze(0)  # (1, 1, P, H)
        grad_time_content = (grad_theta * w_exp).sum(dim=-1)  # (B, L, P)
        grad_x_imag = grad_time_content * pos_freq.unsqueeze(0)  # (B, L, P)
        
        # Apply log gradient scaling AFTER full chain rule
        grad_x_imag = torch.log1p(grad_x_imag.abs()) * grad_x_imag.sign()
        grad_w = torch.log1p(grad_w.abs()) * grad_w.sign()
        
        return grad_x_imag, None, grad_w, grad_b, None


def _omniware_gate_with_log_grad(
    x_imag: torch.Tensor,
    pos_freq: torch.Tensor,
    w: torch.Tensor,
    b: torch.Tensor,
    energy_scale: float,
) -> torch.Tensor:
    """Compute omniware gate with log gradient scaling."""
    return _OmniwareGateWithLogGrad.apply(x_imag, pos_freq, w, b, energy_scale)


# =============================================================================
# UNIFIED RESONANT FFN
# =============================================================================

GateMode = Literal['content', 'time', 'parallel', 'omniware']


class ResonantFFN(nn.Module):
    """
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║  CRITICAL ARCHITECTURE INVARIANT - DO NOT MODIFY WITHOUT UNDERSTANDING:      ║
    ║                                                                              ║
    ║  The gate path uses DIRECT element-wise wavelength modulation on x_imag.     ║
    ║  There is NO linear projection before the interference computation.          ║
    ║  This is the ONLY correct implementation of the resonant MLP.                ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    
    Architecture Pseudocode (AUTHORITATIVE):
    ─────────────────────────────────────────
    
        # Inputs:
        #   x_real: (B, L, d_model)  - content stream (e.g., 512-dim)
        #   x_imag: (B, L, n_phase)  - phase stream (e.g., 64-dim, can equal d_model)
        #   hidden_dim = d_model * expansion (e.g., 2048)
        
        # ═══════════════════════════════════════════════════════════════════
        # VALUE PATH: Linear projection with optional activation
        # ═══════════════════════════════════════════════════════════════════
        value = x_real @ W_up                    # (B, L, d_model) @ (d_model, H) -> (B, L, H)
        # NOTE: No activation on value by default (like GLU gate path)
        
        # ═══════════════════════════════════════════════════════════════════
        # GATE PATH: TRUE INTERFERENCE - NO LINEAR PROJECTION!
        # ═══════════════════════════════════════════════════════════════════
        # Learnable parameters (per input-dim, per neuron):
        #   wavelength_raw: (n_phase, hidden_dim) - controls oscillation frequency
        #   phase_offset:   (n_phase, hidden_dim) - learned phase bias
        
        wavelength = 1 / (1 + softplus(wavelength_raw))   # (P, H) bounded (0, 1]
        
        # Element-wise modulation - THIS IS THE KEY OPERATION:
        # Each input dimension contributes its own wave to each neuron
        theta = x_imag.unsqueeze(-1) * wavelength + phase_offset
        #       (B, L, P, 1)         * (P, H)     + (P, H)
        #       -> (B, L, P, H)  via broadcasting
        
        # TRUE interference: sum of cosines across input dimensions
        # Each neuron receives interference pattern from P input waves
        cos_theta = cos(theta)                   # (B, L, P, H)
        cos_sum = cos_theta.sum(dim=-2)          # (B, L, H) - sum over P dimension
        gate = cos_sum * energy_scale            # energy_scale = 1/sqrt(n_phase)
        
        # ═══════════════════════════════════════════════════════════════════
        # GATED OUTPUT
        # ═══════════════════════════════════════════════════════════════════
        out = value * gate                       # (B, L, H) element-wise
        
        # ═══════════════════════════════════════════════════════════════════
        # SEPARATE DOWN PROJECTIONS (information mixing between streams)
        # ═══════════════════════════════════════════════════════════════════
        res_real = out  @ W_down_real            # (B, L, H) @ (H, d_model) -> (B, L, d_model)
        res_imag = gate @ W_down_imag            # (B, L, H) @ (H, n_phase) -> (B, L, n_phase)
        
        # Additive residual
        out_real = x_real + res_real
        out_imag = x_imag + res_imag
        
        return out_real, out_imag
    
    Why NO Linear Projection on Gate Path?
    ──────────────────────────────────────
    1. The interference pattern emerges from DIRECT modulation of input phases
    2. A linear projection would destroy the per-dimension wave structure
    3. Each input dimension d contributes: cos(wavelength[d,n] * x_imag[d] + B[d,n])
    4. The SUM of these cosines creates constructive/destructive interference
    5. This is fundamentally different from (and richer than) a linear transform
    
    Gate Modes:
    ───────────
    - 'content': Original resonant gating (content-only)
    - 'time': Position-aware gating (RoPE-style frequencies)
    - 'parallel': Multiplicative time × content gating
    - 'omniware': Unified time*content in single theta (recommended)
    
    Triton Acceleration:
    ────────────────────
    When use_triton=True (default) and gate_mode='omniware':
    - Uses V2 optimized kernels with autotuned block sizes
    - Two-pass backward without atomic operations (~3x faster)
    - <2x overhead vs SwiGLU baseline
    
    Memory Note:
    ────────────
    The naive implementation materializes (B, L, n_phase, hidden_dim) tensor.
    For B=32, L=2048, P=64, H=2048: this is 32GB!
    Use Triton kernels for production (avoids this entirely).
    
    Args:
        d_model: Dimension of real/content stream
        n_phase: Dimension of imaginary/phase stream (can differ from d_model)
        expansion: Hidden dimension multiplier (hidden_dim = d_model * expansion)
        gate_mode: Gating strategy ('content', 'time', 'parallel', 'omniware')
        use_triton: Use Triton kernels for acceleration (only for 'omniware')
        log_grad: Apply log gradient scaling for 'omniware' mode (recommended)
        max_seq_len: Maximum sequence length for position cache
        base_freq: Base for RoPE-style frequencies
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        n_phase: int = None,
        expansion: int = 4,
        gate_mode: GateMode = 'omniware',
        use_triton: bool = True,
        log_grad: bool = True,
        max_seq_len: int = 8192,
        base_freq: float = 10000.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_phase = n_phase if n_phase is not None else d_model
        self.hidden_dim = d_model * expansion
        self.energy_scale = 1.0 / math.sqrt(self.n_phase)
        self.gate_mode = gate_mode
        self.use_triton = use_triton and gate_mode == 'omniware'
        self.log_grad = log_grad
        
        # Check Triton availability
        self._triton_available = False
        if self.use_triton:
            try:
                from .kernels.ffn import omniware_ffn_gate_forward_v2
                self._triton_available = True
            except ImportError:
                self._triton_available = False
                self.use_triton = False
        
        # ═══════════════════════════════════════════════════════════════════
        # VALUE PATH: x_real -> hidden_dim
        # ═══════════════════════════════════════════════════════════════════
        self.W_up = nn.Parameter(torch.empty(d_model, self.hidden_dim))
        
        # ═══════════════════════════════════════════════════════════════════
        # GATE PATH PARAMETERS (mode-dependent)
        # ═══════════════════════════════════════════════════════════════════
        if gate_mode == 'content':
            # Content-only: wavelength modulation
            self.wavelength_raw = nn.Parameter(torch.empty(self.n_phase, self.hidden_dim))
            self.phase_offset = nn.Parameter(torch.empty(self.n_phase, self.hidden_dim))
        elif gate_mode == 'time':
            # Time-only: spectral weight modulates position
            self.spectral_weight = nn.Parameter(torch.empty(self.n_phase, self.hidden_dim))
            self.phase_offset = nn.Parameter(torch.empty(self.n_phase, self.hidden_dim))
        elif gate_mode == 'parallel':
            # Parallel: separate time and content parameters
            self.w_time = nn.Parameter(torch.empty(self.n_phase, self.hidden_dim))
            self.b_time = nn.Parameter(torch.empty(self.n_phase, self.hidden_dim))
            self.wavelength_raw = nn.Parameter(torch.empty(self.n_phase, self.hidden_dim))
            self.b_content = nn.Parameter(torch.empty(self.n_phase, self.hidden_dim))
        elif gate_mode == 'omniware':
            # Omniware: unified time*content
            self.w = nn.Parameter(torch.empty(self.n_phase, self.hidden_dim))
            self.b = nn.Parameter(torch.empty(self.n_phase, self.hidden_dim))
        else:
            raise ValueError(f"Unknown gate_mode: {gate_mode}")
        
        # ═══════════════════════════════════════════════════════════════════
        # DOWN PROJECTIONS: Separate for real and imag (information mixing)
        # ═══════════════════════════════════════════════════════════════════
        self.W_down_real = nn.Parameter(torch.empty(self.hidden_dim, d_model))
        self.W_down_imag = nn.Parameter(torch.empty(self.hidden_dim, self.n_phase))
        
        # Pre-norm for value path stability
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Position frequencies for time-aware modes
        if gate_mode in ('time', 'parallel', 'omniware'):
            inv_freq = 1.0 / (base_freq ** (
                torch.arange(0, self.n_phase, 2).float() / self.n_phase
            ))
            full_inv_freq = torch.zeros(self.n_phase)
            full_inv_freq[0::2] = inv_freq[:self.n_phase // 2]
            full_inv_freq[1::2] = inv_freq[:self.n_phase // 2]
            self.register_buffer('inv_freq', full_inv_freq)
            
            positions = torch.arange(max_seq_len).float()
            pos_freqs = torch.outer(positions, full_inv_freq)
            self.register_buffer('pos_freqs', pos_freqs)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize parameters for stable training."""
        nn.init.xavier_uniform_(self.W_up)
        
        if self.gate_mode == 'content':
            nn.init.normal_(self.wavelength_raw, mean=0.0, std=0.1)
            nn.init.uniform_(self.phase_offset, -math.pi, math.pi)
        elif self.gate_mode == 'time':
            nn.init.normal_(self.spectral_weight, mean=1.0, std=0.1)
            nn.init.uniform_(self.phase_offset, -math.pi, math.pi)
        elif self.gate_mode == 'parallel':
            nn.init.normal_(self.w_time, mean=1.0, std=0.1)
            nn.init.uniform_(self.b_time, -math.pi, math.pi)
            nn.init.normal_(self.wavelength_raw, mean=0.0, std=0.1)
            nn.init.uniform_(self.b_content, -math.pi, math.pi)
        elif self.gate_mode == 'omniware':
            nn.init.normal_(self.w, mean=1.0, std=0.1)
            nn.init.uniform_(self.b, -math.pi, math.pi)
        
        nn.init.xavier_uniform_(self.W_down_real)
        nn.init.xavier_uniform_(self.W_down_imag)
        self.W_down_real.data *= 0.5
        self.W_down_imag.data *= 0.5
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with TRUE holographic interference gating.
        
        Args:
            x_real: Content stream (B, L, d_model)
            x_imag: Phase stream (B, L, n_phase)
            
        Returns:
            out_real: (B, L, d_model) - content output with residual
            out_imag: (B, L, n_phase) - phase output with residual
        """
        B, L, _ = x_real.shape
        
        # ═══════════════════════════════════════════════════════════════════
        # VALUE PATH: PreNorm + linear projection (no activation)
        # ═══════════════════════════════════════════════════════════════════
        x_real_normed = self.norm(x_real)
        value = x_real_normed @ self.W_up  # (B, L, H)
        
        # ═══════════════════════════════════════════════════════════════════
        # GATE PATH: Mode-dependent computation
        # ═══════════════════════════════════════════════════════════════════
        gate = self._compute_gate(x_imag, L)
        
        # ═══════════════════════════════════════════════════════════════════
        # GATED OUTPUT + DOWN PROJECTIONS
        # ═══════════════════════════════════════════════════════════════════
        out = value * gate  # (B, L, H)
        
        res_real = out @ self.W_down_real   # (B, L, d_model)
        res_imag = gate @ self.W_down_imag  # (B, L, n_phase)
        
        res_real = self.dropout(res_real)
        
        out_real = x_real + res_real
        out_imag = x_imag + res_imag
        
        return out_real, out_imag
    
    def _compute_gate(self, x_imag: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Compute interference gate based on gate_mode."""
        B = x_imag.shape[0]
        
        if self.gate_mode == 'content':
            return self._gate_content(x_imag)
        elif self.gate_mode == 'time':
            return self._gate_time(x_imag, seq_len, B)
        elif self.gate_mode == 'parallel':
            return self._gate_parallel(x_imag, seq_len, B)
        elif self.gate_mode == 'omniware':
            return self._gate_omniware(x_imag, seq_len)
    
    def _gate_content(self, x_imag: torch.Tensor) -> torch.Tensor:
        """Content-only gating (original ResonantFFN)."""
        wavelength = 1.0 / (1.0 + F.softplus(self.wavelength_raw))  # (P, H)
        x_imag_exp = x_imag.unsqueeze(-1)  # (B, L, P, 1)
        theta = x_imag_exp * wavelength + self.phase_offset  # (B, L, P, H)
        cos_theta = torch.cos(theta)
        cos_sum = cos_theta.sum(dim=-2)  # (B, L, H)
        return cos_sum * self.energy_scale
    
    def _gate_time(self, x_imag: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        """Time/position-aware gating."""
        pos_freqs = self.pos_freqs[:seq_len]  # (L, P)
        pos_freqs_exp = pos_freqs.unsqueeze(0).unsqueeze(-1)  # (1, L, P, 1)
        spectral_exp = self.spectral_weight.unsqueeze(0).unsqueeze(0)  # (1, 1, P, H)
        offset_exp = self.phase_offset.unsqueeze(0).unsqueeze(0)  # (1, 1, P, H)
        
        theta = spectral_exp * pos_freqs_exp + offset_exp  # (1, L, P, H)
        theta = theta.expand(batch_size, -1, -1, -1)  # (B, L, P, H)
        
        cos_theta = torch.cos(theta)
        cos_sum = cos_theta.sum(dim=-2)  # (B, L, H)
        return cos_sum * self.energy_scale
    
    def _gate_parallel(self, x_imag: torch.Tensor, seq_len: int, batch_size: int) -> torch.Tensor:
        """Parallel time × content multiplicative gating."""
        # Time path
        pos_freqs = self.pos_freqs[:seq_len]  # (L, P)
        pos_freqs_exp = pos_freqs.unsqueeze(0).unsqueeze(-1)  # (1, L, P, 1)
        w_time_exp = self.w_time.unsqueeze(0).unsqueeze(0)  # (1, 1, P, H)
        b_time_exp = self.b_time.unsqueeze(0).unsqueeze(0)
        
        theta_time = w_time_exp * pos_freqs_exp + b_time_exp  # (1, L, P, H)
        cos_time = torch.cos(theta_time)
        gate_time = cos_time.sum(dim=-2)  # (1, L, H)
        
        # Content path
        wavelength = 1.0 / (1.0 + F.softplus(self.wavelength_raw))  # (P, H)
        x_imag_exp = x_imag.unsqueeze(-1)  # (B, L, P, 1)
        b_content_exp = self.b_content.unsqueeze(0).unsqueeze(0)
        
        theta_content = x_imag_exp * wavelength + b_content_exp  # (B, L, P, H)
        cos_content = torch.cos(theta_content)
        gate_content = cos_content.sum(dim=-2)  # (B, L, H)
        
        # Multiplicative combination
        return gate_time * gate_content * (self.energy_scale ** 2)
    
    def _gate_omniware(self, x_imag: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Omniware: unified time*content gating."""
        pos_freqs = self.pos_freqs[:seq_len]  # (L, P)
        
        # Try Triton kernel first
        if self.use_triton and self._triton_available and x_imag.is_cuda:
            from .kernels.ffn import omniware_ffn_gate_forward_v2
            return omniware_ffn_gate_forward_v2(
                x_imag, pos_freqs, self.w, self.b,
                self.energy_scale, self.log_grad
            )
        
        # PyTorch fallback
        if self.log_grad:
            return _omniware_gate_with_log_grad(
                x_imag, pos_freqs, self.w, self.b, self.energy_scale
            )
        else:
            time_content = x_imag * pos_freqs.unsqueeze(0)  # (B, L, P)
            time_content_exp = time_content.unsqueeze(-1)  # (B, L, P, 1)
            w_exp = self.w.unsqueeze(0).unsqueeze(0)  # (1, 1, P, H)
            b_exp = self.b.unsqueeze(0).unsqueeze(0)  # (1, 1, P, H)
            theta = w_exp * time_content_exp + b_exp  # (B, L, P, H)
            cos_theta = torch.cos(theta)
            cos_sum = cos_theta.sum(dim=-2)  # (B, L, H)
            return cos_sum * self.energy_scale
    
    def extra_repr(self) -> str:
        triton_info = ', triton=True' if self.use_triton else ''
        log_info = ', log_grad=True' if self.log_grad and self.gate_mode == 'omniware' else ''
        return (f'd_model={self.d_model}, n_phase={self.n_phase}, '
                f'hidden_dim={self.hidden_dim}, gate_mode={self.gate_mode}'
                f'{triton_info}{log_info}')


# =============================================================================
# CONVENIENCE ALIASES
# =============================================================================

# For backward compatibility with old imports
ResonantFFN_Content = lambda **kw: ResonantFFN(gate_mode='content', **kw)
ResonantFFN_TimeAware = lambda **kw: ResonantFFN(gate_mode='time', **kw)
ResonantFFN_ParallelGate = lambda **kw: ResonantFFN(gate_mode='parallel', **kw)
ResonantFFN_Omniware = lambda **kw: ResonantFFN(gate_mode='omniware', **kw)


__all__ = [
    'ResonantFFN',
    'ResonantFFN_Content',
    'ResonantFFN_TimeAware',
    'ResonantFFN_ParallelGate',
    'ResonantFFN_Omniware',
]
