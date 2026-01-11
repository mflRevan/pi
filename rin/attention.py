"""
Holographic Attention - Unified Attention Modules

This module provides attention implementations for the Holographic Transformer:

- PureInterferenceAttention: Phase-only position encoding (fastest)
- HolographicAttention: Content + Phase blended scoring (most expressive)

Usage:
    from rin.attention import HolographicAttention, PureInterferenceAttention
    
    # Pure interference (fastest, phase-only)
    attn = PureInterferenceAttention(d_model=512, n_heads=8)
    
    # Holographic (blended content + phase)
    attn = HolographicAttention(d_model=512, n_heads=8, per_head_alpha=True)
    
    # Forward pass
    out_real, out_imag = attn(x_real, x_imag)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AttentionConfig:
    """Configuration for attention modules."""
    d_model: int = 512
    n_heads: int = 8
    n_phase: int = 64           # Phase features (typically 2x heads)
    dropout: float = 0.0
    causal: bool = True
    max_seq_len: int = 8192
    use_flash: bool = True      # Use Flash Attention when available


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_position_phases(max_len: int, n_phase: int, base: float = 10000.0) -> torch.Tensor:
    """Create position-dependent phase values.
    
    Uses RoPE-style frequency computation for multi-scale position encoding.
    
    Args:
        max_len: Maximum sequence length
        n_phase: Number of phase features
        base: Base for frequency computation (like RoPE)
        
    Returns:
        Tensor of shape (max_len, n_phase) with position phases
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, n_phase, 2).float() / n_phase))
    position = torch.arange(max_len).float()
    sinusoid = torch.einsum('i,j->ij', position, inv_freq)
    # Interleave sin and cos phases for diversity
    pos_phase = torch.zeros(max_len, n_phase)
    pos_phase[:, 0::2] = sinusoid[:, :n_phase // 2]
    pos_phase[:, 1::2] = sinusoid[:, :n_phase // 2]
    return pos_phase


# =============================================================================
# PURE INTERFERENCE ATTENTION
# =============================================================================

class PureInterferenceAttention(nn.Module):
    """
    Pure Interference Attention - Phase-only position encoding.
    
    This is the fastest variant that uses ONLY phase-based scoring:
        score[i,j] = cos(Q_i)·cos(K_j) + sin(Q_i)·sin(K_j)
                   = cos(Q_i - K_j)  [by trig identity]
    
    Position is encoded additively in the phase computation:
        Q_theta = W_q @ x_imag + bias_q + pos[i]
        K_theta = W_k @ x_imag + bias_k + pos[j]
    
    This creates a natural relative position encoding:
        score depends on pos[i] - pos[j]
    
    Key insight: Unlike RoPE which rotates embeddings, we inject position
    ADDITIVELY into the phase space. This is more stable and maintains
    the real/imag separation.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_phase: Number of phase features (default: 8 * n_heads)
        dropout: Attention dropout
        causal: Whether to apply causal masking
        max_seq_len: Maximum sequence length for position encoding
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_phase: int = None,
        dropout: float = 0.0,
        causal: bool = True,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_phase = n_phase if n_phase else n_heads * 8
        assert self.n_phase % n_heads == 0
        self.phase_per_head = self.n_phase // n_heads
        self.causal = causal
        
        self.scale = 1.0 / math.sqrt(self.phase_per_head)
        
        # Phase projections: x_imag -> theta
        # Separate Q and K projections for flexibility
        self.Q_phase = nn.Linear(d_model, self.n_phase, bias=True)
        self.K_phase = nn.Linear(d_model, self.n_phase, bias=True)
        
        # Value projection: x_real -> V
        self.V_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.O_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Positional phases - registered as buffer (not parameter)
        pos_phase = create_position_phases(max_seq_len, self.n_phase)
        self.register_buffer('pos_phase', pos_phase)
        
        # Post-attention normalization
        self.norm = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training dynamics."""
        nn.init.xavier_uniform_(self.Q_phase.weight)
        nn.init.xavier_uniform_(self.K_phase.weight)
        nn.init.xavier_uniform_(self.V_proj.weight)
        nn.init.xavier_uniform_(self.O_proj.weight, gain=1 / math.sqrt(2))
        
        # Initialize biases as random phases in [-π, π]
        nn.init.uniform_(self.Q_phase.bias, -math.pi, math.pi)
        nn.init.uniform_(self.K_phase.bias, -math.pi, math.pi)
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with pure interference attention.
        
        Args:
            x_real: Content stream (B, L, D) - used for values
            x_imag: Phase stream (B, L, D) - used for Q/K phases
            mask: Optional attention mask (B, 1, L, L) or broadcastable
            
        Returns:
            out_real: Attended content (B, L, D)
            out_imag: Unchanged phase stream (B, L, D)
        """
        B, L, D = x_real.shape
        
        # Compute Q/K phases from x_imag + positional phases
        Q_theta = self.Q_phase(x_imag) + self.pos_phase[:L]  # (B, L, n_phase)
        K_theta = self.K_phase(x_imag) + self.pos_phase[:L]  # (B, L, n_phase)
        
        # Reshape for multi-head: (B, L, n_phase) -> (B, H, L, phase_per_head)
        Q_theta = Q_theta.view(B, L, self.n_heads, self.phase_per_head).transpose(1, 2)
        K_theta = K_theta.view(B, L, self.n_heads, self.phase_per_head).transpose(1, 2)
        
        # Compute cos/sin
        Q_cos, Q_sin = torch.cos(Q_theta), torch.sin(Q_theta)
        K_cos, K_sin = torch.cos(K_theta), torch.sin(K_theta)
        
        # Interference scores: cos(Q)·cos(K)^T + sin(Q)·sin(K)^T
        scores = (torch.matmul(Q_cos, K_cos.transpose(-2, -1)) +
                  torch.matmul(Q_sin, K_sin.transpose(-2, -1))) * self.scale
        
        # Apply masks
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x_real.device, dtype=torch.bool), 1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Softmax attention
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Value projection and attention
        V = self.V_proj(x_real).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.O_proj(out)
        
        # Residual + norm
        out_real = self.norm(x_real + self.dropout(out))
        out_imag = x_imag  # Phase stream unchanged
        
        return out_real, out_imag
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, n_heads={self.n_heads}, n_phase={self.n_phase}'


# =============================================================================
# HOLOGRAPHIC ATTENTION
# =============================================================================

class HolographicAttention(nn.Module):
    """
    Holographic Attention - Content + Phase blended scoring.
    
    Combines standard content-based attention with phase interference:
        score = (1-α) * content_score + α * phase_score
        
    Where:
        content_score = Q_real @ K_real^T / √d
        phase_score = cos(Q) @ cos(K)^T + sin(Q) @ sin(K)^T / √p
    
    The blend factor α is learnable, starting at 0.5.
    When per_head_alpha=True (default), each head learns its own α,
    allowing specialization in content vs positional matching.
    
    This variant is more expressive than pure interference but slightly slower.
    Use when you need both content matching AND positional relationships.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        n_phase: Number of phase features (default: 8 * n_heads)
        dropout: Attention dropout
        causal: Whether to apply causal masking
        max_seq_len: Maximum sequence length for position encoding
        per_head_alpha: Whether each head learns its own blend factor
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_phase: int = None,
        dropout: float = 0.0,
        causal: bool = True,
        max_seq_len: int = 8192,
        per_head_alpha: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_phase = n_phase if n_phase else n_heads * 8
        assert self.n_phase % n_heads == 0
        self.phase_per_head = self.n_phase // n_heads
        self.causal = causal
        self.per_head_alpha = per_head_alpha
        
        self.content_scale = 1.0 / math.sqrt(self.head_dim)
        self.phase_scale = 1.0 / math.sqrt(self.phase_per_head)
        
        # Content projections (from x_real)
        self.Q_content = nn.Linear(d_model, d_model, bias=False)
        self.K_content = nn.Linear(d_model, d_model, bias=False)
        self.V_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Phase projections (from x_imag)
        self.Q_phase = nn.Linear(d_model, self.n_phase, bias=True)
        self.K_phase = nn.Linear(d_model, self.n_phase, bias=True)
        
        # Positional phases
        pos_phase = create_position_phases(max_seq_len, self.n_phase)
        self.register_buffer('pos_phase', pos_phase)
        
        # Learnable blend factor (initialized to 0.5 via sigmoid(0))
        # Per-head alpha allows each head to specialize in content vs phase
        if per_head_alpha:
            self.alpha_logit = nn.Parameter(torch.zeros(n_heads, 1, 1))  # (H, 1, 1) for broadcasting
        else:
            self.alpha_logit = nn.Parameter(torch.zeros(1))
        
        # Output
        self.O_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.Q_content.weight)
        nn.init.xavier_uniform_(self.K_content.weight)
        nn.init.xavier_uniform_(self.V_proj.weight)
        nn.init.xavier_uniform_(self.Q_phase.weight)
        nn.init.xavier_uniform_(self.K_phase.weight)
        nn.init.xavier_uniform_(self.O_proj.weight, gain=1 / math.sqrt(2))
        nn.init.uniform_(self.Q_phase.bias, -math.pi, math.pi)
        nn.init.uniform_(self.K_phase.bias, -math.pi, math.pi)
    
    @property
    def alpha(self) -> torch.Tensor:
        """Blend factor between content and phase scores.
        
        Returns:
            If per_head_alpha: tensor of shape (H, 1, 1) for broadcasting over (B, H, L, L)
            Otherwise: scalar tensor
        """
        return torch.sigmoid(self.alpha_logit)
    
    def get_alpha_values(self) -> torch.Tensor:
        """Return alpha values for monitoring (detached)."""
        return self.alpha.detach().squeeze()
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with holographic (blended) attention.
        
        Args:
            x_real: Content stream (B, L, D) - used for Q/K content and values
            x_imag: Phase stream (B, L, D) - used for Q/K phases
            mask: Optional attention mask (B, 1, L, L) or broadcastable
            
        Returns:
            out_real: Attended content (B, L, D)
            out_imag: Unchanged phase stream (B, L, D)
        """
        B, L, D = x_real.shape
        
        # Content scores from x_real
        Q = self.Q_content(x_real).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.K_content(x_real).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.V_proj(x_real).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        
        content_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.content_scale
        
        # Phase scores from x_imag
        Q_theta = self.Q_phase(x_imag) + self.pos_phase[:L]
        K_theta = self.K_phase(x_imag) + self.pos_phase[:L]
        
        Q_theta = Q_theta.view(B, L, self.n_heads, self.phase_per_head).transpose(1, 2)
        K_theta = K_theta.view(B, L, self.n_heads, self.phase_per_head).transpose(1, 2)
        
        phase_scores = (torch.matmul(torch.cos(Q_theta), torch.cos(K_theta).transpose(-2, -1)) +
                       torch.matmul(torch.sin(Q_theta), torch.sin(K_theta).transpose(-2, -1))) * self.phase_scale
        
        # Blend scores
        alpha = self.alpha
        scores = (1 - alpha) * content_scores + alpha * phase_scores
        
        # Masking
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(L, L, device=x_real.device, dtype=torch.bool), 1
            )
            scores = scores.masked_fill(causal_mask, float('-inf'))
        
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        
        # Attention
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).reshape(B, L, D)
        out = self.O_proj(out)
        
        out_real = self.norm(x_real + self.dropout(out))
        out_imag = x_imag
        
        return out_real, out_imag
    
    def extra_repr(self) -> str:
        return (f'd_model={self.d_model}, n_heads={self.n_heads}, '
                f'n_phase={self.n_phase}, per_head_alpha={self.per_head_alpha}')


__all__ = [
    'AttentionConfig',
    'create_position_phases',
    'PureInterferenceAttention',
    'HolographicAttention',
]
