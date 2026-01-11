"""
Holographic Block - Transformer Block with Attention and FFN

This module provides the HolographicBlock which combines:
- Holographic or Pure Interference attention
- Resonant FFN with configurable gating mode
- Pre-norm architecture

Usage:
    from rin.block import HolographicBlock
    
    block = HolographicBlock(
        d_model=512,
        n_heads=8,
        gate_mode='omniware',
        use_triton=True,
    )
    
    out_real, out_imag = block(x_real, x_imag)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

from .attention import HolographicAttention, PureInterferenceAttention
from .ffn import ResonantFFN


class HolographicBlock(nn.Module):
    """
    Full transformer block with holographic attention and resonant FFN.
    
    Combines:
    - Holographic or Pure Interference attention
    - Resonant FFN with configurable gating mode
    - Pre-norm architecture
    
    FFN Gate Modes (propagated to ResonantFFN):
    - 'content': Original content-only gating (no position awareness)
    - 'time': Time/position-aware gating (RoPE-style)
    - 'parallel': Multiplicative time × content gating
    - 'omniware': Unified time × content theta (most expressive, default)
    
    For 'omniware' mode, log_grad (default True) enables logarithmic gradient
    scaling which compresses large gradients from fast frequencies while
    preserving small gradients from slow frequencies (~40-500x ratio reduction).
    
    Args:
        d_model: Model dimension (for both real and imag streams)
        n_heads: Number of attention heads
        n_phase: Number of phase features for attention (default: 8 * n_heads)
        expansion: FFN expansion factor
        dropout: Dropout rate
        causal: Whether to use causal masking
        max_seq_len: Maximum sequence length for position encoding
        use_pure_interference: If True, use PureInterferenceAttention
        gate_mode: FFN gating mode ('content', 'time', 'parallel', 'omniware')
        use_triton: Whether to use Triton kernels for FFN
        log_grad: Enable log gradient scaling for 'omniware' mode
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_phase: int = None,
        expansion: int = 4,
        dropout: float = 0.0,
        causal: bool = True,
        max_seq_len: int = 8192,
        use_pure_interference: bool = False,
        gate_mode: str = 'omniware',
        use_triton: bool = True,
        log_grad: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.gate_mode = gate_mode
        self.use_triton = use_triton
        self.log_grad = log_grad
        
        # Pre-attention norm
        self.norm1 = nn.LayerNorm(d_model)
        
        # Attention
        if use_pure_interference:
            self.attn = PureInterferenceAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_phase=n_phase,
                dropout=dropout,
                causal=causal,
                max_seq_len=max_seq_len,
            )
        else:
            self.attn = HolographicAttention(
                d_model=d_model,
                n_heads=n_heads,
                n_phase=n_phase,
                dropout=dropout,
                causal=causal,
                max_seq_len=max_seq_len,
            )
        
        # Pre-FFN norm
        self.norm2 = nn.LayerNorm(d_model)
        
        # Resonant FFN with unified entry point
        # Note: FFN uses d_model for n_phase (input stream dimension)
        self.ffn = ResonantFFN(
            d_model=d_model,
            n_phase=d_model,  # FFN operates on d_model-dim phase stream
            expansion=expansion,
            gate_mode=gate_mode,
            use_triton=use_triton,
            log_grad=log_grad,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
    
    def forward(
        self,
        x_real: torch.Tensor,
        x_imag: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the block.
        
        Args:
            x_real: Content stream (B, L, d_model)
            x_imag: Phase stream (B, L, d_model)
            mask: Optional attention mask
            
        Returns:
            out_real: Processed content stream (B, L, d_model)
            out_imag: Processed phase stream (B, L, d_model)
        """
        # Pre-norm attention
        normed = self.norm1(x_real)
        x_real, x_imag = self.attn(normed, x_imag, mask)
        
        # Pre-norm FFN
        normed = self.norm2(x_real)
        x_real, x_imag = self.ffn(normed, x_imag)
        
        return x_real, x_imag
    
    def extra_repr(self) -> str:
        return f'd_model={self.d_model}, gate_mode={self.gate_mode}'


__all__ = ['HolographicBlock']
