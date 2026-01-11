"""
Triton Kernels for Holographic Transformer

This package contains optimized Triton kernels for:
- FFN gate computation (Resonant, Omniware)
- Attention (Phase projection, Interference scores)

Usage:
    from rin.kernels import (
        # FFN kernels
        omniware_ffn_gate_forward_v2,
        TritonOmniwareFFN,
        
        # Attention kernels  
        fused_phase_projection,
        interference_scores,
    )
"""

from .ffn import (
    # V2 optimized kernels (recommended)
    omniware_ffn_gate_forward_v2,
    OmniwareFFNGateFunctionV2,
    TritonOmniwareFFN,
    
    # Legacy V1 kernels (for reference only)
    resonant_ffn_gate_forward,
    omniware_ffn_gate_forward,
    omniware_ffn_gate_forward_with_grad,
)

from .attention import (
    fused_phase_projection,
    interference_scores,
)

from .utils import (
    get_cos_sin_lut,
    LUT_SIZE,
    LUT_SCALE,
    TRITON_AVAILABLE,
)

__all__ = [
    # V2 FFN (recommended)
    'omniware_ffn_gate_forward_v2',
    'OmniwareFFNGateFunctionV2',
    'TritonOmniwareFFN',
    
    # Legacy FFN
    'resonant_ffn_gate_forward',
    'omniware_ffn_gate_forward',
    'omniware_ffn_gate_forward_with_grad',
    
    # Attention
    'fused_phase_projection',
    'interference_scores',
    
    # Utilities
    'get_cos_sin_lut',
    'LUT_SIZE',
    'LUT_SCALE',
    'TRITON_AVAILABLE',
]
