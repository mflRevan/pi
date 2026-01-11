"""
Holographic Transformer (RIN) - V7 Modular Architecture

A neural network architecture based on complex-valued representations with:
    - Clean separation of concerns: Real = CONTENT, Imaginary = TIMING
    - Resonant FFN with TRUE interference gating
    - Holographic Attention with content + phase blending
    - Modular, maintainable code structure

Core Components:
    - ffn.py:           ResonantFFN (unified entry point for all FFN variants)
    - attention.py:     HolographicAttention, PureInterferenceAttention
    - block.py:         HolographicBlock (attention + FFN)
    - transformer.py:   HolographicTransformer, SwiGLUTransformer (baseline)
    - kernels/          Triton kernels for acceleration

FFN Gate Modes:
    - 'content': Original content-only gating
    - 'time': Position-aware gating (RoPE-style)
    - 'parallel': Multiplicative time × content gating
    - 'omniware': Unified time × content (recommended, default)

Triton Acceleration:
    When available, Triton kernels provide <2x overhead vs SwiGLU baseline.
    Automatically enabled for 'omniware' mode on CUDA.

Usage:
    from rin import HolographicTransformer, ResonantFFN, HolographicAttention
    
    # Full model
    model = HolographicTransformer(
        vocab_size=50257,
        d_model=512,
        n_layers=6,
        gate_mode='omniware',
        use_triton=True,
    )
    
    # Components
    ffn = ResonantFFN(d_model=512, gate_mode='omniware')
    attn = HolographicAttention(d_model=512, n_heads=8)
"""

import math

# Constants
PHI = (1 + math.sqrt(5)) / 2  # Golden ratio
GOLDEN_RATIO = PHI

# =============================================================================
# PRIMARY EXPORTS (New Modular Architecture)
# =============================================================================

# FFN - Single entry point
from .ffn import (
    ResonantFFN,
    ResonantFFN_Content,
    ResonantFFN_TimeAware,
    ResonantFFN_ParallelGate,
    ResonantFFN_Omniware,
)

# Attention
from .attention import (
    AttentionConfig,
    PureInterferenceAttention,
    HolographicAttention,
    create_position_phases,
)

# Block
from .block import HolographicBlock

# Transformer
from .transformer import (
    SwiGLUTransformer,
    HolographicTransformer,
)

# Utilities
from .utils import (
    create_inv_freq,
    create_pos_freqs,
    compute_energy_scale,
)

# =============================================================================
# TRITON KERNELS (Optional)
# =============================================================================

TRITON_AVAILABLE = False
try:
    import triton
    TRITON_AVAILABLE = True
    
    from .kernels import (
        # FFN kernels (V2 recommended)
        omniware_ffn_gate_forward_v2,
        OmniwareFFNGateFunctionV2,
        TritonOmniwareFFN,
        
        # Attention kernels
        fused_phase_projection,
        interference_scores,
        
        # Utilities
        get_cos_sin_lut,
    )
except ImportError:
    pass

# =============================================================================
# LEGACY EXPORTS (Backward Compatibility)
# =============================================================================
# These imports from optimized.py are deprecated but kept for compatibility

try:
    from .optimized import (
        PureInterferenceAttention as _LegacyPureInterferenceAttention,
        HolographicAttention as _LegacyHolographicAttention,
        ResonantFFN as _LegacyResonantFFN,
        HolographicBlock as _LegacyHolographicBlock,
        HolographicTransformer as _LegacyHolographicTransformer,
    )
except ImportError:
    pass


__version__ = "7.0.0"  # V7: Modular Architecture

__all__ = [
    # Constants
    'PHI',
    'GOLDEN_RATIO',
    'TRITON_AVAILABLE',
    
    # FFN (primary)
    'ResonantFFN',
    'ResonantFFN_Content',
    'ResonantFFN_TimeAware',
    'ResonantFFN_ParallelGate',
    'ResonantFFN_Omniware',
    
    # Attention (primary)
    'AttentionConfig',
    'PureInterferenceAttention',
    'HolographicAttention',
    
    # Block & Transformer (primary)
    'HolographicBlock',
    'HolographicTransformer',
    'SwiGLUTransformer',
    
    # Utilities
    'create_position_phases',
    'create_inv_freq',
    'create_pos_freqs',
    'compute_energy_scale',
]

# Conditional Triton exports
if TRITON_AVAILABLE:
    __all__.extend([
        'omniware_ffn_gate_forward_v2',
        'OmniwareFFNGateFunctionV2',
        'TritonOmniwareFFN',
        'fused_phase_projection',
        'interference_scores',
        'get_cos_sin_lut',
    ])

__all__ = [
    # Constants
    "PHI",
    "GOLDEN_RATIO",
    "TRITON_AVAILABLE",
    
    # Core: Resonant Layer (clean implementation)
    "RMSNorm",
    "ResonantLayer",
    "ResonantBlock",
    
    # Attention
    "ComplexAttention",
    "SDPAComplexAttention", 
    "FullComplexAttention",
    "StandardAttention",
    "get_attention",
    
    # Optimized Holographic/Interference
    "PureInterferenceAttention",
    "HolographicAttention",
    "ResonantFFN",
    "HolographicBlock",
    "HolographicTransformer",
    
    # Model
    "ComplexEmbedding",
    "ComplexPositionalEncoding",
    "ComplexTransformerBlock",
    "ComplexResonantTransformer",
    "StandardTransformer",
    
    # Utils
    "wrap_time_periodic",
    "count_parameters",
    
    # Config
    "ModelConfig",
    "TrainingConfig",
    "DataConfig",
]
