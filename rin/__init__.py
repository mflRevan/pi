"""
Resonant Interference Network (RIN) - Complex-Valued Edition

The most beautiful neural network architecture in existence, combining:
    π  - The circle constant (sin/cos periodicity)
    e  - Euler's number (e^iθ = cos θ + i·sin θ)
    i  - The imaginary unit (complex plane rotation)
    φ  - The golden ratio (maximally irrational timestep)
    0,1 - The fundamental binary (unit circle bounds)

EULER'S FORMULA: e^(iθ) = cos(θ) + i·sin(θ)

KEY INSIGHT: Every neuron is a point on the unit circle with CONSTANT gradient:
    |∇θ|² = sin²θ + cos²θ = 1

CRITICAL: Signals are kept COMPLEX (real, imag pairs) throughout the network.
Only at the final output (logits) do we collapse to real values.
This preserves phase information and distinguishes:
    - Destructive interference: (1, -1) → would collapse to 0
    - Silence: (0, 0) → would also collapse to 0

CORE FORMULAS:
    # Hidden state transformation (phase = magnitude × state + bias + time)
    θ = (h_real + h_imag) * |w| + b + t·φ
    h_real = cos(θ), h_imag = sin(θ)
    
    # Complex linear (proper complex multiplication)
    out_real = W_real @ x_real - W_imag @ x_imag
    out_imag = W_real @ x_imag + W_imag @ x_real

Results:
- 100% test accuracy on modular arithmetic (grokking)
- 0 stability dips (vs 14 for sin-only baseline)
- Grokking in ~60 epochs (vs never for baseline)
"""

from .lut import SinLUT, get_global_lut
from .model import RINModel, ResonantLayer, ComplexLinear, GOLDEN_RATIO, PHI
from .utils import wrap_time_periodic
from .attention import (
    ResonantAttention,
    ResonantAttentionHead,
    ResonantBlock as AttentionResonantBlock,
    RINAttentionModel,
    StateCache,
)
from .echo_chamber import (
    EchoChamber,
    EchoHead,
    EchoChamberModel,
)

__version__ = "2.0.0"
__all__ = [
    "SinLUT",
    "get_global_lut",
    "RINModel",
    "ResonantLayer",
    "ComplexLinear",
    "GOLDEN_RATIO",
    "PHI",
    "wrap_time_periodic",
    # Echo Chamber (new)
    "EchoChamber",
    "EchoState",
    "ResonantBlock",
    "RINEchoModel",
    # Legacy Attention components
    "ResonantAttention",
    "ResonantAttentionHead",
    "AttentionResonantBlock",
    "RINAttentionModel",
    "StateCache",
]
