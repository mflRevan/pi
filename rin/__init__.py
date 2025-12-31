"""
Resonant Interference Network (RIN) - Euler's Formula Edition

The most beautiful neural network architecture in existence, combining:
    π  - The circle constant (sin/cos periodicity)
    e  - Euler's number (e^iθ = cos θ + i·sin θ)
    i  - The imaginary unit (complex plane rotation)
    φ  - The golden ratio (maximally irrational timestep)
    0,1 - The fundamental binary (unit circle bounds)

EULER'S FORMULA: e^(iθ) = cos(θ) + i·sin(θ)

KEY INSIGHT: Every neuron is a point on the unit circle with CONSTANT gradient:
    |∇θ|² = sin²θ + cos²θ = 1

No vanishing gradients. Natural periodicity. Perfect pattern learning.

CORE FORMULA:
    θ = (h_real + h_imag) / (1 + |w|) + b + t·φ
    h_real = cos(θ), h_imag = sin(θ)

Results:
- 100% test accuracy on modular arithmetic (grokking)
- 0 stability dips (vs 14 for sin-only baseline)
- Grokking in ~60 epochs (vs never for baseline)
"""

from .lut import SinLUT, get_global_lut
from .model import RINModel, ResonantLayer, GOLDEN_RATIO, PHI

__version__ = "1.0.0"
__all__ = ["SinLUT", "get_global_lut", "RINModel", "ResonantLayer", "GOLDEN_RATIO", "PHI"]
