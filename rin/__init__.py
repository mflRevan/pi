"""
Resonant Interference Network (RIN)

A novel neural network architecture that treats information as continuous 
interference fields of complex waveforms, using harmonic resonance instead 
of attention mechanisms.

Key concepts:
- Sin-based neurons with LUT acceleration
- STDP-like phase-error learning for frequency weights
- O(n) computational efficiency
- Infinite sequence generalization
"""

from .lut import SinLUT
from .layers import SinLayer, ResonantBlock
from .model import RINModel

__version__ = "0.1.0"
__all__ = ["SinLUT", "SinLayer", "ResonantBlock", "RINModel"]
