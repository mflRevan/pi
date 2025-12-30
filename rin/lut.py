"""
Sine Look-Up Table (LUT) Module

Provides a precomputed sine table for efficient forward pass computation.
Uses linear interpolation for sub-index precision.

The LUT is a universal, persistent tensor that maps phase values to sine outputs
without expensive trigonometric computation at runtime.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class SinLUT(nn.Module):
    """
    Sine Look-Up Table with linear interpolation.
    
    Precomputes sin values across [0, 2π) and provides fast lookup
    with phase wrapping (modulo 2π) and linear interpolation.
    
    Args:
        resolution: Number of discrete samples in the LUT (default 512)
        device: Device to store the LUT on
        dtype: Data type for the LUT (default float32)
    """
    
    def __init__(
        self, 
        resolution: int = 512, 
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        self.resolution = resolution
        self.dtype = dtype
        
        # Precompute the sine table: sin values for phases [0, 2π)
        # We store resolution samples evenly distributed
        phases = torch.linspace(0, 2 * math.pi, resolution + 1, dtype=dtype)[:-1]
        sin_table = torch.sin(phases)
        
        # Also precompute cosine for backward pass
        cos_table = torch.cos(phases)
        
        # Register as buffers (persistent, but not parameters)
        self.register_buffer('sin_table', sin_table)
        self.register_buffer('cos_table', cos_table)
        
        # Precompute scaling factor: how many LUT indices per radian
        self.register_buffer(
            'scale', 
            torch.tensor(resolution / (2 * math.pi), dtype=dtype)
        )
        self.register_buffer(
            'two_pi',
            torch.tensor(2 * math.pi, dtype=dtype)
        )
        
    def _phase_to_index(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert phase values to LUT indices with interpolation weights.
        
        Args:
            phase: Phase values (any shape), will be wrapped to [0, 2π)
            
        Returns:
            idx_low: Lower index for interpolation
            frac: Fractional part for interpolation weight
        """
        # Wrap phase to [0, 2π) using modulo
        wrapped_phase = torch.fmod(phase, self.two_pi)
        # Handle negative phases
        wrapped_phase = torch.where(
            wrapped_phase < 0, 
            wrapped_phase + self.two_pi, 
            wrapped_phase
        )
        
        # Convert to continuous index
        continuous_idx = wrapped_phase * self.scale
        
        # Get integer indices
        idx_low = continuous_idx.long()
        idx_high = (idx_low + 1) % self.resolution
        
        # Get fractional part for interpolation
        frac = continuous_idx - idx_low.float()
        
        # Clamp idx_low to valid range (should already be, but safety first)
        idx_low = idx_low % self.resolution
        
        return idx_low, idx_high, frac
    
    def lookup_sin(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Look up sin values with linear interpolation.
        
        Args:
            phase: Phase values in radians (any shape)
            
        Returns:
            Interpolated sin values
        """
        idx_low, idx_high, frac = self._phase_to_index(phase)
        
        # Linear interpolation: (1-frac) * low + frac * high
        sin_low = self.sin_table[idx_low]
        sin_high = self.sin_table[idx_high]
        
        return sin_low + frac * (sin_high - sin_low)
    
    def lookup_cos(self, phase: torch.Tensor) -> torch.Tensor:
        """
        Look up cos values with linear interpolation.
        Used in backward pass for gradients.
        
        Args:
            phase: Phase values in radians (any shape)
            
        Returns:
            Interpolated cos values
        """
        idx_low, idx_high, frac = self._phase_to_index(phase)
        
        cos_low = self.cos_table[idx_low]
        cos_high = self.cos_table[idx_high]
        
        return cos_low + frac * (cos_high - cos_low)
    
    def lookup_sin_cos(self, phase: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Look up both sin and cos values efficiently (single index computation).
        
        Args:
            phase: Phase values in radians (any shape)
            
        Returns:
            Tuple of (sin_values, cos_values)
        """
        idx_low, idx_high, frac = self._phase_to_index(phase)
        
        sin_low = self.sin_table[idx_low]
        sin_high = self.sin_table[idx_high]
        sin_val = sin_low + frac * (sin_high - sin_low)
        
        cos_low = self.cos_table[idx_low]
        cos_high = self.cos_table[idx_high]
        cos_val = cos_low + frac * (cos_high - cos_low)
        
        return sin_val, cos_val
    
    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        """Forward pass: lookup sin values."""
        return self.lookup_sin(phase)
    
    def __repr__(self) -> str:
        return f"SinLUT(resolution={self.resolution})"


# Global LUT instance for efficiency (created on first use)
_GLOBAL_LUT: Optional[SinLUT] = None
_GLOBAL_LUT_RESOLUTION: int = 512


def get_global_lut(
    resolution: int = 512, 
    device: Optional[torch.device] = None,
    force_recreate: bool = False
) -> SinLUT:
    """
    Get or create the global SinLUT instance.
    
    This provides a universal, persistent LUT that can be shared across
    all SinLayer instances for memory efficiency.
    
    Args:
        resolution: LUT resolution (only used on first creation)
        device: Target device
        force_recreate: Force recreation of the LUT
        
    Returns:
        Global SinLUT instance
    """
    global _GLOBAL_LUT, _GLOBAL_LUT_RESOLUTION
    
    if _GLOBAL_LUT is None or force_recreate or resolution != _GLOBAL_LUT_RESOLUTION:
        _GLOBAL_LUT = SinLUT(resolution=resolution)
        _GLOBAL_LUT_RESOLUTION = resolution
        
    if device is not None and _GLOBAL_LUT.sin_table.device != device:
        _GLOBAL_LUT = _GLOBAL_LUT.to(device)
        
    return _GLOBAL_LUT


def reset_global_lut():
    """Reset the global LUT (useful for testing or device changes)."""
    global _GLOBAL_LUT
    _GLOBAL_LUT = None
