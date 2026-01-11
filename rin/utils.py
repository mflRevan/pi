"""
Utilities for Holographic Transformer

This module provides centralized utility functions used across the package:
- Position encoding utilities
- Gradient scaling functions
- Configuration helpers

Usage:
    from rin.utils import create_position_phases, create_inv_freq
"""

import torch
import math


def create_position_phases(max_len: int, n_phase: int, base: float = 10000.0) -> torch.Tensor:
    """
    Create position-dependent phase values.
    
    Uses RoPE-style frequency computation for multi-scale position encoding.
    The phases are interleaved (sin and cos together) for diversity.
    
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
    # Interleave phases for diversity
    pos_phase = torch.zeros(max_len, n_phase)
    pos_phase[:, 0::2] = sinusoid[:, :n_phase // 2]
    pos_phase[:, 1::2] = sinusoid[:, :n_phase // 2]
    return pos_phase


def create_inv_freq(n_phase: int, base: float = 10000.0) -> torch.Tensor:
    """
    Create inverse frequency tensor for RoPE-style position encoding.
    
    Args:
        n_phase: Number of phase features (should be even)
        base: Base for frequency computation
        
    Returns:
        Tensor of shape (n_phase,) with inverse frequencies
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, n_phase, 2).float() / n_phase))
    full_inv_freq = torch.zeros(n_phase)
    full_inv_freq[0::2] = inv_freq[:n_phase // 2]
    full_inv_freq[1::2] = inv_freq[:n_phase // 2]
    return full_inv_freq


def create_pos_freqs(max_len: int, n_phase: int, base: float = 10000.0) -> torch.Tensor:
    """
    Create position × frequency matrix for Omniware-style time encoding.
    
    pos_freqs[l, p] = l * inv_freq[p]
    
    Args:
        max_len: Maximum sequence length
        n_phase: Number of phase features
        base: Base for frequency computation
        
    Returns:
        Tensor of shape (max_len, n_phase)
    """
    inv_freq = create_inv_freq(n_phase, base)
    positions = torch.arange(max_len).float()
    return torch.outer(positions, inv_freq)


def compute_energy_scale(n_phase: int) -> float:
    """
    Compute energy normalization factor for interference patterns.
    
    The sum of n_phase cosines is normalized by 1/sqrt(n_phase) to maintain
    stable variance across different n_phase values.
    
    Args:
        n_phase: Number of phase features
        
    Returns:
        Energy scale factor (1/sqrt(n_phase))
    """
    return 1.0 / math.sqrt(n_phase)


__all__ = [
    'create_position_phases',
    'create_inv_freq',
    'create_pos_freqs',
    'compute_energy_scale',
]
