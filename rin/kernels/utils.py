"""
Kernel Utilities - LUT, common helpers
"""

import torch
import math

# Check Triton availability
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


# =============================================================================
# COSINE LUT SETUP
# =============================================================================
# 64K entry LUT for cos, covering [0, 2π] with wrapping
# In Triton, we precompute and store in global memory, then load into SRAM

LUT_SIZE = 65536  # 64K entries
LUT_SCALE = LUT_SIZE / (2 * math.pi)  # Convert radians to LUT index

# Precompute the LUT once (on CPU, then move to GPU)
_COS_LUT = None
_SIN_LUT = None


def get_cos_sin_lut(device: torch.device):
    """
    Get or create cos/sin LUT on the specified device.
    
    The LUT provides fast approximate sin/cos computation for Triton kernels.
    64K entries covering [0, 2π] gives ~0.0001 rad precision.
    
    Args:
        device: Target device for the LUT tensors
        
    Returns:
        Tuple of (cos_lut, sin_lut), each shape (65536,)
    """
    global _COS_LUT, _SIN_LUT
    if _COS_LUT is None or _COS_LUT.device != device:
        angles = torch.arange(LUT_SIZE, dtype=torch.float32) * (2 * math.pi / LUT_SIZE)
        _COS_LUT = torch.cos(angles).to(device)
        _SIN_LUT = torch.sin(angles).to(device)
    return _COS_LUT, _SIN_LUT


def clear_lut_cache():
    """Clear the LUT cache to free GPU memory."""
    global _COS_LUT, _SIN_LUT
    _COS_LUT = None
    _SIN_LUT = None
