"""
Utility Functions for Resonant Interference Network

Includes:
- Visualization tools
- Analysis functions
- Helper utilities
- Time wrapping for periodic networks
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Dict, List, Tuple, Any


TWO_PI = 2 * math.pi


def wrap_time_periodic(t: torch.Tensor) -> torch.Tensor:
    """
    Wrap time to [0, 2π) with detached modulo for gradient flow.
    
    The key insight: since we use periodic functions (sin/cos via Euler's formula),
    t mod 2π gives identical forward values AND gradients as unwrapped t.
    
    By detaching the modulo operation:
    - Forward: sees t mod 2π (bounded, no numerical issues at large t)
    - Backward: sees original t (gradients flow through as if no modulo)
    
    This works because:
    - sin(t) = sin(t mod 2π) and d/dt sin(t) = cos(t) = cos(t mod 2π)
    - The periodic structure means "direction" is relative, not absolute
    - Golden ratio timescale avoids standing waves, so model learns relative flow
    
    Args:
        t: Time tensor (any shape), already scaled by φ (golden ratio)
        
    Returns:
        Time wrapped to [0, 2π) with gradient passthrough
    """
    # Compute wrapped value
    t_wrapped = torch.fmod(t, TWO_PI)
    # Handle negative values
    t_wrapped = torch.where(t_wrapped < 0, t_wrapped + TWO_PI, t_wrapped)
    
    # Detach the modulo: gradient flows through t, not through the modulo op
    # This is: t_wrapped_detached = t + (t_wrapped - t).detach()
    # Forward: returns t_wrapped
    # Backward: gradient of t_wrapped_detached w.r.t. t is 1 (identity)
    return t + (t_wrapped - t).detach()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters in a model by component.
    
    Returns:
        Dictionary with parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Count by module type
    by_type = {}
    for name, module in model.named_modules():
        module_params = sum(p.numel() for p in module.parameters(recurse=False))
        module_type = type(module).__name__
        by_type[module_type] = by_type.get(module_type, 0) + module_params
    
    return {
        "total": total,
        "trainable": trainable,
        "by_type": by_type,
    }


def analyze_sin_layer_phases(
    layer,
    sample_input: torch.Tensor,
    timestep: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Analyze the phase distribution in a SinLayer.
    
    Args:
        layer: SinLayer instance
        sample_input: Sample input tensor (batch, seq_len, embed_dim)
        timestep: Which timestep to analyze
        
    Returns:
        Dictionary with phase analysis
    """
    with torch.no_grad():
        w = layer.w  # (embed_dim, num_neurons)
        b = layer.b  # (embed_dim, num_neurons)
        
        # Get input for specific timestep
        x = sample_input[:, timestep, :]  # (batch, embed_dim)
        
        # Compute phases
        # x: (batch, embed_dim) -> (batch, embed_dim, 1)
        x_expanded = x.unsqueeze(-1)
        # phase: (batch, embed_dim, num_neurons)
        phase = w.unsqueeze(0) * x_expanded + b.unsqueeze(0) + timestep
        
        # Wrap to [0, 2π)
        wrapped_phase = torch.fmod(phase, 2 * math.pi)
        wrapped_phase = torch.where(wrapped_phase < 0, wrapped_phase + 2 * math.pi, wrapped_phase)
        
        return {
            "raw_phase": phase,
            "wrapped_phase": wrapped_phase,
            "phase_mean": wrapped_phase.mean(),
            "phase_std": wrapped_phase.std(),
            "w_stats": {"mean": w.mean(), "std": w.std(), "min": w.min(), "max": w.max()},
            "b_stats": {"mean": b.mean(), "std": b.std(), "min": b.min(), "max": b.max()},
        }


def visualize_lut_accuracy(lut, num_samples: int = 1000) -> Dict[str, float]:
    """
    Compare LUT accuracy against torch.sin.
    
    Args:
        lut: SinLUT instance
        num_samples: Number of random samples to test
        
    Returns:
        Dictionary with accuracy metrics
    """
    with torch.no_grad():
        # Random phases
        phases = torch.rand(num_samples) * 4 * math.pi - 2 * math.pi  # [-2π, 2π]
        
        # LUT values
        lut_sin = lut.lookup_sin(phases)
        lut_cos = lut.lookup_cos(phases)
        
        # True values
        true_sin = torch.sin(phases)
        true_cos = torch.cos(phases)
        
        # Errors
        sin_error = (lut_sin - true_sin).abs()
        cos_error = (lut_cos - true_cos).abs()
        
        return {
            "sin_max_error": sin_error.max().item(),
            "sin_mean_error": sin_error.mean().item(),
            "cos_max_error": cos_error.max().item(),
            "cos_mean_error": cos_error.mean().item(),
            "lut_resolution": lut.resolution,
        }


def estimate_memory_usage(
    vocab_size: int,
    embed_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_heads: int = 4,
    neurons_per_head: int = 128,
    batch_size: int = 32,
    seq_len: int = 512,
    dtype_bytes: int = 4,  # float32
) -> Dict[str, float]:
    """
    Estimate memory usage for RIN model.
    
    Returns:
        Dictionary with memory estimates in MB
    """
    # Embedding table
    embed_params = vocab_size * embed_dim
    
    # Each SinLayer: w (embed_dim * neurons) + b (embed_dim * neurons)
    neurons_total = num_heads * neurons_per_head * num_layers
    sin_params = 2 * embed_dim * neurons_total
    
    # Combine layer (for multi-head)
    combine_params = num_heads * neurons_per_head * embed_dim * num_layers
    
    # Layer norms
    ln_params = 2 * embed_dim * (num_layers + 1)  # 2 for gamma, beta
    
    # Output projection (tied with embedding)
    output_params = 0  # Tied
    
    total_params = embed_params + sin_params + combine_params + ln_params
    
    # Activation memory (forward pass)
    # Input embeddings
    embed_act = batch_size * seq_len * embed_dim
    # Sin layer intermediates (phase computation)
    # phase: (batch, seq_len, embed_dim, num_neurons)
    phase_act = batch_size * seq_len * embed_dim * neurons_total // num_layers
    # Output per layer
    layer_output_act = batch_size * seq_len * embed_dim
    
    total_activation = embed_act + phase_act + layer_output_act * num_layers
    
    return {
        "params_mb": total_params * dtype_bytes / 1e6,
        "activation_mb": total_activation * dtype_bytes / 1e6,
        "total_mb": (total_params + total_activation) * dtype_bytes / 1e6,
        "num_params": total_params,
    }


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_gradient_stats(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """
    Compute gradient statistics for debugging.
    
    Returns:
        Dictionary with gradient stats per parameter
    """
    stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            stats[name] = {
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "min": grad.min().item(),
                "max": grad.max().item(),
                "norm": grad.norm().item(),
            }
    return stats


def analyze_phase_alignment(
    model,
    input_ids: torch.Tensor,
    target_phase: float = 0.0,
) -> Dict[str, float]:
    """
    Analyze how well the model's phases align with target.
    
    This is useful for understanding the STDP-like learning.
    """
    model.eval()
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    
    alignments = []
    
    with torch.no_grad():
        # Get embeddings
        x = model.token_embedding(input_ids)
        seq_len = input_ids.shape[1]
        t = torch.arange(seq_len, dtype=x.dtype, device=device)
        
        # Analyze each layer
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'heads'):
                # Multi-head layer
                for j, head in enumerate(layer.heads):
                    w = head.w
                    b = head.b
                    
                    # Compute phases
                    x_exp = x.unsqueeze(-1)
                    t_exp = t.view(1, -1, 1, 1)
                    phase = w.unsqueeze(0).unsqueeze(0) * x_exp + b.unsqueeze(0).unsqueeze(0) + t_exp
                    
                    # Phase error from target
                    phase_error = torch.fmod(phase - target_phase, math.pi)
                    alignment = 1.0 - torch.abs(phase_error).mean().item() / (math.pi / 2)
                    alignments.append(alignment)
    
    return {
        "mean_alignment": np.mean(alignments),
        "min_alignment": np.min(alignments),
        "max_alignment": np.max(alignments),
        "per_head_alignment": alignments,
    }


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(
        self,
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == "min":
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        
        return self.should_stop


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_model_summary(model: nn.Module, input_shape: Tuple[int, ...] = (1, 128)):
    """Print a summary of the model architecture."""
    print("\n" + "="*60)
    print("MODEL SUMMARY")
    print("="*60)
    
    print(f"\n{model}")
    
    print("\n" + "-"*40)
    print("Parameter counts:")
    stats = count_parameters(model)
    print(f"  Total: {stats['total']:,}")
    print(f"  Trainable: {stats['trainable']:,}")
    print("\n  By module type:")
    for mod_type, count in sorted(stats['by_type'].items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {mod_type}: {count:,}")
    
    print("\n" + "="*60)
