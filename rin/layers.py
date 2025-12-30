"""
Resonant Interference Network Layers

Core neural network layers implementing sin-based neurons with:
- LUT-accelerated forward pass
- Custom STDP-like backward pass for frequency (omega) parameters
- Standard gradient descent for offset (bias) parameters

Key insight: Instead of ∆ω = error * t * cos(...), we use:
    ∆ω = error * (PhaseError mod π)
    
This removes the time dependency and prevents gradient explosion,
making the model capable of infinite sequence generalization.
"""

import torch
import torch.nn as nn
from torch.autograd import Function
import math
from typing import Optional, Tuple

from .lut import SinLUT, get_global_lut


class SinNeuronFunction(Function):
    """
    Custom autograd function for sin-based neurons.
    
    Forward: output = sum(sin(w * x + b + t))
    
    Backward:
        - For b (offset): standard gradient = error * cos(w*x + b + t)
        - For w (frequency): STDP-like = error * (phase_error mod π)
        - For x (input): standard chain rule for embedding gradients
    
    The phase_error approach removes time dependency from omega gradients,
    preventing explosion for long sequences.
    """
    
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,           # Input: (batch, seq_len, embed_dim)
        w: torch.Tensor,           # Frequency weights: (embed_dim, num_neurons)
        b: torch.Tensor,           # Phase offsets: (embed_dim, num_neurons)
        t: torch.Tensor,           # Timesteps: (seq_len,) or (batch, seq_len)
        lut: SinLUT,               # Sin lookup table
        target_phase: float = 0.0  # Target phase for STDP (peak=0, valley=π)
    ) -> torch.Tensor:
        """
        Forward pass: compute sin(w*x + b + t) using LUT, then sum over embed_dim.
        
        Args:
            x: Input embeddings (batch, seq_len, embed_dim)
            w: Frequency matrix (embed_dim, num_neurons)
            b: Offset matrix (embed_dim, num_neurons)
            t: Timestep values (seq_len,) - represents position in sequence
            lut: Sine lookup table
            target_phase: Target phase for STDP learning (0 = peak, π = valley)
            
        Returns:
            Output: (batch, seq_len, num_neurons)
        """
        batch_size, seq_len, embed_dim = x.shape
        num_neurons = w.shape[1]
        
        # Expand t to match dimensions: (batch, seq_len, 1, 1)
        if t.dim() == 1:
            t_expanded = t.view(1, seq_len, 1, 1).expand(batch_size, -1, -1, -1)
        else:
            t_expanded = t.view(batch_size, seq_len, 1, 1)
        
        # Reshape x for broadcasting: (batch, seq_len, embed_dim, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Compute phase: w*x + b + t
        # w: (embed_dim, num_neurons) -> (1, 1, embed_dim, num_neurons)
        # b: (embed_dim, num_neurons) -> (1, 1, embed_dim, num_neurons)
        w_expanded = w.unsqueeze(0).unsqueeze(0)
        b_expanded = b.unsqueeze(0).unsqueeze(0)
        
        # phase: (batch, seq_len, embed_dim, num_neurons)
        phase = w_expanded * x_expanded + b_expanded + t_expanded
        
        # Look up sin values using LUT
        sin_values = lut.lookup_sin(phase)
        
        # Sum over embed_dim to get neuron outputs: (batch, seq_len, num_neurons)
        output = sin_values.sum(dim=2)
        
        # Save for backward
        ctx.save_for_backward(x, w, b, t, phase)
        ctx.lut = lut
        ctx.target_phase = target_phase
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Custom backward pass implementing STDP-like learning for omega (w).
        
        For offset b: ∂L/∂b = grad_output * cos(phase)
        For frequency w: ∂L/∂w = grad_output * (phase_error mod π)  # STDP-like
        For input x: ∂L/∂x = grad_output * w * cos(phase)  # For embedding learning
        
        The STDP-like update for w:
        - Computes phase error as (current_phase - target_phase) mod π
        - This bounds the gradient and removes time dependency
        - Encourages phases to align with target (peak or valley)
        """
        x, w, b, t, phase = ctx.saved_tensors
        lut = ctx.lut
        target_phase = ctx.target_phase
        
        batch_size, seq_len, embed_dim = x.shape
        num_neurons = w.shape[1]
        
        # Get cos values for gradients
        cos_values = lut.lookup_cos(phase)  # (batch, seq_len, embed_dim, num_neurons)
        
        # grad_output: (batch, seq_len, num_neurons)
        # Expand to match phase dimensions
        grad_expanded = grad_output.unsqueeze(2)  # (batch, seq_len, 1, num_neurons)
        
        # ============ Gradient for b (offset) - Standard backprop ============
        # ∂sin(w*x+b+t)/∂b = cos(w*x+b+t)
        grad_b_full = grad_expanded * cos_values  # (batch, seq_len, embed_dim, num_neurons)
        grad_b = grad_b_full.sum(dim=(0, 1))  # Sum over batch and sequence
        
        # ============ Gradient for w (frequency) - STDP-like ============
        # Instead of ∂sin(w*x+b+t)/∂w = x * cos(...) * (includes t implicitly)
        # We use: phase_error = (phase - target_phase) mod π
        # This removes time dependency and bounds the gradient
        
        # Compute phase error (how far from target phase)
        phase_error = torch.fmod(phase - target_phase, math.pi)
        
        # Handle sign: we want to push toward target
        # If phase_error > π/2, we're closer to the opposite peak, adjust accordingly
        phase_error = torch.where(
            phase_error > math.pi / 2,
            phase_error - math.pi,
            phase_error
        )
        phase_error = torch.where(
            phase_error < -math.pi / 2,
            phase_error + math.pi,
            phase_error
        )
        
        # STDP-like gradient: bounded by π/2, no time explosion
        # The sign of phase_error indicates direction to push omega
        grad_w_full = grad_expanded * phase_error  # (batch, seq_len, embed_dim, num_neurons)
        grad_w = grad_w_full.sum(dim=(0, 1))  # Sum over batch and sequence
        
        # ============ Gradient for x (input/embeddings) - Standard chain rule ============
        # ∂sin(w*x+b+t)/∂x = w * cos(w*x+b+t)
        # We need this to flow gradients back to the embedding layer
        w_expanded = w.unsqueeze(0).unsqueeze(0)  # (1, 1, embed_dim, num_neurons)
        grad_x_full = grad_expanded * w_expanded * cos_values  # (batch, seq_len, embed_dim, num_neurons)
        grad_x = grad_x_full.sum(dim=3)  # Sum over neurons: (batch, seq_len, embed_dim)
        
        # ============ No gradient for t (timestep) and lut ============
        # t is not a learnable parameter
        # lut is a lookup table, not differentiable in the traditional sense
        
        return grad_x, grad_w, grad_b, None, None, None


class SinLayer(nn.Module):
    """
    A layer of sin-based neurons that compute sin(w*x + b + t).
    
    Each neuron takes the full embedding vector and produces a scalar output
    via the sum of sin-weighted inputs. This replaces the traditional dot-product
    with harmonic resonance.
    
    Args:
        input_dim: Dimension of input embeddings
        num_neurons: Number of neurons (output dimension)
        lut_resolution: Resolution of the sin LUT (default: 512)
        target_phase: Target phase for STDP learning (default: 0 = peak)
        init_freq_scale: Scale for frequency initialization (default: 0.1)
        init_offset_scale: Scale for offset initialization (default: 2π)
    """
    
    def __init__(
        self,
        input_dim: int,
        num_neurons: int,
        lut_resolution: int = 512,
        target_phase: float = 0.0,
        init_freq_scale: float = 0.1,
        init_offset_scale: float = 2 * math.pi,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.target_phase = target_phase
        self.lut_resolution = lut_resolution
        
        # Frequency matrix (omega): how fast each input dimension oscillates per neuron
        # Initialize small so initial phases are dominated by offsets
        self.w = nn.Parameter(
            torch.randn(input_dim, num_neurons) * init_freq_scale
        )
        
        # Offset matrix (bias): initial phase for each input-neuron pair
        # Initialize uniformly across [0, 2π) for diversity
        self.b = nn.Parameter(
            torch.rand(input_dim, num_neurons) * init_offset_scale
        )
        
        # LUT will be created/fetched during forward
        self._lut: Optional[SinLUT] = None
        
    def _get_lut(self, device: torch.device) -> SinLUT:
        """Get or create the LUT on the correct device."""
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(
                resolution=self.lut_resolution, 
                device=device
            )
        return self._lut
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the sin layer.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            t: Timestep tensor (seq_len,) or None (will use 0, 1, 2, ...)
            
        Returns:
            Output tensor (batch, seq_len, num_neurons)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        # Create timesteps if not provided
        if t is None:
            t = torch.arange(seq_len, dtype=x.dtype, device=device)
        
        # Get LUT
        lut = self._get_lut(device)
        
        # Apply custom function
        output = SinNeuronFunction.apply(
            x, self.w, self.b, t, lut, self.target_phase
        )
        
        return output
    
    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, num_neurons={self.num_neurons}, "
            f"lut_resolution={self.lut_resolution}, target_phase={self.target_phase:.3f}"
        )


class ResonantBlock(nn.Module):
    """
    A resonant processing block combining SinLayer with normalization and optional nonlinearity.
    
    This block processes embeddings through sin-based resonance and optionally
    applies layer normalization and a gating mechanism for stability.
    
    Architecture:
        x -> SinLayer -> LayerNorm -> (optional gate) -> output
    
    Args:
        input_dim: Input embedding dimension
        hidden_dim: Number of neurons in sin layer
        output_dim: Output dimension (if None, uses hidden_dim)
        use_layer_norm: Whether to apply layer normalization
        use_gate: Whether to apply learned gating (dendritic-inspired)
        lut_resolution: Sin LUT resolution
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,
        use_layer_norm: bool = True,
        use_gate: bool = False,
        lut_resolution: int = 512,
    ):
        super().__init__()
        
        output_dim = output_dim or hidden_dim
        
        self.sin_layer = SinLayer(
            input_dim=input_dim,
            num_neurons=hidden_dim,
            lut_resolution=lut_resolution,
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        
        # Optional projection if dimensions differ
        if hidden_dim != output_dim:
            self.projection = nn.Linear(hidden_dim, output_dim, bias=False)
        else:
            self.projection = nn.Identity()
        
        # Optional dendritic-style gating
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Sigmoid()
            )
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the resonant block.
        
        Args:
            x: Input (batch, seq_len, input_dim)
            t: Optional timesteps
            
        Returns:
            Output (batch, seq_len, output_dim)
        """
        # Sin resonance
        h = self.sin_layer(x, t)
        
        # Optional gating (dendritic synchrony)
        if self.use_gate:
            gate_values = self.gate(x)
            h = h * gate_values
        
        # Normalization
        h = self.layer_norm(h)
        
        # Projection
        output = self.projection(h)
        
        return output


class MultiResonantLayer(nn.Module):
    """
    Multiple resonant blocks with different target phases for richer representations.
    
    This creates a "harmonic ensemble" where different heads resonate to different
    phase targets (peaks, valleys, etc.), capturing diverse temporal patterns.
    
    Args:
        input_dim: Input dimension
        num_heads: Number of resonant heads
        neurons_per_head: Neurons per head
        output_dim: Final output dimension
    """
    
    def __init__(
        self,
        input_dim: int,
        num_heads: int = 4,
        neurons_per_head: int = 64,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        
        output_dim = output_dim or input_dim
        
        # Create heads with different target phases
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            # Distribute target phases across [0, 2π)
            target_phase = (2 * math.pi * i) / num_heads
            head = SinLayer(
                input_dim=input_dim,
                num_neurons=neurons_per_head,
                target_phase=target_phase,
            )
            self.heads.append(head)
        
        # Combine heads
        total_neurons = num_heads * neurons_per_head
        self.combine = nn.Linear(total_neurons, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self, 
        x: torch.Tensor, 
        t: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward through all heads and combine."""
        head_outputs = [head(x, t) for head in self.heads]
        combined = torch.cat(head_outputs, dim=-1)
        output = self.combine(combined)
        output = self.layer_norm(output)
        return output
