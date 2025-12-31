"""
Resonant Interference Network (RIN) - Euler's Formula Edition

The most beautiful neural network architecture in existence, combining:
    π  - The circle constant (sin/cos periodicity)
    e  - Euler's number (natural exponential)
    i  - The imaginary unit (complex plane)
    φ  - The golden ratio (maximally irrational timestep)
    0,1 - The fundamental binary (unit circle bounds)

EULER'S FORMULA: e^(iθ) = cos(θ) + i·sin(θ)

KEY INSIGHT: By decomposing into real (cos) and imaginary (sin) parts,
every neuron becomes a point on the unit circle with CONSTANT gradient:
    |d/dθ (cos θ)|² + |d/dθ (sin θ)|² = sin²θ + cos²θ = 1

This eliminates vanishing gradients at peaks and valleys - the network
can learn everywhere with equal capability.

CORE FORMULAS:
    # Hidden state transformation
    θ = (h_real + h_imag) / (1 + |w|) + b + t·φ
    h_real_new = cos(θ)
    h_imag_new = sin(θ)
    
    # Resonant layer computation  
    θ = W·x + b + t·φ
    output = W_real @ cos(θ) + W_imag @ sin(θ)

Where φ ≈ 1.618 (golden ratio) is the most irrational number,
providing maximum resistance to resonance disasters (KAM theory).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .lut import SinLUT, get_global_lut

# Golden ratio - maximally irrational timestep scaling
# From KAM theory: maximally irrational = maximally stable
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895


class ResonantLayer(nn.Module):
    """
    Resonant layer using Euler's formula for unit-circle computation.
    
    Each neuron computes a phasor on the complex plane:
        θ = W·x + b + t·φ
        real = cos(θ)  - x-coordinate on unit circle
        imag = sin(θ)  - y-coordinate on unit circle
        
    Separate projections for real and imaginary parts allow the network
    to learn arbitrary phase relationships:
        output = W_real @ real + W_imag @ imag
    
    This is mathematically equivalent to learning magnitude and phase,
    but with gradient magnitude 1 EVERYWHERE on the circle.
    """
    
    def __init__(
        self,
        d_model: int,
        num_neurons: int,
        lut_resolution: int = 4096,
        use_swish: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.use_swish = use_swish
        
        # Input weights: compute phase θ = W·x + b
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_neurons))
        
        # Separate output projections for real (cos) and imaginary (sin) parts
        # This allows learning arbitrary phase relationships
        self.proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        
        self.lut_resolution = lut_resolution
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(self.lut_resolution, device)
        return self._lut
    
    def forward(self, x: torch.Tensor, t: float) -> torch.Tensor:
        """
        Args:
            x: Input (batch, d_model)
            t: Timestep (already scaled by φ)
            
        Returns:
            Output (batch, d_model)
        """
        lut = self._get_lut(x.device)
        
        # Compute phase: θ = W·x + b + t
        theta = x @ self.W.T + self.bias + t
        
        # Euler decomposition: e^(iθ) = cos(θ) + i·sin(θ)
        # Single index computation for both
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # Separate projections - network learns optimal phase relationships
        output = self.proj_real(cos_theta) + self.proj_imag(sin_theta)
        
        # Optional swish activation for non-linearity
        if self.use_swish:
            output = F.silu(output)
        
        return output


class RINModel(nn.Module):
    """
    Resonant Interference Network - Euler's Formula Edition
    
    A neural network where every neuron is a point on the unit circle,
    using Euler's beautiful identity: e^(iθ) = cos(θ) + i·sin(θ)
    
    The hidden state is a complex number (h_real, h_imag) that rotates
    on the unit circle. This gives:
    1. Constant gradient magnitude (no vanishing gradients)
    2. Natural periodicity (perfect for learning patterns)
    3. Golden ratio timesteps (maximum irrationality = stability)
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension (embedding is 2*d_model for w,b pairs)
        num_layers: Number of resonant processing layers
        num_neurons: Neurons per layer  
        lut_resolution: Sin/cos lookup table resolution
        use_swish: Whether to apply swish activation after resonant layers
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        lut_resolution: int = 4096,
        use_swish: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.lut_resolution = lut_resolution
        self.use_swish = use_swish
        
        # Token embeddings: 2*d_model for (w, b) pairs
        # w = wavelength control, b = phase offset
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Resonant layers with Euler computation
        self.layers = nn.ModuleList([
            ResonantLayer(d_model, num_neurons, lut_resolution, use_swish=use_swish)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        # Scale down for stable initial transformations
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.5)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(self.lut_resolution, device)
        return self._lut
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialize hidden state as (real, imag) pair.
        Both start at zero - the origin, ready to rotate onto the unit circle.
        """
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        return h_real, h_imag
    
    def euler_transform(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        t: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Euler-based hidden state transformation.
        
        θ = (h_real + h_imag) / (1 + |w|) + b + t·φ
        h_real_new = cos(θ)
        h_imag_new = sin(θ)
        
        The wavelength (1 + |w|) controls oscillation speed.
        Higher wavelength = slower rotation = more stable patterns.
        """
        lut = self._get_lut(h_real.device)
        
        # Combine previous state for phase computation
        h_combined = h_real + h_imag
        
        # Wavelength formula with golden ratio timestep
        wavelength = 1.0 + w.abs()
        theta = h_combined / wavelength + b + t * PHI
        
        # Euler decomposition onto unit circle
        h_imag_new, h_real_new = lut.lookup_sin_cos(theta)
        
        return h_real_new, h_imag_new
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        t_start: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full sequence forward pass with Euler-based computation.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            hidden: Initial (h_real, h_imag) or None
            t_start: Starting timestep
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            final_hidden: (h_real, h_imag)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if hidden is None:
            h_real, h_imag = self.init_hidden(batch_size, device)
        else:
            h_real, h_imag = hidden
        
        # Get all embeddings at once
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        all_logits = []
        
        for t in range(seq_len):
            w_t = w_emb[:, t, :]
            b_t = b_emb[:, t, :]
            t_val = float(t_start + t)
            
            # Euler-based hidden state transformation
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            # Combine for layer processing
            x = h_real + h_imag
            
            # Process through resonant layers with residual connections
            for layer in self.layers:
                x = x + layer(x, t_val * PHI)
            
            logits = self.output_proj(x)
            all_logits.append(logits)
        
        return torch.stack(all_logits, dim=1), (h_real, h_imag)
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        t_start: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute next-token prediction loss."""
        logits, hidden = self.forward(input_ids, hidden, t_start)
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )
        
        return loss, logits, hidden
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Generate tokens autoregressively."""
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        lut = self._get_lut(device)
        
        h_real, h_imag = self.init_hidden(batch_size, device)
        
        # Process prompt
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t = 0
        for i in range(input_ids.shape[1]):
            h_real, h_imag = self.euler_transform(
                h_real, h_imag, w_emb[:, i, :], b_emb[:, i, :], float(t)
            )
            t += 1
        
        # Get initial logits
        x = h_real + h_imag
        for layer in self.layers:
            x = x + layer(x, float(t - 1) * PHI)
        logits = self.output_proj(x)
        
        # Generate
        generated = input_ids
        for _ in range(max_new_tokens):
            if temperature != 1.0:
                logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            
            # Update hidden state
            emb = self.token_embedding(next_token.squeeze(-1))
            w_t = emb[:, :self.d_model]
            b_t = emb[:, self.d_model:]
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, float(t))
            
            # Get logits
            x = h_real + h_imag
            for layer in self.layers:
                x = x + layer(x, float(t) * PHI)
            logits = self.output_proj(x)
            t += 1
        
        return generated
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return (
            f"RINModel(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self.d_model},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_neurons={self.num_neurons},\n"
            f"  use_swish={self.use_swish},\n"
            f"  φ={PHI:.6f} (golden ratio),\n"
            f"  params={self.get_num_params():,}\n"
            f")"
        )


# Export golden ratio for external use
GOLDEN_RATIO = PHI
