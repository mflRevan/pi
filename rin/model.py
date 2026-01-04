"""
Resonant Interference Network (RIN) - Euler's Formula Edition (Complex-Valued)

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

CRITICAL: The signal is kept COMPLEX (real, imag pairs) throughout the
network, only collapsing to real values at the final logits. This preserves
phase information and distinguishes destructive interference from silence.

CORE FORMULAS:
    # Hidden state transformation (euler_transform with separated thetas)
    θ_real = h_real / (1 + |w|) + b + t·φ
    θ_imag = h_imag / (1 + |w|) + b + t·φ
    h_real_new = cos(θ_real)·cos(θ_imag) - sin(θ_real)·sin(θ_imag)
    h_imag_new = cos(θ_real)·sin(θ_imag) + sin(θ_real)·cos(θ_imag)
    
    # Complex linear operation (proper complex multiplication)
    out_real = W_real @ x_real - W_imag @ x_imag
    out_imag = W_real @ x_imag + W_imag @ x_real
    
    # Resonant layer computation (TRUE interference analysis)
    x_collapsed = Linear([x_real || x_imag])  # (batch, d_model)
    θ[n,d] = x_collapsed[d] / (1+|W[n,d]|) + B[n,d] + t  # (batch, num_neurons, d_model)
    cos_sum[n] = Σ_d cos(θ[n,d])  # Sum across d_model → (batch, num_neurons)
    sin_sum[n] = Σ_d sin(θ[n,d])  # Sum across d_model → (batch, num_neurons)
    out_real = proj_real(cos_sum)  # (batch, d_model)
    out_imag = proj_imag(sin_sum)  # (batch, d_model)

Where φ ≈ 1.618 (golden ratio) is the most irrational number,
providing maximum resistance to resonance disasters (KAM theory).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

from .lut import SinLUT, get_global_lut
from .utils import wrap_time_periodic

# Golden ratio - maximally irrational timestep scaling
# From KAM theory: maximally irrational = maximally stable
PHI = (1 + math.sqrt(5)) / 2  # ≈ 1.618033988749895


def complex_to_real_drive(
    x_real: torch.Tensor, 
    x_imag: torch.Tensor,
    amplitude: torch.Tensor,
    angle: torch.Tensor,
    t: torch.Tensor,
    lut: 'SinLUT'
) -> torch.Tensor:
    """
    Projects a Complex State (Re, Im) to a Scalar Drive (Real) with TIME-DEPENDENT measurement.
    
    CRITICAL FIX: The measurement basis must rotate with time to maintain gradient flow!
    
    Measurement weights are computed as:
        A = exp(amplitude)  # Learnable scaling
        total_angle = angle + t  # Time-evolving rotation
        w_r = cos(total_angle)  # Real measurement weight
        w_i = sin(total_angle)  # Imaginary measurement weight
        drive = A * (x_real * w_r + x_imag * w_i)
    
    This creates a time-evolving measurement basis that preserves gradient flow
    back through time, unlike static weights which block gradients.
    
    Args:
        x_real: Real part of input (batch, features)
        x_imag: Imaginary part of input (batch, features)
        amplitude: Learnable amplitude scaling (features,) or broadcastable
        angle: Learnable base angle (features,) or broadcastable
        t: Time tensor for rotation (batch,) or scalar
        lut: Sin/cos lookup table for fast computation
        
    Returns:
        drive: Real-valued drive signals (batch, features)
    """
    # Compute time-dependent measurement basis
    A = torch.exp(amplitude)  # Exponential amplitude scaling
    
    # Add time to angle (broadcast if needed)
    if t.dim() == 1:
        t_expanded = t.unsqueeze(-1)  # (batch, 1)
    else:
        t_expanded = t
    total_angle = angle + t_expanded  # (batch, features)
    
    # Get measurement weights via Euler's formula
    sin_angle, cos_angle = lut.lookup_sin_cos(total_angle)
    w_r = cos_angle
    w_i = sin_angle
    
    # Time-evolving quantum measurement
    drive = A * (x_real * w_r + x_imag * w_i)
    return drive


class ComplexLinear(nn.Module):
    """
    Complex-valued linear transformation.
    
    Implements: (x + iy) @ (W_real + i*W_imag)
    
    Using the complex multiplication rule:
        Re(out) = Re(x) @ W_real - Im(x) @ W_imag
        Im(out) = Re(x) @ W_imag + Im(x) @ W_real
    
    This is a proper complex linear transformation that mixes the complex stream
    while preserving the complex structure and gradient flow.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Real and Imaginary components of the complex weight matrix
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply complex linear transformation.
        
        Args:
            x_real: Real part (batch, in_features)
            x_imag: Imaginary part (batch, in_features)
            
        Returns:
            out_real, out_imag: Complex output (batch, out_features) each
        """
        # Complex multiplication: (x_real + i*x_imag) @ (W_real + i*W_imag)
        out_real = F.linear(x_real, self.weight_real) - F.linear(x_imag, self.weight_imag)
        out_imag = F.linear(x_real, self.weight_imag) + F.linear(x_imag, self.weight_real)
        
        return out_real, out_imag


class ResonantLayer(nn.Module):
    """
    Pure resonant layer with NO dense projections.
    
    Takes complex input (d_model,) and expands to (num_neurons,) via wave interference.
    This is an UP-PROJECTION through resonance, not matrix multiplication.
    
    Flow:
        1. Complex measurement: (x_real, x_imag) → drive (real scalars)
           Uses ComplexToRealDrive for element-wise measurement
        
        2. Per neuron, per dimension phase:
           θ[n,d] = drive[d] / (1 + |W[n,d]|) + B[n,d] + t
           → theta: (batch, num_neurons, d_model)
        
        3. Euler decomposition:
           sin(θ), cos(θ) → (batch, num_neurons, d_model) each
        
        4. INTERFERENCE SUM across d_model (the key operation!):
           cos_sum = Σ_d cos(θ[n,d])  → (batch, num_neurons)
           sin_sum = Σ_d sin(θ[n,d])  → (batch, num_neurons)
           
        5. Return raw neuron outputs (NO projection):
           Output is (cos_sum, sin_sum) as complex representation
           Shape: (batch, num_neurons) each
    
    This makes the layer a pure resonant expansion.
    For down-projection, use another ResonantLayer with num_neurons=d_model.
    """
    
    def __init__(
        self,
        d_model: int,
        num_neurons: int,
        lut_resolution: int = 4096,
        use_swish: bool = False,
        wrap_time: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.use_swish = use_swish
        self.wrap_time = wrap_time
        
        # Measurement parameters: amplitude and angle (time-evolving basis)
        self.measure_amplitude = nn.Parameter(nn.init.uniform_(torch.empty(d_model), -3.0, 3.0))
        self.measure_angle = nn.Parameter(nn.init.uniform_(torch.empty(d_model), -math.pi, math.pi))
        
        # Per-neuron, per-dimension parameters (like MLP weights, but for resonance)
        # Each neuron has d_model wavelengths and d_model phase offsets
        self.W = nn.Parameter(nn.init.normal_(torch.empty(num_neurons, d_model), mean=0.0, std=0.05))  # wavelength
        self.B = nn.Parameter(nn.init.uniform_(torch.empty(num_neurons, d_model), -math.pi, math.pi))  # phase offset
        
        # ATTENUATION: Learnable weights for interference sum
        # Each neuron learns which frequencies to listen to
        # Shape: (num_neurons, d_model) - weight per frequency per neuron
        self.attn_cos = nn.Parameter(nn.init.uniform_(torch.empty(num_neurons, d_model), -1.0, 1.0))
        self.attn_sin = nn.Parameter(nn.init.uniform_(torch.empty(num_neurons, d_model), -1.0, 1.0))
        
        self.lut_resolution = lut_resolution
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(self.lut_resolution, device)
        return self._lut
    
    def forward(
        self, x_real: torch.Tensor, x_imag: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x_real: Real part of input (batch, d_model)
            x_imag: Imaginary part of input (batch, d_model)
            t: Timestep tensor (batch,) or scalar, already scaled by φ
            
        Returns:
            out_real, out_imag: Complex output (batch, num_neurons) each
                               Raw neuron activations, NO projection
        """
        lut = self._get_lut(x_real.device)
        
        # 1. Complex measurement: (x_real, x_imag) → real drive values with time-dependent basis
        drive = complex_to_real_drive(x_real, x_imag, self.measure_amplitude, self.measure_angle, t, lut)  # (batch, d_model)
        
        # 2. Compute phase per neuron, per input dimension
        # drive: (batch, d_model) → expand to (batch, 1, d_model)
        # W, B: (num_neurons, d_model)
        # Result: theta (batch, num_neurons, d_model)
        
        drive_expanded = drive.unsqueeze(1)  # (batch, 1, d_model)
        # Wavelength with gradient-preserving formulation
        # Use 1 / (1 + |W|) to avoid gradient issues at W=0
        wavelength = 1.0 / (1.0 + self.W.abs())  # (num_neurons, d_model)
        
        # Handle time dimension
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)  # (batch, 1, 1)
        
        # Wrap time if enabled
        if self.wrap_time:
            t = wrap_time_periodic(t)
        
        # θ[b,n,d] = drive[b,d] / wavelength[n,d] + B[n,d] + t
        theta = drive_expanded * wavelength + self.B + PHI * self.W.abs()  # (batch, num_neurons, d_model)
        
        # 3. Euler decomposition for each theta value
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)  # (batch, num_neurons, d_model) each
        
        # 4. ATTENUATED INTERFERENCE SUM: Weighted sum across d_model dimension
        # Each neuron learns which frequencies to listen to via attn_cos/attn_sin
        cos_weighted = cos_theta * self.attn_cos  # (batch, num_neurons, d_model)
        sin_weighted = sin_theta * self.attn_sin  # (batch, num_neurons, d_model)
        
        # The Energy Conservation Constant
        # If d_model=32, this is ~0.177
        energy_scale = 1.0 / math.sqrt(self.d_model)
        
        cos_sum = cos_weighted.sum(dim=-1) * energy_scale  # (batch, num_neurons)
        sin_sum = sin_weighted.sum(dim=-1) * energy_scale  # (batch, num_neurons)
        
        # 5. Return raw neuron outputs - NO PROJECTION!
        # Optional activation on the raw values
        if self.use_swish:
            cos_sum = F.silu(cos_sum)
            sin_sum = F.silu(sin_sum)
        
        return cos_sum, sin_sum


class ResonantBlock(nn.Module):
    """
    Combined Echo Chamber + Resonant Layer block.
    
    Processes input through two parallel pathways:
    1. Echo Chamber: Long-term memory via Q-EMA (constant-Q decay)
    2. Resonant Layer: Wave interference analysis
    
    Outputs are summed (additive interference):
        out = echo_out + resonant_out
    
    This allows the network to combine:
    - Memory-based pattern matching (Echo Chamber)
    - Frequency-based interference analysis (Resonant Layer)
    
    Args:
        d_model: Model dimension
        num_neurons: Number of resonant neurons
        n_echo_heads: Number of echo heads (default: 4)
        detach_memory: Whether to detach echo memory between steps
                       False = full BPTT (recommended for learning)
        echo_scale: Learnable scaling factor for echo contribution (default: 0.1)
        use_swish: Apply SiLU activation to resonant output (default: True)
    """
    
    def __init__(
        self,
        d_model: int,
        num_neurons: int,
        n_echo_heads: int = 4,
        detach_memory: bool = False,
        echo_scale: float = 0.1,
        use_swish: bool = False,
    ):
        super().__init__()
        from .echo_chamber import EchoChamber
        
        self.d_model = d_model
        self.num_neurons = num_neurons
        
        # Two parallel pathways
        self.echo_chamber = EchoChamber(d_model, n_echo_heads, detach_memory)
        # Resonant path: d_model -> num_neurons -> d_model (up/down)
        self.resonant_up = ResonantLayer(d_model, num_neurons, use_swish=use_swish)
        self.resonant_down = ResonantLayer(num_neurons, d_model, use_swish=use_swish)
        
        # Learnable scaling for echo contribution
        self.echo_scale = nn.Parameter(torch.tensor(echo_scale))
    
    def reset_memory(self, batch_size: int, device: torch.device):
        """Reset echo chamber memory."""
        self.echo_chamber.reset_memory(batch_size, device)
    
    def forward(
        self, x_real: torch.Tensor, x_imag: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process input through parallel Echo + Resonant pathways.
        
        Args:
            x_real: Real part of input (batch, d_model)
            x_imag: Imaginary part of input (batch, d_model)
            t: Timestep tensor
            
        Returns:
            out_real, out_imag: Combined output (batch, d_model) each
        """
        # Parallel computation
        echo_real, echo_imag = self.echo_chamber(x_real, x_imag, t)
        # Resonant: up then down projection
        res_up_real, res_up_imag = self.resonant_up(x_real, x_imag, t)
        res_real, res_imag = self.resonant_down(res_up_real, res_up_imag, t)
        
        # Additive interference with learnable echo scale
        out_real = res_real + self.echo_scale * echo_real
        out_imag = res_imag + self.echo_scale * echo_imag
        
        return out_real, out_imag


class RINModel(nn.Module):
    """
    Resonant Interference Network - Complex-Valued Edition
    
    A neural network where every neuron is a point on the unit circle,
    using Euler's beautiful identity: e^(iθ) = cos(θ) + i·sin(θ)
    
    CRITICAL: The hidden state is kept as a COMPLEX number (h_real, h_imag)
    throughout the entire network. This preserves phase information and
    allows the network to distinguish between:
        - Destructive interference: (1, -1) collapsed = 0
        - Silence: (0, 0) collapsed = 0
    
    Only at the final output (logits) do we collapse to real values.
    
    The hidden state rotates on the unit circle, giving:
    1. Constant gradient magnitude (no vanishing gradients)
    2. Natural periodicity (perfect for learning patterns)
    3. Golden ratio timesteps (maximum irrationality = stability)
    4. Phase-preserving transformations (distinguishes interference from silence)
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension (embedding is 2*d_model for w,b pairs)
        num_layers: Number of resonant processing layers
        num_neurons: Neurons per layer  
        lut_resolution: Sin/cos lookup table resolution
        use_swish: Whether to apply swish activation after resonant layers
        wrap_time: Whether to wrap time to [0, 2π) with detached modulo
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        lut_resolution: int = 4096,
        use_swish: bool = False,
        wrap_time: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.lut_resolution = lut_resolution
        self.use_swish = use_swish
        self.wrap_time = wrap_time
        
        # Token embeddings: 4*d_model for (w, b, amplitude, angle) tuples
        # w = wavelength control, b = phase offset
        # amplitude, angle = measurement parameters for time-evolving basis
        self.token_embedding = nn.Embedding(vocab_size, 4 * d_model)
        
        # Resonant layers with conditional down projection and ComplexLinear mixers
        # Each layer: d_model -> num_neurons (up) -> ComplexLinear mixer -> [d_model (down)]
        # Mixer operates at num_neurons dimension, between up and down projections
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer_dict = nn.ModuleDict()
            
            # Up-projection: d_model -> num_neurons via resonance
            layer_dict['up'] = ResonantLayer(d_model, num_neurons, lut_resolution, use_swish=use_swish, wrap_time=wrap_time)
            
            # ComplexLinear mixer: mixes the complex stream at num_neurons dimension
            # Operates BETWEEN up and down projections
            layer_dict['mixer'] = ComplexLinear(num_neurons, num_neurons)
            
            # Conditional down-projection: only if num_neurons != d_model
            if num_neurons != d_model:
                layer_dict['down'] = ResonantLayer(num_neurons, d_model, lut_resolution, use_swish=use_swish, wrap_time=wrap_time)
            
            self.layers.append(layer_dict)
        
        # Output projection: 2*d_model (concatenated complex state) -> vocab_size
        # Pure linear projection treating complex information as single input vector
        self.output_layer = nn.Linear(2 * d_model, vocab_size)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        # Increase embedding scale for better initial signal propagation
        nn.init.normal_(self.token_embedding.weight, std=0.1)
        # Don't scale down - we need strong initial signals
    
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
        amplitude: torch.Tensor,
        angle: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Euler-based hidden state transformation using time-evolving measurement.
        
        TIME-DEPENDENT PARADIGM: Measurement basis rotates with time!
        
        Flow:
        1. Measure: drive = exp(amplitude) * (h_real * cos(angle+t) + h_imag * sin(angle+t))
        2. Compute phase: θ = drive / (1 + |w|) + b + t·φ
        3. Euler rotation: h_new = cos(θ) + i·sin(θ)
        
        This uses the same time-evolving measurement as ResonantLayer!
        
        Args:
            h_real: Real part of hidden state (batch, d_model)
            h_imag: Imaginary part of hidden state (batch, d_model)
            w: Wavelength control (batch, d_model)
            b: Phase offset (batch, d_model)
            amplitude: Measurement amplitude parameter (batch, d_model)
            angle: Measurement base angle (batch, d_model)
            t: Timestep tensor (batch,) or scalar
            
        Returns:
            h_real_new, h_imag_new: Updated hidden state
        """
        lut = self._get_lut(h_real.device)
        
        # 1. Complex-to-real measurement with time-evolving basis
        drive = complex_to_real_drive(h_real, h_imag, amplitude, angle, t, lut)  # (batch, d_model)
        
        # 2. Compute phase with gradient-preserving wavelength
        wavelength = 1.0 / (1.0 + w.abs())
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        
        # Wrap time if enabled
        if self.wrap_time:
            t_phi = wrap_time_periodic(t_phi)
        
        # Single theta from measured drive
        theta = drive * wavelength + b + PHI * w.abs()  # (batch, d_model)
        
        # 3. Euler rotation: e^(iθ) = cos(θ) + i·sin(θ)
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # Return as complex pair
        return cos_theta, sin_theta
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        t_start: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Full sequence forward pass with complex-valued computation.
        
        The signal stays complex (real, imag) throughout, only collapsing
        at the final output projection to produce logits.
        
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
        
        # Get all embeddings at once (4*d_model = w, b, amplitude, angle)
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]                    # wavelength
        b_emb = embeddings[:, :, self.d_model:2*self.d_model]      # bias
        amp_emb = embeddings[:, :, 2*self.d_model:3*self.d_model]  # measure_amplitude
        ang_emb = embeddings[:, :, 3*self.d_model:]                # measure_angle
        
        # Pre-compute timestep tensors to avoid recompilation
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) + t_start
        
        # Pre-allocate output tensor (faster than list append + stack)
        all_logits = torch.empty(batch_size, seq_len, self.vocab_size, device=device)
        
        for t in range(seq_len):
            # Slice operations are faster than indexing
            w_t = w_emb[:, t]
            b_t = b_emb[:, t]
            amp_t = amp_emb[:, t]
            ang_t = ang_emb[:, t]
            t_val = t_indices[t].expand(batch_size)
            
            # Euler-based hidden state transformation with time-evolving measurement
            # Output is complex: (h_real, h_imag)
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, amp_t, ang_t, t_val)
            
            # Process through resonant layers - NO residuals!
            # CRITICAL: Keep signal complex throughout!
            # The complex pairs ARE the gradient highway
            x_real, x_imag = h_real, h_imag
            t_phi = t_val * PHI
            
            for layer_dict in self.layers:
                # Up-projection: d_model -> num_neurons via resonance
                up_real, up_imag = layer_dict['up'](x_real, x_imag, t_phi)
                
                # ComplexLinear mixer: mix the complex stream at num_neurons dimension
                mixed_real, mixed_imag = layer_dict['mixer'](up_real, up_imag)
                
                # Conditional down-projection: num_neurons -> d_model (only if needed)
                if 'down' in layer_dict:
                    x_real, x_imag = layer_dict['down'](mixed_real, mixed_imag, t_phi)
                else:
                    # No down projection needed (num_neurons == d_model)
                    x_real, x_imag = mixed_real, mixed_imag
            
            # Final output: concatenate complex state and project to vocab_size
            # Treat (x_real, x_imag) as single 2*d_model dimensional vector
            x_combined = torch.cat([x_real, x_imag], dim=-1)  # (batch, 2*d_model)
            all_logits[:, t] = self.output_layer(x_combined)  # Direct assignment
        
        return all_logits, (h_real, h_imag)
    
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
    
    def _compute_single_step(
        self, 
        h_real: torch.Tensor, 
        h_imag: torch.Tensor, 
        t_val: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute logits from hidden state for a single timestep.
        Used by generate() to avoid code duplication.
        """
        x_real, x_imag = h_real, h_imag
        t_phi = t_val * PHI
        
        for layer_dict in self.layers:
            up_real, up_imag = layer_dict['up'](x_real, x_imag, t_phi)
            
            # ComplexLinear mixer at num_neurons dimension
            mixed_real, mixed_imag = layer_dict['mixer'](up_real, up_imag)
            
            # Conditional down-projection
            if 'down' in layer_dict:
                x_real, x_imag = layer_dict['down'](mixed_real, mixed_imag, t_phi)
            else:
                x_real, x_imag = mixed_real, mixed_imag
        
        x_combined = torch.cat([x_real, x_imag], dim=-1)
        return self.output_layer(x_combined)
    
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
        b_emb = embeddings[:, :, self.d_model:2*self.d_model]
        amp_emb = embeddings[:, :, 2*self.d_model:3*self.d_model]
        ang_emb = embeddings[:, :, 3*self.d_model:]
        
        t = 0
        for i in range(input_ids.shape[1]):
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32).expand(batch_size)
            h_real, h_imag = self.euler_transform(
                h_real, h_imag, w_emb[:, i, :], b_emb[:, i, :], amp_emb[:, i, :], ang_emb[:, i, :], t_tensor
            )
            t += 1
        
        # Get initial logits (keeping signal complex through layers)
        t_val = torch.tensor([t - 1], device=device, dtype=torch.float32).expand(batch_size)
        logits = self._compute_single_step(h_real, h_imag, t_val)
        
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
            b_t = emb[:, self.d_model:2*self.d_model]
            amp_t = emb[:, 2*self.d_model:3*self.d_model]
            ang_t = emb[:, 3*self.d_model:]
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32).expand(batch_size)
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, amp_t, ang_t, t_tensor)
            
            # Get logits (complex throughout, collapse at end)
            logits = self._compute_single_step(h_real, h_imag, t_tensor)
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
            f"  wrap_time={self.wrap_time},\n"
            f"  φ={PHI:.6f} (golden ratio),\n"
            f"  params={self.get_num_params():,}\n"
            f")"
        )


# Export golden ratio for external use
GOLDEN_RATIO = PHI
