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


class ComplexLinear(nn.Module):
    """
    Complex-valued linear layer using proper complex multiplication.
    
    Models the weight as a complex matrix: W = W_real + i·W_imag
    
    Complex multiplication formula:
        (a + bi)(c + di) = (ac - bd) + (ad + bc)i
        
    So for input (x_real, x_imag) and weight (W_real, W_imag):
        out_real = W_real @ x_real - W_imag @ x_imag
        out_imag = W_real @ x_imag + W_imag @ x_real
    """
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Real and imaginary parts of the complex weight matrix
        self.W_real = nn.Linear(in_features, out_features, bias=False)
        self.W_imag = nn.Linear(in_features, out_features, bias=False)
        
        # Optional bias (applied to both real and imag)
        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)
        
        self._init_weights()
    
    def _init_weights(self):
        # Xavier-like initialization scaled for complex
        nn.init.xavier_uniform_(self.W_real.weight, gain=0.5)
        nn.init.xavier_uniform_(self.W_imag.weight, gain=0.5)
    
    def forward(
        self, x_real: torch.Tensor, x_imag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply complex linear transformation.
        
        Args:
            x_real: Real part of input (batch, in_features)
            x_imag: Imaginary part of input (batch, in_features)
            
        Returns:
            out_real, out_imag: Complex output parts (batch, out_features)
        """
        # Complex multiplication: (W_r + iW_i)(x_r + ix_i)
        # Real: W_r·x_r - W_i·x_i
        out_real = self.W_real(x_real) - self.W_imag(x_imag)
        # Imag: W_r·x_i + W_i·x_r
        out_imag = self.W_real(x_imag) + self.W_imag(x_real)
        
        if self.bias_real is not None:
            out_real = out_real + self.bias_real
            out_imag = out_imag + self.bias_imag
        
        return out_real, out_imag


class ResonantLayer(nn.Module):
    """
    True resonant layer with per-neuron, per-dimension wavelength and phase.
    
    Like an MLP, but instead of weighted sums, we do INTERFERENCE analysis.
    
    Each neuron has its own:
        - W: (d_model,) wavelengths - one per input dimension
        - B: (d_model,) phase offsets - one per input dimension
    
    Giving full parameter matrices: W, B both (num_neurons, d_model)
    
    Flow:
        1. Input collapse: [x_real || x_imag] → Linear → x_collapsed (batch, d_model)
        
        2. Per neuron, per dimension phase:
           θ[n,d] = x_collapsed[d] / (1 + |W[n,d]|) + B[n,d] + t
           → theta: (batch, num_neurons, d_model)
        
        3. Euler decomposition:
           sin(θ), cos(θ) → (batch, num_neurons, d_model) each
        
        4. INTERFERENCE SUM across d_model (the key operation!):
           cos_sum = Σ_d cos(θ[n,d])  → (batch, num_neurons) scalar per neuron
           sin_sum = Σ_d sin(θ[n,d])  → (batch, num_neurons) scalar per neuron
           
           This is like a dot product but summing wave interference instead of
           weighted values. Constructive interference → large magnitude.
           Destructive interference → values cancel out.
        
        5. Project back: Linear(num_neurons, d_model) for real and imag separately
    """
    
    def __init__(
        self,
        d_model: int,
        num_neurons: int,
        lut_resolution: int = 4096,
        use_swish: bool = True,
        wrap_time: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.use_swish = use_swish
        self.wrap_time = wrap_time
        
        # Input collapse: complex plane → single vector for phase computation
        self.input_collapse = nn.Linear(2 * d_model, d_model, bias=True)
        
        # Per-neuron, per-dimension parameters (like MLP weights, but for resonance)
        # Each neuron has d_model wavelengths and d_model phase offsets
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)  # wavelength
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))  # phase offset
        
        # ATTENUATION: Learnable weights for interference sum
        # Each neuron learns which frequencies to listen to
        # Shape: (num_neurons, d_model) - weight per frequency per neuron
        self.attn_cos = nn.Parameter(torch.ones(num_neurons, d_model))
        self.attn_sin = nn.Parameter(torch.ones(num_neurons, d_model))
        
        # Output projections: interference scalars → d_model
        self.out_proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.out_proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        
        self.lut_resolution = lut_resolution
        self._lut = None
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_collapse.weight, gain=0.5)
        nn.init.zeros_(self.input_collapse.bias)
        nn.init.xavier_uniform_(self.out_proj_real.weight, gain=0.5)
        nn.init.xavier_uniform_(self.out_proj_imag.weight, gain=0.5)
    
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
            out_real, out_imag: Complex output (batch, d_model) each
        """
        lut = self._get_lut(x_real.device)
        
        # 1. Collapse complex plane into single phase input
        x_combined = torch.cat([x_real, x_imag], dim=-1)  # (batch, 2*d_model)
        x_collapsed = self.input_collapse(x_combined)  # (batch, d_model)
        
        # 2. Compute phase per neuron, per input dimension
        # x_collapsed: (batch, d_model) → expand to (batch, 1, d_model)
        # W, B: (num_neurons, d_model)
        # Result: theta (batch, num_neurons, d_model)
        
        x_expanded = x_collapsed.unsqueeze(1)  # (batch, 1, d_model)
        wavelength = 1.0 + self.W.abs()  # (num_neurons, d_model)
        
        # Handle time dimension
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)  # (batch, 1, 1)
        
        # Wrap time if enabled
        if self.wrap_time:
            t = wrap_time_periodic(t)
        
        # θ[b,n,d] = x[b,d] / wavelength[n,d] + B[n,d] + t
        theta = x_expanded / wavelength + self.B + t  # (batch, num_neurons, d_model)
        
        # 3. Euler decomposition for each theta value
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)  # (batch, num_neurons, d_model) each
        
        # 4. ATTENUATED INTERFERENCE SUM: Weighted sum across d_model dimension
        # Each neuron learns which frequencies to listen to via attn_cos/attn_sin
        # This allows neurons to selectively tune into specific frequency bands
        cos_weighted = cos_theta * self.attn_cos  # (batch, num_neurons, d_model)
        sin_weighted = sin_theta * self.attn_sin  # (batch, num_neurons, d_model)
        
        cos_sum = cos_weighted.sum(dim=-1)  # (batch, num_neurons)
        sin_sum = sin_weighted.sum(dim=-1)  # (batch, num_neurons)
        
        # 5. Project interference scalars back to d_model
        out_real = self.out_proj_real(cos_sum)  # (batch, d_model)
        out_imag = self.out_proj_imag(sin_sum)  # (batch, d_model)
        
        # Optional activation
        if self.use_swish:
            out_real = F.silu(out_real)
            out_imag = F.silu(out_imag)
        
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
        use_swish: bool = True,
        wrap_time: bool = False,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.lut_resolution = lut_resolution
        self.use_swish = use_swish
        self.wrap_time = wrap_time
        
        # Token embeddings: 2*d_model for (w, b) pairs
        # w = wavelength control, b = phase offset
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Resonant layers - now operate on complex signals
        self.layers = nn.ModuleList([
            ResonantLayer(d_model, num_neurons, lut_resolution, use_swish=use_swish, wrap_time=wrap_time)
            for _ in range(num_layers)
        ])
        
        # Output projection: collapses complex signal to real logits
        # Uses ComplexLinear to properly combine real and imag parts
        self.output_proj_complex = ComplexLinear(d_model, vocab_size, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
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
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Euler-based hidden state transformation with proper complex propagation.
        
        CRITICAL FIX: We compute SEPARATE thetas for real and imaginary components,
        then combine via proper complex multiplication. This preserves phase information
        and distinguishes constructive from destructive interference.
        
        θ_real = h_real / (1 + |w|) + b + t·φ
        θ_imag = h_imag / (1 + |w|) + b + t·φ
        
        Then combine via complex multiplication of unit phasors:
        e^(iθ_real) x e^(iθ_imag) = e^(i(θ_real + θ_imag))
        
        But computed through separate gradient paths:
        h_real_new = cos(θ_real)·cos(θ_imag) - sin(θ_real)·sin(θ_imag)
        h_imag_new = cos(θ_real)·sin(θ_imag) + sin(θ_real)·cos(θ_imag)
        
        This is proper complex multiplication: e^(iθ_r) x e^(iθ_i) = e^(i(θ_r + θ_i))
        But computed through separate gradient paths to preserve information.
        
        The wavelength (1 + |w|) controls oscillation speed:
        - Larger |w| = slower rotation = more stable patterns
        
        Args:
            t: Timestep tensor (batch,) or scalar tensor
        """
        lut = self._get_lut(h_real.device)
        
        # Phase formula with SEPARATE thetas - no information collapse!
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        
        # Wrap time to [0, 2π) if enabled (detached modulo for gradient flow)
        if self.wrap_time:
            t_phi = wrap_time_periodic(t_phi)
        
        # Separate theta computation preserves real/imag distinction
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        # Euler decomposition for each component
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        # Complex multiplication: (cos_r + i·sin_r) x (cos_i + i·sin_i)
        # = (cos_r·cos_i - sin_r·sin_i) + i(cos_r·sin_i + sin_r·cos_i)
        # This preserves BOTH gradient paths through h_real and h_imag
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        
        return h_real_new, h_imag_new
    
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
        
        # Get all embeddings at once
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        # Pre-compute timestep tensors to avoid recompilation
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) + t_start
        
        all_logits = []
        
        for t in range(seq_len):
            w_t = w_emb[:, t, :]
            b_t = b_emb[:, t, :]
            t_val = t_indices[t].expand(batch_size)
            
            # Euler-based hidden state transformation
            # Output is complex: (h_real, h_imag)
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            # Process through resonant layers with residual connections
            # CRITICAL: Keep signal complex throughout!
            x_real, x_imag = h_real, h_imag
            t_phi = t_val * PHI
            
            for layer in self.layers:
                # Layer outputs complex (delta_real, delta_imag)
                delta_real, delta_imag = layer(x_real, x_imag, t_phi)
                # Residual connection in complex space
                x_real = x_real + delta_real
                x_imag = x_imag + delta_imag
            
            # Final collapse: complex -> real logits
            # Use complex projection then sum components
            logits_real, logits_imag = self.output_proj_complex(x_real, x_imag)
            # Collapse to real: this is the ONLY place we sum real + imag
            logits = logits_real + logits_imag
            
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
        
        for layer in self.layers:
            delta_real, delta_imag = layer(x_real, x_imag, t_phi)
            x_real = x_real + delta_real
            x_imag = x_imag + delta_imag
        
        logits_real, logits_imag = self.output_proj_complex(x_real, x_imag)
        return logits_real + logits_imag
    
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
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32).expand(batch_size)
            h_real, h_imag = self.euler_transform(
                h_real, h_imag, w_emb[:, i, :], b_emb[:, i, :], t_tensor
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
            b_t = emb[:, self.d_model:]
            t_tensor = torch.tensor([t], device=device, dtype=torch.float32).expand(batch_size)
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_tensor)
            
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
