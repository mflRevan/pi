"""
RIN Echo Model - Resonant Interference Network with Echo Chambers

A novel architecture combining:
1. Euler-based hidden state transformation (phase rotation)
2. Echo Chambers with EMA value states and Flash Attention
3. Parallel resonant + attention paths (not sequential like Transformer)

ARCHITECTURE:
    Input -> Embedding -> [Echo Chamber || ResonantLayer] x N -> Output
    
    Each block processes:
    - Resonant path: ComplexLinear -> ResonantLayer (Euler transform)
    - Echo path: ComplexLinear -> Attention with EMA values
    - Combination: multiplicative gating or additive interference
    
This preserves the beautiful properties of RIN while adding
attention-based memory through the Echo Chamber mechanism.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List

from .lut import get_global_lut
from .utils import wrap_time_periodic
from .model import ComplexLinear, PHI
from .echo_chamber import EchoChamber, ResonantBlock, EchoState


class RINEchoModel(nn.Module):
    """
    Resonant Interference Network with Echo Chambers.
    
    Combines Euler-based recurrent state evolution with Echo Chamber
    attention for enhanced memory and retrieval capabilities.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_layers: Number of ResonantBlocks
        num_neurons: Neurons per resonant layer
        n_heads: Attention heads per echo chamber
        alpha: EMA decay rate for echo states
        learnable_alpha: Whether alpha is learnable per head
        output_mode: 'complex_linear' or 'resonant' for echo output
        gate_mode: 'multiplicative', 'additive', or 'glu'
        lut_resolution: Sin/cos lookup table resolution
        use_swish: Whether to use swish activation
        wrap_time: Whether to wrap time to [0, 2π)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        n_heads: int = 8,
        alpha: float = 0.1,
        learnable_alpha: bool = True,
        output_mode: str = 'complex_linear',
        gate_mode: str = 'multiplicative',
        lut_resolution: int = 4096,
        use_swish: bool = True,
        wrap_time: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.n_heads = n_heads
        self.alpha = alpha
        self.output_mode = output_mode
        self.gate_mode = gate_mode
        self.lut_resolution = lut_resolution
        self.use_swish = use_swish
        self.wrap_time = wrap_time
        
        # Token embedding: 2*d_model for (w, b) pairs
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Resonant Blocks (Echo Chamber + ResonantLayer)
        self.blocks = nn.ModuleList([
            ResonantBlock(
                d_model=d_model,
                num_neurons=num_neurons,
                n_heads=n_heads,
                alpha=alpha,
                learnable_alpha=learnable_alpha,
                output_mode=output_mode,
                lut_resolution=lut_resolution,
                use_swish=use_swish,
                wrap_time=wrap_time,
                dropout=dropout,
                gate_mode=gate_mode,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection: complex -> logits
        self.output_proj = ComplexLinear(d_model, vocab_size, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.5)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(self.lut_resolution, device)
        return self._lut
    
    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize complex hidden state."""
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        return h_real, h_imag
    
    def init_echo_states(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Initialize echo states for all blocks."""
        return [
            block.echo_chamber.init_state(batch_size, device)
            for block in self.blocks
        ]
    
    def euler_transform(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Euler-based hidden state transformation with separated thetas.
        
        θ_real = h_real / (1 + |w|) + b + t·φ
        θ_imag = h_imag / (1 + |w|) + b + t·φ
        
        Complex multiplication:
        h_real_new = cos(θ_real)·cos(θ_imag) - sin(θ_real)·sin(θ_imag)
        h_imag_new = cos(θ_real)·sin(θ_imag) + sin(θ_real)·cos(θ_imag)
        """
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() == 1 else t * PHI
        
        if self.wrap_time:
            t_phi = wrap_time_periodic(t_phi)
        
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        
        return h_real_new, h_imag_new
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        echo_states: Optional[List[torch.Tensor]] = None,
        t_start: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass with Echo Chambers.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            hidden: Initial (h_real, h_imag) or None
            echo_states: Initial echo states or None
            t_start: Starting timestep
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            final_hidden: (h_real, h_imag)
            final_echo_states: Updated echo states
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if hidden is None:
            h_real, h_imag = self.init_hidden(batch_size, device)
        else:
            h_real, h_imag = hidden
        
        if echo_states is None:
            echo_states = self.init_echo_states(batch_size, device)
        
        # Get embeddings
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) + t_start
        
        all_logits = []
        
        for t in range(seq_len):
            w_t = w_emb[:, t, :]
            b_t = b_emb[:, t, :]
            t_val = t_indices[t].expand(batch_size)
            
            # Euler transform on hidden state
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            # Process through Resonant Blocks
            x_real, x_imag = h_real, h_imag
            new_echo_states = []
            
            for i, block in enumerate(self.blocks):
                (x_real, x_imag), new_state = block.forward_step(
                    x_real, x_imag, echo_states[i], t_val * PHI
                )
                new_echo_states.append(new_state)
            
            echo_states = new_echo_states
            
            # Output projection
            logits_real, logits_imag = self.output_proj(x_real, x_imag)
            logits = logits_real + logits_imag
            all_logits.append(logits)
        
        return torch.stack(all_logits, dim=1), (h_real, h_imag), echo_states
    
    def forward_sequence(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        t_start: int = 0,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass using Flash Attention for sequences.
        
        More efficient for training on full sequences.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if hidden is None:
            h_real, h_imag = self.init_hidden(batch_size, device)
        else:
            h_real, h_imag = hidden
        
        # Get embeddings
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) + t_start
        
        # Process Euler transform for each step (maintains recurrence)
        h_states_real = []
        h_states_imag = []
        
        for t in range(seq_len):
            w_t = w_emb[:, t, :]
            b_t = b_emb[:, t, :]
            t_val = t_indices[t].expand(batch_size)
            
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            h_states_real.append(h_real)
            h_states_imag.append(h_imag)
        
        # Stack for sequence processing
        x_real = torch.stack(h_states_real, dim=1)  # (batch, seq_len, d_model)
        x_imag = torch.stack(h_states_imag, dim=1)
        
        # Process through blocks with Flash Attention
        for block in self.blocks:
            x_real, x_imag = block.forward_sequence(x_real, x_imag, t_start=t_start)
        
        # Output projection
        x_real_flat = x_real.view(-1, self.d_model)
        x_imag_flat = x_imag.view(-1, self.d_model)
        logits_real, logits_imag = self.output_proj(x_real_flat, x_imag_flat)
        logits = (logits_real + logits_imag).view(batch_size, seq_len, self.vocab_size)
        
        return logits, (h_real, h_imag)
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        t_start: int = 0,
        use_flash: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Compute next-token prediction loss."""
        if use_flash:
            logits, hidden = self.forward_sequence(input_ids, hidden, t_start)
        else:
            logits, hidden, _ = self.forward(input_ids, hidden, t_start=t_start)
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )
        
        return loss, logits, hidden
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def get_echo_alphas(self) -> List[torch.Tensor]:
        """Get the learned alpha values from each echo chamber."""
        return [block.echo_chamber.alpha for block in self.blocks]
    
    def __repr__(self) -> str:
        return (
            f"RINEchoModel(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self.d_model},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_neurons={self.num_neurons},\n"
            f"  n_heads={self.n_heads},\n"
            f"  alpha={self.alpha},\n"
            f"  output_mode='{self.output_mode}',\n"
            f"  gate_mode='{self.gate_mode}',\n"
            f"  wrap_time={self.wrap_time},\n"
            f"  φ={PHI:.6f} (golden ratio),\n"
            f"  params={self.get_num_params():,}\n"
            f")"
        )
