"""
Resonant Attention Layer - State-Based Interference Attention

A novel attention mechanism that leverages the interference properties of
high-dimensional complex state vectors for self-gating behavior.

KEY INSIGHT: Interference Is The Gate!
=======================================
- Coherent context (strong signal): Sharp softmax → high-magnitude output
- Diffuse context (noise): Flat softmax → random phase cancellation → near-zero

This eliminates the need for explicit gating mechanisms.

ARCHITECTURE:
    ResonantAttention + ResonantLayer form a block (like Transformer)
    
    ResonantAttention:
        - n attention heads dividing d_model
        - Each head: query = euler_transform(input_patch, w_head, b_head, t)
        - Compare queries against cached historical states via dot product
        - Softmax over similarity scores
        - Retrieve weighted sum of full states
        - Sum all head outputs → project → add to stream

STATE CACHE:
    - Caches entire history of states over t
    - Each cached state linked to its input embedding (w, b)
    - Gradient flows through current state only (history detached)
    - Gradient does flow to embedding transformation at t

GRADIENT FLOW (Training mode):
    x_t (current) ← GRADIENT FLOWS
    states[0:t-1] ← GRADIENT FLOWS (learns from attention retrieval!)
    embeddings ← GRADIENT FLOWS (learns from both evolution & attention)
    
    This enables the model to learn "memorable" representations:
    - Tokens can learn to create states that are easily retrieved
    - Credit assignment flows through attention back to earlier tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, NamedTuple
from dataclasses import dataclass

from .lut import get_global_lut

# Golden ratio
PHI = (1 + math.sqrt(5)) / 2


class StateCache:
    """
    State history cache for resonant attention with gradient control.
    
    During training, states RETAIN gradients to allow learning through
    attention retrieval. This enables the model to learn:
    - "Make this token memorable for later retrieval"
    - Credit assignment through the attention mechanism
    
    During inference, states are DETACHED to save memory.
    
    KEY INSIGHT: In Transformers, causal masking creates token-to-token
    gradient paths. In RIN, state evolution IS the causal path, and
    attention retrieval should also provide gradient paths.
    
    Attributes:
        states: List of state tensors (batch, d_model)
        embeddings_w: List of w embeddings (retain grad)
        embeddings_b: List of b embeddings (retain grad)
        max_len: Maximum cache length (None = unlimited)
        training: Whether to preserve gradients (True) or detach (False)
    """
    
    def __init__(self, max_len: Optional[int] = None, training: bool = True):
        self.max_len = max_len
        self.training = training
        self.reset()
    
    def reset(self):
        """Clear the cache."""
        self.states: List[torch.Tensor] = []
        self.embeddings_w: List[torch.Tensor] = []
        self.embeddings_b: List[torch.Tensor] = []
        self._stacked_states: Optional[torch.Tensor] = None
        self._stacked_valid = False
    
    def append(
        self,
        state: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
    ):
        """
        Add a new state to the cache.
        
        Args:
            state: Current state (batch, d_model)
                   - Preserved during training for gradient flow
                   - Detached during inference for memory efficiency
            w: Embedding w component (batch, d_model) - always retains grad
            b: Embedding b component (batch, d_model) - always retains grad
        """
        # During training: preserve gradients for attention retrieval
        # During inference: detach to save memory
        if self.training:
            self.states.append(state)  # Keep computation graph!
        else:
            self.states.append(state.detach())
        
        # Embeddings always retain gradients for learning
        self.embeddings_w.append(w)
        self.embeddings_b.append(b)
        self._stacked_valid = False
        
        # Enforce max length
        if self.max_len is not None and len(self.states) > self.max_len:
            self.states.pop(0)
            self.embeddings_w.pop(0)
            self.embeddings_b.pop(0)
    
    def get_stacked_states(self) -> Optional[torch.Tensor]:
        """
        Get all states as a single tensor.
        
        Returns:
            Stacked states (batch, seq_len, d_model) or None if empty
        """
        if len(self.states) == 0:
            return None
        
        if not self._stacked_valid:
            # Stack along sequence dimension
            self._stacked_states = torch.stack(self.states, dim=1)
            self._stacked_valid = True
        
        return self._stacked_states
    
    def __len__(self) -> int:
        return len(self.states)


class ResonantAttentionHead(nn.Module):
    """
    Single attention head using resonance-based query/key matching.
    
    KEY INSIGHT: Both query and keys should be Euler-transformed to enable
    proper resonance matching. When query and key have similar phases, their
    dot product is large (constructive interference). Different phases → small.
    
    Query/Key computation:
        θ_q = x_patch / (1 + |w_q|) + b_q + t·φ
        θ_k = k_patch / (1 + |w_k|) + b_k  (keys use cached time)
        
        q = [cos(θ_q), sin(θ_q)]  -- keep complex structure!
        k = [cos(θ_k), sin(θ_k)]
        
        score = q · k = cos(θ_q)·cos(θ_k) + sin(θ_q)·sin(θ_k) = cos(θ_q - θ_k)
        
    This creates true phase-based matching: similar phases → high score.
    """
    
    def __init__(
        self,
        d_model: int,
        d_head: int,
        head_idx: int,
        lut_resolution: int = 4096,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_head = d_head
        self.head_idx = head_idx
        self.start_idx = head_idx * d_head
        self.end_idx = (head_idx + 1) * d_head
        
        # Query transformation parameters
        self.w_query = nn.Parameter(torch.randn(d_head) * 0.02)
        self.b_query = nn.Parameter(torch.zeros(d_head))
        
        # Key transformation parameters (separate from query!)
        self.w_key = nn.Parameter(torch.randn(d_head) * 0.02)
        self.b_key = nn.Parameter(torch.zeros(d_head))
        
        # Scale for attention scores (2*d_head because we concat cos and sin)
        self.scale = math.sqrt(2 * d_head)
        
        self.lut_resolution = lut_resolution
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(self.lut_resolution, device)
        return self._lut
    
    def compute_query(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute query vector using Euler transform.
        
        Returns [cos(θ), sin(θ)] concatenated to preserve complex structure.
        """
        lut = self._get_lut(x.device)
        
        # Extract this head's patch
        x_patch = x[:, self.start_idx:self.end_idx]
        
        # Euler transform
        wavelength = 1.0 + self.w_query.abs()
        if t.dim() == 0:
            t_phi = t * PHI
        elif t.dim() == 1:
            t_phi = t.unsqueeze(-1) * PHI
        else:
            t_phi = t * PHI
        theta = x_patch / wavelength + self.b_query + t_phi
        
        # Keep cos and sin separate for proper resonance matching
        sin_q, cos_q = lut.lookup_sin_cos(theta)
        query = torch.cat([cos_q, sin_q], dim=-1)  # (batch, 2*d_head)
        
        return query
    
    def compute_keys(
        self,
        cached_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute key vectors for cached states using Euler transform.
        
        Returns [cos(θ), sin(θ)] concatenated for each historical state.
        """
        lut = self._get_lut(cached_states.device)
        
        # Extract patches: (batch, history, d_head)
        k_patches = cached_states[:, :, self.start_idx:self.end_idx]
        
        # Key transform (no time term - keys are static representations)
        wavelength = 1.0 + self.w_key.abs()
        theta = k_patches / wavelength + self.b_key
        
        sin_k, cos_k = lut.lookup_sin_cos(theta)
        keys = torch.cat([cos_k, sin_k], dim=-1)  # (batch, history, 2*d_head)
        
        return keys
    
    def forward(
        self,
        x: torch.Tensor,
        cached_states: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention output with resonance-based matching.
        
        The dot product of [cos θ_q, sin θ_q] · [cos θ_k, sin θ_k] equals
        cos(θ_q - θ_k) by the cosine difference identity. This creates
        true phase-based attention: similar phases → high attention.
        """
        batch_size, history_len, _ = cached_states.shape
        
        # Transform query and keys
        query = self.compute_query(x, t)  # (batch, 2*d_head)
        keys = self.compute_keys(cached_states)  # (batch, history, 2*d_head)
        
        # Attention scores via dot product
        # This computes cos(θ_q - θ_k) due to trig identity!
        scores = torch.bmm(
            query.unsqueeze(1),  # (batch, 1, 2*d_head)
            keys.transpose(1, 2)  # (batch, 2*d_head, history)
        ).squeeze(1)  # (batch, history)
        
        # Scale and softmax
        scores = scores / self.scale
        weights = F.softmax(scores, dim=-1)
        
        # Retrieve weighted sum of FULL states
        output = torch.bmm(
            weights.unsqueeze(1),
            cached_states
        ).squeeze(1)
        
        return output, weights


class ResonantAttention(nn.Module):
    """
    Multi-head Resonant Attention layer.
    
    Combines multiple attention heads, each computing Euler-transformed
    queries to attend to historical states. The interference property
    of high-dimensional vectors provides natural gating:
    
    - Coherent retrieval → high-magnitude output
    - Diffuse retrieval → phase cancellation → near-zero output
    
    Architecture:
        1. Each of n_heads computes query via Euler transform on its patch
        2. Queries attend to full historical states via softmax
        3. All head outputs are summed (interference!)
        4. Final projection and residual connection
        
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads (d_model must be divisible)
        lut_resolution: LUT resolution for Euler computation
        dropout: Dropout probability (default 0.0)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        lut_resolution: int = 4096,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
        # Create attention heads
        self.heads = nn.ModuleList([
            ResonantAttentionHead(d_model, self.d_head, i, lut_resolution)
            for i in range(n_heads)
        ])
        
        # Output projection: context → stream
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Optional dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.out_proj.weight, std=0.02)
        # Scale down output for stable residual addition
        with torch.no_grad():
            self.out_proj.weight.mul_(0.5)
    
    def forward(
        self,
        x: torch.Tensor,
        cache: StateCache,
        t: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass with attention to cached states.
        
        Args:
            x: Current state (batch, d_model)
            cache: StateCache with historical states
            t: Current timestep
            return_weights: Whether to return attention weights for analysis
            
        Returns:
            output: Transformed state (batch, d_model)
            weights: List of attention weights per head (if return_weights)
        """
        # If no history, return input unchanged (no context to attend to)
        cached_states = cache.get_stacked_states()
        if cached_states is None or len(cache) == 0:
            if return_weights:
                return x, None
            return x, None
        
        # Collect outputs from all heads
        head_outputs = []
        all_weights = [] if return_weights else None
        
        for head in self.heads:
            output, weights = head(x, cached_states, t)
            head_outputs.append(output)
            if return_weights:
                all_weights.append(weights)
        
        # Sum all head outputs (interference happens here!)
        # Coherent heads reinforce, incoherent heads cancel
        context = torch.stack(head_outputs, dim=0).sum(dim=0)  # (batch, d_model)
        
        # Project and apply dropout
        context = self.dropout(self.out_proj(context))
        
        # Residual connection: x = x + attention(x)
        output = x + context
        
        return output, all_weights
    
    def get_attention_stats(
        self,
        weights_list: List[torch.Tensor],
    ) -> dict:
        """
        Compute statistics about attention patterns for analysis.
        
        Args:
            weights_list: List of attention weights from each head
            
        Returns:
            Dictionary with entropy, max_weight, etc. per head
        """
        stats = {}
        for i, weights in enumerate(weights_list):
            # Entropy of attention distribution (higher = more diffuse)
            entropy = -(weights * (weights + 1e-10).log()).sum(dim=-1).mean()
            # Maximum attention weight (higher = more focused)
            max_weight = weights.max(dim=-1)[0].mean()
            # Effective number of attended positions
            effective_n = (1.0 / (weights ** 2).sum(dim=-1)).mean()
            
            stats[f'head_{i}'] = {
                'entropy': entropy.item(),
                'max_weight': max_weight.item(),
                'effective_n': effective_n.item(),
            }
        return stats


class ResonantBlock(nn.Module):
    """
    A single Resonant Block = ResonantAttention + ResonantLayer
    
    Analogous to Transformer block = Attention + FFN
    
    NOTE: This block operates on COLLAPSED (scalar) states for attention,
    because attention needs to compare against cached historical states.
    The resonant layer here uses a simplified scalar interface.
    
    For full complex-valued processing, use the main RINModel.
    
    Architecture:
        x → ResonantAttention(x, cache, t) → ResonantLayer(x, t) → output
        
    Both sublayers use residual connections.
    """
    
    def __init__(
        self,
        d_model: int,
        num_neurons: int,
        n_heads: int = 8,
        lut_resolution: int = 4096,
        use_swish: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_neurons = num_neurons
        self.use_swish = use_swish
        
        self.attention = ResonantAttention(
            d_model=d_model,
            n_heads=n_heads,
            lut_resolution=lut_resolution,
            dropout=dropout,
        )
        
        # Resonant layer with proper per-neuron, per-dimension interference analysis
        # Each neuron has d_model wavelengths and d_model phase offsets
        self.W = nn.Parameter(torch.randn(num_neurons, d_model) * 0.02)  # wavelength
        self.B = nn.Parameter(torch.zeros(num_neurons, d_model))  # phase offset
        self.proj_real = nn.Linear(num_neurons, d_model, bias=False)
        self.proj_imag = nn.Linear(num_neurons, d_model, bias=False)
        
        self.lut_resolution = lut_resolution
        self._lut = None
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(self.lut_resolution, device)
        return self._lut
    
    def resonant_forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Resonant layer with proper per-dimension interference analysis.
        
        Each neuron computes phase per input dimension, then sums the
        interference across dimensions (the key insight!).
        
        θ[n,d] = x[d] / (1 + |W[n,d]|) + B[n,d] + t
        cos_sum[n] = Σ_d cos(θ[n,d])
        sin_sum[n] = Σ_d sin(θ[n,d])
        """
        lut = self._get_lut(x.device)
        
        # x: (batch, d_model) -> expand to (batch, 1, d_model)
        # W, B: (num_neurons, d_model)
        x_expanded = x.unsqueeze(1)  # (batch, 1, d_model)
        wavelength = 1.0 + self.W.abs()  # (num_neurons, d_model)
        
        if t.dim() == 1:
            t = t.unsqueeze(-1).unsqueeze(-1)  # (batch, 1, 1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)
        
        # θ[b,n,d] = x[b,d] / wavelength[n,d] + B[n,d] + t
        theta = x_expanded / wavelength + self.B + t  # (batch, num_neurons, d_model)
        
        # Euler decomposition
        sin_theta, cos_theta = lut.lookup_sin_cos(theta)
        
        # INTERFERENCE SUM: Sum across d_model dimension
        # This is the TRUE interference - waves combining constructively/destructively
        cos_sum = cos_theta.sum(dim=-1)  # (batch, num_neurons)
        sin_sum = sin_theta.sum(dim=-1)  # (batch, num_neurons)
        
        # Project back to d_model
        output = self.proj_real(cos_sum) + self.proj_imag(sin_sum)
        
        if self.use_swish:
            output = F.silu(output)
        return output
    
    def forward(
        self,
        x: torch.Tensor,
        cache: StateCache,
        t: torch.Tensor,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through attention then resonant layer.
        
        Args:
            x: Input state (batch, d_model) - scalar/collapsed
            cache: State history cache
            t: Timestep (should be scaled by PHI for resonant layer)
            return_weights: Whether to return attention weights
            
        Returns:
            output: Transformed state (batch, d_model)
            weights: Attention weights if requested
        """
        # Attention sublayer
        x, weights = self.attention(x, cache, t, return_weights)
        
        # Resonant sublayer with residual (scalar interface)
        x = x + self.resonant_forward(x, t)
        
        return x, weights


class RINAttentionModel(nn.Module):
    """
    Resonant Interference Network with Attention - Full Model
    
    Combines the recurrent Euler state transformation with resonant
    attention blocks for enhanced long-range memory.
    
    Architecture:
        1. Token embedding → (w, b) pairs
        2. Euler state transformation: h_real, h_imag = euler(h, w, b, t)
        3. Cache state for attention
        4. Process through ResonantBlocks (attention + resonant layers)
        5. Output projection
        
    The state cache enables attention to historical states while
    maintaining the recurrent nature of the architecture.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension  
        num_layers: Number of ResonantBlocks
        num_neurons: Neurons per resonant layer
        n_heads: Attention heads per block
        lut_resolution: LUT resolution
        use_swish: Whether to use swish activation
        max_cache_len: Maximum state cache length (None = unlimited)
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 128,
        num_layers: int = 2,
        num_neurons: int = 256,
        n_heads: int = 8,
        lut_resolution: int = 4096,
        use_swish: bool = True,
        max_cache_len: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_neurons = num_neurons
        self.n_heads = n_heads
        self.lut_resolution = lut_resolution
        self.use_swish = use_swish
        self.max_cache_len = max_cache_len
        
        # Token embeddings: 2*d_model for (w, b) pairs
        self.token_embedding = nn.Embedding(vocab_size, 2 * d_model)
        
        # Resonant blocks (attention + resonant layers)
        self.blocks = nn.ModuleList([
            ResonantBlock(
                d_model=d_model,
                num_neurons=num_neurons,
                n_heads=n_heads,
                lut_resolution=lut_resolution,
                use_swish=use_swish,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)
        
        self._lut = None
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.output_proj.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight.mul_(0.5)
    
    def _get_lut(self, device):
        if self._lut is None or self._lut.sin_table.device != device:
            self._lut = get_global_lut(self.lut_resolution, device)
        return self._lut
    
    def init_hidden(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize hidden state."""
        h_real = torch.zeros(batch_size, self.d_model, device=device)
        h_imag = torch.zeros(batch_size, self.d_model, device=device)
        return h_real, h_imag
    
    def create_cache(self, training: Optional[bool] = None) -> StateCache:
        """
        Create a new state cache.
        
        Args:
            training: Whether to preserve gradients. If None, uses self.training.
        
        Returns:
            StateCache configured for training or inference mode.
        """
        if training is None:
            training = self.training
        return StateCache(max_len=self.max_cache_len, training=training)
    
    def euler_transform(
        self,
        h_real: torch.Tensor,
        h_imag: torch.Tensor,
        w: torch.Tensor,
        b: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Euler-based state transformation with separate theta propagation.
        
        CRITICAL: Compute SEPARATE thetas for real and imaginary components,
        then combine via proper complex multiplication. This preserves phase
        information and distinguishes constructive from destructive interference.
        
        θ_real = h_real / (1 + |w|) + b + t·φ
        θ_imag = h_imag / (1 + |w|) + b + t·φ
        
        Complex multiplication: e^(iθ_real) × e^(iθ_imag)
        h_real_new = cos(θ_real)·cos(θ_imag) - sin(θ_real)·sin(θ_imag)
        h_imag_new = cos(θ_real)·sin(θ_imag) + sin(θ_real)·cos(θ_imag)
        """
        lut = self._get_lut(h_real.device)
        
        wavelength = 1.0 + w.abs()
        t_phi = t.unsqueeze(-1) * PHI if t.dim() >= 1 else t * PHI
        
        # Separate theta computation preserves real/imag distinction
        theta_real = h_real / wavelength + b + t_phi
        theta_imag = h_imag / wavelength + b + t_phi
        
        # Euler decomposition for each component
        sin_real, cos_real = lut.lookup_sin_cos(theta_real)
        sin_imag, cos_imag = lut.lookup_sin_cos(theta_imag)
        
        # Complex multiplication: (cos_r + i·sin_r) × (cos_i + i·sin_i)
        # Preserves BOTH gradient paths through h_real and h_imag
        h_real_new = cos_real * cos_imag - sin_real * sin_imag
        h_imag_new = cos_real * sin_imag + sin_real * cos_imag
        
        return h_real_new, h_imag_new
    
    def forward(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[StateCache] = None,
        t_start: int = 0,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], StateCache]:
        """
        Forward pass with attention to state history.
        
        Args:
            input_ids: Token IDs (batch, seq_len)
            hidden: Initial (h_real, h_imag) or None
            cache: Existing StateCache or None (creates new)
            t_start: Starting timestep
            return_attention: Whether to return attention weights
            
        Returns:
            logits: (batch, seq_len, vocab_size)
            final_hidden: (h_real, h_imag)
            cache: Updated StateCache
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        if hidden is None:
            h_real, h_imag = self.init_hidden(batch_size, device)
        else:
            h_real, h_imag = hidden
        
        if cache is None:
            cache = self.create_cache(training=self.training)
        
        # Get embeddings
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        # Pre-compute timesteps
        t_indices = torch.arange(seq_len, device=device, dtype=torch.float32) + t_start
        
        all_logits = []
        all_attention = [] if return_attention else None
        
        for t_idx in range(seq_len):
            w_t = w_emb[:, t_idx]
            b_t = b_emb[:, t_idx]
            t_val = t_indices[t_idx].expand(batch_size)
            
            # Euler state transformation
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_val)
            
            # Combine for processing
            x = h_real + h_imag
            
            # Cache current state BEFORE attention (with embeddings for gradient)
            # The embeddings retain gradients, state is detached
            cache.append(x, w_t, b_t)
            
            # Process through blocks
            t_phi = t_val * PHI
            block_attention = []
            for block in self.blocks:
                x, weights = block(x, cache, t_phi, return_weights=return_attention)
                if return_attention and weights is not None:
                    block_attention.append(weights)
            
            if return_attention:
                all_attention.append(block_attention)
            
            logits = self.output_proj(x)
            all_logits.append(logits)
        
        output = torch.stack(all_logits, dim=1)
        
        if return_attention:
            return output, (h_real, h_imag), cache, all_attention
        return output, (h_real, h_imag), cache
    
    def compute_loss(
        self,
        input_ids: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache: Optional[StateCache] = None,
        t_start: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], StateCache]:
        """Compute next-token prediction loss."""
        logits, hidden, cache = self.forward(input_ids, hidden, cache, t_start)
        
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_labels.view(-1),
        )
        
        return loss, logits, hidden, cache
    
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
        
        h_real, h_imag = self.init_hidden(batch_size, device)
        cache = self.create_cache(training=False)  # Inference mode - detach states
        
        # Process prompt
        embeddings = self.token_embedding(input_ids)
        w_emb = embeddings[:, :, :self.d_model]
        b_emb = embeddings[:, :, self.d_model:]
        
        t = 0
        for i in range(input_ids.shape[1]):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_emb[:, i], b_emb[:, i], t_tensor)
            x = h_real + h_imag
            cache.append(x, w_emb[:, i], b_emb[:, i])
            t += 1
        
        # Get initial logits
        x = h_real + h_imag
        t_phi = torch.full((batch_size,), (t - 1) * PHI, device=device, dtype=torch.float32)
        for block in self.blocks:
            x, _ = block(x, cache, t_phi)
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
            
            # Update
            emb = self.token_embedding(next_token.squeeze(-1))
            w_t = emb[:, :self.d_model]
            b_t = emb[:, self.d_model:]
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            h_real, h_imag = self.euler_transform(h_real, h_imag, w_t, b_t, t_tensor)
            x = h_real + h_imag
            cache.append(x, w_t, b_t)
            
            t_phi = t_tensor * PHI
            for block in self.blocks:
                x, _ = block(x, cache, t_phi)
            logits = self.output_proj(x)
            t += 1
        
        return generated
    
    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
    
    def __repr__(self) -> str:
        return (
            f"RINAttentionModel(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  d_model={self.d_model},\n"
            f"  num_layers={self.num_layers},\n"
            f"  num_neurons={self.num_neurons},\n"
            f"  n_heads={self.n_heads},\n"
            f"  use_swish={self.use_swish},\n"
            f"  max_cache_len={self.max_cache_len},\n"
            f"  φ={PHI:.6f} (golden ratio),\n"
            f"  params={self.get_num_params():,}\n"
            f")"
        )
